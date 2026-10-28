"""
scripts/retrieval/plain_rag.py

Pure vector-only retrieval — Qdrant Cloud edition.

No Neo4j, no query parsing, no graph expansion, no fusion boosts.
This is the baseline for comparing against GraphRAG.

Returns the same response shape as GraphRAGHybridRetriever.search()
so the API layer can swap modes transparently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

STATUTE_COLLECTION  = "legal_statute_chunks"
JUDGMENT_COLLECTION = "legal_judgment_chunks"
EMBED_MODEL         = "all-MiniLM-L6-v2"

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "")
    key = os.getenv("QDRANT_API_KEY", "")
    if not url or not key:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set in the environment."
        )
    return QdrantClient(url=url, port=443, api_key=key, prefer_grpc=False)


class PlainRAGRetriever:
    """
    Qdrant-only retrieval.  No graph, no reranking — pure cosine similarity.

    The results use the same dict shape so the FastAPI layer can call
    either retriever interchangeably.
    """

    def __init__(self) -> None:
        self._qdrant = _qdrant_client()
        self._embed  = _get_embed_model()

        existing = {c.name for c in self._qdrant.get_collections().collections}
        if STATUTE_COLLECTION not in existing:
            raise RuntimeError(
                f"Qdrant collection '{STATUTE_COLLECTION}' not found. "
                "Run migrate_chroma_to_qdrant.py first."
            )
        self._judgment_available = JUDGMENT_COLLECTION in existing

    # ------------------------------------------------------------------
    # Public API — same signature as GraphRAGHybridRetriever.search()
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k_statutes: int = 3,
        top_k_judgments: int = 3,
    ) -> Dict[str, Any]:
        """Return statute and judgment results using pure vector similarity."""
        query_vector = self._embed.encode(query).tolist()

        statute_results  = self._query_statutes(query_vector, top_k=top_k_statutes)
        judgment_results = self._query_judgments(query_vector, top_k=top_k_judgments)

        return {
            "query":            query,
            "parsed_intent":    {},
            "graph_expansion":  {},
            "statute_results":  statute_results,
            "judgment_results": judgment_results,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _query_statutes(self, query_vector: list, top_k: int) -> List[Dict[str, Any]]:
        results = self._qdrant.query_points(
            collection_name=STATUTE_COLLECTION,
            query=query_vector,
            limit=max(top_k, 1),
            with_payload=True,
        )

        out: List[Dict[str, Any]] = []
        for i, pt in enumerate(results.points):
            payload    = pt.payload or {}
            chunk_id   = payload.get("chunk_id", str(pt.id))
            section_id = payload.get("section_id") or payload.get("act_id", "")
            distance   = round(1.0 - float(pt.score), 6)
            out.append({
                "id":              chunk_id,
                "section_id":      section_id,
                "act":             payload.get("act_id") or payload.get("act"),
                "section_number":  payload.get("section_number"),
                "title":           payload.get("title"),
                "text":            payload.get("text", ""),
                "metadata":        payload,
                "vector_distance": distance,
                "vector_rank":     i + 1,
                "type":            "statute",
                "source":          "vector",
                # Neutral scores — no boost applied
                "rrf_score":       round(1.0 / (60 + i + 1), 6),
                "boost_applied":   0.0,
                "boost_reasons":   [],
            })
        return out

    def _query_judgments(self, query_vector: list, top_k: int) -> List[Dict[str, Any]]:
        if not self._judgment_available:
            return []

        results = self._qdrant.query_points(
            collection_name=JUDGMENT_COLLECTION,
            query=query_vector,
            limit=max(top_k, 1),
            with_payload=True,
        )

        out: List[Dict[str, Any]] = []
        for i, pt in enumerate(results.points):
            payload  = pt.payload or {}
            chunk_id = payload.get("chunk_id", str(pt.id))
            distance = round(1.0 - float(pt.score), 6)
            out.append({
                "id":               chunk_id,
                "case_id":          payload.get("case_id"),
                "chunk_type":       payload.get("chunk_type"),
                "legal_references": payload.get("legal_references"),
                "text":             payload.get("text", ""),
                "metadata":         payload,
                "vector_distance":  distance,
                "vector_rank":      i + 1,
                "rrf_score":        round(1.0 / (60 + i + 1), 6),
                "type":             "judgment",
            })
        return out

    def close(self) -> None:
        """No persistent connections to close (QdrantClient is stateless)."""
        pass
