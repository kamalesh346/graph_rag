"""
plain_rag.py

Pure vector-only retrieval using ChromaDB.
No Neo4j, no query parsing, no graph expansion, no fusion boosts.
This is the baseline for comparing against GraphRAG.

Returns the same response shape as GraphRAGHybridRetriever.search()
so the API layer can swap modes transparently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb

CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
STATUTE_COLLECTION = "legal_statute_chunks"
JUDGMENT_COLLECTION = "legal_judgment_chunks"


class PlainRAGRetriever:
    """
    ChromaDB-only retrieval.  No graph, no reranking — pure cosine similarity.

    This is the control group for the GraphRAG comparison.
    The results use the same dict shape so the FastAPI layer can call
    either retriever interchangeably.
    """

    def __init__(self) -> None:
        if not CHROMA_DB_DIR.exists():
            raise FileNotFoundError(f"ChromaDB directory not found at {CHROMA_DB_DIR}")

        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.statute_col = self.chroma_client.get_collection(name=STATUTE_COLLECTION)

        try:
            self.judgment_col = self.chroma_client.get_collection(name=JUDGMENT_COLLECTION)
        except Exception:
            self.judgment_col = None

    # ------------------------------------------------------------------
    # Public API — same signature as GraphRAGHybridRetriever.search()
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k_statutes: int = 3,
        top_k_judgments: int = 3,
    ) -> Dict[str, Any]:
        """
        Return statute and judgment results using pure vector similarity only.

        The result dict uses the same keys as GraphRAGHybridRetriever so the
        FastAPI _provision_payload / _judgment_payload helpers work unchanged.
        """
        statute_results = self._query_statutes(query, top_k=top_k_statutes)
        judgment_results = self._query_judgments(query, top_k=top_k_judgments)

        return {
            "query": query,
            # No parsed_intent or graph_expansion — this is plain RAG
            "parsed_intent": {},
            "graph_expansion": {},
            "statute_results": statute_results,
            "judgment_results": judgment_results,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _query_statutes(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        n_results = min(top_k, self.statute_col.count())
        if n_results == 0:
            return []

        res = self.statute_col.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        results: List[Dict[str, Any]] = []
        if res and res["ids"] and res["ids"][0]:
            for i, doc_id in enumerate(res["ids"][0]):
                meta = res["metadatas"][0][i]
                section_id = meta.get("section_id", doc_id)
                results.append({
                    "id": doc_id,
                    "section_id": section_id,
                    "act": meta.get("act"),
                    "section_number": meta.get("section_number"),
                    "title": meta.get("title"),
                    "text": res["documents"][0][i],
                    "metadata": meta,
                    "vector_distance": res["distances"][0][i],
                    "vector_rank": i + 1,
                    "type": "statute",
                    "source": "vector",
                    # Neutral scores — no boost applied
                    "rrf_score": round(1.0 / (60 + i + 1), 6),
                    "boost_applied": 0.0,
                    "boost_reasons": [],
                })

        return results

    def _query_judgments(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.judgment_col:
            return []

        n_results = min(top_k, self.judgment_col.count())
        if n_results == 0:
            return []

        res = self.judgment_col.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        results: List[Dict[str, Any]] = []
        if res and res["ids"] and res["ids"][0]:
            for i, doc_id in enumerate(res["ids"][0]):
                meta = res["metadatas"][0][i]
                results.append({
                    "id": doc_id,
                    "case_id": meta.get("case_id"),
                    "chunk_type": meta.get("chunk_type"),
                    "legal_references": meta.get("legal_references"),
                    "text": res["documents"][0][i],
                    "metadata": meta,
                    "vector_distance": res["distances"][0][i],
                    "vector_rank": i + 1,
                    "rrf_score": round(1.0 / (60 + i + 1), 6),
                    "type": "judgment",
                })

        return results

    def close(self) -> None:
        """No persistent connections to close (ChromaDB client is stateless)."""
        pass
