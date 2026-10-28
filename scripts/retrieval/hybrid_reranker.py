"""
scripts/retrieval/hybrid_reranker.py

GraphRAG Hybrid Retriever — Qdrant Cloud edition.

Vector store: Qdrant Cloud (was: local ChromaDB)
Embedding  : all-MiniLM-L6-v2 via sentence-transformers (unchanged)
All other logic (Neo4j, RRF fusion, metadata boosts) is unchanged.
"""

import sys
import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
RETRIEVAL_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Set

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from scripts.retrieval.query_parser import LegalQueryParser
from scripts.retrieval.graph_retriever import Neo4jGraphRetriever
from scripts.retrieval.fusion import fuse_candidates

STATUTE_COLLECTION  = "legal_statute_chunks"
JUDGMENT_COLLECTION = "legal_judgment_chunks"
EMBED_MODEL         = "all-MiniLM-L6-v2"

# ── Shared embedding model (loaded once per process) ──────────────────────────
_embed_model: SentenceTransformer | None = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _qdrant_client() -> QdrantClient:
    """Build a Qdrant Cloud client from environment variables."""
    url = os.getenv("QDRANT_URL", "")
    key = os.getenv("QDRANT_API_KEY", "")
    if not url or not key:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set in the environment."
        )
    return QdrantClient(url=url, port=443, api_key=key, prefer_grpc=False)


def _qdrant_to_statute(point, rank_idx: int) -> Dict[str, Any]:
    """Map a Qdrant ScoredPoint to the statute dict shape used by the pipeline."""
    payload   = point.payload or {}
    chunk_id  = payload.get("chunk_id", str(point.id))
    section_id = payload.get("section_id") or payload.get("act_id", "")
    # Qdrant score is cosine similarity (higher = better).
    # Store 1 - score as vector_distance so existing display code works.
    distance = round(1.0 - float(point.score), 6)
    return {
        "id":              chunk_id,
        "section_id":      section_id,
        "act":             payload.get("act_id") or payload.get("act"),
        "section_number":  payload.get("section_number"),
        "title":           payload.get("title"),
        "text":            payload.get("text", ""),
        "metadata":        payload,
        "vector_distance": distance,
        "vector_rank":     rank_idx + 1,
        "type":            "statute",
        "source":          "vector",
    }


def _qdrant_to_judgment(point, rank_idx: int, extracted_secs: List[str], rrf_k: int) -> Dict[str, Any]:
    """Map a Qdrant ScoredPoint to the judgment dict shape used by the pipeline."""
    payload  = point.payload or {}
    chunk_id = payload.get("chunk_id", str(point.id))
    ref_str  = str(payload.get("legal_references", ""))
    distance = round(1.0 - float(point.score), 6)

    sec_match_boost = sum(0.5 for sec in extracted_secs if sec in ref_str)
    rrf_score = (1.0 / (rrf_k + rank_idx + 1)) + sec_match_boost

    return {
        "id":               chunk_id,
        "case_id":          payload.get("case_id"),
        "chunk_type":       payload.get("chunk_type"),
        "legal_references": payload.get("legal_references"),
        "text":             payload.get("text", ""),
        "metadata":         payload,
        "vector_distance":  distance,
        "vector_rank":      rank_idx + 1,
        "rrf_score":        round(rrf_score, 6),
        "type":             "judgment",
    }


class GraphRAGHybridRetriever:
    """Integrated GraphRAG Hybrid Search Engine — Qdrant Cloud vector store."""

    def __init__(self):
        self.parser         = LegalQueryParser()
        self.graph_retriever = Neo4jGraphRetriever()
        self._qdrant        = _qdrant_client()
        self._embed         = _get_embed_model()

        # Confirm collections are reachable (read-only check)
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if STATUTE_COLLECTION not in existing:
            raise RuntimeError(
                f"Qdrant collection '{STATUTE_COLLECTION}' not found. "
                "Run migrate_chroma_to_qdrant.py first."
            )
        self.judgment_col_available = JUDGMENT_COLLECTION in existing

    # Public property kept for admin_service direct-lookup compatibility
    @property
    def vector_retriever(self):
        """Compatibility shim: returns self so callers can access .statute_collection."""
        return self

    @property
    def statute_collection(self):
        """Compatibility shim for code that calls retriever.vector_retriever.statute_collection."""
        return _QdrantCollectionProxy(self._qdrant, STATUTE_COLLECTION)

    def search(
        self,
        query: str,
        top_k_statutes: int = 3,
        top_k_judgments: int = 3,
        rrf_k: int = 60,
    ) -> Dict[str, Any]:
        """
        GraphRAG hybrid retrieval.
        1. Parse legal query.
        2. Neo4j graph expansion.
        3. Qdrant vector search (statutes + judgments).
        4. RRF fusion + metadata boosts.
        """
        parsed         = self.parser.parse(query)
        extracted_secs = parsed["extracted_sections"]
        primary_target = parsed["primary_target_section"]

        graph_expansion = self.graph_retriever.expand_sections(extracted_secs)
        all_graph_secs  = set(graph_expansion["all_expanded_sections"])
        corresponds_secs = set(graph_expansion["corresponds_to"])

        # ── Statute vector search ─────────────────────────────────────────────
        query_vector     = self._embed.encode(query).tolist()
        n_fetch_statutes = max(10, top_k_statutes * 3)

        s_results = self._qdrant.query_points(
            collection_name=STATUTE_COLLECTION,
            query=query_vector,
            limit=n_fetch_statutes,
            with_payload=True,
        )

        raw_statute_candidates = [
            _qdrant_to_statute(pt, idx)
            for idx, pt in enumerate(s_results.points)
        ]

        # ── Rerank + RRF for statutes ─────────────────────────────────────────
        reranked_statutes = self._rerank_statutes(
            raw_statute_candidates,
            extracted_secs=extracted_secs,
            primary_target=primary_target,
            corresponds_secs=corresponds_secs,
            all_graph_secs=all_graph_secs,
            rrf_k=rrf_k,
        )

        # ── Neo4j graph fusion ────────────────────────────────────────────────
        if self.graph_retriever.is_connected():
            anchors = self.graph_retriever.find_graph_anchors(query, parsed_query=parsed)
            graph_candidates = []
            for anchor_list in anchors.values():
                for anchor in anchor_list:
                    traversal = self.graph_retriever.traverse_anchor(anchor)
                    graph_candidates.append({
                        "statutes":    traversal.get("statutes", set()),
                        "judgments":   traversal.get("judgments", set()),
                        "cited_cases": traversal.get("cited_cases", set()),
                        "relevance":   0.0,
                    })
            fused_statutes = fuse_candidates(reranked_statutes, graph_candidates, query)
        else:
            fused_statutes = reranked_statutes

        # ── Judgment vector search ────────────────────────────────────────────
        reranked_judgments: List[Dict[str, Any]] = []
        if self.judgment_col_available:
            try:
                n_fetch_j = max(10, top_k_judgments * 3)
                j_results = self._qdrant.query_points(
                    collection_name=JUDGMENT_COLLECTION,
                    query=query_vector,
                    limit=n_fetch_j,
                    with_payload=True,
                )
                reranked_judgments = [
                    _qdrant_to_judgment(pt, idx, extracted_secs, rrf_k)
                    for idx, pt in enumerate(j_results.points)
                ]
                reranked_judgments.sort(key=lambda x: x["rrf_score"], reverse=True)
            except Exception as exc:
                print(f"Warning: Judgment vector retrieval warning: {exc}")

        return {
            "query":            query,
            "parsed_intent":    parsed,
            "graph_expansion":  graph_expansion,
            "statute_results":  fused_statutes[:top_k_statutes],
            "judgment_results": reranked_judgments[:top_k_judgments],
        }

    def _rerank_statutes(
        self,
        candidates: List[Dict[str, Any]],
        extracted_secs: List[str],
        primary_target: Optional[str],
        corresponds_secs: Set[str],
        all_graph_secs: Set[str],
        rrf_k: int,
    ) -> List[Dict[str, Any]]:
        """RRF score + metadata boosts — logic unchanged from ChromaDB version."""
        for item in candidates:
            sec_id   = item["section_id"]
            vec_rank = item["vector_rank"]

            base_rrf = 1.0 / (rrf_k + vec_rank)
            boost    = 0.0
            reasons  = []

            if primary_target and sec_id == primary_target:
                boost += 2.0
                reasons.append(f"Exact Target Section Match ({sec_id})")
            elif sec_id in extracted_secs:
                boost += 1.5
                reasons.append(f"Extracted Query Section Match ({sec_id})")

            if sec_id in corresponds_secs:
                boost += 1.0
                reasons.append(f"Graph Act-Correspondence Match ({sec_id})")
            elif sec_id in all_graph_secs:
                boost += 0.5
                reasons.append(f"Graph Related Section ({sec_id})")

            item["rrf_score"]    = round(base_rrf + boost, 6)
            item["boost_applied"] = round(boost, 4)
            item["boost_reasons"] = reasons

        candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        return candidates

    def close(self):
        if self.graph_retriever:
            self.graph_retriever.close()


class _QdrantCollectionProxy:
    """
    Minimal proxy that mimics the ChromaDB collection.get() interface
    used by backend/main.py for direct section lookup by metadata filter.
    """

    def __init__(self, qdrant: QdrantClient, collection_name: str):
        self._qdrant = qdrant
        self._name   = collection_name

    def get(self, where: dict, include: list | None = None) -> dict:
        """
        Emulates: col.get(where={"section_id": sid}, include=["documents","metadatas"])

        Accepts ChromaDB-style `where` dict with either:
          {"section_id": "BNS:115"}           — direct equality
          {"section_id": {"$eq": "BNS:115"}}  — ChromaDB operator syntax
        """
        # Normalise filter value
        field_filters = []
        for key, val in where.items():
            if isinstance(val, dict):
                # ChromaDB operator syntax: {"$eq": "..."}
                val = val.get("$eq", val.get("$in", [None])[0] if "$in" in val else None)
            if val is not None:
                field_filters.append(
                    FieldCondition(key=key, match=MatchValue(value=val))
                )

        scroll_filter = Filter(must=field_filters) if field_filters else None

        results, _ = self._qdrant.scroll(
            collection_name=self._name,
            scroll_filter=scroll_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            return {"ids": [], "documents": [[]], "metadatas": [{}]}

        pt      = results[0]
        payload = pt.payload or {}
        text    = payload.get("text", "")
        meta    = {k: v for k, v in payload.items() if k != "text"}

        # Return shape compatible with what main.py expects from chroma col.get()
        return {
            "ids":       [payload.get("chunk_id", str(pt.id))],
            "documents": [[text]],
            "metadatas": [meta],
        }


if __name__ == "__main__":
    retriever = GraphRAGHybridRetriever()
    queries = [
        "What are the exceptions to murder under Section 300?",
        "Punishment for culpable homicide not amounting to murder under Section 304 or BNS 105",
        "Procedure for statement examination under CrPC 313 or BNSS 351",
    ]
    for q in queries:
        res = retriever.search(q, top_k_statutes=3, top_k_judgments=2)
        print(f"\n==================================================")
        print(f"QUERY: '{q}'")
        print(f"Target Section Extracted: {res['parsed_intent']['extracted_sections']}")
        print(f"Graph Expansion         : {res['graph_expansion']['corresponds_to']}")
        print("TOP RETRIEVED STATUTES:")
        for idx, s in enumerate(res["statute_results"]):
            print(
                f"  #{idx+1} [{s['section_id']}] "
                f"(Score: {s['rrf_score']} | Boost: {s['boost_applied']} | Dist: {s['vector_distance']:.4f})"
            )
            print(f"      Reasons: {s['boost_reasons']}")
            print(f"      Text   : {s['text'][:110]}...")
    retriever.close()
