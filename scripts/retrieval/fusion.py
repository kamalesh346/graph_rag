"""
fusion.py

Rule‑based fusion of retrieval candidates from the vector (ChromaDB) and
graph (Neo4j) branches.

The design treats graph traversal results **as a signal** that boosts
already‑retrieved vector candidates rather than inserting raw graph dicts.

Boost constants are tuned for the current benchmark – they can be tweaked
later if needed.
"""

import re
from typing import Any, Dict, List, Set

# -----------------------------------------------------------------------
# Boost configuration (tweakable)
# -----------------------------------------------------------------------
GRAPH_BOOST = 0.50          # extra score when a statute appears in graph traversal
STATUTE_ID_BOOST = 0.20    # when the query contains a statutory identifier
CASE_KEYWORD_BOOST = 0.10  # when the query looks case‑oriented


# -----------------------------------------------------------------------
# Query‑pattern detectors
# -----------------------------------------------------------------------
def _has_statute_id(query: str) -> bool:
    """Detect patterns like IPC:300, CRPC:313, BNS:101 in the query."""
    return bool(re.search(r"[A-Z]{2,5}:\d+[A-Z]?", query))


def _has_case_keywords(query: str) -> bool:
    """Return True if the query looks case‑oriented."""
    keywords = ["case", "judgment", "precedent", "court", "conviction", "acquittal"]
    lower = query.lower()
    return any(k in lower for k in keywords)


# -----------------------------------------------------------------------
# Graph signal extractor
# -----------------------------------------------------------------------
def _extract_graph_statute_ids(graph_hits: List[Dict[str, Any]]) -> Set[str]:
    """Flatten all statute IDs mentioned across every graph traversal result."""
    ids: Set[str] = set()
    for hit in graph_hits:
        ids.update(hit.get("statutes", set()))
    return ids


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------
def fuse_candidates(
    vector_hits: List[Dict[str, Any]],
    graph_hits: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """Re‑rank *vector_hits* using graph traversal as a boosting signal.

    Parameters
    ----------
    vector_hits:
        Candidates from ChromaDB / the hybrid reranker.  Each dict must
        contain either ``rrf_score`` or ``similarity`` (higher = better) and a
        ``section_id`` field.
    graph_hits:
        Output of ``traverse_anchor`` calls.  Each dict may contain:
        ``statutes`` (set of IDs), ``judgments`` (set), ``cited_cases`` (set).
        These are **only** used as signals – they are **not** inserted into the
        result list.
    query:
        The original user query string, used for pattern‑based boosts.

    Returns
    -------
    List[Dict]
        The same vector candidates, re‑ranked by ``final_score``.  Each entry
        gains the keys ``source``, ``final_score``, and ``fusion_boosts`` (for
        inspection / tracing).
    """
    # Collect statutes that the graph says are relevant – this is the boost
    # signal for matching vector candidates.
    graph_statute_ids = _extract_graph_statute_ids(graph_hits)
    has_id = _has_statute_id(query)
    has_case = _has_case_keywords(query)

    merged: List[Dict[str, Any]] = []
    for hit in vector_hits:
        # Base score prefers a pre‑computed RRF score; otherwise fall back to the
        # raw similarity.
        base = hit.get("rrf_score", hit.get("similarity", 0.0))
        boosts: Dict[str, float] = {}

        sec_id = hit.get("section_id", "")
        if sec_id and sec_id in graph_statute_ids:
            boosts["graph_traversal"] = GRAPH_BOOST
        if has_id:
            boosts["statute_id_in_query"] = STATUTE_ID_BOOST
        if has_case:
            boosts["case_keyword_in_query"] = CASE_KEYWORD_BOOST

        final = round(base + sum(boosts.values()), 6)
        merged.append({
            **hit,
            "source": hit.get("source", "vector"),
            "final_score": final,
            "fusion_boosts": boosts,
        })

    merged.sort(key=lambda x: x["final_score"], reverse=True)
    return merged
