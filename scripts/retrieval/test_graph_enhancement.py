# scripts/retrieval/test_graph_enhancement.py
"""Unit tests for the new GraphRAG enhancement components.

Tests exercise:
- Neo4jGraphRetriever.find_graph_anchors
- Neo4jGraphRetriever.traverse_anchor
- fusion.fuse_candidates  (graph-as-signal / boost-only design)

All Neo4j calls are mocked; tests are fast and environment-independent.
"""

from unittest.mock import MagicMock, patch
import pytest


# -----------------------------------------------------------------------
# Mock helpers
# -----------------------------------------------------------------------
class MockRecord:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def data(self):
        return self._data


class MockResult:
    def __init__(self, records):
        self._records = [MockRecord(r) for r in records]

    def __iter__(self):
        return iter(self._records)

    def data(self):
        return [r.data() for r in self._records]

    def single(self):
        return self._records[0] if self._records else None


def _make_retriever_with_session(session_factory):
    """Create a Neo4jGraphRetriever with a mocked session."""
    from scripts.retrieval.graph_retriever import Neo4jGraphRetriever

    mock_driver = MagicMock()
    mock_driver.session = session_factory

    with patch.object(Neo4jGraphRetriever, "_connect",
                      lambda self: setattr(self, "driver", mock_driver)):
        retr = Neo4jGraphRetriever()
    return retr


# -----------------------------------------------------------------------
# 1. find_graph_anchors
# -----------------------------------------------------------------------
def test_find_graph_anchors_basic():
    """Explicit, concept, fulltext, and case anchors are returned."""

    QUERY = "Rape and murder cases, IPC:300"

    def make_session():
        class S:
            def __enter__(self):  return self
            def __exit__(self, *_): return False
            def run(self, query, **params):
                q = " ".join(query.split())
                if "StatuteSection" in q and params.get("sid") == "IPC:300":
                    return MockResult([{"s": {"id": "IPC:300", "text": "Statute text"}}])
                if "LegalConcept" in q:
                    w = params.get("w", "")
                    if w == "murder":
                        return MockResult([{"c": {"id": "LC001", "name": "Murder"}}])
                    return MockResult([])
                if "Chunk" in q and "chk.text" in q:
                    return MockResult([{"chk": {"id": "CHK42", "text": "Rape case"}}])
                if ":Case" in q and "c.id" in q and "CitedCase" not in q:
                    return MockResult([{"c": {"id": "CASE123", "name": "Some case"}}])
                if "CitedCase" in q:
                    return MockResult([{"cc": {"id": "CIT456", "name": "Cited case"}}])
                return MockResult([])
        return S()

    retr = _make_retriever_with_session(make_session)
    assert retr.is_connected()

    anchors = retr.find_graph_anchors(
        query=QUERY,
        parsed_query={"section_ids": ["IPC:300"]},
    )

    assert isinstance(anchors, dict)
    assert set(anchors.keys()) == {"explicit", "concept", "fulltext", "case"}
    assert any(r.get("s", {}).get("id") == "IPC:300" for r in anchors["explicit"])
    assert len(anchors["fulltext"]) >= 1
    assert len(anchors["case"]) >= 1

    retr.close()


# -----------------------------------------------------------------------
# 2. traverse_anchor
# -----------------------------------------------------------------------
def test_traverse_anchor_expands_correctly():
    """Traversal should collect statutes, judgments and cited cases."""

    traverse_data = {
        "statutes": ["STAT456", "STAT789"],
        "judgments": ["CHK1", "CHK2"],
        "cited_cases": ["CIT1"],
    }

    def make_session():
        class S:
            def __enter__(self):  return self
            def __exit__(self, *_): return False
            def run(self, query, **params):
                if params.get("id") == "STAT123":
                    return MockResult([traverse_data])
                return MockResult([])
        return S()

    retr = _make_retriever_with_session(make_session)
    results = retr.traverse_anchor({"s": {"id": "STAT123"}}, max_depth=3)

    assert results["statutes"] == {"STAT456", "STAT789"}
    assert results["judgments"] == {"CHK1", "CHK2"}
    assert results["cited_cases"] == {"CIT1"}

    retr.close()


# -----------------------------------------------------------------------
# 3. fuse_candidates — graph-as-signal design
# -----------------------------------------------------------------------
def test_fuse_candidates_graph_boost():
    """Graph traversal IDs should boost matching vector hits."""
    from scripts.retrieval.fusion import fuse_candidates

    vector_hits = [
        {"id": "v1", "section_id": "IPC:300", "rrf_score": 0.50, "source": "vector"},
        {"id": "v2", "section_id": "IPC:302", "rrf_score": 0.60, "source": "vector"},
        {"id": "v3", "section_id": "IPC:304", "rrf_score": 0.45, "source": "vector"},
    ]

    # Graph says IPC:300 and IPC:304 are relevant
    graph_hits = [
        {"statutes": {"IPC:300", "IPC:304"}, "judgments": set(), "cited_cases": set()},
    ]

    merged = fuse_candidates(vector_hits, graph_hits, query="general homicide inquiry")

    # IPC:302 (0.60) had no graph boost → stays at 0.60
    # IPC:300 (0.50 + 0.30 graph boost = 0.80) → should be first
    # IPC:304 (0.45 + 0.30 graph boost = 0.75) → should be second
    section_order = [c["section_id"] for c in merged]
    assert section_order[0] == "IPC:300", f"Expected IPC:300 first, got {section_order}"
    assert section_order[1] == "IPC:304", f"Expected IPC:304 second, got {section_order}"
    assert section_order[2] == "IPC:302", f"Expected IPC:302 third, got {section_order}"


def test_fuse_candidates_all_vector_when_no_graph():
    """When no graph hits, vector order is preserved."""
    from scripts.retrieval.fusion import fuse_candidates

    vector_hits = [
        {"id": "v1", "section_id": "IPC:300", "rrf_score": 0.65, "source": "vector"},
        {"id": "v2", "section_id": "IPC:302", "rrf_score": 0.55, "source": "vector"},
    ]

    merged = fuse_candidates(vector_hits, [], query="no graph hits here")
    assert merged[0]["section_id"] == "IPC:300"
    assert merged[1]["section_id"] == "IPC:302"
    # All sources remain "vector"
    for c in merged:
        assert c["source"] == "vector"


def test_fuse_candidates_statute_id_boost():
    """Presence of statutory ID in query adds boost to all vector hits."""
    from scripts.retrieval.fusion import fuse_candidates

    vector_hits = [
        {"id": "v1", "section_id": "IPC:300", "rrf_score": 0.50, "source": "vector"},
    ]
    merged_with = fuse_candidates(vector_hits, [], query="What does IPC:300 say?")
    merged_without = fuse_candidates(vector_hits, [], query="What does the law say?")

    assert merged_with[0]["final_score"] > merged_without[0]["final_score"]
    assert merged_with[0]["fusion_boosts"].get("statute_id_in_query", 0) == 0.20


def test_fuse_candidates_preserves_section_id():
    """All returned items must have a section_id (no raw graph dicts inserted)."""
    from scripts.retrieval.fusion import fuse_candidates

    vector_hits = [
        {"id": "v1", "section_id": "IPC:376", "rrf_score": 0.70, "source": "vector"},
    ]
    graph_hits = [
        {"statutes": {"IPC:376"}, "judgments": {"CHK1"}, "cited_cases": set()},
    ]
    merged = fuse_candidates(vector_hits, graph_hits, query="rape punishment IPC:376")

    for item in merged:
        assert "section_id" in item, f"Item missing section_id: {item}"
        assert item["section_id"] != "", f"Empty section_id found: {item}"
