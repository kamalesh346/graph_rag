"""
tests/test_qdrant_inference_migration.py

Migration test suite for the Qdrant Cloud Inference refactor.

Unit tests use mocks — no network calls, no production data modification.
Integration tests (marked with @pytest.mark.integration) make real Qdrant
Cloud API calls and require QDRANT_URL + QDRANT_API_KEY to be set.

Run all unit tests:
    pytest tests/test_qdrant_inference_migration.py -v

Run with integration tests (requires live Qdrant credentials):
    pytest tests/test_qdrant_inference_migration.py -v -m integration
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_scored_point(chunk_id="IPC_300_SECTION_001", score=0.85, extra_payload=None):
    """Return a minimal mock ScoredPoint object."""
    payload = {
        "chunk_id":      chunk_id,
        "section_id":    "IPC:300",
        "act_id":        "IPC",
        "section_number": "300",
        "text":          "Murder definition text.",
        "document_type": "statute",
    }
    if extra_payload:
        payload.update(extra_payload)
    pt = SimpleNamespace(id=str(uuid.uuid4()), score=score, payload=payload)
    return pt


# ── 1. chunk_id_to_uuid determinism ───────────────────────────────────────────

class TestChunkIdToUuid:
    def test_deterministic(self):
        from scripts.retrieval.qdrant_inference import chunk_id_to_uuid
        a = chunk_id_to_uuid("IPC_300_SECTION_001")
        b = chunk_id_to_uuid("IPC_300_SECTION_001")
        assert a == b

    def test_different_ids_produce_different_uuids(self):
        from scripts.retrieval.qdrant_inference import chunk_id_to_uuid
        a = chunk_id_to_uuid("IPC_300_SECTION_001")
        b = chunk_id_to_uuid("BNS_101_SECTION_001")
        assert a != b

    def test_uuid_format(self):
        from scripts.retrieval.qdrant_inference import chunk_id_to_uuid
        result = chunk_id_to_uuid("TEST_001")
        uuid.UUID(result)  # raises ValueError if not valid UUID


# ── 2. get_qdrant_client — env-var validation ─────────────────────────────────

class TestGetQdrantClient:
    def test_raises_without_env(self, monkeypatch):
        """Client factory must raise if QDRANT_URL or QDRANT_API_KEY are absent."""
        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)

        # Reset singleton so missing-env check runs
        import scripts.retrieval.qdrant_inference as qi
        original = qi._client
        qi._client = None
        try:
            with pytest.raises(RuntimeError, match="QDRANT_URL"):
                qi.get_qdrant_client()
        finally:
            qi._client = original

    def test_returns_singleton(self):
        """Two calls must return the same object."""
        import scripts.retrieval.qdrant_inference as qi
        with patch("scripts.retrieval.qdrant_inference.QdrantClient") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            qi._client = None  # reset singleton

            c1 = qi.get_qdrant_client()
            c2 = qi.get_qdrant_client()
            assert c1 is c2
            assert mock_cls.call_count == 1
            qi._client = None  # clean up after test


# ── 3. query_collection — validates input & delegates to client ───────────────

class TestQueryCollection:
    def test_empty_text_raises_value_error(self):
        from scripts.retrieval.qdrant_inference import query_collection
        with pytest.raises(ValueError):
            query_collection("legal_statute_chunks", "")

    def test_whitespace_text_raises_value_error(self):
        from scripts.retrieval.qdrant_inference import query_collection
        with pytest.raises(ValueError):
            query_collection("legal_statute_chunks", "   ")

    def test_calls_query_points(self):
        import scripts.retrieval.qdrant_inference as qi
        fake_client = MagicMock()
        fake_client.query_points.return_value = SimpleNamespace(
            points=[_make_scored_point()]
        )
        with patch.object(qi, "get_qdrant_client", return_value=fake_client):
            points = qi.query_collection("legal_statute_chunks", "murder", limit=5)
        assert len(points) == 1
        fake_client.query_points.assert_called_once()
        call_kwargs = fake_client.query_points.call_args
        assert call_kwargs.kwargs["collection_name"] == "legal_statute_chunks"
        assert call_kwargs.kwargs["limit"] == 5

    def test_uses_document_object(self):
        """query_collection must pass a Document (not a raw vector)."""
        import scripts.retrieval.qdrant_inference as qi
        from qdrant_client.models import Document
        fake_client = MagicMock()
        fake_client.query_points.return_value = SimpleNamespace(points=[])
        with patch.object(qi, "get_qdrant_client", return_value=fake_client):
            qi.query_collection("legal_statute_chunks", "test query")
        query_arg = fake_client.query_points.call_args.kwargs["query"]
        assert isinstance(query_arg, Document), (
            f"Expected Document object, got {type(query_arg)}"
        )
        assert query_arg.text == "test query"
        assert "minilm" in query_arg.model.lower()

    def test_api_failure_raises_inference_error(self):
        import scripts.retrieval.qdrant_inference as qi
        fake_client = MagicMock()
        fake_client.query_points.side_effect = RuntimeError("Connection refused")
        with patch.object(qi, "get_qdrant_client", return_value=fake_client):
            with pytest.raises(qi.QdrantInferenceError):
                qi.query_collection("legal_statute_chunks", "murder query")


# ── 4. upsert_chunks — embedding via hosted inference ─────────────────────────

class TestUpsertChunks:
    def test_empty_list_returns_zero(self):
        import scripts.retrieval.qdrant_inference as qi
        fake_client = MagicMock()
        with patch.object(qi, "get_qdrant_client", return_value=fake_client):
            result = qi.upsert_chunks("legal_statute_chunks", [])
        assert result == 0
        fake_client.upsert.assert_not_called()

    def test_skips_empty_text_chunks(self):
        import scripts.retrieval.qdrant_inference as qi
        fake_client = MagicMock()
        with patch.object(qi, "get_qdrant_client", return_value=fake_client):
            result = qi.upsert_chunks(
                "legal_statute_chunks",
                [{"chunk_id": "TEST_001", "text": ""}]
            )
        assert result == 0

    def test_upserts_with_document_vector(self):
        """Ensure PointStruct.vector is a Document, not a float list."""
        import scripts.retrieval.qdrant_inference as qi
        from qdrant_client.models import Document, PointStruct
        fake_client = MagicMock()
        with patch.object(qi, "get_qdrant_client", return_value=fake_client):
            qi.upsert_chunks(
                "legal_statute_chunks",
                [{"chunk_id": "IPC_300_SECTION_001", "text": "Murder definition."}]
            )
        fake_client.upsert.assert_called_once()
        points_arg = fake_client.upsert.call_args.kwargs["points"]
        assert len(points_arg) == 1
        point = points_arg[0]
        assert isinstance(point.vector, Document), (
            f"Expected Document for hosted inference, got {type(point.vector)}"
        )
        assert "minilm" in point.vector.model.lower()

    def test_batches_large_uploads(self):
        """More than _UPSERT_BATCH_SIZE chunks must result in multiple upsert calls."""
        import scripts.retrieval.qdrant_inference as qi
        fake_client = MagicMock()
        chunks = [
            {"chunk_id": f"CHUNK_{i:04d}", "text": f"Text chunk number {i}."}
            for i in range(70)
        ]
        with patch.object(qi, "get_qdrant_client", return_value=fake_client):
            result = qi.upsert_chunks("legal_statute_chunks", chunks)
        assert result == 70
        assert fake_client.upsert.call_count >= 2  # must have been batched

    def test_upsert_failure_raises_inference_error(self):
        import scripts.retrieval.qdrant_inference as qi
        fake_client = MagicMock()
        fake_client.upsert.side_effect = RuntimeError("Qdrant error")
        with patch.object(qi, "get_qdrant_client", return_value=fake_client):
            with pytest.raises(qi.QdrantInferenceError):
                qi.upsert_chunks(
                    "legal_statute_chunks",
                    [{"chunk_id": "X_001", "text": "some text"}]
                )


# ── 5. Statute ingestion — no local model called ──────────────────────────────

class TestStatuteIngestion:
    def test_no_sentence_transformer_import_during_ingest(self):
        """
        Importing admin_service must not trigger SentenceTransformer loading.
        SentenceTransformer.__init__ must never be called in production path.
        """
        with patch("sentence_transformers.SentenceTransformer.__init__") as mock_init:
            # Force re-import by manipulating sys.modules is fragile;
            # instead, verify that after importing admin_service, no local
            # model has been instantiated.
            import backend.admin_service  # noqa: F401
            mock_init.assert_not_called()

    def test_ingest_statute_calls_upsert_inference(self):
        """ingest_statute must call _upsert_inference_chunks, not embed.encode()."""
        import backend.admin_service as admin
        mock_upsert = MagicMock(return_value=1)
        mock_scroll = MagicMock(return_value=[])
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(admin, "_upsert_inference_chunks", mock_upsert),
            patch.object(admin, "_scroll_by_filter", mock_scroll),
            patch.object(admin, "_neo4j_driver", return_value=mock_driver),
        ):
            result = admin.ingest_statute(
                act="TEST", section="999", title="Test Section",
                text="This is a test statute text.", overwrite=False,
            )

        mock_upsert.assert_called_once()
        assert result["indexed"] is True

    def test_duplicate_statute_raises_without_overwrite(self):
        import backend.admin_service as admin
        existing_pt = SimpleNamespace(id="some-uuid")
        mock_scroll = MagicMock(return_value=[existing_pt])

        with patch.object(admin, "_scroll_by_filter", mock_scroll):
            with pytest.raises(ValueError, match="already exists"):
                admin.ingest_statute(
                    act="IPC", section="300", title="Murder",
                    text="Murder text.", overwrite=False,
                )


# ── 6. Judgment ingestion — no local model called ─────────────────────────────

class TestJudgmentIngestion:
    def test_ingest_judgment_calls_upsert_inference(self):
        """ingest_judgment must call _upsert_inference_chunks for all chunks."""
        import backend.admin_service as admin
        mock_upsert  = MagicMock(return_value=2)
        mock_scroll  = MagicMock(return_value=[])
        mock_driver  = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        long_text = "\n\n".join([f"Paragraph {i} with some legal content." * 5 for i in range(10)])
        with (
            patch.object(admin, "_upsert_inference_chunks", mock_upsert),
            patch.object(admin, "_scroll_by_filter", mock_scroll),
            patch.object(admin, "_delete_points",    MagicMock()),
            patch.object(admin, "_neo4j_driver",     return_value=mock_driver),
            patch.object(admin, "_upsert_jsonl",     MagicMock()),
        ):
            result = admin.ingest_judgment(case_id="2025 INSC 001", text=long_text)

        mock_upsert.assert_called_once()
        assert result["indexed"] is True

    def test_empty_text_raises_value_error(self):
        import backend.admin_service as admin
        with pytest.raises(ValueError):
            admin.ingest_judgment(case_id="CASE_001", text="")

    def test_empty_case_id_raises_value_error(self):
        import backend.admin_service as admin
        with pytest.raises(ValueError):
            admin.ingest_judgment(case_id="", text="Some judgment text.")


# ── 7. hybrid_reranker — no embed_query, uses query_collection ───────────────

class TestHybridReranker:
    def test_no_embed_query_import(self):
        """hybrid_reranker must not import embed_query from embedder in production path."""
        import scripts.retrieval.hybrid_reranker as hr
        # After the migration, embed_query should not be a direct attribute
        assert not hasattr(hr, "embed_query"), (
            "embed_query should NOT be directly imported in hybrid_reranker after migration"
        )

    def test_uses_query_col(self):
        """GraphRAGHybridRetriever.search must call _query_col, not embed_query."""
        import scripts.retrieval.hybrid_reranker as hr

        fake_client  = MagicMock()
        fake_client.get_collections.return_value = SimpleNamespace(
            collections=[
                SimpleNamespace(name="legal_statute_chunks"),
                SimpleNamespace(name="legal_judgment_chunks"),
            ]
        )
        mock_points = [_make_scored_point()]

        with (
            patch("scripts.retrieval.qdrant_inference.get_qdrant_client", return_value=fake_client),
            patch("scripts.retrieval.qdrant_inference._client", fake_client),
            patch("scripts.retrieval.hybrid_reranker._query_col", return_value=mock_points) as mock_qc,
            patch("scripts.retrieval.hybrid_reranker.Neo4jGraphRetriever") as mock_neo4j,
            patch("scripts.retrieval.hybrid_reranker.LegalQueryParser") as mock_parser,
        ):
            mock_neo4j.return_value.is_connected.return_value = False
            mock_neo4j.return_value.expand_sections.return_value = {
                "corresponds_to": [], "all_expanded_sections": [], "refers_to": [], "concepts": []
            }
            mock_parser.return_value.parse.return_value = {
                "extracted_sections": [], "primary_target_section": None, "section_ids": []
            }
            retriever = hr.GraphRAGHybridRetriever()
            result    = retriever.search("What is murder?")

        assert mock_qc.called
        assert "statute_results" in result
        assert "judgment_results" in result


# ── 8. plain_rag — uses query_collection ─────────────────────────────────────

class TestPlainRAG:
    def test_search_uses_query_col(self):
        import scripts.retrieval.plain_rag as pr

        fake_client = MagicMock()
        fake_client.get_collections.return_value = SimpleNamespace(
            collections=[
                SimpleNamespace(name="legal_statute_chunks"),
                SimpleNamespace(name="legal_judgment_chunks"),
            ]
        )
        mock_pt = _make_scored_point()

        with (
            patch("scripts.retrieval.qdrant_inference.get_qdrant_client", return_value=fake_client),
            patch("scripts.retrieval.qdrant_inference._client", fake_client),
            patch("scripts.retrieval.plain_rag._query_col", return_value=[mock_pt]) as mock_qc,
        ):
            retriever = pr.PlainRAGRetriever()
            result    = retriever.search("murder definition")

        assert mock_qc.called
        assert len(result["statute_results"]) == 1
        assert result["statute_results"][0]["section_id"] == "IPC:300"


# ── 9. scroll_by_filter — returns empty list on errors ───────────────────────

class TestScrollByFilter:
    def test_returns_empty_on_exception(self):
        import scripts.retrieval.qdrant_inference as qi
        fake_client = MagicMock()
        fake_client.scroll.side_effect = RuntimeError("Connection failed")
        with patch.object(qi, "get_qdrant_client", return_value=fake_client):
            result = qi.scroll_by_filter("legal_statute_chunks", "section_id", "IPC:300")
        assert result == []


# ── 10. delete_points — raises on failure ─────────────────────────────────────

class TestDeletePoints:
    def test_raises_on_failure(self):
        import scripts.retrieval.qdrant_inference as qi
        fake_client = MagicMock()
        fake_client.delete.side_effect = RuntimeError("Cannot delete")
        with patch.object(qi, "get_qdrant_client", return_value=fake_client):
            with pytest.raises(qi.QdrantInferenceError):
                qi.delete_points("legal_statute_chunks", ["some-id"])

    def test_no_op_on_empty_list(self):
        import scripts.retrieval.qdrant_inference as qi
        fake_client = MagicMock()
        with patch.object(qi, "get_qdrant_client", return_value=fake_client):
            qi.delete_points("legal_statute_chunks", [])
        fake_client.delete.assert_not_called()


# ── 11. Backend starts without loading SentenceTransformer ───────────────────

class TestNoLocalModelOnStartup:
    def test_no_local_model_in_production_path(self):
        """
        Importing the full backend stack must not instantiate SentenceTransformer.
        This verifies the migration prevents the OOM issue on Render.
        """
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            # Import (or re-examine) the modules in the production call path
            import scripts.retrieval.qdrant_inference  # noqa: F401
            import scripts.retrieval.hybrid_reranker   # noqa: F401
            import scripts.retrieval.plain_rag         # noqa: F401
            import backend.admin_service               # noqa: F401
            mock_st.assert_not_called()


# ── 12. Integration tests — require live Qdrant Cloud credentials ─────────────

HAVE_QDRANT_CREDS = bool(
    os.getenv("QDRANT_URL", "").strip() and os.getenv("QDRANT_API_KEY", "").strip()
)


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_QDRANT_CREDS, reason="No Qdrant credentials in environment")
class TestIntegrationQdrantInference:
    """
    Read-only integration tests against the live Qdrant Cloud cluster.
    These tests never insert, update, or delete production data.
    """

    def test_statute_collection_query(self):
        """Query inference works for legal_statute_chunks."""
        import scripts.retrieval.qdrant_inference as qi
        qi._client = None  # force fresh client
        points = qi.query_collection(
            "legal_statute_chunks", "murder and culpable homicide", limit=3
        )
        assert len(points) >= 1, "Expected at least one result from statute collection"
        assert points[0].score > 0.0

    def test_judgment_collection_query(self):
        """Query inference works for legal_judgment_chunks."""
        import scripts.retrieval.qdrant_inference as qi
        qi._client = None
        points = qi.query_collection(
            "legal_judgment_chunks", "criminal appeal conviction", limit=3
        )
        # Collection may be empty in test env — just verify the call succeeds
        assert isinstance(points, list)

    def test_statute_payload_schema_preserved(self):
        """Existing statute metadata schema must be intact."""
        import scripts.retrieval.qdrant_inference as qi
        qi._client = None
        points = qi.query_collection(
            "legal_statute_chunks", "murder section 300", limit=1
        )
        if not points:
            pytest.skip("No points returned — collection may be empty")
        payload = points[0].payload
        required_fields = ["chunk_id", "section_id", "act_id", "text", "document_type"]
        for field in required_fields:
            assert field in payload, f"Missing required field '{field}' in statute payload"

    def test_judgment_payload_schema_preserved(self):
        """Existing judgment metadata schema must be intact."""
        import scripts.retrieval.qdrant_inference as qi
        qi._client = None
        points = qi.query_collection(
            "legal_judgment_chunks", "supreme court criminal", limit=1
        )
        if not points:
            pytest.skip("No points returned — collection may be empty")
        payload = points[0].payload
        required_fields = ["chunk_id", "case_id", "text", "document_type"]
        for field in required_fields:
            assert field in payload, f"Missing required field '{field}' in judgment payload"

    def test_get_collection_stats(self):
        """Collection stats must return non-negative integers."""
        import scripts.retrieval.qdrant_inference as qi
        qi._client = None
        s = qi.get_collection_stats("legal_statute_chunks")
        j = qi.get_collection_stats("legal_judgment_chunks")
        assert s >= 0
        assert j >= 0
        print(f"\nCollection stats — statutes: {s}, judgments: {j}")
