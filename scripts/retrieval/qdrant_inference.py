"""
scripts/retrieval/qdrant_inference.py

Centralized Qdrant Cloud Inference service for Legal GraphRAG.

All embedding — at query time and during document ingestion — is performed
by Qdrant's hosted inference endpoint.  No local SentenceTransformer model
is loaded anywhere in the production request path.

Model: sentence-transformers/all-minilm-l6-v2
  - Same weights used for the existing 384-dim vectors in Qdrant Cloud
  - Fully compatible with existing collection vectors (no re-indexing needed)

Public interface
----------------
  get_qdrant_client()         → singleton QdrantClient (cloud_inference=True)
  query_collection(...)       → run a text-based vector query via hosted inference
  upsert_chunks(...)          → embed & upsert a batch of chunks via hosted inference
  scroll_by_filter(...)       → check for an existing point by payload filter
  delete_points(...)          → delete specific point IDs from a collection
  get_collection_stats(...)   → return point count for a collection
  chunk_id_to_uuid(...)       → stable UUID5 from a chunk_id string

Environment variables required
-------------------------------
  QDRANT_URL       – full HTTPS URL to the Qdrant Cloud cluster
  QDRANT_API_KEY   – Qdrant Cloud API key (JWT token)
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Document,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
    PointStruct,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME          = "sentence-transformers/all-minilm-l6-v2"
STATUTE_COLLECTION  = "legal_statute_chunks"
JUDGMENT_COLLECTION = "legal_judgment_chunks"

# Batch size: keeps single Qdrant request payload reasonable
_UPSERT_BATCH_SIZE = 32

# Singleton — one client per process
_client: QdrantClient | None = None


# ── Exceptions ─────────────────────────────────────────────────────────────────

class QdrantInferenceError(RuntimeError):
    """
    Raised when Qdrant Cloud Inference or the Qdrant API itself fails.
    Callers should surface this as an HTTP 503 / service-unavailable response.
    """


# ── Client singleton ───────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    """
    Return the shared Qdrant Cloud client, creating it on first call.

    The client is initialised with cloud_inference=True so that
    Document(text=..., model=...) objects can be passed directly to
    query_points() and upsert().

    Raises RuntimeError if QDRANT_URL or QDRANT_API_KEY are absent.
    """
    global _client
    if _client is not None:
        return _client

    url = os.getenv("QDRANT_URL", "").strip().strip('"').strip("'")
    key = os.getenv("QDRANT_API_KEY", "").strip().strip('"').strip("'")
    if not url or not key:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY environment variables must be set "
            "to use Qdrant Cloud Inference."
        )

    try:
        _client = QdrantClient(
            url=url,
            api_key=key,
            cloud_inference=True,
        )
        logger.info(
            "Qdrant Cloud client initialised (cloud_inference=True, model=%s).", MODEL_NAME
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise Qdrant client: {exc}") from exc

    return _client


# ── ID helpers ─────────────────────────────────────────────────────────────────

def chunk_id_to_uuid(chunk_id: str) -> str:
    """
    Deterministic UUID5 from a chunk_id string.
    Matches the strategy used in migrate_chroma_to_qdrant.py and the
    original admin_service.py so that point IDs are stable across re-uploads.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"legal_graphrag.{chunk_id}"))


# ── Query-time inference ───────────────────────────────────────────────────────

def query_collection(
    collection_name: str,
    text: str,
    limit: int = 10,
    query_filter: Filter | None = None,
    with_payload: bool = True,
) -> list:
    """
    Run a text-based vector search against a Qdrant collection using
    Qdrant Cloud Inference.

    Parameters
    ----------
    collection_name : str
        Name of the Qdrant collection to search.
    text : str
        The user query or document text to embed and search with.
    limit : int
        Maximum number of results to return.
    query_filter : Filter | None
        Optional Qdrant filter to apply alongside the vector search.
    with_payload : bool
        Whether to include point payloads in the response.

    Returns
    -------
    list[ScoredPoint]
        Ordered list of matching points (highest score first).

    Raises
    ------
    ValueError
        If text is empty or not a string.
    QdrantInferenceError
        If the Qdrant API call fails.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("query_collection requires a non-empty text string.")

    client = get_qdrant_client()
    try:
        result = client.query_points(
            collection_name=collection_name,
            query=Document(text=text.strip(), model=MODEL_NAME),
            limit=limit,
            query_filter=query_filter,
            with_payload=with_payload,
        )
        return result.points
    except Exception as exc:
        snippet = text[:80].replace("\n", " ")
        logger.error(
            "Qdrant query_points failed on '%s' (text snippet: '%.80s...'): %s",
            collection_name, snippet, exc,
        )
        raise QdrantInferenceError(
            f"Qdrant Cloud Inference query failed for collection '{collection_name}': {exc}"
        ) from exc


# ── Ingestion-time inference ───────────────────────────────────────────────────

def upsert_chunks(
    collection_name: str,
    chunks: list[dict[str, Any]],
    text_field: str = "text",
    id_field: str = "chunk_id",
) -> int:
    """
    Embed and upsert a list of chunk dicts into Qdrant using Cloud Inference.

    Each chunk dict must contain:
      - text_field (default "text")      : the string to embed
      - id_field   (default "chunk_id")  : stable string ID used to derive a UUID5

    All other keys in the dict are stored as the point payload.

    Chunks with empty or whitespace-only text are skipped and logged.
    Batches of up to _UPSERT_BATCH_SIZE points are inserted per request.

    Parameters
    ----------
    collection_name : str
        Target Qdrant collection.
    chunks : list[dict]
        List of chunk dicts to embed and insert.
    text_field : str
        Key in each chunk dict that holds the text to embed.
    id_field : str
        Key in each chunk dict that holds the stable string ID.

    Returns
    -------
    int
        Number of chunks successfully upserted.

    Raises
    ------
    QdrantInferenceError
        If a batch upsert fails. Already-inserted batches are not rolled back.
    """
    if not chunks:
        return 0

    client      = get_qdrant_client()
    inserted    = 0
    skipped     = 0
    batch: list[PointStruct] = []

    def _flush(batch_to_flush: list[PointStruct]) -> None:
        nonlocal inserted
        if not batch_to_flush:
            return
        try:
            client.upsert(collection_name=collection_name, points=batch_to_flush)
            inserted += len(batch_to_flush)
        except Exception as exc:
            logger.error(
                "Qdrant upsert failed for collection '%s' (batch of %d): %s",
                collection_name, len(batch_to_flush), exc,
            )
            raise QdrantInferenceError(
                f"Qdrant upsert failed for '{collection_name}': {exc}"
            ) from exc

    for chunk in chunks:
        text   = chunk.get(text_field, "")
        id_str = chunk.get(id_field, "")

        if not isinstance(text, str) or not text.strip():
            logger.warning(
                "Skipping chunk with empty text (id_field='%s').", id_str or "<no id>"
            )
            skipped += 1
            continue

        point_id = chunk_id_to_uuid(id_str) if id_str else str(uuid.uuid4())
        payload  = dict(chunk)

        # Use Document so Qdrant Cloud generates the embedding server-side
        point = PointStruct(
            id=point_id,
            vector=Document(text=text.strip(), model=MODEL_NAME),
            payload=payload,
        )
        batch.append(point)

        if len(batch) >= _UPSERT_BATCH_SIZE:
            _flush(batch)
            batch = []

    _flush(batch)  # flush any remainder

    if skipped:
        logger.warning(
            "upsert_chunks: skipped %d chunk(s) with empty text in '%s'.",
            skipped, collection_name,
        )
    logger.info(
        "upsert_chunks: %d chunk(s) inserted into '%s'.", inserted, collection_name
    )
    return inserted


# ── Utility helpers ────────────────────────────────────────────────────────────

def scroll_by_filter(
    collection_name: str,
    field: str,
    value: str,
    limit: int = 1,
    with_vectors: bool = False,
) -> list:
    """
    Scroll Qdrant for points matching a single field equality filter.

    Returns a list of Record objects (up to `limit`).
    Returns [] on any error (non-fatal for existence checks).
    """
    client = get_qdrant_client()
    try:
        results, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key=field, match=MatchValue(value=value))
            ]),
            limit=limit,
            with_payload=False,
            with_vectors=with_vectors,
        )
        return results
    except Exception as exc:
        logger.warning(
            "scroll_by_filter failed on '%s' (field=%s, value=%s): %s",
            collection_name, field, value, exc,
        )
        return []


def delete_points(collection_name: str, point_ids: list) -> None:
    """Delete a list of point IDs from a collection. Raises QdrantInferenceError on failure."""
    if not point_ids:
        return
    client = get_qdrant_client()
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=point_ids),
        )
        logger.info(
            "Deleted %d point(s) from '%s'.", len(point_ids), collection_name
        )
    except Exception as exc:
        logger.error(
            "Failed to delete %d point(s) from '%s': %s",
            len(point_ids), collection_name, exc,
        )
        raise QdrantInferenceError(
            f"Failed to delete points from '{collection_name}': {exc}"
        ) from exc


def get_collection_stats(collection_name: str) -> int:
    """Return the point count for a collection, or 0 on error."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection_name)
        return info.points_count or 0
    except Exception as exc:
        logger.warning("get_collection_stats failed for '%s': %s", collection_name, exc)
        return 0
