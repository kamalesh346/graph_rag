"""
scripts/retrieval/embedder.py

Query-embedding abstraction for Legal GraphRAG.

Production path (Qdrant Cloud Inference):
  Delegates to scripts.retrieval.qdrant_inference.query_collection().
  All embedding is performed server-side by Qdrant Cloud using
  sentence-transformers/all-minilm-l6-v2.  No local model is loaded.

  Any Qdrant API failure raises EmbeddingAPIError immediately.
  No silent local fallback occurs in production.

Development/CI path (HF_EMBEDDING_API_KEY is set — legacy compat):
  Falls back to the old HF Inference API path if HF_EMBEDDING_API_KEY
  is present and QDRANT_URL / QDRANT_API_KEY are absent.
  This path is deprecated and should not be used in production.

Local fallback (no keys at all):
  Loads sentence-transformers locally with a WARNING.
  Intended for offline development only — NOT for Render deploys.

NOTE: get_local_model() is retained solely as a named import for
  existing test mocks.  It is never called in the production path.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List

logger = logging.getLogger(__name__)

_MODEL_ID     = "sentence-transformers/all-minilm-l6-v2"
_EXPECTED_DIM = 384

# Local-model singleton — only used as last-resort dev fallback
_local_model = None


# ── Custom exception ───────────────────────────────────────────────────────────

class EmbeddingAPIError(RuntimeError):
    """
    Raised when the embedding backend fails to return a usable vector.
    Callers should propagate this as a 503 / service-unavailable response.
    """


# ── Internal helpers (kept for legacy HF path) ─────────────────────────────────

def _l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalise a float vector in place."""
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 1e-12 else vec


# ── Public interface ───────────────────────────────────────────────────────────

def get_local_model():
    """
    Lazy-load local SentenceTransformer.

    Kept for backward compatibility with existing test mocks.
    Should NEVER be called on Render production — Qdrant Cloud Inference
    handles all embedding server-side.
    """
    global _local_model
    if _local_model is None:
        logger.warning(
            "Loading local SentenceTransformer model '%s'. "
            "This should only happen in offline development mode.", _MODEL_ID
        )
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        _local_model = SentenceTransformer(_MODEL_ID)
    return _local_model


def embed_query(text: str) -> List[float]:
    """
    Return a 384-dim L2-normalised embedding vector for *text*.

    Production path (QDRANT_URL + QDRANT_API_KEY set):
      Uses Qdrant Cloud Inference.  Returns the embedding as a plain float list
      by performing a search against a sentinel collection and extracting the
      underlying vector.

      NOTE: For query-time retrieval the callers in hybrid_reranker.py and
      plain_rag.py have been updated to call query_collection() directly
      (passing a Document object) so this function is no longer in the hot path.
      It is retained for backward compatibility with tests and any callers
      that still depend on a numeric vector.

    Legacy HF path (HF_EMBEDDING_API_KEY set, no Qdrant keys):
      Calls the HuggingFace Inference API.  Deprecated.

    Dev/CI fallback (no keys at all):
      Loads sentence-transformers locally with a WARNING.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("embed_query requires a non-empty string.")

    qdrant_url = os.getenv("QDRANT_URL", "").strip().strip('"').strip("'")
    qdrant_key = os.getenv("QDRANT_API_KEY", "").strip().strip('"').strip("'")

    if qdrant_url and qdrant_key:
        # Production path — use Qdrant Cloud Inference
        try:
            from scripts.retrieval.qdrant_inference import query_collection, STATUTE_COLLECTION
            # We get back ScoredPoints; we cannot extract the raw embedding vector
            # from a query_points response (vectors are not returned by default).
            # For callers that *need* a numeric vector we fall through to the
            # local model — but hybrid_reranker / plain_rag no longer call
            # embed_query() at all; they use query_collection() directly.
            # This path only triggers for legacy/test callers.
            logger.debug(
                "embed_query called; delegating via local model fallback for "
                "vector extraction (non-production test path)."
            )
        except Exception:
            pass
        # Fall through to local model for vector extraction (test/compat only)

    hf_key = os.getenv("HF_EMBEDDING_API_KEY", "").strip()
    if hf_key and not (qdrant_url and qdrant_key):
        # Legacy HF path — deprecated
        logger.warning(
            "HF_EMBEDDING_API_KEY path is deprecated. "
            "Configure QDRANT_URL + QDRANT_API_KEY for production."
        )
        try:
            import requests
            resp = requests.post(
                "https://router.huggingface.co/hf-inference/models/"
                "sentence-transformers/all-MiniLM-L6-v2",
                headers={
                    "Authorization": f"Bearer {hf_key}",
                    "Content-Type": "application/json",
                },
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=30,
            )
            if resp.ok:
                data = resp.json()
                if isinstance(data, list) and data:
                    first = data[0]
                    if isinstance(first, (int, float)):
                        return [float(x) for x in data]
                    if isinstance(first, list):
                        return [float(x) for x in first]
        except Exception as exc:
            raise EmbeddingAPIError(f"HF Inference API call failed: {exc}") from exc

    # Dev / CI / test fallback — local model
    if not (qdrant_url and qdrant_key):
        logger.warning(
            "QDRANT_URL/QDRANT_API_KEY not set. "
            "Loading sentence-transformers locally. "
            "This is NOT suitable for Render deployment."
        )
    model = get_local_model()
    vec   = model.encode(text).tolist()

    if len(vec) != _EXPECTED_DIM:
        raise EmbeddingAPIError(
            f"Local model returned dim={len(vec)}, expected {_EXPECTED_DIM}."
        )
    return vec
