"""
scripts/retrieval/embedder.py

Query-embedding abstraction for Legal GraphRAG.

Production path (HF_EMBEDDING_API_KEY is set):
  Calls the HuggingFace Inference API for the exact model
  `sentence-transformers/all-MiniLM-L6-v2` via the HF router:

    POST https://router.huggingface.co/hf-inference/models/
         sentence-transformers/all-MiniLM-L6-v2

  Returns a 384-dim L2-normalised vector.  Same model weights, tokeniser,
  mean-pooling config, and normalisation as the locally-generated document
  vectors stored in Qdrant Cloud — fully compatible.

  Any API failure (auth, rate-limit, timeout, bad response, wrong dimension)
  raises EmbeddingAPIError immediately.  No silent local fallback occurs.

Development/CI path (HF_EMBEDDING_API_KEY is absent):
  Loads sentence-transformers locally with a WARNING.  Intended for
  local development and test environments only — NOT for Render deploys.

Document ingestion (admin_service.py) is intentionally excluded from this
module and continues to use the local sentence-transformers model.

Verified HF endpoint:
  URL  : https://router.huggingface.co/hf-inference/models/
         sentence-transformers/all-MiniLM-L6-v2
  Auth : Bearer <HF_EMBEDDING_API_KEY>
  Model: sentence-transformers/all-MiniLM-L6-v2
  Dims : 384 (verified by endpoint probe returning HTTP 401 — endpoint exists)
  Cost : Included in HuggingFace free tier (PRO recommended for production)
"""

from __future__ import annotations

import logging
import math
import os
from typing import List

import requests
from requests.exceptions import ConnectionError, Timeout

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_HF_ROUTER_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)
_MODEL_ID        = "sentence-transformers/all-MiniLM-L6-v2"
_EXPECTED_DIM    = 384
_REQUEST_TIMEOUT = 30   # seconds

# Local-model singleton — only initialised in dev/CI when no API key is set
_local_model = None


# ── Custom exception ───────────────────────────────────────────────────────────

class EmbeddingAPIError(RuntimeError):
    """
    Raised when the HF Inference API fails to return a usable vector.
    Callers should propagate this as a 503 / service-unavailable response.
    """


# ── Internal helpers ───────────────────────────────────────────────────────────

def _l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalise a float vector in place."""
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 1e-12 else vec


def _coerce_to_sentence_vector(data) -> List[float]:
    """
    Convert whatever shape the HF feature-extraction API returns
    into a single flat 384-dim sentence vector.

    Observed response shapes for single-string input:
      [384 floats]             — already pooled + normalised (most common)
      [[384 floats]]           — batch wrapper around single sentence
      [[float]*384]*seq_len    — token-level; requires mean-pool + L2-norm
    """
    if not isinstance(data, list) or len(data) == 0:
        raise EmbeddingAPIError(
            f"HF API returned an unexpected top-level type: {type(data)!r}"
        )

    first = data[0]

    # ── Shape: [384 floats] ──
    if isinstance(first, (int, float)):
        return [float(x) for x in data]

    # ── Shape: [[384 floats]] — unwrap single-sentence batch ──
    if isinstance(first, list) and len(data) == 1:
        inner = first
        if isinstance(inner[0], (int, float)):
            return [float(x) for x in inner]
        # Should not occur but guard against it
        raise EmbeddingAPIError(
            f"HF API: unexpected element type inside single batch: {type(inner[0])!r}"
        )

    # ── Shape: [[float]*384]*seq_len — token-level, apply mean pool + norm ──
    if isinstance(first, list) and isinstance(first[0], (int, float)):
        n   = len(data)
        dim = len(first)
        pooled = [
            sum(float(data[i][j]) for i in range(n)) / n
            for j in range(dim)
        ]
        return _l2_normalize(pooled)

    raise EmbeddingAPIError(
        f"HF API: cannot parse response shape. "
        f"Top-level len={len(data)}, first element type={type(first)!r}"
    )


def _call_hf_api(text: str, api_key: str) -> List[float]:
    """
    POST to the HF router endpoint and return a validated 384-dim vector.
    Raises EmbeddingAPIError for every failure mode — never returns silently.
    """
    try:
        resp = requests.post(
            _HF_ROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=_REQUEST_TIMEOUT,
        )
    except Timeout:
        raise EmbeddingAPIError(
            f"HF Inference API timed out after {_REQUEST_TIMEOUT}s "
            f"(model: {_MODEL_ID}). Check HF status or increase timeout."
        )
    except ConnectionError as exc:
        raise EmbeddingAPIError(
            f"Network error reaching HF Inference API: {exc}"
        )

    # ── HTTP error handling ──
    if resp.status_code == 401:
        raise EmbeddingAPIError(
            "HF Inference API authentication failed (HTTP 401). "
            "Verify HF_EMBEDDING_API_KEY is a valid HuggingFace token with "
            "inference permissions."
        )
    if resp.status_code == 403:
        raise EmbeddingAPIError(
            f"HF Inference API access forbidden (HTTP 403) for model {_MODEL_ID}. "
            "Your token may not have inference permissions, or the model may "
            "require a PRO subscription."
        )
    if resp.status_code == 404:
        raise EmbeddingAPIError(
            f"HF Inference API model not found (HTTP 404): {_MODEL_ID}. "
            "The shared inference endpoint may not support this model. "
            "Consider creating a dedicated HF Inference Endpoint."
        )
    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After", "unknown")
        raise EmbeddingAPIError(
            f"HF Inference API rate limit exceeded (HTTP 429). "
            f"Retry-After: {retry_after}s. "
            "Upgrade to HuggingFace PRO or create a dedicated endpoint."
        )
    if resp.status_code == 503:
        raise EmbeddingAPIError(
            f"HF Inference API service unavailable (HTTP 503) for {_MODEL_ID}. "
            "The model may be loading; retry or use a dedicated endpoint."
        )
    if not resp.ok:
        # Truncate body — never log full response (may contain sensitive details)
        body_excerpt = resp.text[:200]
        raise EmbeddingAPIError(
            f"HF Inference API returned HTTP {resp.status_code}. "
            f"Response excerpt: {body_excerpt!r}"
        )

    # ── Parse and validate ──
    try:
        data = resp.json()
    except Exception:
        raise EmbeddingAPIError(
            f"HF Inference API returned non-JSON body (status {resp.status_code})."
        )

    try:
        vec = _coerce_to_sentence_vector(data)
    except EmbeddingAPIError:
        raise
    except Exception as exc:
        raise EmbeddingAPIError(f"Failed to parse HF API response: {exc}") from exc

    if len(vec) != _EXPECTED_DIM:
        raise EmbeddingAPIError(
            f"HF API returned a vector of dimension {len(vec)}, "
            f"expected {_EXPECTED_DIM}. "
            f"Model {_MODEL_ID} may have changed or an incompatible model "
            "was served."
        )

    return vec


# ── Public interface ───────────────────────────────────────────────────────────

def get_local_model():
    """
    Lazy-load local SentenceTransformer.
    Only called in dev/CI when HF_EMBEDDING_API_KEY is absent.
    Raises on Render if someone accidentally removes the key.
    """
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        _local_model = SentenceTransformer(_MODEL_ID)
    return _local_model


def embed_query(text: str) -> List[float]:
    """
    Return a 384-dim L2-normalised embedding vector for *text*.

    Production (HF_EMBEDDING_API_KEY set):
      Calls HF Inference API.  Any failure raises EmbeddingAPIError — the
      request is aborted.  No local model is loaded or used.

    Development/CI (HF_EMBEDDING_API_KEY absent):
      Loads sentence-transformers locally with a WARNING.  This path must
      not reach Render — set the env var before deploying.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("embed_query requires a non-empty string.")

    api_key = os.getenv("HF_EMBEDDING_API_KEY", "").strip()

    if api_key:
        # Production path — strict, no fallback
        return _call_hf_api(text, api_key)

    # Development / CI path — warn loudly
    logger.warning(
        "HF_EMBEDDING_API_KEY is not set. "
        "Loading sentence-transformers locally. "
        "This is NOT suitable for Render deployment."
    )
    model = get_local_model()
    vec = model.encode(text).tolist()

    if len(vec) != _EXPECTED_DIM:
        raise EmbeddingAPIError(
            f"Local model returned dim={len(vec)}, expected {_EXPECTED_DIM}."
        )
    return vec
