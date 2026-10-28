"""
scripts/retrieval/embedder.py

Single point of truth for query embedding.

Priority:
  1. HuggingFace Inference API   — when HF_EMBEDDING_API_KEY is set
       Endpoint: router.huggingface.co/hf-inference/models/
                 sentence-transformers/all-MiniLM-L6-v2
       Response: 384-dim float list, already mean-pooled + L2-normalized.
       Identical to local SentenceTransformer output — same model weights,
       same tokenizer, same pooling config, same normalization.

  2. Local SentenceTransformer   — fallback when no API key (local dev / CI)

Only query encoding is handled here. Document ingestion (admin_service.py)
continues to use the local model and is intentionally not changed.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List

import requests

logger = logging.getLogger(__name__)

# HF Inference API — new router URL (legacy api-inference.huggingface.co deprecated)
_HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)
_EMBED_MODEL = "all-MiniLM-L6-v2"

# Local model singleton — only loaded when no API key is set
_local_model = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 1e-12 else vec


def _coerce_to_flat_vector(data) -> List[float]:
    """
    Robustly unwrap whatever shape the HF API returns into a flat 384-dim list.

    Observed shapes:
      [384 floats]                — single string input, already sentence-level
      [[384 floats]]              — batch of 1 string, outer batch dim
      [[384 floats], ...]         — batch of N strings (we take index 0)
      [[[token_embs]*seq]*batch]  — token-level (mean-pool each token row)
    """
    if not data:
        raise ValueError("HF API returned empty response.")

    # Unwrap single-element batch
    if isinstance(data[0], list) and len(data) == 1:
        data = data[0]

    if isinstance(data[0], float):
        # Already a 384-dim flat vector
        return data

    if isinstance(data[0], list):
        if isinstance(data[0][0], float):
            # data is [seq_len][384] — token-level, need mean pooling
            n = len(data)
            dim = len(data[0])
            pooled = [sum(data[i][j] for i in range(n)) / n for j in range(dim)]
            return _l2_normalize(pooled)
        # Deeper nesting — shouldn't happen but handle gracefully
        raise ValueError(f"Unexpected HF API nesting depth. data[0][0] type: {type(data[0][0])}")

    raise ValueError(f"Cannot parse HF API response: {type(data[0])}")


# ── Public API ─────────────────────────────────────────────────────────────────

def get_local_model():
    """Lazy-load the local SentenceTransformer. Used for startup preload when no API key."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(_EMBED_MODEL)
    return _local_model


def embed_query(text: str) -> List[float]:
    """
    Embed a single query string into a 384-dim L2-normalized float list.

    Routes to HF Inference API when HF_EMBEDDING_API_KEY is set,
    otherwise falls back to local SentenceTransformer.
    """
    api_key = os.getenv("HF_EMBEDDING_API_KEY", "").strip()

    if api_key:
        try:
            resp = requests.post(
                _HF_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=30,
            )
            resp.raise_for_status()
            return _coerce_to_flat_vector(resp.json())
        except Exception as exc:
            logger.warning(
                "HF Embedding API failed (%s). Falling back to local model.", exc
            )
            # Fall through

    # Local fallback
    model = get_local_model()
    return model.encode(text).tolist()
