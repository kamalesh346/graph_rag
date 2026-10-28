"""
migrate_chroma_to_qdrant.py

One-shot data migration: local ChromaDB → Qdrant Cloud.

Rules:
  - Read-only on existing Qdrant collections (no create/drop/rename).
  - Extracts vectors already stored in ChromaDB — no re-embedding.
  - Upsert is idempotent: re-running produces identical point IDs.
  - Does NOT modify retrieval code; run this before the code migration.

Usage:
    python migrate_chroma_to_qdrant.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ── env guards ────────────────────────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

if not QDRANT_URL or not QDRANT_API_KEY:
    print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set in the environment or .env")
    sys.exit(1)

CHROMA_DB_DIR       = PROJECT_ROOT / "data" / "chroma_db"
STATUTE_COLLECTION  = "legal_statute_chunks"
JUDGMENT_COLLECTION = "legal_judgment_chunks"
BATCH_SIZE          = 50          # points per upsert call


# ── ID helper ─────────────────────────────────────────────────────────────────

def _to_uuid(chunk_id: str) -> str:
    """
    Deterministic UUID5 from the ChromaDB string chunk_id.
    Same input always produces the same UUID — migration is idempotent.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"legal_graphrag.{chunk_id}"))


# ── Core migration ────────────────────────────────────────────────────────────

def migrate_collection(
    chroma_col,
    qdrant_client,
    collection_name: str,
    label: str,
) -> int:
    """
    Read all documents (with stored vectors) from a ChromaDB collection
    and upsert them into the matching Qdrant collection.

    Returns the number of points upserted.
    """
    from qdrant_client.models import PointStruct

    total = chroma_col.count()
    print(f"\n── {label} ({collection_name}) ──────────────────────────")
    print(f"  ChromaDB count : {total}")

    if total == 0:
        print("  Nothing to migrate — collection is empty.")
        return 0

    # ChromaDB stores vectors internally; get() with include=["embeddings"]
    # returns the exact vectors that were generated at index time.
    all_data = chroma_col.get(
        include=["embeddings", "documents", "metadatas"],
    )

    ids        = all_data["ids"]
    embeddings = all_data["embeddings"]
    documents  = all_data["documents"]
    metadatas  = all_data["metadatas"]

    if embeddings is None or len(embeddings) == 0 or embeddings[0] is None:
        print("  ERROR: ChromaDB returned no stored embeddings.")
        print("  The collection may have been built without storing vectors.")
        print("  Re-embedding is required — please run the build script first.")
        return 0

    print(f"  Vector dim     : {len(embeddings[0])}")

    points: list[PointStruct] = []
    upserted = 0

    for i in range(len(ids)):
        chunk_id = ids[i]
        vector   = embeddings[i]
        text     = documents[i]
        meta     = metadatas[i] or {}

        # Payload: all ChromaDB metadata fields + the document text + original ID
        payload: dict[str, Any] = dict(meta)
        payload["text"]     = text
        payload["chunk_id"] = chunk_id       # preserves the original string ID

        # Qdrant IDs must be UUID or unsigned int — use deterministic UUID5
        point_id = _to_uuid(chunk_id)

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        # Flush batch
        if len(points) >= BATCH_SIZE:
            qdrant_client.upsert(collection_name=collection_name, points=points)
            upserted += len(points)
            print(f"  Upserted {upserted}/{total}…")
            points = []

    # Flush remainder
    if points:
        qdrant_client.upsert(collection_name=collection_name, points=points)
        upserted += len(points)

    # Verify
    info = qdrant_client.get_collection(collection_name)
    print(f"  ✅ Done — upserted {upserted} points.")
    print(f"  Qdrant points_count: {info.points_count}")
    return upserted


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=== ChromaDB → Qdrant Cloud Data Migration ===")
    print(f"Source : {CHROMA_DB_DIR}")

    # Validate source
    if not CHROMA_DB_DIR.exists():
        print(f"ERROR: ChromaDB directory not found at {CHROMA_DB_DIR}")
        sys.exit(1)

    import chromadb
    from qdrant_client import QdrantClient

    chroma = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    qdrant = QdrantClient(
        url=QDRANT_URL,
        port=443,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
    )

    # Confirm Qdrant collections exist before touching anything
    existing = {c.name for c in qdrant.get_collections().collections}
    for col_name in [STATUTE_COLLECTION, JUDGMENT_COLLECTION]:
        if col_name not in existing:
            print(f"ERROR: Qdrant collection '{col_name}' does not exist.")
            print("Create the collection before running this migration.")
            sys.exit(1)
    print(f"Qdrant collections confirmed: {sorted(existing)}")

    # Migrate statutes
    statute_col = chroma.get_collection(name=STATUTE_COLLECTION)
    s_count = migrate_collection(
        statute_col, qdrant, STATUTE_COLLECTION, "Statutes"
    )

    # Migrate judgments
    try:
        judgment_col = chroma.get_collection(name=JUDGMENT_COLLECTION)
        j_count = migrate_collection(
            judgment_col, qdrant, JUDGMENT_COLLECTION, "Judgments"
        )
    except Exception as exc:
        print(f"\nWARNING: Could not open judgment collection: {exc}")
        j_count = 0

    print(f"\n=== Migration complete ===")
    print(f"  Statutes  : {s_count} points → {STATUTE_COLLECTION}")
    print(f"  Judgments : {j_count} points → {JUDGMENT_COLLECTION}")
    print(
        "\nNext step: run the code migration to point retrieval at Qdrant Cloud."
    )


if __name__ == "__main__":
    main()
