"""
build_chroma_index.py

Ingests all 348 judgment chunks from data/chunks/all_judgment_chunks.jsonl
into a persistent local ChromaDB collection named 'legal_judgment_chunks'.
"""

import json
import os
import sys
from pathlib import Path

# Fix TF / Keras compatibility environment flags before importing transformers/chromadb
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_JSONL_PATH = PROJECT_ROOT / "data" / "chunks" / "all_judgment_chunks.jsonl"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "legal_judgment_chunks"


def main():
    print(f"=== ChromaDB Vector Index Builder ===")
    print(f"Input file: {CHUNKS_JSONL_PATH}")
    print(f"ChromaDB path: {CHROMA_DB_DIR}")

    if not CHUNKS_JSONL_PATH.exists():
        print(f"ERROR: Chunks file not found at {CHUNKS_JSONL_PATH}")
        sys.exit(1)

    # 1. Load chunks
    chunks = []
    with open(CHUNKS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks from JSONL.")

    # 2. Initialize Chroma persistent client
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # 3. Embedding function (using sentence-transformers default or ONNX)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Create or get collection
    # Reset collection if exists to ensure clean index
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' for clean re-index.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "Legal Supreme Court Judgment Chunks for GraphRAG"}
    )

    # 5. Prepare batch data
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        
        # Format metadata for ChromaDB (Chroma requires primitive types or strings)
        meta = {
            "case_id": chunk.get("case_id", ""),
            "document_type": chunk.get("document_type", "judgment"),
            "chunk_type": chunk.get("chunk_type", "judgment"),
            "source_field": chunk.get("source", {}).get("field", ""),
            "page_start": int(chunk.get("source", {}).get("page_start", 0)),
            "page_end": int(chunk.get("source", {}).get("page_end", 0)),
            "token_count": int(chunk.get("token_count", 0)),
            # Join list fields as comma-separated strings for Chroma metadata filtering
            "legal_references": ",".join(chunk.get("legal_references", [])),
            "legal_concepts": ",".join(chunk.get("legal_concepts", [])),
            "cited_cases": ",".join(chunk.get("cited_cases", [])),
        }

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(meta)

    # 6. Batch add (batch size 100)
    batch_size = 100
    total = len(ids)
    print(f"Ingesting {total} chunks into collection '{COLLECTION_NAME}'...")

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"  Ingested batch {i + 1} to {end} / {total}")

    print(f"\nSUCCESS: ChromaDB collection '{COLLECTION_NAME}' built successfully with {collection.count()} items.")


if __name__ == "__main__":
    main()
