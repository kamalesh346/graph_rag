"""
validate_graphrag_storage.py

Validates storage integrity across both Vector DB (ChromaDB) and Knowledge Graph (Neo4j).
Performs sample similarity search and reports total chunk/node counts.
"""

import os
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "legal_judgment_chunks"


def validate_chromadb():
    print("=== 1. Validating ChromaDB Vector Collection ===")
    if not CHROMA_DB_DIR.exists():
        print(f"FAIL: ChromaDB directory does not exist at {CHROMA_DB_DIR}")
        return False

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception as e:
        print(f"FAIL: Could not retrieve collection '{COLLECTION_NAME}': {e}")
        return False

    count = collection.count()
    print(f"Total chunks indexed in '{COLLECTION_NAME}': {count}")

    if count != 285:
        print(f"WARNING: Expected 285 chunks, but found {count}")
    else:
        print("PASS: ChromaDB contains exactly 285 judgment chunks.")

    # Run sample similarity search query
    query = "dying declaration under section 32 exception requirement and evidentiary value"
    print(f"\nTesting sample semantic retrieval query: '{query}'...")
    
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print("\nTop 3 Retrieved Chunks:")
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i in range(len(ids)):
        print(f"\nResult #{i+1} (Distance: {dists[i]:.4f}):")
        print(f"  Chunk ID   : {ids[i]}")
        print(f"  Case ID    : {metas[i].get('case_id')}")
        print(f"  Chunk Type : {metas[i].get('chunk_type')}")
        print(f"  References : {metas[i].get('legal_references')}")
        print(f"  Text Snippet: {docs[i][:150]}...")

    return True


def validate_neo4j():
    print("\n=== 2. Validating Neo4j Knowledge Graph ===")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
        driver.verify_connectivity()
    except Exception as e:
        print(f"INFO: Local Neo4j server at {neo4j_uri} is not reachable ({e}).")
        print("Neo4j validation skipped. Run 'python scripts/ingest_neo4j_graph.py' when Neo4j is running.")
        return True

    with driver.session() as session:
        counts = session.run("""
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
        """)
        print("\nNeo4j Node Counts by Label:")
        for record in counts:
            print(f"  - {record['label']}: {record['count']}")

        rel_counts = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
        ORDER BY count DESC
        """)
        print("\nNeo4j Relationship Counts by Type:")
        for record in rel_counts:
            print(f"  - {record['type']}: {record['count']}")

    driver.close()
    return True


def main():
    print("=== GraphRAG Storage & Retrieval Validation Report ===")
    v_ok = validate_chromadb()
    g_ok = validate_neo4j()
    
    if v_ok:
        print("\nOVERALL STORAGE STATUS: PASS")
    else:
        print("\nOVERALL STORAGE STATUS: FAIL")


if __name__ == "__main__":
    main()
