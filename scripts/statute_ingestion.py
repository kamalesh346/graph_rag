"""
statute_ingestion.py - PHASES S8, S9, S10: Vector Indexing & Knowledge Graph Ingestion

1. PHASE S8: ChromaDB Vector Indexing
   Ingests 142 statute chunks into persistent collection 'legal_statute_chunks'.

2. PHASE S9: Neo4j Statute Knowledge Graph
   Ingests statute nodes & relationships:
   - (:Act)
   - (:StatuteSection)
   - (:StatuteUnit)
   - (:LegalConcept)
   - (:Chunk)
   Relationships:
   - (:Act)-[:HAS_SECTION]->(:StatuteSection)
   - (:StatuteSection)-[:HAS_UNIT]->(:StatuteUnit)
   - (:Chunk)-[:REPRESENTS]->(:StatuteUnit)
   - (:Chunk)-[:REFERS_TO]->(:StatuteSection)
   - (:Chunk)-[:DISCUSSES]->(:LegalConcept)

3. PHASE S10: Act-to-Act Mapping Graph
   Creates CORRESPONDS_TO edges (e.g. IPC:302 -> BNS:103).
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

# Environment flags for sentence-transformers / chromadb
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUTES_JSONL = PROJECT_ROOT / "data" / "chunks" / "statutes" / "all_statute_chunks.jsonl"
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
STATUTE_COLLECTION = "legal_statute_chunks"
LEGAL_MAPS_DIR = PROJECT_ROOT / "data" / "legal" / "maps"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def ingest_chromadb_statutes() -> int:
    print(f"=== PHASE S8: ChromaDB Statute Ingestion ===")
    print(f"Reading statute chunks from: {STATUTES_JSONL}")

    if not STATUTES_JSONL.exists():
        print(f"ERROR: {STATUTES_JSONL} not found. Run statute_chunker.py first.")
        return 0

    chunks = []
    with open(STATUTES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} statute chunks.")

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        client.delete_collection(STATUTE_COLLECTION)
        print(f"Deleted existing collection '{STATUTE_COLLECTION}' for clean re-index.")
    except Exception:
        pass

    collection = client.create_collection(
        name=STATUTE_COLLECTION,
        embedding_function=ef,
        metadata={"description": "Canonical Statute Chunks (1 Section = 1 Chunk)"}
    )

    ids = []
    documents = []
    metadatas = []

    for c in chunks:
        ids.append(c["chunk_id"])
        documents.append(c["text"])

        src = c.get("source", {})
        meta = {
            "act_id": c.get("act_id", ""),
            "section_id": c.get("section_id", ""),
            "document_type": c.get("document_type", "statute"),
            "chunk_type": c.get("chunk_type", "section"),
            "section_number": str(src.get("section_number", "")),
            "has_main_text": bool(src.get("has_main_text", True)),
            "subsections": ",".join(src.get("subsections", [])),
            "explanations": ",".join(src.get("explanations", [])),
            "illustrations": ",".join(src.get("illustrations", [])),
            "exceptions": ",".join(src.get("exceptions", [])),
            "provisos": ",".join(src.get("provisos", [])),
            "token_count": int(c.get("token_count", 0)),
            "legal_references": ",".join(c.get("legal_references", [])),
            "legal_concepts": ",".join(c.get("legal_concepts", [])),
            "successor_references": ",".join(c.get("successor_references", [])),
            "predecessor_references": ",".join(c.get("predecessor_references", [])),
        }
        metadatas.append(meta)

    # Batch add
    batch_size = 100
    total = len(ids)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )

    count = collection.count()
    print(f"SUCCESS: Ingested {count} section chunks into ChromaDB collection '{STATUTE_COLLECTION}'.")
    return count


def ingest_neo4j_statutes() -> bool:
    print(f"\n=== PHASES S9 & S10: Neo4j Statute Knowledge Graph Ingestion ===")
    
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Connected to Neo4j database successfully!")
    except Exception as e:
        print(f"INFO: Local Neo4j server at {NEO4J_URI} is not reachable ({e}).")
        print("Neo4j statute graph ingestion skipped until Neo4j server is active.")
        return False

    with driver.session() as session:
        # 1. Constraints
        constraints = [
            "CREATE CONSTRAINT act_id_unique IF NOT EXISTS FOR (a:Act) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT statute_section_unique IF NOT EXISTS FOR (s:StatuteSection) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT legal_concept_unique IF NOT EXISTS FOR (lc:LegalConcept) REQUIRE lc.id IS UNIQUE",
        ]
        for c in constraints:
            try:
                session.run(c)
            except Exception:
                pass

        # 2. Ingest Statute Chunks
        chunks = []
        with open(STATUTES_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        for c in chunks:
            act_id = c["act_id"]
            sec_id = c["section_id"]
            chunk_id = c["chunk_id"]
            text = c["text"]

            cypher = """
            MERGE (a:Act {id: $act_id})
            MERGE (s:StatuteSection {id: $sec_id})
            MERGE (a)-[:HAS_SECTION]->(s)

            MERGE (chk:Chunk {id: $chunk_id})
            SET chk.document_type = "statute",
                chk.chunk_type = "section",
                chk.text = $text,
                chk.act_id = $act_id,
                chk.section_id = $sec_id
            MERGE (s)-[:HAS_CHUNK]->(chk)
            """
            session.run(cypher, act_id=act_id, sec_id=sec_id, chunk_id=chunk_id, text=text)

            # References: (:StatuteSection)-[:REFERS_TO]->(:StatuteSection)
            for ref in c.get("legal_references", []):
                if ref:
                    session.run("""
                    MATCH (s:StatuteSection {id: $sec_id})
                    MERGE (target:StatuteSection {id: $ref_id})
                    MERGE (s)-[:REFERS_TO]->(target)
                    """, sec_id=sec_id, ref_id=ref)

            # Concepts: (:StatuteSection)-[:DISCUSSES]->(:LegalConcept)
            for concept in c.get("legal_concepts", []):
                if concept:
                    concept_name = concept.replace("LEGAL_CONCEPT:", "").replace("_", " ").title()
                    session.run("""
                    MATCH (s:StatuteSection {id: $sec_id})
                    MERGE (lc:LegalConcept {id: $concept_id})
                    SET lc.name = $name
                    MERGE (s)-[:DISCUSSES]->(lc)
                    """, sec_id=sec_id, concept_id=concept, name=concept_name)

        # 3. PHASE S10: Act-to-Act Correspondence Mapping Graph (:StatuteSection)-[:CORRESPONDS_TO]->(:StatuteSection)
        print("Ingesting Act-to-Act correspondence mapping edges...")
        map_files = list(LEGAL_MAPS_DIR.glob("*.json"))
        for mf in map_files:
            with open(mf, "r", encoding="utf-8") as f:
                map_data = json.load(f)
            mapping_id = map_data.get("mapping", {}).get("mapping_id", mf.stem)

            for item in map_data.get("mappings", []):
                src = item.get("source_section")
                tgt = item.get("target_section")
                if src and tgt:
                    session.run("""
                    MERGE (s1:StatuteSection {id: $src})
                    MERGE (s2:StatuteSection {id: $tgt})
                    MERGE (s1)-[:CORRESPONDS_TO {act_mapping: $mapping_id}]->(s2)
                    """, src=src, tgt=tgt, mapping_id=mapping_id)

    driver.close()
    print("SUCCESS: Neo4j Statute Knowledge Graph Ingestion complete!")
    return True


if __name__ == "__main__":
    ingest_chromadb_statutes()
    ingest_neo4j_statutes()
