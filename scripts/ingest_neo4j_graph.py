"""
ingest_neo4j_graph.py

Ingests all 348 judgment chunks (from data/chunks/all_judgment_chunks.jsonl)
and statutory mappings (from data/legal/maps/*.json) into Neo4j.

Creates Node Types:
  - (:Case)
  - (:Chunk)
  - (:StatuteSection)
  - (:LegalConcept)
  - (:CitedCase)

Creates Relationships:
  - (:Case)-[:HAS_CHUNK]->(:Chunk)
  - (:Chunk)-[:REFERS_TO]->(:StatuteSection)
  - (:Chunk)-[:DISCUSSES]->(:LegalConcept)
  - (:Case)-[:CITES_CASE]->(:CitedCase)
  - (:Chunk)-[:DISCUSSES_CASE]->(:CitedCase)
  - (:StatuteSection)-[:CORRESPONDS_TO]->(:StatuteSection)
"""

import json
import os
import sys
from pathlib import Path
from neo4j import GraphDatabase, exceptions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_JSONL_PATH = PROJECT_ROOT / "data" / "chunks" / "all_judgment_chunks.jsonl"
LEGAL_MAPS_DIR = PROJECT_ROOT / "data" / "legal" / "maps"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def create_constraints(session):
    print("Creating uniqueness constraints & indexes...")
    constraints = [
        "CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (chk:Chunk) REQUIRE chk.id IS UNIQUE",
        "CREATE CONSTRAINT statute_id_unique IF NOT EXISTS FOR (s:StatuteSection) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (lc:LegalConcept) REQUIRE lc.id IS UNIQUE",
        "CREATE CONSTRAINT cited_case_id_unique IF NOT EXISTS FOR (cc:CitedCase) REQUIRE cc.id IS UNIQUE",
    ]
    for cypher in constraints:
        try:
            session.run(cypher)
        except Exception as e:
            print(f"  Warning setting constraint ({cypher[:30]}...): {e}")


def ingest_statute_mappings(session):
    print("Ingesting statutory act correspondence mappings...")
    map_files = list(LEGAL_MAPS_DIR.glob("*.json"))
    count = 0

    for map_file in map_files:
        with open(map_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        mapping_info = data.get("mapping", {})
        act_mapping_name = mapping_info.get("mapping_id", map_file.stem)
        mappings = data.get("mappings", [])

        for item in mappings:
            src = item.get("source_section")
            tgt = item.get("target_section")
            rel = item.get("relationship", "CORRESPONDS_TO")

            if src and tgt:
                # Merge both sections and create CORRESPONDS_TO edge
                cypher = """
                MERGE (s1:StatuteSection {id: $src})
                MERGE (s2:StatuteSection {id: $tgt})
                MERGE (s1)-[r:CORRESPONDS_TO {act_mapping: $act_mapping}]->(s2)
                """
                session.run(cypher, src=src, tgt=tgt, act_mapping=act_mapping_name)
                count += 1

    print(f"  Ingested {count} statutory correspondence relationships.")


def ingest_judgment_chunks(session):
    print(f"Ingesting judgment chunks from {CHUNKS_JSONL_PATH}...")

    chunks = []
    with open(CHUNKS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    cases_count = 0
    chunks_count = 0
    ref_count = 0
    concept_count = 0
    citation_count = 0

    for chk in chunks:
        case_id = chk["case_id"]
        chunk_id = chk["chunk_id"]
        document_type = chk.get("document_type", "judgment")
        chunk_type = chk.get("chunk_type", "judgment")
        text = chk.get("text", "")
        token_count = chk.get("token_count", 0)
        source_field = chk.get("source", {}).get("field", "")
        page_start = chk.get("source", {}).get("page_start", 0)
        page_end = chk.get("source", {}).get("page_end", 0)

        # 1. Merge Case and Chunk, connect (:Case)-[:HAS_CHUNK]->(:Chunk)
        cypher_case_chunk = """
        MERGE (c:Case {id: $case_id})
        MERGE (chk:Chunk {id: $chunk_id})
        SET chk.case_id = $case_id,
            chk.document_type = $document_type,
            chk.chunk_type = $chunk_type,
            chk.text = $text,
            chk.token_count = $token_count,
            chk.source_field = $source_field,
            chk.page_start = $page_start,
            chk.page_end = $page_end
        MERGE (c)-[:HAS_CHUNK]->(chk)
        """
        session.run(
            cypher_case_chunk,
            case_id=case_id,
            chunk_id=chunk_id,
            document_type=document_type,
            chunk_type=chunk_type,
            text=text,
            token_count=token_count,
            source_field=source_field,
            page_start=page_start,
            page_end=page_end
        )
        chunks_count += 1

        # 2. Statutory references (:Chunk)-[:REFERS_TO]->(:StatuteSection)
        legal_refs = chk.get("legal_references", [])
        for ref in legal_refs:
            if ref:
                cypher_ref = """
                MATCH (chk:Chunk {id: $chunk_id})
                MERGE (s:StatuteSection {id: $ref_id})
                MERGE (chk)-[:REFERS_TO]->(s)
                """
                session.run(cypher_ref, chunk_id=chunk_id, ref_id=ref)
                ref_count += 1

        # 3. Legal concepts (:Chunk)-[:DISCUSSES]->(:LegalConcept)
        concepts = chk.get("legal_concepts", [])
        for concept in concepts:
            if concept:
                concept_name = concept.replace("LEGAL_CONCEPT:", "").replace("_", " ").title()
                cypher_concept = """
                MATCH (chk:Chunk {id: $chunk_id})
                MERGE (lc:LegalConcept {id: $concept_id})
                SET lc.name = $name
                MERGE (chk)-[:DISCUSSES]->(lc)
                """
                session.run(cypher_concept, chunk_id=chunk_id, concept_id=concept, name=concept_name)
                concept_count += 1

        # 4. Cited cases (:Case)-[:CITES_CASE]->(:CitedCase) and (:Chunk)-[:DISCUSSES_CASE]->(:CitedCase)
        cited_cases = chk.get("cited_cases", [])
        for cited_case in cited_cases:
            if cited_case:
                cypher_cited = """
                MATCH (c:Case {id: $case_id})
                MATCH (chk:Chunk {id: $chunk_id})
                MERGE (cc:CitedCase {id: $cited_name})
                SET cc.name = $cited_name
                MERGE (c)-[:CITES_CASE]->(cc)
                MERGE (chk)-[:DISCUSSES_CASE]->(cc)
                """
                session.run(cypher_cited, case_id=case_id, chunk_id=chunk_id, cited_name=cited_case)
                citation_count += 1

    print(f"  Ingested {chunks_count} chunks.")
    print(f"  Created {ref_count} statutory reference edges.")
    print(f"  Created {concept_count} legal concept edges.")
    print(f"  Created {citation_count} case citation edges.")


def main():
    print("=== Neo4j Knowledge Graph Ingestion ===")
    print(f"Connecting to Neo4j at {NEO4J_URI} as user '{NEO4J_USER}'...")

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Successfully connected to Neo4j database!")
    except Exception as e:
        print(f"\n[NEO4J CONNECTION ERROR]: Could not connect to Neo4j at {NEO4J_URI}.")
        print(f"Error details: {e}")
        print("\nPlease ensure Neo4j Desktop or Docker service is running.")
        print("To run via Docker:")
        print("  docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        sys.exit(1)

    with driver.session() as session:
        create_constraints(session)
        ingest_statute_mappings(session)
        ingest_judgment_chunks(session)

    driver.close()
    print("\nSUCCESS: Neo4j Knowledge Graph Ingestion complete!")


if __name__ == "__main__":
    main()
