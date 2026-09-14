"""
graph_retriever.py

Queries Neo4j Knowledge Graph to expand statutory sections and legal queries via:
- :CORRESPONDS_TO edges (IPC ↔ BNS, CrPC ↔ BNSS, IEA ↔ BSA)
- :REFERS_TO cross-reference edges
- :DISCUSSES legal concepts
"""

import os
from typing import Any, Dict, List, Optional, Set

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class Neo4jGraphRetriever:
    """Interface for Graph-Augmented Legal Query Expansion using Neo4j."""

    def __init__(self):
        self.driver = None
        self._connect()

    def _connect(self):
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
            self.driver = driver
        except Exception as e:
            # Gracefully set driver to None if Neo4j is unavailable
            self.driver = None

    def is_connected(self) -> bool:
        return self.driver is not None

    def expand_sections(self, section_ids: List[str]) -> Dict[str, Any]:
        """
        Given a list of section IDs (e.g. ['IPC:300', 'IPC:304']),
        queries Neo4j for corresponding sections, cross-references, and legal concepts.
        """
        if not self.driver or not section_ids:
            return {
                "corresponds_to": [],
                "refers_to": [],
                "concepts": [],
                "all_expanded_sections": list(section_ids),
            }

        corresponds_to = set()
        refers_to = set()
        concepts = set()
        all_expanded = set(section_ids)

        try:
            with self.driver.session() as session:
                for sec_id in section_ids:
                    # 1. Act-to-Act Correspondence (:CORRESPONDS_TO)
                    result_corr = session.run("""
                    MATCH (s:StatuteSection {id: $sec_id})-[r:CORRESPONDS_TO]-(c:StatuteSection)
                    RETURN c.id AS target_id
                    """, sec_id=sec_id)
                    for rec in result_corr:
                        target = rec["target_id"]
                        corresponds_to.add(target)
                        all_expanded.add(target)

                    # 2. Cross-References (:REFERS_TO)
                    result_ref = session.run("""
                    MATCH (s:StatuteSection {id: $sec_id})-[r:REFERS_TO]->(c:StatuteSection)
                    RETURN c.id AS target_id
                    """, sec_id=sec_id)
                    for rec in result_ref:
                        target = rec["target_id"]
                        refers_to.add(target)
                        all_expanded.add(target)

                    # 3. Concepts (:DISCUSSES)
                    result_con = session.run("""
                    MATCH (s:StatuteSection {id: $sec_id})-[:DISCUSSES]->(c:LegalConcept)
                    RETURN c.name AS concept_name
                    """, sec_id=sec_id)
                    for rec in result_con:
                        concepts.add(rec["concept_name"])

        except Exception as e:
            print(f"Warning: Neo4j query error: {e}")

        return {
            "corresponds_to": sorted(list(corresponds_to)),
            "refers_to": sorted(list(refers_to)),
            "concepts": sorted(list(concepts)),
            "all_expanded_sections": sorted(list(all_expanded)),
        }

    def close(self):
        if self.driver:
            self.driver.close()


if __name__ == "__main__":
    retriever = Neo4jGraphRetriever()
    print(f"Neo4j Connected: {retriever.is_connected()}")
    if retriever.is_connected():
        exp = retriever.expand_sections(["IPC:300", "IPC:304", "CRPC:313"])
        print("Expansion test for ['IPC:300', 'IPC:304', 'CRPC:313']:")
        print(f"  Corresponds To : {exp['corresponds_to']}")
        print(f"  Refers To       : {exp['refers_to']}")
        print(f"  Concepts        : {exp['concepts']}")
        print(f"  All Expanded    : {exp['all_expanded_sections']}")
    retriever.close()
