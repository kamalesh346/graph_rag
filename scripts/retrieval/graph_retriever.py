"""
graph_retriever.py

Queries Neo4j Knowledge Graph to expand statutory sections and legal queries via:
- :CORRESPONDS_TO edges (IPC ↔ BNS, CrPC ↔ BNSS, IEA ↔ BSA)
- :REFERS_TO cross-reference edges
- :DISCUSSES legal concepts
"""

import os
from typing import Any, Dict, List, Optional, Set
import re
# Expose Neo4j driver class for test patching
try:
    from neo4j import GraphDatabase as _RealGraphDatabase
except Exception:
    _RealGraphDatabase = None  # type: ignore

class _DummyGraphDatabase:
    @staticmethod
    def driver(*args, **kwargs):
        raise RuntimeError("Neo4j driver not available")

GraphDatabase = _RealGraphDatabase if _RealGraphDatabase is not None else _DummyGraphDatabase

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
            
            driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            driver.verify_connectivity()
            self.driver = driver
            print("Neo4j graph retriever connected successfully.")
        
        except Exception as e:
            self.driver = None
            print(f"Neo4j graph retriever connection failed: {type(e).__name__}:{e}")

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

    def find_graph_anchors(self, query: str, parsed_query: dict | None = None) -> dict:
        """Return anchor nodes for four independent retrieval routes.
        Keys: explicit, concept, fulltext, case – each a list of node records.
        """
        anchors = {"explicit": [], "concept": [], "fulltext": [], "case": []}
        if not self.driver:
            return anchors

        with self.driver.session() as session:
            # A. Explicit statutory references (if parser supplied IDs)
            if parsed_query and parsed_query.get("section_ids"):
                for sid in parsed_query["section_ids"]:
                    res = session.run("MATCH (s:StatuteSection {id: $sid}) RETURN s", sid=sid)
                    anchors["explicit"].extend(res.data())

            # B. Legal concept matching (simple keyword search)
            words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
            if words:
                for w in set(words):
                    res = session.run(
                        """
                        MATCH (c:LegalConcept)
                        WHERE toLower(c.name) CONTAINS $w OR toLower(c.id) CONTAINS $w
                        RETURN c
                        """,
                        w=w,
                    )
                    anchors["concept"].extend(res.data())

            # C. Full‑text chunk search (fallback to CONTAINS on Chunk.text)
            res = session.run(
                """
                MATCH (chk:Chunk)
                WHERE toLower(chk.text) CONTAINS $qt
                RETURN chk LIMIT 20
                """,
                qt=query.lower(),
            )
            anchors["fulltext"].extend(res.data())

            # D. Case / citation matching
            res = session.run(
                """
                MATCH (c:Case)
                WHERE toLower(c.id) CONTAINS $qt
                RETURN c LIMIT 5
                """,
                qt=query.lower(),
            )
            anchors["case"].extend(res.data())

            res = session.run(
                """
                MATCH (cc:CitedCase)
                WHERE toLower(cc.id) CONTAINS $qt OR toLower(cc.name) CONTAINS $qt
                RETURN cc LIMIT 5
                """,
                qt=query.lower(),
            )
            anchors["case"].extend(res.data())

        return anchors

    def traverse_anchor(self, anchor_node: dict, max_depth: int = 3) -> dict:
        """Perform a bounded traversal from a given anchor node.
        Returns dict with sets: statutes, judgments, cited_cases.
        """
        results = {"statutes": set(), "judgments": set(), "cited_cases": set()}
        if not self.driver:
            return results

        # Extract the underlying node and its id
        node_record = next(iter(anchor_node.values()))
        nid = node_record.get("id")
        if not nid:
            return results

        with self.driver.session() as session:
            # Depth must be a literal integer in Cypher – format it in.
            d = int(max_depth)
            cypher = f"""
            CALL {{
                MATCH (s:StatuteSection {{id: $id}})
                OPTIONAL MATCH (s)-[:CORRESPONDS_TO|REFERS_TO|DISCUSSES*1..{d}]-(sec:StatuteSection)
                OPTIONAL MATCH (s)-[:REFERS_TO*1..{d}]-(chk:Chunk)
                OPTIONAL MATCH (c:Case)-[:HAS_CHUNK]-(chk)
                OPTIONAL MATCH (c)-[:CITES_CASE]-(cc:CitedCase)
                RETURN collect(DISTINCT sec.id) AS statutes,
                       collect(DISTINCT chk.id) AS judgments,
                       collect(DISTINCT cc.id) AS cited_cases
                UNION
                MATCH (lc:LegalConcept {{id: $id}})
                OPTIONAL MATCH (lc)<-[:DISCUSSES]-(chk2:Chunk)
                OPTIONAL MATCH (chk2)-[:REFERS_TO]->(s2:StatuteSection)
                OPTIONAL MATCH (c2:Case)-[:HAS_CHUNK]-(chk2)
                OPTIONAL MATCH (c2)-[:CITES_CASE]-(cc2:CitedCase)
                RETURN collect(DISTINCT s2.id) AS statutes,
                       collect(DISTINCT chk2.id) AS judgments,
                       collect(DISTINCT cc2.id) AS cited_cases
                UNION
                MATCH (chk3:Chunk {{id: $id}})
                OPTIONAL MATCH (chk3)-[:REFERS_TO]->(s3:StatuteSection)
                OPTIONAL MATCH (c3:Case)-[:HAS_CHUNK]-(chk3)
                OPTIONAL MATCH (c3)-[:CITES_CASE]-(cc3:CitedCase)
                RETURN collect(DISTINCT s3.id) AS statutes,
                       collect(DISTINCT chk3.id) AS judgments,
                       collect(DISTINCT cc3.id) AS cited_cases
                UNION
                MATCH (cx) WHERE (cx:Case OR cx:CitedCase) AND cx.id = $id
                OPTIONAL MATCH (cx)-[:HAS_CHUNK]->(chk4:Chunk)
                OPTIONAL MATCH (chk4)-[:REFERS_TO]->(s4:StatuteSection)
                OPTIONAL MATCH (cx)-[:CITES_CASE]-(cc4:CitedCase)
                RETURN collect(DISTINCT s4.id) AS statutes,
                       collect(DISTINCT chk4.id) AS judgments,
                       collect(DISTINCT cc4.id) AS cited_cases
            }}
            RETURN
                reduce(a=[], x IN collect(statutes) | a + x) AS statutes,
                reduce(a=[], x IN collect(judgments) | a + x) AS judgments,
                reduce(a=[], x IN collect(cited_cases) | a + x) AS cited_cases
            """
            rec = session.run(cypher, id=nid).single()
            if rec:
                results["statutes"] = set(filter(None, rec["statutes"]))
                results["judgments"] = set(filter(None, rec["judgments"]))
                results["cited_cases"] = set(filter(None, rec["cited_cases"]))

        return results
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
