import sys
user_site = r"C:\Users\kamal\AppData\Roaming\Python\Python312\site-packages"
anaconda_site = r"C:\Users\kamal\anaconda3\Lib\site-packages"
if user_site in sys.path and anaconda_site in sys.path:
    sys.path.remove(anaconda_site)
    sys.path.insert(0, anaconda_site)

import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_DIR = Path(__file__).resolve().parent
if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
from chromadb.utils import embedding_functions

from query_parser import LegalQueryParser
from graph_retriever import Neo4jGraphRetriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
STATUTE_COLLECTION = "legal_statute_chunks"
JUDGMENT_COLLECTION = "legal_judgment_chunks"


class GraphRAGHybridRetriever:
    """Integrated GraphRAG Hybrid Search Engine combining Vector, Graph, & Metadata Boosts."""

    def __init__(self):
        self.parser = LegalQueryParser()
        self.graph_retriever = Neo4jGraphRetriever()
        
        # Initialize ChromaDB client
        if not CHROMA_DB_DIR.exists():
            raise FileNotFoundError(f"ChromaDB directory not found at {CHROMA_DB_DIR}")

        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.statute_col = self.chroma_client.get_collection(name=STATUTE_COLLECTION)
        
        try:
            self.judgment_col = self.chroma_client.get_collection(name=JUDGMENT_COLLECTION)
        except Exception:
            self.judgment_col = None

    def search(
        self,
        query: str,
        top_k_statutes: int = 3,
        top_k_judgments: int = 3,
        rrf_k: int = 60,
    ) -> Dict[str, Any]:
        """
        Executes a GraphRAG hybrid retrieval workflow:
        1. Parses legal entities and intent from query.
        2. Traverses Neo4j knowledge graph for correspondence/references/concepts.
        3. Queries ChromaDB for vector similarity candidates.
        4. Applies metadata-boosting (exact section match boost) & RRF score fusion.
        """
        # 1. Parse Query
        parsed = self.parser.parse(query)
        extracted_secs = parsed["extracted_sections"]
        primary_target = parsed["primary_target_section"]

        # 2. Graph Traversal & Query Expansion
        graph_expansion = self.graph_retriever.expand_sections(extracted_secs)
        all_graph_secs = set(graph_expansion["all_expanded_sections"])
        corresponds_secs = set(graph_expansion["corresponds_to"])

        # 3. Dense Vector Retrieval (Statutes)
        raw_statute_candidates = []
        n_fetch_statutes = max(10, top_k_statutes * 3)
        
        s_res = self.statute_col.query(
            query_texts=[query],
            n_results=min(n_fetch_statutes, self.statute_col.count()),
            include=["documents", "metadatas", "distances"]
        )

        if s_res and s_res["ids"] and s_res["ids"][0]:
            s_ids = s_res["ids"][0]
            s_docs = s_res["documents"][0]
            s_metas = s_res["metadatas"][0]
            s_dists = s_res["distances"][0]

            for rank_idx in range(len(s_ids)):
                sec_id = s_metas[rank_idx].get("section_id", s_ids[rank_idx])
                raw_statute_candidates.append({
                    "id": s_ids[rank_idx],
                    "section_id": sec_id,
                    "act": s_metas[rank_idx].get("act"),
                    "section_number": s_metas[rank_idx].get("section_number"),
                    "title": s_metas[rank_idx].get("title"),
                    "text": s_docs[rank_idx],
                    "metadata": s_metas[rank_idx],
                    "vector_distance": s_dists[rank_idx],
                    "vector_rank": rank_idx + 1,
                    "type": "statute",
                })

        # 4. Rerank & Score Fusion for Statutes
        reranked_statutes = self._rerank_statutes(
            raw_statute_candidates,
            extracted_secs=extracted_secs,
            primary_target=primary_target,
            corresponds_secs=corresponds_secs,
            all_graph_secs=all_graph_secs,
            rrf_k=rrf_k,
        )

        # 5. Dense Vector Retrieval (Judgments)
        reranked_judgments = []
        if self.judgment_col:
            try:
                count_j = self.judgment_col.count()
                if count_j > 0:
                    n_fetch_judgments = max(10, top_k_judgments * 3)
                    j_res = self.judgment_col.query(
                        query_texts=[query],
                        n_results=min(n_fetch_judgments, count_j),
                        include=["documents", "metadatas", "distances"]
                    )
                    if j_res and j_res["ids"] and j_res["ids"][0]:
                        j_ids = j_res["ids"][0]
                        j_docs = j_res["documents"][0]
                        j_metas = j_res["metadatas"][0]
                        j_dists = j_res["distances"][0]

                        for rank_idx in range(len(j_ids)):
                            ref_str = str(j_metas[rank_idx].get("legal_references", ""))
                            sec_match_boost = 0.0
                            for sec in extracted_secs:
                                if sec in ref_str:
                                    sec_match_boost += 0.5

                            rrf_score = (1.0 / (rrf_k + rank_idx + 1)) + sec_match_boost

                            reranked_judgments.append({
                                "id": j_ids[rank_idx],
                                "case_id": j_metas[rank_idx].get("case_id"),
                                "chunk_type": j_metas[rank_idx].get("chunk_type"),
                                "legal_references": j_metas[rank_idx].get("legal_references"),
                                "text": j_docs[rank_idx],
                                "metadata": j_metas[rank_idx],
                                "vector_distance": j_dists[rank_idx],
                                "vector_rank": rank_idx + 1,
                                "rrf_score": round(rrf_score, 6),
                                "type": "judgment",
                            })

                        reranked_judgments.sort(key=lambda x: x["rrf_score"], reverse=True)
            except Exception as e:
                print(f"Warning: Judgment vector retrieval warning: {e}")

        return {
            "query": query,
            "parsed_intent": parsed,
            "graph_expansion": graph_expansion,
            "statute_results": reranked_statutes[:top_k_statutes],
            "judgment_results": reranked_judgments[:top_k_judgments],
        }

    def _rerank_statutes(
        self,
        candidates: List[Dict[str, Any]],
        extracted_secs: List[str],
        primary_target: Optional[str],
        corresponds_secs: Set[str],
        all_graph_secs: Set[str],
        rrf_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Calculates RRF score + Metadata boosts for statute candidate chunks.
        """
        for item in candidates:
            sec_id = item["section_id"]
            vec_rank = item["vector_rank"]

            # Base RRF score from vector rank position
            base_rrf = 1.0 / (rrf_k + vec_rank)

            # Metadata Boosts
            boost = 0.0
            boost_reasons = []

            # 1. Primary Target Section Match (Major Boost)
            if primary_target and sec_id == primary_target:
                boost += 2.0
                boost_reasons.append(f"Exact Target Section Match ({sec_id})")
            elif sec_id in extracted_secs:
                boost += 1.5
                boost_reasons.append(f"Extracted Query Section Match ({sec_id})")

            # 2. Neo4j Act Correspondence Boost (e.g., BNS 101 matching IPC 300)
            if sec_id in corresponds_secs:
                boost += 1.0
                boost_reasons.append(f"Graph Act-Correspondence Match ({sec_id})")
            elif sec_id in all_graph_secs:
                boost += 0.5
                boost_reasons.append(f"Graph Related Section ({sec_id})")

            final_score = base_rrf + boost
            item["rrf_score"] = round(final_score, 6)
            item["boost_applied"] = round(boost, 4)
            item["boost_reasons"] = boost_reasons

        # Sort candidate chunks by final fused score descending
        candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        return candidates

    def close(self):
        if self.graph_retriever:
            self.graph_retriever.close()


if __name__ == "__main__":
    retriever = GraphRAGHybridRetriever()
    queries = [
        "What are the exceptions to murder under Section 300?",
        "Punishment for culpable homicide not amounting to murder under Section 304 or BNS 105",
        "Procedure for statement examination under CrPC 313 or BNSS 351",
    ]
    for q in queries:
        res = retriever.search(q, top_k_statutes=3, top_k_judgments=2)
        print(f"\n==================================================")
        print(f"QUERY: '{q}'")
        print(f"Target Section Extracted: {res['parsed_intent']['extracted_sections']}")
        print(f"Graph Expansion         : {res['graph_expansion']['corresponds_to']}")
        print(f"--------------------------------------------------")
        print("TOP RETRIEVED STATUTES:")
        for idx, s in enumerate(res["statute_results"]):
            print(
                f"  #{idx+1} [{s['section_id']}] (Score: {s['rrf_score']} | Boost: {s['boost_applied']} | Dist: {s['vector_distance']:.4f})"
            )
            print(f"      Reasons: {s['boost_reasons']}")
            print(f"      Text   : {s['text'][:110]}...")
    retriever.close()
