"""
benchmark_modes.py

Compares three retrieval modes side-by-side on the same query set:

  Mode A – Pure Vector
    ChromaDB semantic search only; no graph at all.

  Mode B – Vector + Graph Expansion (existing expand_sections path)
    ChromaDB + Neo4j expand_sections() for section correspondence/cross-refs.
    This is the "classic" GraphRAG approach already wired in hybrid_reranker.

  Mode C – Full GraphRAG Fusion (new anchor + traversal + fuse_candidates)
    Mode B PLUS the new find_graph_anchors / traverse_anchor / fuse_candidates
    pipeline introduced in Phase 3.5.

Metrics computed per mode:
  Hit@1, Hit@3, MRR (Mean Reciprocal Rank), avg latency per query.

Output:
  reports/benchmark_modes.json   -- machine-readable comparison
  reports/benchmark_modes.txt    -- human-readable summary table
"""

import sys
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrieval.hybrid_reranker import GraphRAGHybridRetriever

# -----------------------------------------------------------------------
# Benchmark query set (same gold standard as evaluate_retrieval)
# -----------------------------------------------------------------------
BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "query": "What are the exceptions to murder under Section 300?",
        "expected_targets": ["IPC:300", "BNS:101"],
        "category": "Statutory Exception",
    },
    {
        "id": "Q2",
        "query": "Punishment for murder under Section 302 or BNS 103",
        "expected_targets": ["IPC:302", "BNS:103"],
        "category": "Punishment & Transition",
    },
    {
        "id": "Q3",
        "query": "Culpable homicide not amounting to murder under Section 304 or BNS 105",
        "expected_targets": ["IPC:304", "BNS:105"],
        "category": "Culpable Homicide",
    },
    {
        "id": "Q4",
        "query": "Procedure for statement examination under CrPC 313 or BNSS 351",
        "expected_targets": ["CRPC:313", "BNSS:351"],
        "category": "Criminal Procedure",
    },
    {
        "id": "Q5",
        "query": "Admissibility of confession to police under Section 25 IEA or BSA 23",
        "expected_targets": ["IEA:25", "BSA:23"],
        "category": "Evidence & Confession to Police",
    },
    {
        "id": "Q6",
        "query": "Definition of criminal conspiracy under IPC 120A or BNS 61",
        "expected_targets": ["IPC:120A", "BNS:61"],
        "category": "Criminal Conspiracy",
    },
    {
        "id": "Q7",
        "query": "Punishment for rape and definition under IPC 376 or BNS 64",
        "expected_targets": ["IPC:376", "BNS:64"],
        "category": "Sexual Offences",
    },
    {
        "id": "Q8",
        "query": "Causing death by negligence under IPC 304A or BNS 106",
        "expected_targets": ["IPC:304A", "BNS:106"],
        "category": "Negligent Homicide",
    },
    {
        "id": "Q9",
        "query": "What happens if someone kills a person under grave and sudden provocation during a sudden fight?",
        "expected_targets": ["IPC:300", "BNS:101", "IPC:304", "BNS:105"],
        "category": "Pure Semantic - Provocation & Homicide",
    },
    {
        "id": "Q10",
        "query": "What is the legal procedure for examining the accused person during a criminal trial?",
        "expected_targets": ["CRPC:313", "BNSS:351"],
        "category": "Pure Semantic - Criminal Trial Examination",
    },
    {
        "id": "Q11",
        "query": "Is a confession made by an accused person to a police officer admissible as evidence in court?",
        "expected_targets": ["IEA:25", "BSA:23"],
        "category": "Pure Semantic - Police Confession",
    },
    {
        "id": "Q12",
        "query": "What is the legal definition when two or more persons agree to commit an illegal act together?",
        "expected_targets": ["IPC:120A", "BNS:61"],
        "category": "Pure Semantic - Conspiracy",
    },
    {
        "id": "Q13",
        "query": "What is the punishment for causing the death of a person by a rash or negligent driving act?",
        "expected_targets": ["IPC:304A", "BNS:106"],
        "category": "Pure Semantic - Rash & Negligent Death",
    },
]

TOP_K = 5


# -----------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------
def compute_rank(retrieved_ids: List[str], expected: List[str]) -> Optional[int]:
    """Return 1-indexed rank of first expected hit, or None."""
    for i, sid in enumerate(retrieved_ids):
        if sid in expected:
            return i + 1
    return None


def aggregate_metrics(ranks: List[Optional[int]], n: int) -> Dict[str, Any]:
    hit1 = sum(1 for r in ranks if r == 1)
    hit3 = sum(1 for r in ranks if r is not None and r <= 3)
    rr = [1.0 / r if r else 0.0 for r in ranks]
    return {
        "hit_1_count": hit1,
        "hit_1_rate_pct": round(hit1 / n * 100, 2),
        "hit_3_count": hit3,
        "hit_3_rate_pct": round(hit3 / n * 100, 2),
        "mrr": round(sum(rr) / n, 4),
    }


def _extract_section_ids(statute_results: List[Dict]) -> List[str]:
    return [s.get("section_id", s.get("id", "")) for s in statute_results]


# -----------------------------------------------------------------------
# Mode A: Pure Vector (ChromaDB only, no graph at all)
# -----------------------------------------------------------------------
def run_mode_a(retriever: GraphRAGHybridRetriever, query: str) -> List[str]:
    n_fetch = max(10, TOP_K * 3)
    count = retriever.statute_col.count()
    s_res = retriever.statute_col.query(
        query_texts=[query],
        n_results=min(n_fetch, count),
        include=["metadatas", "distances"],
    )
    ids = []
    if s_res and s_res["ids"] and s_res["ids"][0]:
        for i, sid in enumerate(s_res["ids"][0]):
            meta = s_res["metadatas"][0][i]
            ids.append(meta.get("section_id", sid))
    return ids[:TOP_K]


# -----------------------------------------------------------------------
# Mode B: Vector + expand_sections (classic GraphRAG, no fusion)
# -----------------------------------------------------------------------
def run_mode_b(retriever: GraphRAGHybridRetriever, query: str) -> List[str]:
    """
    Runs the full search pipeline but strips out the fuse_candidates step
    by temporarily patching is_connected to return False, so the code
    falls through to the else branch (fused_statutes = reranked_statutes).
    """
    original = retriever.graph_retriever.is_connected

    # Monkeypatch: disable the graph anchor path but keep expand_sections
    retriever.graph_retriever.is_connected = lambda: False
    try:
        result = retriever.search(query, top_k_statutes=TOP_K, top_k_judgments=0)
    finally:
        retriever.graph_retriever.is_connected = original  # restore

    return _extract_section_ids(result.get("statute_results", []))


# -----------------------------------------------------------------------
# Mode C: Full GraphRAG Fusion (Phase 3.5)
# -----------------------------------------------------------------------
def run_mode_c(retriever: GraphRAGHybridRetriever, query: str) -> List[str]:
    result = retriever.search(query, top_k_statutes=TOP_K, top_k_judgments=0)
    return _extract_section_ids(result.get("statute_results", []))


# -----------------------------------------------------------------------
# Main benchmark driver
# -----------------------------------------------------------------------
def run_benchmark():
    print("=" * 65)
    print("  THREE-MODE GRAPHRAG BENCHMARK")
    print("  Mode A: Pure Vector (ChromaDB only)")
    print("  Mode B: Vector + Graph Expansion (expand_sections)")
    print("  Mode C: Full GraphRAG Fusion (Phase 3.5 anchors + fusion)")
    print("=" * 65)

    retriever = GraphRAGHybridRetriever()
    neo4j_up = retriever.graph_retriever.is_connected()
    print(f"\nNeo4j Connected: {neo4j_up}")
    if not neo4j_up:
        print("  WARNING: Neo4j offline – Modes B & C degrade to pure vector.\n")

    modes = {
        "mode_a": {"label": "Mode A – Pure Vector", "fn": run_mode_a},
        "mode_b": {"label": "Mode B – Vector + expand_sections", "fn": run_mode_b},
        "mode_c": {"label": "Mode C – Full GraphRAG Fusion", "fn": run_mode_c},
    }

    mode_ranks: Dict[str, List] = {m: [] for m in modes}
    mode_latencies: Dict[str, List] = {m: [] for m in modes}
    mode_query_rows: Dict[str, List] = {m: [] for m in modes}

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        query = item["query"]
        expected = item["expected_targets"]

        print(f"\n[{qid}] {query[:80]}")

        for mode_key, mode_info in modes.items():
            t0 = time.time()
            try:
                retrieved = mode_info["fn"](retriever, query)
            except Exception as exc:
                print(f"   {mode_key.upper()}: ERROR – {exc}")
                retrieved = []
            latency = round(time.time() - t0, 3)

            rank = compute_rank(retrieved, expected)
            mode_ranks[mode_key].append(rank)
            mode_latencies[mode_key].append(latency)

            if rank == 1:
                hit_label = "Hit@1 ✓"
            elif rank and rank <= 3:
                hit_label = f"Hit@3 ✓ (rank {rank})"
            else:
                hit_label = f"Miss    (rank {rank})"

            print(f"   {mode_info['label']:<42}  {hit_label:<22}  top3={retrieved[:3]}")

            mode_query_rows[mode_key].append({
                "id": qid,
                "query": query,
                "category": item["category"],
                "expected_targets": expected,
                "retrieved_top5": retrieved,
                "rank": rank,
                "hit_1": rank == 1,
                "hit_3": rank is not None and rank <= 3,
                "latency_s": latency,
            })

    # ---- Aggregate summary ----
    n = len(BENCHMARK_QUERIES)
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)

    header = f"{'Mode':<42}  {'Hit@1':>6}  {'Hit@3':>6}  {'MRR':>7}  {'AvgLat':>8}"
    print(header)
    print("-" * len(header))

    mode_summaries: Dict[str, Dict] = {}
    for mode_key, mode_info in modes.items():
        metrics = aggregate_metrics(mode_ranks[mode_key], n)
        avg_lat = round(sum(mode_latencies[mode_key]) / n, 3)
        mode_summaries[mode_key] = {**metrics, "avg_latency_s": avg_lat}
        print(
            f"{mode_info['label']:<42}  "
            f"{metrics['hit_1_rate_pct']:>5}%  "
            f"{metrics['hit_3_rate_pct']:>5}%  "
            f"{metrics['mrr']:>7.4f}  "
            f"{avg_lat:>7.3f}s"
        )

    # ---- Delta vs Mode A baseline ----
    a_h1 = mode_summaries["mode_a"]["hit_1_rate_pct"]
    a_mrr = mode_summaries["mode_a"]["mrr"]
    print("\n  Delta vs Mode A (baseline):")
    for mk in ["mode_b", "mode_c"]:
        dh1 = round(mode_summaries[mk]["hit_1_rate_pct"] - a_h1, 2)
        dm = round(mode_summaries[mk]["mrr"] - a_mrr, 4)
        sign_h = "+" if dh1 >= 0 else ""
        sign_m = "+" if dm >= 0 else ""
        print(f"    {modes[mk]['label']:<42}  Hit@1 {sign_h}{dh1}%  MRR {sign_m}{dm}")

    # ---- Save reports ----
    report = {
        "benchmark": "Three-Mode GraphRAG Benchmark",
        "neo4j_connected": neo4j_up,
        "total_queries": n,
        "modes": {
            mk: {
                "label": modes[mk]["label"],
                "metrics": mode_summaries[mk],
                "query_results": mode_query_rows[mk],
            }
            for mk in modes
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "benchmark_modes.json"
    txt_path = REPORTS_DIR / "benchmark_modes.txt"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    sep = "=" * 65
    txt_lines = [
        sep,
        "  THREE-MODE GRAPHRAG BENCHMARK RESULTS",
        sep,
        f"Neo4j Connected : {neo4j_up}",
        f"Total Queries   : {n}",
        "",
        f"{'Mode':<42}  {'Hit@1':>6}  {'Hit@3':>6}  {'MRR':>7}  {'AvgLat':>8}",
        "-" * 65,
    ]
    for mk, mode_info in modes.items():
        m = mode_summaries[mk]
        txt_lines.append(
            f"{mode_info['label']:<42}  "
            f"{m['hit_1_rate_pct']:>5}%  "
            f"{m['hit_3_rate_pct']:>5}%  "
            f"{m['mrr']:>7.4f}  "
            f"{m['avg_latency_s']:>7.3f}s"
        )

    txt_lines += ["", "-" * 65, "PER-QUERY DETAIL", "-" * 65]
    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        txt_lines.append(f"\n[{qid}] {item['query']}")
        txt_lines.append(f"  Expected: {item['expected_targets']}")
        for mk, mode_info in modes.items():
            row = next(r for r in mode_query_rows[mk] if r["id"] == qid)
            hit_str = "Hit@1" if row["hit_1"] else ("Hit@3" if row["hit_3"] else f"Miss(rank {row['rank']})")
            txt_lines.append(
                f"  {mode_info['label']:<40}  {hit_str:<15}  top3={row['retrieved_top5'][:3]}"
            )

    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    print(f"\nReports saved:\n  {json_path}\n  {txt_path}")

    retriever.close()
    return report


if __name__ == "__main__":
    run_benchmark()
