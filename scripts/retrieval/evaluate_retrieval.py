"""
evaluate_retrieval.py

Benchmark suite for GraphRAG Hybrid Retrieval & Reranking Engine.
Evaluates Precision@1, Precision@3, Hit@1, Hit@3, and MRR (Mean Reciprocal Rank)
across core legal queries. Exports reports to reports/retrieval_benchmark_report.json.
"""

import sys
user_site = r"C:\Users\kamal\AppData\Roaming\Python\Python312\site-packages"
anaconda_site = r"C:\Users\kamal\anaconda3\Lib\site-packages"
if user_site in sys.path and anaconda_site in sys.path:
    sys.path.remove(anaconda_site)
    sys.path.insert(0, anaconda_site)

import json
import os
import time
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RETRIEVAL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"

if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from hybrid_reranker import GraphRAGHybridRetriever

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
    # --- PURE NATURAL LANGUAGE QUERIES (ZERO SECTION OR ACT MENTIONS) ---
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


def run_retrieval_benchmark():
    print("==================================================")
    print("   GRAPHRAG HYBRID RETRIEVAL BENCHMARK EVALUATION")
    print("==================================================")

    start_time = time.time()
    retriever = GraphRAGHybridRetriever()

    results = []
    reciprocal_ranks = []
    hit_1_count = 0
    hit_3_count = 0

    for item in BENCHMARK_QUERIES:
        q_id = item["id"]
        q_str = item["query"]
        expected = item["expected_targets"]

        res = retriever.search(q_str, top_k_statutes=5, top_k_judgments=3)
        statute_results = res["statute_results"]

        retrieved_sec_ids = [s["section_id"] for s in statute_results]

        # Calculate rank of first expected target hit
        target_rank = None
        for r_idx, sec_id in enumerate(retrieved_sec_ids):
            if sec_id in expected:
                target_rank = r_idx + 1
                break

        if target_rank == 1:
            hit_1_count += 1
            hit_3_count += 1
            reciprocal_ranks.append(1.0)
        elif target_rank in [2, 3]:
            hit_3_count += 1
            reciprocal_ranks.append(1.0 / target_rank)
        elif target_rank is not None:
            reciprocal_ranks.append(1.0 / target_rank)
        else:
            reciprocal_ranks.append(0.0)

        results.append({
            "id": q_id,
            "query": q_str,
            "category": item["category"],
            "expected_targets": expected,
            "retrieved_top_5": retrieved_sec_ids,
            "top_1_hit": (target_rank == 1),
            "top_3_hit": (target_rank is not None and target_rank <= 3),
            "target_rank": target_rank,
            "top_statute": statute_results[0] if statute_results else None,
        })

        print(f"\n[{q_id}] Query: '{q_str}'")
        print(f"  Expected Targets : {expected}")
        print(f"  Retrieved Top 3  : {retrieved_sec_ids[:3]}")
        print(f"  Rank #1 Hit      : {'PASS' if target_rank == 1 else 'FAIL (Rank ' + str(target_rank) + ')'}")

    total_queries = len(BENCHMARK_QUERIES)
    hit_1_rate = round((hit_1_count / total_queries) * 100, 2)
    hit_3_rate = round((hit_3_count / total_queries) * 100, 2)
    mrr = round(sum(reciprocal_ranks) / total_queries, 4)
    elapsed = round(time.time() - start_time, 2)

    print("\n==================================================")
    print("   BENCHMARK EVALUATION SUMMARY METRICS")
    print("==================================================")
    print(f"Total Benchmark Queries Tested : {total_queries}")
    print(f"Hit@1 Accuracy (Rank #1 Correct): {hit_1_rate}% ({hit_1_count}/{total_queries})")
    print(f"Hit@3 Accuracy (Top 3 Correct)  : {hit_3_rate}% ({hit_3_count}/{total_queries})")
    print(f"Mean Reciprocal Rank (MRR)      : {mrr}")
    print(f"Evaluation Execution Time       : {elapsed}s")

    report = {
        "benchmark": "GraphRAG Hybrid Retrieval Engine Evaluation",
        "execution_time_seconds": elapsed,
        "metrics": {
            "total_queries": total_queries,
            "hit_1_count": hit_1_count,
            "hit_1_rate_pct": hit_1_rate,
            "hit_3_count": hit_3_count,
            "hit_3_rate_pct": hit_3_rate,
            "mrr": mrr,
        },
        "query_results": results,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "retrieval_benchmark_report.json"
    txt_path = REPORTS_DIR / "retrieval_benchmark_report.txt"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    txt_lines = [
        "==================================================",
        "   GRAPHRAG HYBRID RETRIEVAL BENCHMARK REPORT",
        "==================================================",
        f"Execution Time       : {elapsed}s",
        f"Total Queries        : {total_queries}",
        f"Hit@1 Accuracy       : {hit_1_rate}% ({hit_1_count}/{total_queries})",
        f"Hit@3 Accuracy       : {hit_3_rate}% ({hit_3_count}/{total_queries})",
        f"MRR (Mean Recip. Rank): {mrr}",
        "--------------------------------------------------",
        "PER-QUERY BENCHMARK RESULTS:",
        "--------------------------------------------------",
    ]
    for r in results:
        status_str = "PASS [Hit@1]" if r["top_1_hit"] else f"FAIL [Rank {r['target_rank']}]"
        txt_lines.append(f"[{r['id']}] {status_str} - '{r['query']}'")
        txt_lines.append(f"     Expected: {r['expected_targets']} | Retrieved: {r['retrieved_top_5'][:3]}")

    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    print(f"\nBenchmark reports saved to {json_path} and {txt_path}.")

    retriever.close()
    return report


if __name__ == "__main__":
    run_retrieval_benchmark()
