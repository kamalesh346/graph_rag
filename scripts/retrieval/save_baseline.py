#!/usr/bin/env python3
"""Baseline preservation script for Phase 3.5.
Runs the current GraphRAGHybridRetriever on the benchmark query set
and stores the raw provision and judgment payloads for later comparison.
"""

import json
import sys
from pathlib import Path

# Ensure the retrieval package is on the path
RETRIEVAL_DIR = Path(__file__).resolve().parent
if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from hybrid_reranker import GraphRAGHybridRetriever
from evaluate_retrieval import BENCHMARK_QUERIES

def run_baseline():
    retriever = GraphRAGHybridRetriever()
    baseline = []
    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        qstr = item["query"]
        # Use the same top‑k settings as the benchmark
        result = retriever.search(qstr, top_k_statutes=5, top_k_judgments=3)
        baseline.append({
            "id": qid,
            "query": qstr,
            "provisions": result.get("statute_results", []),
            "judgments": result.get("judgment_results", []),
        })
    retriever.close()
    return baseline

if __name__ == "__main__":
    data = run_baseline()
    reports_dir = Path(__file__).resolve().parents[2] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "baseline_results.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Baseline results written to {out_path}")
