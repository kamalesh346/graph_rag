"""Validate chunked judgment JSON artifacts and emit human- and machine-readable reports."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = ROOT / "data" / "chunks"
REPORTS_DIR = ROOT / "reports"


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    chunk_files = [f for f in sorted(CHUNKS_DIR.glob("*.json")) if f.name != "all_judgment_chunks.json"]

    case_reports = []
    total_chunks = 0
    total_issues = 0
    total_headnotes = 0
    total_topics = 0
    total_judgments = 0

    for f in chunk_files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        counts = payload["chunk_counts"]
        chunks = payload["chunks"]

        token_counts = [c["token_count"] for c in chunks if c["token_count"] > 0]
        avg_tokens = int(sum(token_counts) / len(token_counts)) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0

        # Check metadata validity
        valid = True
        for c in chunks:
            if not c.get("chunk_id") or not c.get("chunk_type") or "text" not in c:
                valid = False
                break

        case_reports.append({
            "filename": f.name,
            "case_id": payload["case_id"],
            "total_chunks": payload["total_chunks"],
            "chunk_counts": counts,
            "avg_tokens": avg_tokens,
            "max_tokens": max_tokens,
            "valid": valid,
        })

        total_chunks += payload["total_chunks"]
        total_issues += counts["issue"]
        total_headnotes += counts["headnote"]
        total_topics += counts["legal_topic"]
        total_judgments += counts["judgment"]

    report_data = {
        "total_cases_chunked": len(case_reports),
        "total_chunks": total_chunks,
        "summary_counts": {
            "issue": total_issues,
            "headnote": total_headnotes,
            "legal_topic": total_topics,
            "judgment": total_judgments,
        },
        "cases": case_reports,
    }

    (REPORTS_DIR / "chunking_report.json").write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "INTELLIGENT CHUNKING REPORT",
        "=" * 40,
        f"Total Cases Chunked: {len(case_reports)}",
        f"Total Chunks Generated: {total_chunks}",
        f"  - Issue Chunks: {total_issues}",
        f"  - Headnote Chunks: {total_headnotes}",
        f"  - Legal Topic Chunks: {total_topics}",
        f"  - Judgment Chunks: {total_judgments}",
        "",
        "PER-CASE BREAKDOWN:",
        "-" * 40,
    ]
    for cr in case_reports:
        c = cr["chunk_counts"]
        lines.append(
            f"{'PASS' if cr['valid'] else 'FAIL'}: {cr['case_id']} | Total: {cr['total_chunks']} chunks (issue: {c['issue']}, headnote: {c['headnote']}, topic: {c['legal_topic']}, judgment: {c['judgment']}) | Avg tokens: {cr['avg_tokens']} | Max tokens: {cr['max_tokens']}"
        )

    (REPORTS_DIR / "chunking_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
