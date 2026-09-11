"""Validate structured case JSON and emit human- and machine-readable reports."""
from __future__ import annotations

import json
from pathlib import Path

from parse_legal_sections import MANDATORY

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    structured_dir = ROOT / "data" / "extracted" / "structured"
    report_dir = ROOT / "reports"; report_dir.mkdir(exist_ok=True)
    cases = []
    for path in sorted(structured_dir.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        sections = item["sections"]
        found = set(item["processing"].get("sections_found", []))
        missing = sorted(MANDATORY - found)
        chars = sum(len(page.get("text", "")) for page in [item["judgment"]]) + sum(
            len(value) if isinstance(value, str) else sum(len(v) for v in value) for value in sections.values()
        )
        status = "PASS" if not missing and item["judgment"]["text"] else "WARN"
        cases.append({
            "filename": path.name, "case_id": item["metadata"].get("case_id"),
            "pages": item["processing"]["pdf_pages"], "characters": chars,
            "sections_found": sorted(found), "missing_sections": missing,
            "optional_sections_detected": sorted(found & {"websites_cited", "books_cited", "periodicals_cited"}),
            "paragraph_count": len(item["judgment"]["paragraphs"]),
            "tables_detected": item["judgment"]["text"].count("[TABLE]"),
            "document_type": item["metadata"].get("document_type"), "ocr_used": False, "status": status,
        })
    report = {"total_pdfs": len(cases), "successfully_extracted": sum(c["status"] == "PASS" for c in cases), "needs_review": sum(c["status"] != "PASS" for c in cases), "cases": cases}
    (report_dir / "extraction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["LEGAL CORPUS EXTRACTION REPORT", "=" * 40, f"Total PDFs: {report['total_pdfs']}", f"PASS: {report['successfully_extracted']}", f"Needs review: {report['needs_review']}", ""]
    for case in cases:
        lines.append(f"{case['status']}: {case['case_id']} | {case['pages']} pages | {case['paragraph_count']} paragraphs | missing: {', '.join(case['missing_sections']) or 'none'}")
    (report_dir / "extraction_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
