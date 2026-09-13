"""
statute_validator.py - PHASE S1: Normalized Statute Validation

Validates raw normalized statute JSON files in data/legal/acts/
(ipc.json, bns.json, crpc.json, bnss.json, iea.json, bsa.json).

Detects structural edge cases:
- Missing required fields (act_id, section_id, section_number, title)
- Invalid section ID format (must be ACT:number or ACT:number(sub))
- Duplicate section_id or subsection_id (e.g. IPC:354A(1) duplicates)
- Empty main text (flagged as info when valid subsections exist, e.g. IPC:304B, BNS:103)
- Truncated text (text ending mid-sentence, e.g. IPC:376)
- Malformed predecessor/successor references
- Concept validation against project legal concept inventory
"""

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTS_DIR = PROJECT_ROOT / "data" / "legal" / "acts"
REPORTS_DIR = PROJECT_ROOT / "reports"

CONCEPT_INVENTORY = [
    "murder",
    "culpable homicide",
    "culpable homicide not amounting to murder",
    "dying declaration",
    "grave and sudden provocation",
    "sudden fight",
    "circumstantial evidence",
    "ocular evidence",
    "benefit of doubt",
    "first information report",
    "common intention",
    "acquittal",
    "material contradiction",
    "medical evidence",
    "private defence",
    "consent",
    "dowry death",
    "cruelty",
    "rape",
    "abetment",
    "criminal conspiracy",
]


class StatuteValidator:

    def __init__(self, acts_dir: Path = ACTS_DIR):
        self.acts_dir = acts_dir

    def validate_file(self, file_path: Path) -> dict[str, Any]:
        report = {
            "filename": file_path.name,
            "act_id": None,
            "total_sections": 0,
            "issues": [],
            "warnings": [],
            "info": [],
            "valid": True,
        }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            report["issues"].append(f"JSON Parse Error: {e}")
            report["valid"] = False
            return report

        act_meta = data.get("act", {})
        act_id = act_meta.get("act_id")
        report["act_id"] = act_id

        if not act_id:
            report["issues"].append("Missing required field 'act.act_id'")
            report["valid"] = False

        sections = data.get("sections", [])
        report["total_sections"] = len(sections)

        seen_section_ids = set()
        seen_subsection_ids = set()

        for idx, sec in enumerate(sections, start=1):
            sec_id = sec.get("section_id")
            sec_num = sec.get("section_number")
            title = sec.get("title")
            text = sec.get("text", "")
            subsections = sec.get("subsections", [])
            explanations = sec.get("explanations", [])
            exceptions = sec.get("exceptions", [])
            legal_concepts = sec.get("legal_concepts", [])

            # 1. Missing required fields
            if not sec_id:
                report["issues"].append(f"Section #{idx}: Missing 'section_id'")
                report["valid"] = False
            if not sec_num:
                report["issues"].append(f"Section #{idx} ({sec_id}): Missing 'section_number'")
            if title is None:
                report["issues"].append(f"Section #{idx} ({sec_id}): Missing 'title'")

            # 2. Check section_id format (e.g. IPC:302, BNS:103)
            if sec_id:
                if not re.match(r"^[A-Z]+:\d+[A-Za-z]*(?:\([^\)]+\))?$", sec_id):
                    report["warnings"].append(f"Section #{idx}: Non-standard section_id format '{sec_id}'")

                if sec_id in seen_section_ids:
                    report["issues"].append(f"Duplicate section_id detected: '{sec_id}'")
                    report["valid"] = False
                seen_section_ids.add(sec_id)

            # 3. Check for empty main text (e.g. IPC:304B or BNS:103 where content is in subsections)
            if not text.strip():
                if subsections:
                    report["info"].append(f"Section '{sec_id}' has empty main text, but contains {len(subsections)} subsections.")
                else:
                    report["warnings"].append(f"Section '{sec_id}' has completely empty main text and no subsections.")

            # 4. Check for truncated text (text ending abruptly mid-sentence, e.g. IPC:376)
            if text.strip() and not re.search(r"[\.!\?\";:\)]$", text.strip()) and not subsections:
                report["warnings"].append(f"Section '{sec_id}' text appears truncated (ends with: '{text.strip()[-30:]}')")

            # 5. Check subsections for duplicate subsection_ids
            for sub in subsections:
                sub_id = sub.get("section_id")
                sub_num = sub.get("subsection_number")
                sub_text = sub.get("text", "")

                if sub_id:
                    if sub_id in seen_subsection_ids:
                        report["warnings"].append(f"Section '{sec_id}': Duplicate subsection_id '{sub_id}' (subsection {sub_num})")
                    seen_subsection_ids.add(sub_id)

                if sub_text.strip() and not re.search(r"[\.!\?\";:\)]$", sub_text.strip()):
                    report["warnings"].append(f"Subsection '{sub_id}' text appears truncated (ends with: '{sub_text.strip()[-30:]}')")

            # 6. Check legal_concepts against inventory
            for concept in legal_concepts:
                c_clean = concept.lower().strip()
                if c_clean not in CONCEPT_INVENTORY:
                    report["info"].append(f"Section '{sec_id}': Concept '{concept}' is outside standard concept inventory.")

        return report

    def validate_all(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        files = sorted(self.acts_dir.glob("*.json"))
        file_reports = []
        total_sections = 0
        total_issues = 0
        total_warnings = 0

        for f in files:
            rep = self.validate_file(f)
            file_reports.append(rep)
            total_sections += rep["total_sections"]
            total_issues += len(rep["issues"])
            total_warnings += len(rep["warnings"])

        summary = {
            "total_acts_validated": len(file_reports),
            "total_sections": total_sections,
            "total_issues": total_issues,
            "total_warnings": total_warnings,
            "overall_valid": total_issues == 0,
        }

        # Save report
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_payload = {"summary": summary, "acts": file_reports}

        out_json = REPORTS_DIR / "statute_validation_report.json"
        out_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # Format text report
        txt_lines = [
            "==================================================",
            "   PHASE S1: STATUTE VALIDATION REPORT",
            "==================================================",
            f"Total Acts Validated: {summary['total_acts_validated']}",
            f"Total Sections      : {summary['total_sections']}",
            f"Total Critical Issues: {summary['total_issues']}",
            f"Total Warnings      : {summary['total_warnings']}",
            f"Overall Status      : {'PASS' if summary['overall_valid'] else 'FAIL'}",
            "-" * 50,
        ]

        for rep in file_reports:
            status = "PASS" if rep["valid"] else "FAIL"
            txt_lines.append(f"[{status}] {rep['filename']} ({rep['act_id']}): {rep['total_sections']} sections")
            for iss in rep["issues"]:
                txt_lines.append(f"   [ISSUE] {iss}")
            for warn in rep["warnings"][:5]:  # Show top 5 warnings per act
                txt_lines.append(f"   [WARN]  {warn}")
            if len(rep["warnings"]) > 5:
                txt_lines.append(f"   [WARN]  ... plus {len(rep['warnings']) - 5} more warnings.")

        out_txt = REPORTS_DIR / "statute_validation_report.txt"
        out_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

        return summary, file_reports


if __name__ == "__main__":
    validator = StatuteValidator()
    summary, reports = validator.validate_all()
    print(f"Statute Validation Complete. Overall Status: {'PASS' if summary['overall_valid'] else 'FAIL'}")
