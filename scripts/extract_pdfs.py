"""Extract PDF text page by page, parse legal sections, and write JSON artifacts using [CASE_ID.json] naming convention."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fitz  # PyMuPDF
from parse_legal_sections import parse_case

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "judgement_pdf"
if not PDF_DIR.exists():
    PDF_DIR = ROOT / "data" / "pdfs"

EXTRACTED_DIR = ROOT / "data" / "extracted"
JUDGEMENT_TXT_DIR = ROOT / "data" / "judgement_txt"
SELECTED_CASES_PATH = ROOT / "selected_cases.json"


def load_metadata_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for record in records:
        path_key = record.get("path") or ""
        if path_key:
            result[path_key] = record
    return result


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return re.sub(r"_+", "_", cleaned).strip("_")


import re

def main() -> int:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    JUDGEMENT_TXT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_map = load_metadata_map(SELECTED_CASES_PATH)
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    print(f"Processing {len(pdf_files)} PDF files with parse_legal_sections...")
    success_count = 0
    fail_count = 0

    for pdf in pdf_files:
        stem = pdf.stem
        meta = dict(metadata_map.get(stem, {}))
        
        case_id = meta.get("case_id") or stem
        case_id_slug = sanitize_filename(case_id)
        if not case_id_slug:
            case_id_slug = stem

        try:
            with fitz.open(pdf) as doc:
                pages = [{"page_number": idx + 1, "text": page.get_text("text")} for idx, page in enumerate(doc)]
            
            # Save plain text in data/judgement_txt/[CASE_ID].txt
            txt_path = JUDGEMENT_TXT_DIR / f"{case_id_slug}.txt"
            full_text = "\n\n".join([f"--- PAGE {p['page_number']} ---\n{p['text']}" for p in pages])
            txt_path.write_text(full_text, encoding="utf-8")

            # Parse legal sections & enriched metadata
            structured = parse_case(meta, pages)
            
            # Save structured JSON in data/extracted/[CASE_ID].json
            out_json = EXTRACTED_DIR / f"{case_id_slug}.json"
            out_json.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")
            
            total_paras = structured['judgment']['total_paragraphs']
            print(f"[OK] {out_json.name} ({case_id}) -> paras: {total_paras}, sections: {len(structured['processing']['sections_found'])}")
            success_count += 1
        except Exception as e:
            print(f"[FAIL] {pdf.name}: {e}", file=sys.stderr)
            fail_count += 1

    print(f"\nCompleted: {success_count} parsed JSONs saved in data/extracted/, {len(list(JUDGEMENT_TXT_DIR.glob('*.txt')))} .txt files in data/judgement_txt/")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())