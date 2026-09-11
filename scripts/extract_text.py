import os
import json

try:
    import pymupdf as fitz
except ImportError:
    import fitz

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
EXTRACTED_DIR = os.path.join(BASE_DIR, "data", "extracted")
SELECTED_CASES_PATH = os.path.join(BASE_DIR, "selected_cases.json")

os.makedirs(EXTRACTED_DIR, exist_ok=True)

# Map path -> case metadata
metadata_map = {}
if os.path.exists(SELECTED_CASES_PATH):
    with open(SELECTED_CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
        for c in cases:
            if "path" in c:
                metadata_map[c["path"]] = c

pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF files in {PDF_DIR}")

processed = 0
failed = 0

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    case_key = os.path.splitext(pdf_file)[0]
    out_txt_path = os.path.join(EXTRACTED_DIR, f"{case_key}.txt")
    out_json_path = os.path.join(EXTRACTED_DIR, f"{case_key}.json")
    
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages_text.append(f"--- PAGE {page_num + 1} ---\n{text}")
        
        full_text = "\n\n".join(pages_text)
        
        # Save plain text
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        # Save structured json with metadata
        meta = metadata_map.get(case_key, {})
        extracted_data = {
            "case_id": meta.get("case_id", ""),
            "title": meta.get("title", ""),
            "citation": meta.get("citation", ""),
            "decision_date": meta.get("decision_date", ""),
            "year": meta.get("year", ""),
            "judge": meta.get("judge", ""),
            "description": meta.get("description", ""),
            "path": case_key,
            "total_pages": len(doc),
            "char_count": len(full_text),
            "full_text": full_text
        }
        
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
            
        print(f"[EXTRACTED] {case_key}: {len(doc)} pages, {len(full_text)} characters -> {out_txt_path}")
        processed += 1
    except Exception as e:
        print(f"[ERROR] Extracting {pdf_file}: {e}")
        failed += 1

print(f"\nExtraction Summary: {processed} processed, {failed} failed.")
