import json
import os
import urllib.request
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTED_CASES_PATH = os.path.join(BASE_DIR, "selected_cases.json")
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")

os.makedirs(PDF_DIR, exist_ok=True)

with open(SELECTED_CASES_PATH, "r", encoding="utf-8") as f:
    cases = json.load(f)

print(f"Loaded {len(cases)} cases from selected_cases.json")

downloaded = 0
skipped = 0
failed = 0

for case in cases:
    path = case.get("path")
    year = case.get("year")
    case_id = case.get("case_id", "Unknown ID")
    
    if not path or not year:
        print(f"Skipping case {case_id}: missing path or year")
        continue

    dest_filename = f"{path}.pdf"
    dest_path = os.path.join(PDF_DIR, dest_filename)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
        print(f"[EXISTS] {dest_filename} ({os.path.getsize(dest_path)} bytes)")
        skipped += 1
        continue

    s3_url = f"https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com/data/pdf/year={year}/english/{path}_EN.pdf"
    
    print(f"Downloading {case_id} ({path}) from S3...")
    try:
        req = urllib.request.Request(
            s3_url, 
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                data = response.read()
                if data.startswith(b"%PDF"):
                    with open(dest_path, "wb") as out_file:
                        out_file.write(data)
                    print(f"  -> Successfully saved {dest_filename} ({len(data)} bytes)")
                    downloaded += 1
                else:
                    print(f"  -> ERROR: Content downloaded is not a valid PDF binary stream")
                    failed += 1
            else:
                print(f"  -> ERROR: HTTP {response.status}")
                failed += 1
    except Exception as e:
        print(f"  -> ERROR downloading {path}: {e}")
        failed += 1

print(f"\nDownload Summary: {downloaded} downloaded, {skipped} skipped, {failed} failed.")
