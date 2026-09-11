"""Normalize structured legal case JSONs into data/normalized/.

Leaves raw JSONs in data/extracted/ untouched as immutable ground truth.
Standardizes section citations (e.g., '304 Part I' -> 'IPC Section 304, Part I',
'304A' / '304(A)' -> 'IPC Section 304A', '300 Exception 1' -> 'IPC Section 300 Exception 1'),
court names ('Supreme Court of India'), and legal concepts ('First Information Report (FIR)')
directly in the structured text and metadata for downstream chunking, ChromaDB, and Neo4j.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = ROOT / "data" / "extracted"
NORMALIZED_DIR = ROOT / "data" / "normalized"


def normalize_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return text

    # 1. Complex Part/Section phrases
    text = re.sub(r"\bunder\s+Part\s+I\s+of\s+s(?:ec(?:tion)?)?\.?\s*304\s*(?:of\s*(?:the\s*)?IPC)?\b", "under IPC Section 304, Part I", text, flags=re.I)
    text = re.sub(r"\bunder\s+Part\s+II\s+of\s+s(?:ec(?:tion)?)?\.?\s*304\s*(?:of\s*(?:the\s*)?IPC)?\b", "under IPC Section 304, Part II", text, flags=re.I)
    text = re.sub(r"\bPart\s+I\s+of\s+(?:Section|s\.)\s*304\s*(?:of\s*(?:the\s*)?IPC)?\b", "IPC Section 304, Part I", text, flags=re.I)
    text = re.sub(r"\bPart\s+II\s+of\s+(?:Section|s\.)\s*304\s*(?:of\s*(?:the\s*)?IPC)?\b", "IPC Section 304, Part II", text, flags=re.I)

    # 2. Section 304 variations
    text = re.sub(r"\b(?:Section|s\.)\s*304\s*(?:\(\s*Part\s*1\s*\)|Part\s*I|Part\s*1|Part-I)\b", "IPC Section 304, Part I", text, flags=re.I)
    text = re.sub(r"\b(?:Section|s\.)\s*304\s*(?:\(\s*Part\s*2\s*\)|Part\s*II|Part\s*2|Part-II)\b", "IPC Section 304, Part II", text, flags=re.I)
    text = re.sub(r"\b(?:Section|s\.)\s*304\s*(?:\(\s*A\s*\)|-A)\b", "IPC Section 304A", text, flags=re.I)
    text = re.sub(r"\b(?:Section|s\.)\s*304A\b", "IPC Section 304A", text, flags=re.I)
    text = re.sub(r"\b(?:Section|s\.)\s*304-?B\b", "IPC Section 304B", text, flags=re.I)

    # 3. Section 300 Exceptions
    text = re.sub(r"\bException\s+1\s+(?:to|of)\s+(?:Section|s\.)\s*300\s*(?:of\s*(?:the\s*)?IPC)?\b", "IPC Section 300 Exception 1", text, flags=re.I)
    text = re.sub(r"\b(?:Section|s\.)\s*300\s+Exception\s+1\b", "IPC Section 300 Exception 1", text, flags=re.I)
    text = re.sub(r"\bException\s+4\s+(?:to|of)\s+(?:Section|s\.)\s*300\s*(?:of\s*(?:the\s*)?IPC)?\b", "IPC Section 300 Exception 4", text, flags=re.I)
    text = re.sub(r"\b(?:Section|s\.)\s*300\s+Exception\s+(?:4|IV)\b", "IPC Section 300 Exception 4", text, flags=re.I)

    # 4. IPC Sections
    for sec in ["302", "300", "299", "301", "304", "34", "201", "84"]:
        text = re.sub(r"\b(?:u/s\.?|u/section)\s*" + sec + r"\b(?:\s*of\s*(?:the\s*)?IPC)?\b", f"under IPC Section {sec}", text, flags=re.I)
        text = re.sub(r"\b(?:s\.|ss\.)\s*" + sec + r"\b(?:\s*of\s*(?:the\s*)?IPC)?\b", f"IPC Section {sec}", text, flags=re.I)

    text = re.sub(r"\b(?:u/s\.?|u/section)\s*120-?B\b(?:\s*of\s*(?:the\s*)?IPC)?\b", "under IPC Section 120B", text, flags=re.I)
    text = re.sub(r"\b(?:s\.|ss\.)\s*120-?B\b(?:\s*of\s*(?:the\s*)?IPC)?\b", "IPC Section 120B", text, flags=re.I)
    text = re.sub(r"\b(?:u/s\.?|u/section)\s*498-?A\b(?:\s*of\s*(?:the\s*)?IPC)?\b", "under IPC Section 498A", text, flags=re.I)
    text = re.sub(r"\b(?:s\.|ss\.)\s*498-?A\b(?:\s*of\s*(?:the\s*)?IPC)?\b", "IPC Section 498A", text, flags=re.I)

    # 5. CrPC Sections
    for sec in ["313", "161", "162", "164", "173", "207", "209", "230", "232", "366"]:
        text = re.sub(r"\b(?:u/s\.?|u/section)\s*" + sec + r"\b(?:\s*of\s*(?:the\s*)?CrPC)?\b", f"under CrPC Section {sec}", text, flags=re.I)
        text = re.sub(r"\b(?:s\.|ss\.)\s*" + sec + r"\b(?:\s*of\s*(?:the\s*)?CrPC)?\b", f"CrPC Section {sec}", text, flags=re.I)
    text = re.sub(r"\b(?:u/s\.?|u/section)\s*173\s*\(2\)\b(?:\s*of\s*(?:the\s*)?CrPC)?\b", "under CrPC Section 173(2)", text, flags=re.I)

    # 6. Evidence Act Sections
    for sec in ["105", "106", "165", "25", "27", "114", "145", "157"]:
        text = re.sub(r"\b(?:u/s\.?|u/section)\s*" + sec + r"\b(?:\s*of\s*(?:the\s*)?Evidence\s+Act)?\b", f"under Evidence Act Section {sec}", text, flags=re.I)
        text = re.sub(r"\b(?:s\.|ss\.)\s*" + sec + r"\b(?:\s*of\s*(?:the\s*)?Evidence\s+Act)?\b", f"Evidence Act Section {sec}", text, flags=re.I)

    # 7. Concept replacements
    text = re.sub(r"\bFIR\b", "First Information Report (FIR)", text)
    return text


def normalize_value(val: Any) -> Any:
    if isinstance(val, str):
        return normalize_text(val)
    elif isinstance(val, list):
        return [normalize_value(item) for item in val]
    elif isinstance(val, dict):
        return {k: normalize_value(v) for k, v in val.items()}
    return val


def main() -> None:
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(EXTRACTED_DIR.glob("*.json"))
    print(f"Normalizing {len(raw_files)} structured case files into data/normalized/...")

    for path in raw_files:
        raw_case = json.loads(path.read_text(encoding="utf-8"))
        normalized_case = normalize_value(raw_case)

        # Ensure metadata contains canonical court identity
        meta = normalized_case.get("metadata", {})
        meta["court"] = "Supreme Court of India"
        meta["canonical_court_id"] = "COURT:SUPREME_COURT_OF_INDIA"
        normalized_case["metadata"] = meta

        out_path = NORMALIZED_DIR / path.name
        out_path.write_text(json.dumps(normalized_case, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[NORMALIZED] {out_path.name}")

    print(f"\nDone. Saved {len(raw_files)} normalized structured JSONs in data/normalized/.")


if __name__ == "__main__":
    main()
