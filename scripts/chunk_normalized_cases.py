"""Intelligent Chunking Engine for Supreme Court Legal-GraphRAG Corpus.

Processes normalized judgment JSON records from data/normalized/ and produces
independent, retrievable chunk streams:
1. ISSUE (from issue_for_consideration)
2. HEADNOTE (split into distinct legal propositions with related_paragraphs traceability)
3. LEGAL_TOPIC (only generated for distinct topics not already covered in headnotes)
4. JUDGMENT (intelligent paragraph grouping with deterministic sliding 1-paragraph overlap & 1000 token max budget)

Outputs chunk artifacts in data/chunks/[CASE_ID_SLUG]_chunks.json and
a consolidated JSONL file in data/chunks/all_judgment_chunks.jsonl.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_DIR = ROOT / "data" / "normalized"
CHUNKS_DIR = ROOT / "data" / "chunks"

STATUTES = {
    "IPC": (r"\bIPC\b|\bIndian Penal Code\b|\bPenal Code,?\s*1860\b", "Indian Penal Code, 1860"),
    "BNS": (r"\bBNS\b|\bBharatiya Nyaya Sanhita\b", "Bharatiya Nyaya Sanhita, 2023"),
    "CrPC": (r"\bCr\.?P\.?C\.?\b|\bCode of Criminal Procedure\b", "Code of Criminal Procedure, 1973"),
    "BNSS": (r"\bBNSS\b|\bBharatiya Nagarik Suraksha Sanhita\b", "Bharatiya Nagarik Suraksha Sanhita, 2023"),
    "Evidence Act": (r"\bEvidence Act\b|\bIndian Evidence Act\b", "Indian Evidence Act, 1872"),
    "BSA": (r"\bBSA\b|\bBharatiya Sakshya Adhiniyam\b", "Bharatiya Sakshya Adhiniyam, 2023"),
}

CONCEPTS = {
    "Murder": r"\bmurder\b",
    "Culpable homicide": r"\bculpable homicide\b",
    "Culpable homicide not amounting to murder": r"\bculpable homicide not amounting to murder\b",
    "Dying declaration": r"\bdying declaration\b",
    "Grave and sudden provocation": r"\bgrave and sudden provocation\b",
    "Sudden fight": r"\bsudden fight\b",
    "Circumstantial evidence": r"\bcircumstantial evidence\b",
    "Ocular evidence": r"\bocular evidence\b",
    "Benefit of doubt": r"\bbenefit of doubt\b",
    "First information report (FIR)": r"\bFIR\b|\bfirst information report\b",
    "Common intention": r"\bcommon intention\b",
    "Acquittal": r"\bacquittal\b|\bacquitted\b",
    "Material contradiction": r"\bmaterial contradiction\b|\bmaterial omissions?\b",
    "Medical evidence": r"\bmedical evidence\b|\bpost[- ]mortem\b",
}


def normalize_text_spacing(text: str) -> str:
    """Clean single intra-paragraph line wraps into single spaces while preserving double newlines."""
    if not text or not isinstance(text, str):
        return text
    paragraphs = text.split("\n\n")
    cleaned_paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in paragraphs if p.strip()]
    return "\n\n".join(cleaned_paras)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.33)


def canonical_concept_id(name: str) -> str:
    return "LEGAL_CONCEPT:" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def roman_part(value: str) -> str:
    mapping = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "I": "I", "II": "II", "III": "III", "IV": "IV", "V": "V"}
    return mapping.get(value.upper(), value.upper())


def clean_exc(exc_val: str) -> str:
    m = re.search(r"(?:[ivx]+|\d+)", exc_val, re.I)
    val = m.group().lower() if m else "1"
    return {"i": "1", "1": "1", "ii": "2", "2": "2", "iii": "3", "3": "3", "iv": "4", "4": "4", "v": "5", "5": "5"}.get(val, val)


def canonical_provision(act: str, raw: str) -> str:
    value = re.sub(r"\s+", " ", raw).strip()
    act_id = act.replace(" ", "")

    exc_match = re.search(r"(?:(\d+)\s*[-–—]?\s*(?:\(\s*)?exception\s*([ivx0-9]+)(?:\s*\))?|exception\s*([ivx0-9]+)\s*(?:to|of)?\s*(?:s(?:ection)?\.?\s*)?(\d+))", value, re.I)
    if exc_match:
        sec = exc_match.group(1) or exc_match.group(4)
        exc_num = clean_exc(exc_match.group(2) or exc_match.group(3))
        return f"{act_id}:{sec}(Exception {exc_num})"

    part_match = re.search(r"(\d+)\s*[-–—]?\s*(?:\(\s*)?part\s*([ivx0-9]+)(?:\s*\))?", value, re.I)
    if part_match:
        sec, part_val = part_match.group(1), roman_part(part_match.group(2))
        return f"{act_id}:{sec}(Part {part_val})"

    sub_match = re.fullmatch(r"(\d+)\s*[-–—]?\s*\(\s*([0-9]+)\s*\)", value)
    if sub_match:
        return f"{act_id}:{sub_match.group(1)}({sub_match.group(2)})"

    letter_sub = re.fullmatch(r"(\d+)\s*[-–—]?\s*(?:\(\s*)?([A-Za-z]+)(?:\s*\))?", value)
    if letter_sub:
        return f"{act_id}:{letter_sub.group(1)}{letter_sub.group(2).upper()}"

    number = re.match(r"\d+", value)
    sec = number.group() if number else value
    return f"{act_id}:{sec}"


def extract_references(text: str) -> list[str]:
    refs = set()
    for act, (pat, _) in STATUTES.items():
        statute = re.compile(pat, re.I)
        sec_num = re.compile(r"\b(?:Section|Sections|S(?:ec)?\.)\s*(\d+[A-Za-z]*(?:\s*[-–—]?\s*(?:\([^)]*\)|Part\s+[IVX0-9]+|Exception\s+[IVX0-9]+))?)", re.I)
        for clause in re.split(r"[\n.;]", text):
            if not statute.search(clause):
                continue
            for m in sec_num.finditer(clause):
                refs.add(canonical_provision(act, m.group(1)))
            for m in re.finditer(r"\bException\s+([IVX0-9]+)\s+(?:to|of)?\s*(?:Section|Sections|S(?:ec)?\.)\s*(\d+)\b", clause, re.I):
                refs.add(canonical_provision(act, f"{m.group(2)} Exception {m.group(1)}"))
            for m in re.finditer(r"\bs\.\s*(\d+[A-Za-z]*)\b", clause, re.I):
                refs.add(canonical_provision(act, m.group(1)))
    return sorted(list(refs))


def extract_concepts(text: str) -> list[str]:
    found = []
    for name, pat in CONCEPTS.items():
        if re.search(pat, text, re.I):
            found.append(canonical_concept_id(name))
    return sorted(found)


def extract_citations(text: str, master_citations: list[str]) -> list[str]:
    found = []
    for cite in master_citations:
        title_part = cite.split("[")[0].split("(")[0].strip()
        if len(title_part) > 5 and title_part.lower() in text.lower():
            found.append(cite)
    return sorted(list(set(found)))


def sanitize_slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return re.sub(r"_+", "_", cleaned).strip("_")


def parse_headnote_propositions(headnote_text: str) -> list[dict[str, Any]]:
    if not headnote_text or not isinstance(headnote_text, str):
        return []

    text = headnote_text.strip()
    raw_blocks = re.split(r"(\[\s*Paras?\s+[\d,\s\-]+\s*\])", text)
    propositions: list[dict[str, Any]] = []
    
    current_text = ""
    for part in raw_blocks:
        part_str = part.strip()
        if not part_str:
            continue
        para_match = re.fullmatch(r"\[\s*Paras?\s+([\d,\s\-]+)\s*\]", part_str)
        if para_match:
            current_text += " " + part_str
            related_paras = []
            for item in para_match.group(1).split(","):
                item = item.strip()
                if "-" in item:
                    sub = item.split("-")
                    if len(sub) == 2 and sub[0].isdigit() and sub[1].isdigit():
                        related_paras.extend(list(range(int(sub[0]), int(sub[1]) + 1)))
                elif item.isdigit():
                    related_paras.append(int(item))
            related_paras = sorted(list(set(related_paras)))

            clean_prop_text = normalize_text_spacing(current_text.strip())
            first_line = clean_prop_text.split("\n")[0].strip()
            heading_parts = first_line.split("–")
            heading = heading_parts[0].strip() if heading_parts else first_line[:60]
            if len(heading_parts) > 1 and len(heading) < 30:
                heading += " – " + heading_parts[1].strip()

            propositions.append({
                "heading": heading,
                "text": clean_prop_text,
                "related_paragraphs": related_paras
            })
            current_text = ""
        else:
            if current_text:
                current_text += "\n" + part_str
            else:
                current_text = part_str

    if current_text.strip():
        clean_prop_text = normalize_text_spacing(current_text.strip())
        first_line = clean_prop_text.split("\n")[0].strip()
        heading_parts = first_line.split("–")
        heading = heading_parts[0].strip() if heading_parts else first_line[:60]
        propositions.append({
            "heading": heading,
            "text": clean_prop_text,
            "related_paragraphs": []
        })

    return propositions


def group_judgment_paragraphs(
    paragraphs: list[dict[str, Any]],
    case_id: str,
    slug: str,
    master_citations: list[str]
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    
    # 1. Enrich individual paragraphs & clean intra-paragraph line breaks
    enriched_paras = []
    for p in paragraphs:
        p_num = p.get("paragraph_number") or p.get("para_number") or 0
        raw_text = p.get("text", "").strip()
        text = normalize_text_spacing(raw_text)
        page_start = p.get("page_start") or p.get("page_number") or 1
        page_end = p.get("page_end") or page_start
        
        p_tokens = estimate_tokens(text)
        refs = extract_references(text)
        concepts = extract_concepts(text)
        cites = extract_citations(text, master_citations)
        
        enriched_paras.append({
            "paragraph_number": p_num,
            "text": text,
            "page_start": page_start,
            "page_end": page_end,
            "token_count": p_tokens,
            "legal_references": refs,
            "legal_concepts": concepts,
            "cited_cases": cites,
        })

    # 2. Intelligent paragraph grouping loop with deterministic sliding overlap
    curr_paras: list[dict[str, Any]] = []
    curr_tokens = 0

    idx = 0
    seq = 1
    while idx < len(enriched_paras):
        p = enriched_paras[idx]
        
        # Check oversized paragraph handling (>1000 tokens)
        if p["token_count"] > 1000:
            if curr_paras:
                chunk_obj = build_judgment_chunk(case_id, slug, seq, curr_paras)
                chunks.append(chunk_obj)
                seq += 1
                curr_paras = []
                curr_tokens = 0

            sub_chunks = split_oversized_paragraph(p, case_id, slug, seq)
            chunks.extend(sub_chunks)
            seq += len(sub_chunks)
            idx += 1
            continue

        # Check if adding p exceeds hard max token limit (1000) or max paras (8)
        would_exceed_tokens = (curr_tokens + p["token_count"]) > 1000
        would_exceed_paras = len(curr_paras) >= 8

        if curr_paras and (would_exceed_tokens or would_exceed_paras or (curr_tokens >= 500)):
            # Close current chunk
            chunk_obj = build_judgment_chunk(case_id, slug, seq, curr_paras)
            chunks.append(chunk_obj)
            seq += 1

            # Deterministic 1-paragraph sliding overlap with Edge Case Handling
            # Only overlap last_p into the next chunk if last_p.token_count + p.token_count <= 1000 tokens
            last_p = curr_paras[-1]
            if (last_p["token_count"] + p["token_count"]) <= 1000:
                curr_paras = [last_p, p]
                curr_tokens = last_p["token_count"] + p["token_count"]
            else:
                curr_paras = [p]
                curr_tokens = p["token_count"]
            idx += 1
            continue

        curr_paras.append(p)
        curr_tokens += p["token_count"]
        idx += 1

    if curr_paras:
        chunk_obj = build_judgment_chunk(case_id, slug, seq, curr_paras)
        chunks.append(chunk_obj)

    return chunks


def build_judgment_chunk(case_id: str, slug: str, seq: int, paras: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_id = f"{slug}_JUDGMENT_{seq:03d}"
    text = "\n\n".join([p["text"] for p in paras])
    p_nums = [p["paragraph_number"] for p in paras if p["paragraph_number"] > 0]
    p_start = min(p["page_start"] for p in paras)
    p_end = max(p["page_end"] for p in paras)

    refs = sorted(list({ref for p in paras for ref in p["legal_references"]}))
    concepts = sorted(list({c for p in paras for c in p["legal_concepts"]}))
    cites = sorted(list({c for p in paras for c in p["cited_cases"]}))

    return {
        "chunk_id": chunk_id,
        "case_id": case_id,
        "document_type": "judgment",
        "chunk_type": "judgment",
        "text": text,
        "source": {
            "field": "judgment.paragraphs",
            "paragraphs": p_nums,
            "page_start": p_start,
            "page_end": p_end,
        },
        "legal_references": refs,
        "legal_concepts": concepts,
        "cited_cases": cites,
        "token_count": estimate_tokens(text),
    }


def split_oversized_paragraph(p: dict[str, Any], case_id: str, slug: str, start_seq: int) -> list[dict[str, Any]]:
    text = p["text"]
    sentences = re.split(r"(?<=\.)\s+", text)
    sub_texts: list[str] = []
    curr: list[str] = []
    curr_t = 0
    
    for s in sentences:
        st = estimate_tokens(s)
        if curr_t + st > 800 and curr:
            sub_texts.append(" ".join(curr))
            curr = [s]
            curr_t = st
        else:
            curr.append(s)
            curr_t += st
    if curr:
        sub_texts.append(" ".join(curr))

    sub_chunks = []
    p_num = p["paragraph_number"]
    for idx, stext in enumerate(sub_texts, start=1):
        sub_id = f"{slug}_JUDGMENT_{(start_seq + idx - 1):03d}"
        sub_chunks.append({
            "chunk_id": sub_id,
            "case_id": case_id,
            "document_type": "judgment",
            "chunk_type": "judgment",
            "text": stext,
            "source": {
                "field": "judgment.paragraphs",
                "paragraphs": [p_num],
                "parent_paragraph": p_num,
                "sub_part": f"{p_num}-{chr(64 + idx)}",
                "page_start": p["page_start"],
                "page_end": p["page_end"],
            },
            "legal_references": extract_references(stext),
            "legal_concepts": extract_concepts(stext),
            "cited_cases": p["cited_cases"],
            "token_count": estimate_tokens(stext),
        })
    return sub_chunks


def process_case(file_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    meta = raw_data.get("metadata", {})
    case_id = meta.get("case_id") or file_path.stem
    slug = sanitize_slug(case_id)
    sections = raw_data.get("sections", {})
    judgment_obj = raw_data.get("judgment", {})
    canon_refs = raw_data.get("canonical_references", {})
    master_citations = canon_refs.get("case_citations", [])

    all_chunks: list[dict[str, Any]] = []

    # Stream 1: Issue Chunking
    issue_text = normalize_text_spacing(sections.get("issue_for_consideration", "").strip())
    if issue_text:
        issue_chunk = {
            "chunk_id": f"{slug}_ISSUE_001",
            "case_id": case_id,
            "document_type": "judgment",
            "chunk_type": "issue",
            "text": issue_text,
            "source": {
                "field": "issue_for_consideration"
            },
            "legal_references": extract_references(issue_text),
            "legal_concepts": extract_concepts(issue_text),
            "cited_cases": extract_citations(issue_text, master_citations),
            "token_count": estimate_tokens(issue_text),
        }
        all_chunks.append(issue_chunk)

    # Stream 2: Headnote Proposition Chunking
    headnote_text = sections.get("headnotes", "").strip()
    headnote_props = []
    if headnote_text:
        headnote_props = parse_headnote_propositions(headnote_text)
        for idx, prop in enumerate(headnote_props, start=1):
            h_chunk = {
                "chunk_id": f"{slug}_HEADNOTE_{idx:03d}",
                "case_id": case_id,
                "document_type": "judgment",
                "chunk_type": "headnote",
                "topic_heading": prop["heading"],
                "text": prop["text"],
                "source": {
                    "field": "headnotes"
                },
                "related_paragraphs": prop["related_paragraphs"],
                "legal_references": extract_references(prop["text"]),
                "legal_concepts": extract_concepts(prop["text"]),
                "cited_cases": extract_citations(prop["text"], master_citations),
                "token_count": estimate_tokens(prop["text"]),
            }
            all_chunks.append(h_chunk)

    # Stream 3: Legal Topic Chunking (Deduplicated against Headnotes)
    # Only generate legal_topic chunks if headnotes are missing OR if topic is not a duplicate
    headnote_texts = [p["text"].lower() for p in headnote_props]
    legal_topics = sections.get("legal_topics", [])
    
    topic_seq = 1
    for topic in legal_topics:
        if isinstance(topic, dict):
            t_heading = topic.get("heading", f"Legal Topic {topic_seq}")
            t_text = normalize_text_spacing(topic.get("text", "").strip())
        else:
            t_str = str(topic)
            t_parts = t_str.split("–", 1)
            t_heading = t_parts[0].strip() if len(t_parts) > 1 else f"Legal Topic {topic_seq}"
            t_text = normalize_text_spacing(t_str.strip())

        # Deduplication check against headnote propositions
        is_duplicate = any(t_text.lower() in h_txt or h_txt[:100] in t_text.lower() for h_txt in headnote_texts)
        if headnote_props and is_duplicate:
            continue

        topic_chunk = {
            "chunk_id": f"{slug}_LEGAL_TOPIC_{topic_seq:03d}",
            "case_id": case_id,
            "document_type": "judgment",
            "chunk_type": "legal_topic",
            "topic_heading": t_heading,
            "text": t_text,
            "source": {
                "field": "legal_topics"
            },
            "legal_references": extract_references(t_text),
            "legal_concepts": extract_concepts(t_text),
            "cited_cases": extract_citations(t_text, master_citations),
            "token_count": estimate_tokens(t_text),
        }
        all_chunks.append(topic_chunk)
        topic_seq += 1

    # Stream 4: Judgment Paragraph Grouping
    paragraphs = judgment_obj.get("paragraphs", [])
    if paragraphs:
        j_chunks = group_judgment_paragraphs(paragraphs, case_id, slug, master_citations)
        all_chunks.extend(j_chunks)

    payload = {
        "case_id": case_id,
        "total_chunks": len(all_chunks),
        "chunk_counts": {
            "issue": sum(1 for c in all_chunks if c["chunk_type"] == "issue"),
            "headnote": sum(1 for c in all_chunks if c["chunk_type"] == "headnote"),
            "legal_topic": sum(1 for c in all_chunks if c["chunk_type"] == "legal_topic"),
            "judgment": sum(1 for c in all_chunks if c["chunk_type"] == "judgment"),
        },
        "chunks": all_chunks,
    }
    return payload, all_chunks


def main() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(NORMALIZED_DIR.glob("*.json"))
    print(f"Processing intelligent chunking across {len(files)} normalized judgment records...")

    master_jsonl = CHUNKS_DIR / "all_judgment_chunks.jsonl"
    jsonl_lines = []

    total_chunks = 0
    total_issues = 0
    total_headnotes = 0
    total_topics = 0
    total_judgments = 0

    for file_path in files:
        payload, chunks = process_case(file_path)
        slug = sanitize_slug(payload["case_id"])
        out_json = CHUNKS_DIR / f"{slug}_chunks.json"
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        for c in chunks:
            jsonl_lines.append(json.dumps(c, ensure_ascii=False))

        counts = payload["chunk_counts"]
        total_chunks += payload["total_chunks"]
        total_issues += counts["issue"]
        total_headnotes += counts["headnote"]
        total_topics += counts["legal_topic"]
        total_judgments += counts["judgment"]

        print(f"[CHUNKED] {file_path.name} -> {payload['total_chunks']} chunks (issue: {counts['issue']}, headnote: {counts['headnote']}, topic: {counts['legal_topic']}, judgment: {counts['judgment']})")

    master_jsonl.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
    print(f"\nCompleted Refined Intelligent Chunking Engine!")
    print(f"Total Cases: {len(files)}")
    print(f"Total Chunks Generated: {total_chunks}")
    print(f"  - Issue Chunks: {total_issues}")
    print(f"  - Headnote Chunks: {total_headnotes}")
    print(f"  - Legal Topic Chunks: {total_topics}")
    print(f"  - Judgment Chunks: {total_judgments}")
    print(f"Saved artifacts in data/chunks/*.json and data/chunks/all_judgment_chunks.jsonl.")


if __name__ == "__main__":
    main()
