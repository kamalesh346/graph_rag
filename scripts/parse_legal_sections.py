"""Deterministic parser for Supreme Court legal-report layout.

Extracts structured legal sections, enriched case metadata (bench/judges,
appeal_number, citation, court, title normalization), clean legal topics,
and full sequential judgment paragraphs.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

SECTION_ORDER = [
    "issue_for_consideration", "headnotes", "legal_topics", "case_law_cited",
    "acts_cited", "keywords", "case_arising_from", "appearances",
    "websites_cited", "books_cited", "periodicals_cited", "judgment",
]


def _normalise_heading(line: str) -> str:
    line = line.upper().replace("–", "-").replace("—", "-")
    return re.sub(r"[^A-Z0-9]+", " ", line).strip()


def classify_heading(line: str) -> str | None:
    """Return a canonical section name only for a recognisable heading line."""
    heading = _normalise_heading(line)
    compact = heading.replace(" ", "")
    if "ISSUE FOR CONSIDERATION" in heading:
        return "issue_for_consideration"
    if heading == "HEADNOTES" or heading.startswith("HEADNOTES "):
        return "headnotes"
    if "CASE LAW CITED" in heading:
        return "case_law_cited"
    if "LIST OF ACTS" in heading:
        return "acts_cited"
    if "LIST OF KEYWORDS" in heading:
        return "keywords"
    if "CASE ARISING FROM" in heading:
        return "case_arising_from"
    if "APPEARANCES FOR PARTIES" in heading:
        return "appearances"
    if "LIST OF WEBSITES" in heading:
        return "websites_cited"
    if "LIST OF BOOKS" in heading:
        return "books_cited"
    if "LIST OF PERIODICALS" in heading:
        return "periodicals_cited"
    if ("JUDGMENT" in heading or "ORDER" in heading) and (
        "SUPREME COURT" in heading or compact in {"JUDGMENT", "ORDER"}
    ):
        return "judgment"
    return None


def clean_pages(raw_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise spacing and strip page header/footer noise."""
    pages: list[dict[str, Any]] = []
    header_patterns = [
        re.compile(r"^\[\d{4}\]\s*\d+\s*S\.C\.R\.?.*$", re.IGNORECASE),
        re.compile(r"^Digital Supreme Court Reports$", re.IGNORECASE),
        re.compile(r"^Supreme Court Reports$", re.IGNORECASE),
        re.compile(r"^\d{3,5}\x08?$"),  # Page number lines like "1877\b" or "1878"
        re.compile(r"^\*\s*Author$", re.IGNORECASE),
        re.compile(r"^\*\u2003Author$", re.IGNORECASE),
    ]
    
    line_sets: list[list[str]] = []
    for page in raw_pages:
        text = (page.get("text") or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        line_sets.append(lines)

    edge_counts: Counter[str] = Counter()
    for lines in line_sets:
        nonempty = [line for line in lines if line]
        for line in nonempty[:3] + nonempty[-3:]:
            if len(line) >= 15 and not re.fullmatch(r"\d+", line):
                edge_counts[line] += 1
    threshold = max(3, (len(raw_pages) + 1) // 2)
    repeated = {line for line, count in edge_counts.items() if count >= threshold}

    for p_idx, (page, lines) in enumerate(zip(raw_pages, line_sets)):
        filtered = []
        for l_idx, line in enumerate(lines):
            if not line:
                continue
            if line in repeated:
                continue
            if p_idx > 0 and l_idx < 4:
                if any(pat.match(line) for pat in header_patterns) or re.search(r"\bv\.\s|\bversus\b", line, re.I):
                    continue
            if any(pat.match(line) for pat in header_patterns):
                continue
            filtered.append(line)
            
        text = "\n".join(filtered)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        pages.append({"page_number": page["page_number"], "text": text})
    return pages


def _split_items(text: str) -> list[str]:
    """Clean legal items (acts, cited cases, keywords) removing header/footnote noise."""
    lines = [re.sub(r"^[\s•·\-–—]+", "", line).strip() for line in text.splitlines()]
    items = []
    for line in lines:
        if not line:
            continue
        if "Digital Supreme Court Reports" in line or "Supreme Court Reports" in line:
            continue
        if re.match(r"^\[\d{4}\]\s*\d+\s*S\.C\.R\.", line, re.I):
            continue
        if re.match(r"^\d{1,2}\s*\[\d{4}\]", line):  # Footnotes like "1 [2003] Supp. 4 SCR 995"
            continue
        if re.match(r"^\d{3,5}\x08?$", line):  # Page numbers
            continue
        items.append(line)
    return items


def _extract_legal_topics(headnotes_text: str) -> list[str]:
    """Extract individual legal topic entries from headnotes text."""
    if not headnotes_text:
        return []
        
    topic_blocks = re.split(r"(\[Paras?\s+[^\]]+\])", headnotes_text)
    topics = []
    
    for i in range(0, len(topic_blocks) - 1, 2):
        t_text = topic_blocks[i].strip()
        t_paras = topic_blocks[i+1].strip()
        full_topic = re.sub(r"\s+", " ", f"{t_text} {t_paras}").strip()
        if full_topic:
            topics.append(full_topic)
            
    if not topics and headnotes_text:
        # Fallback if no [Paras ...] markers
        lines = [l.strip() for l in headnotes_text.splitlines() if l.strip()]
        curr_topic = []
        for line in lines:
            if re.match(r"^[A-Z][A-Za-z\s–—\-]+(?:–|-|:)", line) and curr_topic:
                topics.append(re.sub(r"\s+", " ", " ".join(curr_topic)))
                curr_topic = [line]
            else:
                curr_topic.append(line)
        if curr_topic:
            topics.append(re.sub(r"\s+", " ", " ".join(curr_topic)))
            
    return topics


def _extract_sequential_paragraphs(judgment_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract full sequential paragraphs (1..N) across judgment pages."""
    found: list[dict[str, Any]] = []
    curr: dict[str, Any] | None = None
    expected_next = 1
    marker = re.compile(r"^\s*(\d{1,4})\s*[.\t]?\s*(.*)$")

    for page in judgment_pages:
        lines = page["text"].splitlines()
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            match = marker.match(line_str)
            if match and match.group(1):
                num = int(match.group(1))
                if num == expected_next or (expected_next == 1 and num <= 3):
                    rest = match.group(2).strip()
                    if curr:
                        if "paragraph_number" not in curr:
                            curr["paragraph_number"] = 1
                        curr["text"] = " ".join(curr.pop("_lines")).strip()
                        found.append(curr)
                    curr = {
                        "paragraph_number": num,
                        "page_start": page["page_number"],
                        "page_end": page["page_number"],
                        "_lines": [],
                    }
                    if rest:
                        curr["_lines"].append(rest)
                    expected_next = num + 1
                    continue

            if curr:
                curr["_lines"].append(line_str)
                curr["page_end"] = page["page_number"]
            else:
                curr = {
                    "paragraph_number": 1,
                    "page_start": page["page_number"],
                    "page_end": page["page_number"],
                    "_lines": [line_str],
                }

    if curr:
        if "paragraph_number" not in curr:
            curr["paragraph_number"] = expected_next
        curr["text"] = " ".join(curr.pop("_lines")).strip()
        found.append(curr)

    return found


def _parse_judges(judge_str: str) -> list[str]:
    if not judge_str:
        return []
    cleaned = re.sub(r"\[|\bJJ\.?|\bJ\.?|\*|\]", "", judge_str).strip()
    parts = re.split(r",| and | & ", cleaned)
    judges = [p.strip() for p in parts if p.strip()]
    return judges


def _normalize_title(raw_title: str) -> str:
    if not raw_title:
        return ""
    t = raw_title.replace("versus", "v.").replace("VERSUS", "v.").strip()
    return re.sub(r"\s+", " ", t)


def parse_case(metadata: dict[str, Any], raw_pages: list[dict[str, Any]]) -> dict[str, Any]:
    pages = clean_pages(raw_pages)
    
    buckets: dict[str, list[str]] = {key: [] for key in SECTION_ORDER}
    judgment_pages: list[dict[str, Any]] = []
    current_section: str | None = None
    in_judgment = False
    
    for page in pages:
        page_judgment_lines: list[str] = []
        for line in page["text"].splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            if in_judgment:
                if "Result of the case:" in line_str or "Result of the appeal:" in line_str or "†Headnotes prepared by:" in line_str:
                    in_judgment = False
                    current_section = None
                    continue
                buckets["judgment"].append(line_str)
                page_judgment_lines.append(line_str)
            else:
                detected = classify_heading(line_str)
                if detected:
                    current_section = detected
                    if detected == "judgment":
                        in_judgment = True
                    continue

                if current_section:
                    buckets[current_section].append(line_str)

        if page_judgment_lines:
            judgment_pages.append({"page_number": page["page_number"], "text": "\n".join(page_judgment_lines)})

    section_text = {key: "\n".join(value).strip() for key, value in buckets.items()}
    
    # Enriched Metadata extraction
    page1_text = raw_pages[0]["text"] if raw_pages else ""
    full_doc_text = "\n".join([p["text"] for p in pages])
    
    parsed_metadata = dict(metadata)
    
    case_id = parsed_metadata.get("case_id") or metadata.get("path") or "Unknown"
    parsed_metadata["case_id"] = case_id
    parsed_metadata["title"] = _normalize_title(metadata.get("title", ""))
    parsed_metadata["citation"] = metadata.get("citation", "")
    parsed_metadata["decision_date"] = metadata.get("decision_date", "")
    parsed_metadata["year"] = metadata.get("year", 0)
    parsed_metadata["court"] = "Supreme Court of India"
    parsed_metadata["document_type"] = "Judgment"
    
    # Extract judges/bench from Page 1
    bench_match = re.search(r"\[(.*?\bJJ\.?|.*?\bJ\.?)\]", page1_text)
    if bench_match:
        parsed_metadata["judges"] = _parse_judges(bench_match.group(1))
    else:
        parsed_metadata["judges"] = _parse_judges(metadata.get("judge", ""))
        
    # Extract appeal number
    app_match = re.search(r"(\bCriminal\s+Appeal\s+No\.\s*\d+\s+of\s+\d{4}\b|\bCivil\s+Appeal\s+No\.\s*\d+\s+of\s+\d{4}\b)", full_doc_text, re.I)
    if app_match:
        parsed_metadata["appeal_number"] = re.sub(r"\s+", " ", app_match.group(1)).strip()
        
    # Extract High Court origin
    hc_match = re.search(r"High\s+Court\s+of\s+([^\n.]+?)(?=\s+in|\s+at|\s+dated|\n|\.)", full_doc_text, re.I)
    if hc_match:
        parsed_metadata["high_court"] = re.sub(r"\s+", " ", hc_match.group(0)).strip()

    paragraphs = _extract_sequential_paragraphs(judgment_pages)

    # Clean legal topics
    legal_topics = _extract_legal_topics(section_text["headnotes"])

    return {
        "metadata": parsed_metadata,
        "sections": {
            "issue_for_consideration": section_text["issue_for_consideration"],
            "headnotes": section_text["headnotes"],
            "legal_topics": legal_topics,
            "case_law_cited": _split_items(section_text["case_law_cited"]),
            "acts_cited": _split_items(section_text["acts_cited"]),
            "keywords": _split_items(section_text["keywords"]),
            "case_arising_from": section_text["case_arising_from"],
            "appearances": section_text["appearances"],
            "websites_cited": _split_items(section_text["websites_cited"]),
            "books_cited": _split_items(section_text["books_cited"]),
            "periodicals_cited": _split_items(section_text["periodicals_cited"]),
        },
        "judgment": {
            "total_paragraphs": len(paragraphs),
            "text": section_text["judgment"],
            "paragraphs": paragraphs,
        },
        "processing": {
            "pdf_pages": len(raw_pages),
            "extraction_method": "PyMuPDF",
            "ocr_used": False,
            "sections_found": [k for k, v in buckets.items() if v],
        },
    }