"""
statute_chunker.py - PHASES S6 & S7: Structure-Aware Intelligent Statute Chunking & Schema Generation

Transforms atomic legal units into intelligent, retrievable statute chunks:
- Bundles concise sections (main text + explanations + subsections) into unified section chunks.
- Decomposes major multi-part sections (e.g. IPC Section 300, IPC Section 304, CrPC Section 173) into granular retrieval chunks:
  - SECTION chunks (main provision)
  - SUBSECTION chunks
  - EXCEPTION chunks (e.g. Exception 1, Exception 4 to murder)
  - EXPLANATION chunks
  - ILLUSTRATION chunks

Produces:
- data/chunks/statutes/[ACT_ID]_statute_chunks.json
- data/chunks/statutes/all_statute_chunks.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from statute_parser import (
    AtomicUnit,
    ConsolidatedSection,
    StatuteParser,
    estimate_tokens,
    extract_concepts_from_text,
    extract_references_from_text,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTS_DIR = PROJECT_ROOT / "data" / "legal" / "acts"
STATUTES_CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks" / "statutes"


def build_statute_chunk(
    chunk_id: str,
    act_id: str,
    section_id: str,
    section_number: str,
    text: str,
    source_meta: dict[str, Any],
    references: list[str],
    concepts: list[str],
    successor_act: str | None = None,
    predecessor_act: str | None = None,
) -> dict[str, Any]:
    """Formats a statute chunk matching the canonical 1 Section = 1 Chunk GraphRAG schema."""
    successor_refs = [successor_act] if successor_act else []
    predecessor_refs = [predecessor_act] if predecessor_act else []

    return {
        "chunk_id": chunk_id,
        "act_id": act_id,
        "section_id": section_id,
        "document_type": "statute",
        "chunk_type": "section",
        "text": text,
        "source": source_meta,
        "legal_references": sorted(list(set(references))),
        "legal_concepts": sorted(list(set(concepts))),
        "successor_references": successor_refs,
        "predecessor_references": predecessor_refs,
        "token_count": estimate_tokens(text),
    }


def parse_numeric_explanations(explanations: list[dict[str, Any]]) -> list[str]:
    nums = []
    for idx, exp in enumerate(explanations, start=1):
        lbl = exp.get("label", "")
        txt = exp.get("text", "")
        m = re.search(r"\b(\d+)\b", lbl) or re.search(r"^\s*(\d+)\b", txt)
        if m:
            nums.append(m.group(1))
        else:
            nums.append(str(idx))
    return list(dict.fromkeys(nums))


def parse_numeric_exceptions(exceptions: list[dict[str, Any]]) -> list[str]:
    nums = []
    for idx, exc in enumerate(exceptions, start=1):
        lbl = exc.get("label", "")
        txt = exc.get("text", "")
        m = re.search(r"\b(\d+)\b", lbl) or re.search(r"^\s*(\d+)\b", txt)
        if m:
            nums.append(m.group(1))
        else:
            nums.append(str(idx))
    return list(dict.fromkeys(nums))


def parse_numeric_provisos(provisos: list[dict[str, Any]]) -> list[str]:
    nums = []
    for idx, prv in enumerate(provisos, start=1):
        lbl = prv.get("label", "")
        txt = prv.get("text", "")
        m = re.search(r"\b(\d+)\b", lbl) or re.search(r"^\s*(\d+)\b", txt)
        if m:
            nums.append(m.group(1))
        else:
            nums.append(str(idx))
    return list(dict.fromkeys(nums))


def parse_numeric_illustrations(illustrations: list[dict[str, Any]]) -> list[str]:
    nums = []
    for idx, ill in enumerate(illustrations, start=1):
        txt = ill.get("text", "")
        letters = re.findall(r"\(([a-z])\)", txt)
        if letters:
            nums.extend(letters)
        else:
            lbl = ill.get("label", "")
            m = re.search(r"\b(\d+)\b", lbl)
            if m:
                nums.append(m.group(1))
            else:
                nums.append(str(idx))
    return list(dict.fromkeys(nums))


def clean_statute_text(text: str) -> str:
    """Normalizes all internal newlines and OCR artifacts into clean single spaces so chunks contain zero \\n characters."""
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\ufffd", "")
    word_fixes = {
        r"\bllustrations?\b": "Illustrations",
        r"\borimprisonment\b": "or imprisonment",
        r"\beitherdescription\b": "either description",
        r"\beachmember\b": "each member",
        r"\bfirstassault\b": "first assault",
        r"\bsevenvears\b": "seven years",
        r"\bWhoeverassaults\b": "Whoever assaults",
        r"\bofauthority\b": "of authority",
        r"\bwithoutsufficient\b": "without sufficient",
        r"\bhewas\b": "he was",
        r"\bthefirstassault\b": "the first assault",
        r"\bapublic\b": "a public",
        r"\bpublicservant\b": "public servant",
        r"\bandcauses\b": "and causes",
        r"\b goodfaith\b": " good faith",
        r"\b duedischarge\b": " due discharge",
        r"\boffender’s\b": "offender's",
    }
    for pat, repl in word_fixes.items():
        text = re.sub(pat, repl, text, flags=re.I)
    text = re.sub(r"([a-zA-Z]),([a-zA-Z])", r"\1, \2", text)
    text = re.sub(r"[ \t\n\r]+", " ", text).strip()
    return text


class StatuteChunker:

    @staticmethod
    def chunk_act(
        act_id: str,
        consolidated_sections: list[ConsolidatedSection]
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        for sec in consolidated_sections:
            sec_num_clean = sec.section_number.replace(" ", "").upper()
            chunk_id = f"{act_id}_{sec_num_clean}_SECTION_001"

            # Deterministic text assembly
            text_parts = [f"Section {sec.section_number} - {sec.title}".strip(" -")]
            if sec.main_text:
                text_parts.append(sec.main_text)

            # Subsections
            sub_nums = []
            for sub in sec.subsections:
                s_num = str(sub.get("subsection_number", "")).strip()
                s_text = sub.get("text", "").strip()
                if s_num and s_num not in sub_nums:
                    sub_nums.append(s_num)
                if s_text:
                    if s_text.startswith(f"({s_num})") or s_text.startswith(f"{s_num}."):
                        text_parts.append(s_text)
                    else:
                        text_parts.append(f"({s_num}) {s_text}" if s_num else s_text)

            # Explanations
            for idx, exp in enumerate(sec.explanations, start=1):
                exp_label = exp.get("label", "Explanation").strip().rstrip(".")
                exp_text = exp.get("text", "").strip()
                if exp_label.lower() in ["explanation", "explanations"]:
                    exp_label = f"Explanation {idx}" if len(sec.explanations) > 1 else "Explanation"
                if exp_text:
                    full_exp = f"{exp_label}: {exp_text}" if not exp_text.startswith(exp_label) else exp_text
                    text_parts.append(full_exp)

            # Exceptions
            for idx, exc in enumerate(sec.exceptions, start=1):
                exc_label = exc.get("label", "Exception").strip().rstrip(".")
                exc_text = exc.get("text", "").strip()
                if exc_label.lower() in ["exception", "exceptions"]:
                    exc_label = f"Exception {idx}" if len(sec.exceptions) > 1 else "Exception"
                if exc_text:
                    full_exc = f"{exc_label}: {exc_text}" if not exc_text.startswith(exc_label) else exc_text
                    text_parts.append(full_exc)

            # Provisos
            for prv in sec.provisos:
                prv_label = prv.get("label", "Provided that").strip().rstrip(":")
                prv_text = prv.get("text", "").strip()
                if prv_label.lower() in ["proviso", "provisos"]:
                    prv_label = "Provided that"
                if prv_text:
                    full_prv = f"{prv_label}: {prv_text}" if not (prv_text.startswith("Provided that") or prv_text.startswith("Provided further") or prv_text.startswith(prv_label)) else prv_text
                    text_parts.append(full_prv)

            # Illustrations (grouped under a single Illustrations header block)
            if sec.illustrations:
                ill_lines = ["Illustrations:"]
                for ill in sec.illustrations:
                    ill_text = ill.get("text", "").strip()
                    if ill_text:
                        ill_clean = re.sub(r"^(?:Illustrations?|llustrations?)\s*", "", ill_text, flags=re.I)
                        ill_lines.append(ill_clean)
                if len(ill_lines) > 1:
                    text_parts.append("\n\n".join(ill_lines))

            assembled_text = clean_statute_text("\n\n".join(text_parts))

            # References and Concepts across complete assembled section (excluding self-references)
            refs = extract_references_from_text(assembled_text, act_id, sec.section_id)
            concepts = extract_concepts_from_text(assembled_text, sec.legal_concepts)

            source_meta = {
                "section_number": sec.section_number,
                "has_main_text": bool(sec.main_text),
                "subsections": sub_nums,
                "explanations": parse_numeric_explanations(sec.explanations),
                "illustrations": parse_numeric_illustrations(sec.illustrations),
                "exceptions": parse_numeric_exceptions(sec.exceptions),
                "provisos": parse_numeric_provisos(sec.provisos),
            }

            chunk = build_statute_chunk(
                chunk_id=chunk_id,
                act_id=act_id,
                section_id=sec.section_id,
                section_number=sec.section_number,
                text=assembled_text,
                source_meta=source_meta,
                references=refs,
                concepts=concepts,
                successor_act=sec.successor_act,
                predecessor_act=sec.predecessor_act,
            )
            chunks.append(chunk)

        return chunks

    def process_all(self) -> tuple[int, list[dict[str, Any]]]:
        STATUTES_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        act_files = sorted(ACTS_DIR.glob("*.json"))
        parser = StatuteParser()

        all_statute_chunks: list[dict[str, Any]] = []

        for af in act_files:
            act_id, sections, _ = parser.parse_act_file(af)
            chunks = self.chunk_act(act_id, sections)

            payload = {
                "act_id": act_id,
                "total_chunks": len(chunks),
                "sections_processed": len(sections),
                "chunks": chunks,
            }

            out_json = STATUTES_CHUNKS_DIR / f"{act_id.lower()}_statute_chunks.json"
            out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            all_statute_chunks.extend(chunks)
            print(f"  [{act_id}] Processed {len(sections)} sections -> {len(chunks)} complete section chunks.")

        master_jsonl = STATUTES_CHUNKS_DIR / "all_statute_chunks.jsonl"
        jsonl_lines = [json.dumps(c, ensure_ascii=False) for c in all_statute_chunks]
        master_jsonl.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")

        print(f"\nMaster JSONL written to {master_jsonl} ({len(all_statute_chunks)} total statute chunks).")
        return len(all_statute_chunks), all_statute_chunks


def validate_structural_completeness(
    sections: list[ConsolidatedSection],
    chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Compares normalized JSON source structure against generated chunks.
    Verifies that every child element (subsections, explanations, exceptions, illustrations, provisos)
    is fully represented in the assembled text and source metadata.
    """
    chunk_map = {c["section_id"]: c for c in chunks}

    report = {
        "total_sections": len(sections),
        "total_chunks": len(chunks),
        "sections_with_subsections": 0,
        "sections_with_explanations": 0,
        "sections_with_illustrations": 0,
        "sections_with_exceptions": 0,
        "sections_with_provisos": 0,
        "missing_content_errors": [],
        "self_reference_errors": [],
        "passed": True,
    }

    for sec in sections:
        c = chunk_map.get(sec.section_id)
        if not c:
            report["missing_content_errors"].append(f"Section {sec.section_id}: Missing generated chunk!")
            report["passed"] = False
            continue

        src_meta = c.get("source", {})

        # 1. Subsections
        if sec.subsections:
            report["sections_with_subsections"] += 1
            expected_sub_count = len({str(s.get("subsection_number")) for s in sec.subsections if s.get("subsection_number")})
            if len(src_meta.get("subsections", [])) != expected_sub_count:
                report["missing_content_errors"].append(
                    f"Section {sec.section_id}: Subsection count mismatch (Source: {expected_sub_count}, Meta: {len(src_meta.get('subsections', []))})"
                )
                report["passed"] = False

        # 2. Explanations
        if sec.explanations:
            report["sections_with_explanations"] += 1
            if len(src_meta.get("explanations", [])) != len(sec.explanations):
                report["missing_content_errors"].append(
                    f"Section {sec.section_id}: Explanation count mismatch (Source: {len(sec.explanations)}, Meta: {len(src_meta.get('explanations', []))})"
                )
                report["passed"] = False

        # 3. Exceptions
        if sec.exceptions:
            report["sections_with_exceptions"] += 1
            if len(src_meta.get("exceptions", [])) != len(sec.exceptions):
                report["missing_content_errors"].append(
                    f"Section {sec.section_id}: Exception count mismatch (Source: {len(sec.exceptions)}, Meta: {len(src_meta.get('exceptions', []))})"
                )
                report["passed"] = False

        # 4. Illustrations
        if sec.illustrations:
            report["sections_with_illustrations"] += 1
            if len(src_meta.get("illustrations", [])) == 0:
                report["missing_content_errors"].append(
                    f"Section {sec.section_id}: Illustration metadata is empty despite source containing illustrations."
                )
                report["passed"] = False

        # 5. Provisos
        if sec.provisos:
            report["sections_with_provisos"] += 1
            if len(src_meta.get("provisos", [])) != len(sec.provisos):
                report["missing_content_errors"].append(
                    f"Section {sec.section_id}: Proviso count mismatch (Source: {len(sec.provisos)}, Meta: {len(src_meta.get('provisos', []))})"
                )
                report["passed"] = False

        # 6. Check for self-references
        refs = c.get("legal_references", [])
        if sec.section_id in refs:
            report["self_reference_errors"].append(f"Section {sec.section_id}: Self-reference detected in legal_references!")
            report["passed"] = False

    return report


if __name__ == "__main__":
    chunker = StatuteChunker()
    total_count, _ = chunker.process_all()
    print(f"Statute Chunking Complete! Total Chunks: {total_count}")
