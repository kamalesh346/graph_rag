"""
statute_parser.py - PHASES S2, S3, S4, S5: Structural Extraction, Atomic Units, References & Concepts

1. PHASE S2: Structural Extraction
   Converts normalized JSON section entries into canonical internal representations.
   Safely merges split section objects sharing the same section_id (e.g. BNS:61 part 1 + part 2).

2. PHASE S3: Atomic Legal Units
   Decomposes sections into granular atomic units:
   - SECTION (main provision)
   - SUBSECTION (e.g. BNS:103(1), BNS:103(2))
   - EXPLANATION (e.g. IPC:300_EXPLANATION_1)
   - EXCEPTION (e.g. IPC:300_EXCEPTION_4)
   - PROVISO
   - ILLUSTRATION

3. PHASE S4: Reference Enrichment
   Extracts explicit statutory cross-references (e.g. IPC:304 -> IPC:299, IPC:300).

4. PHASE S5: Concept Enrichment
   Annotates matching legal concepts from the project inventory.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTS_DIR = PROJECT_ROOT / "data" / "legal" / "acts"

CONCEPT_PATTERNS = {
    "murder": r"\bmurder\b",
    "culpable homicide": r"\bculpable homicide\b",
    "culpable homicide not amounting to murder": r"\bculpable homicide not amounting to murder\b",
    "dying declaration": r"\bdying declaration\b",
    "grave and sudden provocation": r"\bgrave and sudden provocation\b",
    "sudden fight": r"\bsudden fight\b|\bheat of passion\b",
    "circumstantial evidence": r"\bcircumstantial evidence\b",
    "ocular evidence": r"\bocular evidence\b|\beyewitness\b",
    "benefit of doubt": r"\bbenefit of doubt\b",
    "first information report": r"\bFIR\b|\bfirst information report\b",
    "common intention": r"\bcommon intention\b",
    "acquittal": r"\bacquittal\b|\bacquitted\b",
    "material contradiction": r"\bmaterial contradiction\b|\bmaterial omissions?\b",
    "medical evidence": r"\bmedical evidence\b|\bpost[- ]mortem\b|\bdoctor\b",
    "private defence": r"\bprivate defence\b|\bself[- ]defence\b",
    "consent": r"\bconsent\b",
    "dowry death": r"\bdowry death\b",
    "cruelty": r"\bcruelty\b",
    "rape": r"\brape\b",
    "abetment": r"\babetment\b|\babets?\b",
    "criminal conspiracy": r"\bcriminal conspiracy\b",
}


@dataclass
class AtomicUnit:
    unit_id: str
    act_id: str
    section_id: str
    parent_section_id: str
    unit_type: str  # 'section', 'subsection', 'explanation', 'exception', 'proviso', 'illustration'
    unit_number: str
    title: str
    text: str
    legal_references: list[str] = field(default_factory=list)
    legal_concepts: list[str] = field(default_factory=list)
    predecessor_act: str | None = None
    successor_act: str | None = None


@dataclass
class ConsolidatedSection:
    act_id: str
    section_id: str
    section_number: str
    title: str
    main_text: str
    predecessor_act: str | None = None
    successor_act: str | None = None
    subsections: list[dict[str, Any]] = field(default_factory=list)
    explanations: list[dict[str, Any]] = field(default_factory=list)
    illustrations: list[dict[str, Any]] = field(default_factory=list)
    exceptions: list[dict[str, Any]] = field(default_factory=list)
    provisos: list[dict[str, Any]] = field(default_factory=list)
    legal_concepts: list[str] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.33)


def canonical_concept_id(name: str) -> str:
    return "LEGAL_CONCEPT:" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def extract_references_from_text(text: str, current_act: str, current_sec_id: str | None = None) -> list[str]:
    """Extract explicit statutory section references from statute text, excluding self-references."""
    if not text:
        return []

    refs = set()
    sec_pattern = re.compile(
        r"\b(?:section|sections|s(?:ec)?\.)\s*(\d+[A-Za-z]*(?:\s*\(\s*\d+\s*\))?)", re.I
    )
    for m in sec_pattern.finditer(text):
        sec_str = m.group(1).replace(" ", "")
        ref_id = f"{current_act}:{sec_str}"
        refs.add(ref_id)

    # Cross-act references like "section 302 of the Indian Penal Code"
    if "penal code" in text.lower() or "ipc" in text.lower():
        for m in re.finditer(r"\b(?:section|s(?:ec)?\.)\s*(\d+[A-Za-z]*)", text, re.I):
            refs.add(f"IPC:{m.group(1)}")

    # Exclude self-references (e.g. IPC:302 referring to IPC:302)
    if current_sec_id:
        clean_current = current_sec_id.split("(")[0]
        filtered = {r for r in refs if r != current_sec_id and r.split("(")[0] != clean_current}
        return sorted(list(filtered))

    return sorted(list(refs))


def extract_concepts_from_text(text: str, declared_concepts: list[str]) -> list[str]:
    """Annotate concepts from text and merge with declared JSON concepts."""
    found = set()

    for concept in declared_concepts:
        if concept:
            found.add(canonical_concept_id(concept))

    for concept_name, pat in CONCEPT_PATTERNS.items():
        if re.search(pat, text, re.I):
            found.add(canonical_concept_id(concept_name))

    return sorted(list(found))


class StatuteParser:

    @staticmethod
    def stitch_subsections(raw_subs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stitches split subsection text fragments into clean, cohesive subsection dicts."""
        if not raw_subs:
            return []
        stitched: list[dict[str, Any]] = []
        seen_nums: set[str] = set()
        for s in raw_subs:
            num = str(s.get("subsection_number", "1"))
            txt = s.get("text", "").strip()
            if not txt:
                continue
            if not stitched:
                stitched.append({
                    "subsection_number": num,
                    "text": txt,
                    "section_id": s.get("section_id"),
                })
                seen_nums.add(num)
            else:
                prev = stitched[-1]
                prev_txt = prev["text"]
                # Check if this item is a continuation of prev
                is_continuation = False
                if prev_txt.endswith("sub-section") or prev_txt.endswith("section") or prev_txt.endswith("clause"):
                    is_continuation = True
                elif txt[0] in [",", ".", ";", ")", "]"] or txt[0].islower() or txt.startswith("of ") or txt.startswith("is "):
                    is_continuation = True

                if is_continuation:
                    if txt[0] in [",", ".", ";", ")", "]"] or prev_txt.endswith("sub-section") or prev_txt.endswith("-"):
                        prev["text"] = prev_txt + " " + txt if not prev_txt.endswith("-") else prev_txt + txt
                    else:
                        prev["text"] = prev_txt + " " + txt
                else:
                    if num in seen_nums:
                        continue
                    stitched.append({
                        "subsection_number": num,
                        "text": txt,
                        "section_id": s.get("section_id"),
                    })
                    seen_nums.add(num)
        return stitched

    @staticmethod
    def stitch_explanations(raw_exps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stitches split explanation fragments into cohesive explanation dicts."""
        if not raw_exps:
            return []
        stitched: list[dict[str, Any]] = []
        for exp in raw_exps:
            lbl = exp.get("label", "").strip()
            txt = exp.get("text", "").strip()
            if not txt:
                continue
            
            is_continuation = False
            if stitched:
                if (lbl.lower() in ["explanation", "explanations"]) and not re.search(r"\b\d+\b", lbl) and not re.search(r"^\s*\d+[\.\-]", txt):
                    is_continuation = True
                elif txt[0].islower() or txt.startswith("is ") or txt.startswith("or "):
                    is_continuation = True

            if is_continuation and stitched:
                stitched[-1]["text"] = (stitched[-1]["text"] + " " + txt).strip()
            else:
                stitched.append({"label": lbl, "text": txt})
        return stitched

    @staticmethod
    def stitch_exceptions(raw_excs: list[dict[str, Any]], main_text: str) -> tuple[list[dict[str, Any]], str]:
        """
        Stitches split exception fragments and re-integrates misclassified sentence continuations
        (e.g., CrPC 296, BSA 108, IEA 105) back into main_text.
        """
        if not raw_excs:
            return [], main_text
        stitched: list[dict[str, Any]] = []
        updated_main = main_text

        for exc in raw_excs:
            lbl = exc.get("label", "").strip()
            txt = exc.get("text", "").strip()
            if not txt:
                continue

            has_num = bool(re.search(r"\b\d+\b", lbl) or re.search(r"^\s*\d+[\.\-]", txt))
            if not has_num and not stitched:
                # No preceding numbered exception -> text continuation of main_text
                updated_main = (updated_main + " " + txt).strip() if updated_main else txt
            elif not has_num and stitched:
                # Preceding numbered exception exists -> continuation of previous exception text
                stitched[-1]["text"] = (stitched[-1]["text"] + " " + txt).strip()
            else:
                stitched.append({"label": lbl, "text": txt})

        return stitched, updated_main

    @staticmethod
    def parse_act_file(file_path: Path) -> tuple[str, list[ConsolidatedSection], list[AtomicUnit]]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        act_meta = data.get("act", {})
        act_id = act_meta.get("act_id", file_path.stem.upper())
        raw_sections = data.get("sections", [])

        # 1. PHASE S2: Structural Extraction & Consolidation
        # Group raw sections by section_id to handle split JSON entries cleanly (e.g. BNS:61 part 1 & part 2)
        consolidated_map: dict[str, ConsolidatedSection] = {}

        for sec in raw_sections:
            sec_id = sec.get("section_id")
            if not sec_id:
                sec_id = f"{act_id}:{sec.get('section_number', '0')}"

            if sec_id not in consolidated_map:
                consolidated_map[sec_id] = ConsolidatedSection(
                    act_id=act_id,
                    section_id=sec_id,
                    section_number=sec.get("section_number", ""),
                    title=sec.get("title", ""),
                    main_text=sec.get("text", "").strip(),
                    predecessor_act=sec.get("predecessor_act"),
                    successor_act=sec.get("successor_act"),
                    subsections=list(sec.get("subsections", [])),
                    explanations=list(sec.get("explanations", [])),
                    illustrations=list(sec.get("illustrations", [])),
                    exceptions=list(sec.get("exceptions", [])),
                    provisos=list(sec.get("provisos", [])),
                    legal_concepts=list(sec.get("legal_concepts", [])),
                )
            else:
                existing = consolidated_map[sec_id]
                # Append text if main text was split
                if sec.get("text", "").strip():
                    if existing.main_text and sec.get("text", "").strip() not in existing.main_text:
                        existing.main_text += "\n" + sec.get("text", "").strip()
                    elif not existing.main_text:
                        existing.main_text = sec.get("text", "").strip()

                # Extend sub-elements with deduplication
                existing_sub_nums = {str(s.get("subsection_number")) for s in existing.subsections}
                for sub in sec.get("subsections", []):
                    s_num = str(sub.get("subsection_number"))
                    if s_num not in existing_sub_nums:
                        existing.subsections.append(sub)
                        existing_sub_nums.add(s_num)

                existing_exp_keys = {(e.get("label"), e.get("text", "").strip()) for e in existing.explanations}
                for exp in sec.get("explanations", []):
                    key = (exp.get("label"), exp.get("text", "").strip())
                    if key not in existing_exp_keys:
                        existing.explanations.append(exp)
                        existing_exp_keys.add(key)

                existing_ill_keys = {(i.get("label"), i.get("text", "").strip()) for i in existing.illustrations}
                for ill in sec.get("illustrations", []):
                    key = (ill.get("label"), ill.get("text", "").strip())
                    if key not in existing_ill_keys:
                        existing.illustrations.append(ill)
                        existing_ill_keys.add(key)

                existing_exc_keys = {(e.get("label"), e.get("text", "").strip()) for e in existing.exceptions}
                for exc in sec.get("exceptions", []):
                    key = (exc.get("label"), exc.get("text", "").strip())
                    if key not in existing_exc_keys:
                        existing.exceptions.append(exc)
                        existing_exc_keys.add(key)

                existing_prv_keys = {(p.get("label"), p.get("text", "").strip()) for p in existing.provisos}
                for prv in sec.get("provisos", []):
                    key = (prv.get("label"), prv.get("text", "").strip())
                    if key not in existing_prv_keys:
                        existing.provisos.append(prv)
                        existing_prv_keys.add(key)

                for c in sec.get("legal_concepts", []):
                    if c and c not in existing.legal_concepts:
                        existing.legal_concepts.append(c)

                if not existing.successor_act and sec.get("successor_act"):
                    existing.successor_act = sec.get("successor_act")
                if not existing.predecessor_act and sec.get("predecessor_act"):
                    existing.predecessor_act = sec.get("predecessor_act")

        consolidated_sections = list(consolidated_map.values())

        # Stitch subsections, explanations, and exceptions across consolidated sections to fix split text fragments
        for sec in consolidated_sections:
            sec.subsections = StatuteParser.stitch_subsections(sec.subsections)
            sec.explanations = StatuteParser.stitch_explanations(sec.explanations)
            sec.exceptions, sec.main_text = StatuteParser.stitch_exceptions(sec.exceptions, sec.main_text)

        # 2. PHASE S3, S4, S5: Extract Atomic Units with References & Concepts
        atomic_units: list[AtomicUnit] = []

        for sec in consolidated_sections:
            sec_id_slug = sec.section_id.replace(":", "_").replace("(", "_").replace(")", "")

            # A. Main Provision Unit (if text exists)
            if sec.main_text:
                refs = extract_references_from_text(sec.main_text, act_id)
                concepts = extract_concepts_from_text(sec.main_text, sec.legal_concepts)

                unit = AtomicUnit(
                    unit_id=f"{sec_id_slug}_MAIN",
                    act_id=act_id,
                    section_id=sec.section_id,
                    parent_section_id=sec.section_id,
                    unit_type="section",
                    unit_number="main",
                    title=sec.title,
                    text=sec.main_text,
                    legal_references=refs,
                    legal_concepts=concepts,
                    predecessor_act=sec.predecessor_act,
                    successor_act=sec.successor_act,
                )
                atomic_units.append(unit)

            # B. Subsection Units
            for sub in sec.subsections:
                sub_num = str(sub.get("subsection_number", "1"))
                sub_text = sub.get("text", "").strip()
                sub_sec_id = sub.get("section_id") or f"{sec.section_id}({sub_num})"

                if sub_text:
                    refs = extract_references_from_text(sub_text, act_id)
                    concepts = extract_concepts_from_text(sub_text, sec.legal_concepts)

                    unit = AtomicUnit(
                        unit_id=f"{sec_id_slug}_SUB_{sub_num}",
                        act_id=act_id,
                        section_id=sub_sec_id,
                        parent_section_id=sec.section_id,
                        unit_type="subsection",
                        unit_number=sub_num,
                        title=f"{sec.title} (Sub-section {sub_num})",
                        text=sub_text,
                        legal_references=refs,
                        legal_concepts=concepts,
                        predecessor_act=sec.predecessor_act,
                        successor_act=sec.successor_act,
                    )
                    atomic_units.append(unit)

            # C. Explanation Units
            for idx, exp in enumerate(sec.explanations, start=1):
                exp_text = exp.get("text", "").strip()
                exp_label = exp.get("label", f"Explanation {idx}")

                if exp_text:
                    full_exp_text = f"{exp_label}: {exp_text}" if not exp_text.startswith(exp_label) else exp_text
                    refs = extract_references_from_text(full_exp_text, act_id)
                    concepts = extract_concepts_from_text(full_exp_text, sec.legal_concepts)

                    unit = AtomicUnit(
                        unit_id=f"{sec_id_slug}_EXP_{idx}",
                        act_id=act_id,
                        section_id=sec.section_id,
                        parent_section_id=sec.section_id,
                        unit_type="explanation",
                        unit_number=str(idx),
                        title=f"{sec.title} ({exp_label})",
                        text=full_exp_text,
                        legal_references=refs,
                        legal_concepts=concepts,
                        predecessor_act=sec.predecessor_act,
                        successor_act=sec.successor_act,
                    )
                    atomic_units.append(unit)

            # D. Exception Units
            for idx, exc in enumerate(sec.exceptions, start=1):
                exc_text = exc.get("text", "").strip()
                exc_label = exc.get("label", f"Exception {idx}")

                if exc_text:
                    full_exc_text = f"{exc_label} {exc_text}" if not exc_text.startswith(exc_label) else exc_text
                    refs = extract_references_from_text(full_exc_text, act_id)
                    concepts = extract_concepts_from_text(full_exc_text, sec.legal_concepts)

                    unit = AtomicUnit(
                        unit_id=f"{sec_id_slug}_EXC_{idx}",
                        act_id=act_id,
                        section_id=sec.section_id,
                        parent_section_id=sec.section_id,
                        unit_type="exception",
                        unit_number=str(idx),
                        title=f"{sec.title} ({exc_label})",
                        text=full_exc_text,
                        legal_references=refs,
                        legal_concepts=concepts,
                        predecessor_act=sec.predecessor_act,
                        successor_act=sec.successor_act,
                    )
                    atomic_units.append(unit)

            # E. Illustration Units
            for idx, ill in enumerate(sec.illustrations, start=1):
                ill_text = ill.get("text", "").strip()
                ill_label = ill.get("label", f"Illustration {idx}")

                if ill_text:
                    full_ill_text = f"{ill_label}\n{ill_text}" if not ill_text.startswith(ill_label) else ill_text
                    refs = extract_references_from_text(full_ill_text, act_id)
                    concepts = extract_concepts_from_text(full_ill_text, sec.legal_concepts)

                    unit = AtomicUnit(
                        unit_id=f"{sec_id_slug}_ILL_{idx}",
                        act_id=act_id,
                        section_id=sec.section_id,
                        parent_section_id=sec.section_id,
                        unit_type="illustration",
                        unit_number=str(idx),
                        title=f"{sec.title} ({ill_label})",
                        text=full_ill_text,
                        legal_references=refs,
                        legal_concepts=concepts,
                        predecessor_act=sec.predecessor_act,
                        successor_act=sec.successor_act,
                    )
                    atomic_units.append(unit)

            # F. Proviso Units
            for idx, prv in enumerate(sec.provisos, start=1):
                prv_text = prv.get("text", "").strip()
                prv_label = prv.get("label", f"Proviso {idx}")

                if prv_text:
                    full_prv_text = f"{prv_label}: {prv_text}" if not prv_text.startswith(prv_label) else prv_text
                    refs = extract_references_from_text(full_prv_text, act_id)
                    concepts = extract_concepts_from_text(full_prv_text, sec.legal_concepts)

                    unit = AtomicUnit(
                        unit_id=f"{sec_id_slug}_PRV_{idx}",
                        act_id=act_id,
                        section_id=sec.section_id,
                        parent_section_id=sec.section_id,
                        unit_type="proviso",
                        unit_number=str(idx),
                        title=f"{sec.title} ({prv_label})",
                        text=full_prv_text,
                        legal_references=refs,
                        legal_concepts=concepts,
                        predecessor_act=sec.predecessor_act,
                        successor_act=sec.successor_act,
                    )
                    atomic_units.append(unit)

        return act_id, consolidated_sections, atomic_units


if __name__ == "__main__":
    parser = StatuteParser()
    act_files = sorted(ACTS_DIR.glob("*.json"))
    print(f"Parsing {len(act_files)} statute files...")

    total_consolidated = 0
    total_atomic = 0

    for af in act_files:
        act_id, sections, units = parser.parse_act_file(af)
        total_consolidated += len(sections)
        total_atomic += len(units)
        print(f"  [{act_id}] {af.name}: {len(sections)} consolidated sections -> {len(units)} atomic units.")

    print(f"\nParse complete: Total {total_consolidated} sections -> {total_atomic} atomic units.")
