"""Mine legal references from normalized case JSONs.

Performs entity disambiguation and format standardization, updating normalized
case artifacts in data/normalized/ and generating graph-ready legal vocabulary
and reference reports using clean human-readable Neo4j canonical IDs.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = ROOT / "data" / "extracted"
NORMALIZED_DIR = ROOT / "data" / "normalized"
ANALYSIS_DIR = ROOT / "data" / "analysis"
REPORTS_DIR = ROOT / "reports"

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

OFFENCES = {
    "Murder": r"\bmurder\b",
    "Culpable homicide": r"\bculpable homicide\b",
}

COURT_ALIASES = {
    "supreme court of india": "Supreme Court of India",
    "supreme court": "Supreme Court of India",
    "sci": "Supreme Court of India",
    "sc": "Supreme Court of India",
}

CONCEPT_ALIASES = {
    "Murder": ["murder"],
    "Culpable homicide": ["culpable homicide"],
    "Culpable homicide not amounting to murder": ["culpable homicide not amounting to murder"],
    "Dying declaration": ["dying declaration"],
    "Grave and sudden provocation": ["grave and sudden provocation"],
    "Sudden fight": ["sudden fight"],
    "Circumstantial evidence": ["circumstantial evidence"],
    "Ocular evidence": ["ocular evidence"],
    "Benefit of doubt": ["benefit of doubt"],
    "First information report (FIR)": ["FIR", "First Information Report"],
    "Common intention": ["common intention"],
    "Acquittal": ["acquittal", "acquitted"],
    "Material contradiction": ["material contradiction", "material omission", "material omissions"],
    "Medical evidence": ["medical evidence", "post-mortem", "post mortem"],
}


def roman_part(value: str) -> str:
    mapping = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "I": "I", "II": "II", "III": "III", "IV": "IV", "V": "V"}
    return mapping.get(value.upper(), value.upper())


def clean_exc(exc_val: str) -> str:
    m = re.search(r"(?:[ivx]+|\d+)", exc_val, re.I)
    val = m.group().lower() if m else "1"
    return {"i": "1", "1": "1", "ii": "2", "2": "2", "iii": "3", "3": "3", "iv": "4", "4": "4", "v": "5", "5": "5"}.get(val, val)


def canonical_provision(act: str, raw: str) -> dict[str, Any]:
    """Map typography variants to clean, human-readable graph canonical IDs."""
    value = re.sub(r"\s+", " ", raw).strip()
    act_id = act.replace(" ", "")
    
    # Exception pattern
    exc_match = re.search(r"(?:(\d+)\s*[-–—]?\s*(?:\(\s*)?exception\s*([ivx0-9]+)(?:\s*\))?|exception\s*([ivx0-9]+)\s*(?:to|of)?\s*(?:s(?:ection)?\.?\s*)?(\d+))", value, re.I)
    if exc_match:
        sec = exc_match.group(1) or exc_match.group(4)
        raw_exc = exc_match.group(2) or exc_match.group(3)
        exc_num = clean_exc(raw_exc)
        return {
            "canonical_id": f"{act_id}:{sec}(Exception {exc_num})",
            "act": act,
            "section": sec,
            "exception": exc_num,
            "display_name": f"{act} Section {sec} Exception {exc_num}",
        }

    # Part pattern (e.g. IPC:304(Part I))
    part_match = re.search(r"(\d+)\s*[-–—]?\s*(?:\(\s*)?part\s*([ivx0-9]+)(?:\s*\))?", value, re.I)
    if part_match:
        sec, part_val = part_match.group(1), roman_part(part_match.group(2))
        return {
            "canonical_id": f"{act_id}:{sec}(Part {part_val})",
            "act": act,
            "section": sec,
            "part": part_val,
            "display_name": f"{act} Section {sec} Part {part_val}",
        }

    # Subsection in parentheses (e.g. CrPC:173(2))
    sub_match = re.fullmatch(r"(\d+)\s*[-–—]?\s*\(\s*([0-9]+)\s*\)", value)
    if sub_match:
        sec, sub_val = sub_match.group(1), sub_match.group(2)
        return {
            "canonical_id": f"{act_id}:{sec}({sub_val})",
            "act": act,
            "section": sec,
            "subsection": sub_val,
            "display_name": f"{act} Section {sec}({sub_val})",
        }

    # Letter compact / hyphenated subsection (e.g. IPC:304A, IPC:120B)
    letter_sub = re.fullmatch(r"(\d+)\s*[-–—]?\s*(?:\(\s*)?([A-Za-z]+)(?:\s*\))?", value)
    if letter_sub:
        sec, letter = letter_sub.group(1), letter_sub.group(2).upper()
        return {
            "canonical_id": f"{act_id}:{sec}{letter}",
            "act": act,
            "section": f"{sec}{letter}",
            "subsection": letter,
            "display_name": f"{act} Section {sec}{letter}",
        }

    # Standard section number
    number = re.match(r"\d+", value)
    sec = number.group() if number else value
    return {
        "canonical_id": f"{act_id}:{sec}",
        "act": act,
        "section": sec,
        "display_name": f"{act} Section {sec}",
    }


def canonical_concept_id(name: str) -> str:
    return "LEGAL_CONCEPT:" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def canonical_court(value: str | None) -> str:
    if not value:
        return "Supreme Court of India"
    clean = re.sub(r"\s+", " ", value).strip()
    return COURT_ALIASES.get(clean.casefold(), "Supreme Court of India")


def case_text(case: dict[str, Any]) -> str:
    sections = case.get("sections", {})
    values: list[str] = []
    for value in sections.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            for item in value:
                values.append(item if isinstance(item, str) else item.get("heading", "") + "\n" + item.get("text", ""))
    if "judgment" in case and isinstance(case["judgment"], dict):
        values.append(case["judgment"].get("text", ""))
    return "\n".join(values)


def provisions_for_statute(text: str, statute_pattern: str) -> set[str]:
    found: set[str] = set()
    statute = re.compile(statute_pattern, re.I)
    section_number = re.compile(
        r"\b(?:Section|Sections|S(?:ec)?\.)\s*(\d+[A-Za-z]*(?:\s*[-–—]?\s*(?:\([^)]*\)|Part\s+[IVX0-9]+|Exception\s+[IVX0-9]+))?)",
        re.I,
    )
    
    for clause in re.split(r"[\n.;]", text):
        statute_matches = list(statute.finditer(clause))
        if not statute_matches:
            continue
            
        for match in section_number.finditer(clause):
            found.add(re.sub(r"\s+", " ", match.group(1)).strip())
            
        for match in re.finditer(r"\bException\s+([IVX0-9]+)\s+(?:to|of)?\s*(?:Section|Sections|S(?:ec)?\.)\s*(\d+)\b", clause, re.I):
            found.add(f"{match.group(2)} Exception {match.group(1)}")
            
        for match in re.finditer(r"\b(?:Section|Sections|S(?:ec)?\.)\s*(\d+)\s*(?:[-–—]?\s*)?Exception\s+([IVX0-9]+)\b", clause, re.I):
            found.add(f"{match.group(1)} Exception {match.group(2)}")

        for statute_match in statute_matches:
            prefix = clause[max(0, statute_match.start() - 100):statute_match.start()]
            anchors = list(re.finditer(r"\b(?:Section|Sections|S(?:ec)?\.)\s+", prefix, re.I))
            if anchors:
                list_text = re.split(r"[\[(]", prefix[anchors[-1].end():])[0]
                for number in re.findall(r"\b\d{1,4}[A-Za-z]*\b", list_text):
                    if not re.fullmatch(r"(?:18|19|20)\d{2}", number):
                        found.add(number)
        for match in re.finditer(r"\bs\.\s*(\d+[A-Za-z]*)\b", clause, re.I):
            found.add(match.group(1))
    return found


def cited_cases(case: dict[str, Any]) -> list[str]:
    sections = case.get("sections", {})
    raw_list = sections.get("case_law_cited", [])
    raw = " ".join(raw_list) if isinstance(raw_list, list) else str(raw_list)
    return [re.sub(r"\s+", " ", item).strip(" .") for item in raw.split(";") if item.strip()]


def provision_entries(items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {key: value for key, value in item.items() if key != "case_ids"} | {
                "case_count": len(item["case_ids"]),
                "case_ids": sorted(item["case_ids"]),
                "raw_variants": sorted(item["raw_variants"]),
            }
            for item in items.values()
        ],
        key=lambda item: (-item["case_count"], int(re.match(r"\d+", item["section"]).group()), item["canonical_id"]),
    )


def concept_entries(items: dict[str, set[str]]) -> list[dict[str, Any]]:
    return [
        {
            "canonical_id": canonical_concept_id(name),
            "name": name,
            "aliases": CONCEPT_ALIASES.get(name, [name]),
            "case_count": len(case_ids),
            "case_ids": sorted(case_ids),
        }
        for name, case_ids in sorted(items.items(), key=lambda item: (-len(item[1]), item[0]))
    ]


def frequency_entries(items: dict[str, set[str]]) -> list[dict[str, Any]]:
    return [
        {"name": name, "case_count": len(case_ids), "case_ids": sorted(case_ids)}
        for name, case_ids in sorted(items.items(), key=lambda item: (-len(item[1]), item[0]))
    ]


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    input_dir = NORMALIZED_DIR if (NORMALIZED_DIR.exists() and list(NORMALIZED_DIR.glob("*.json"))) else EXTRACTED_DIR
    json_files = sorted(input_dir.glob("*.json"))
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in json_files]
    
    provisions: dict[str, dict[str, dict[str, Any]]] = {key: {} for key in STATUTES}
    concepts: dict[str, set[str]] = defaultdict(set)
    offences: dict[str, set[str]] = defaultdict(set)
    citations: dict[str, set[str]] = defaultdict(set)
    courts: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"case_ids": set(), "raw_variants": set()})
    per_case: list[dict[str, Any]] = []

    for file_path, case in zip(json_files, cases):
        case_id = case.get("metadata", {}).get("case_id") or case.get("metadata", {}).get("title") or file_path.stem
        text = case_text(case)
        case_provisions: dict[str, list[dict[str, Any]]] = {}

        for key, (pattern, _) in STATUTES.items():
            matches = provisions_for_statute(text, pattern)
            if matches:
                canonical_matches = {canonical_provision(key, raw)["canonical_id"]: canonical_provision(key, raw) for raw in matches}
                case_provisions[key] = sorted(canonical_matches.values(), key=lambda item: (int(re.match(r"\d+", item["section"]).group()), item["canonical_id"]))
                for raw_provision in matches:
                    normalized = canonical_provision(key, raw_provision)
                    aggregate = provisions[key].setdefault(normalized["canonical_id"], {**normalized, "case_ids": set(), "raw_variants": set()})
                    aggregate["case_ids"].add(case_id)
                    aggregate["raw_variants"].add(raw_provision)

        matched_concepts = []
        for name, pattern in CONCEPTS.items():
            if re.search(pattern, text, re.I):
                concepts[name].add(case_id)
                matched_concepts.append(name)
                
        matched_offences = []
        for name, pattern in OFFENCES.items():
            if re.search(pattern, text, re.I):
                offences[name].add(case_id)
                matched_offences.append(name)
                
        case_citations = cited_cases(case)
        for citation in case_citations:
            citations[citation].add(case_id)
            
        raw_court = case.get("metadata", {}).get("court")
        normalized_court = canonical_court(raw_court)
        courts[normalized_court]["case_ids"].add(case_id)
        if raw_court:
            courts[normalized_court]["raw_variants"].add(raw_court)
        else:
            courts[normalized_court]["raw_variants"].add("Supreme Court of India")

        per_case_analysis = {
            "case_id": case_id,
            "court": normalized_court,
            "canonical_court_id": "COURT:SUPREME_COURT_OF_INDIA",
            "statutes": case_provisions,
            "concepts": [canonical_concept_id(c) for c in sorted(matched_concepts)],
            "offences": sorted(matched_offences),
            "case_citations": case_citations,
        }
        per_case.append(per_case_analysis)

        # Write per-case reference analysis artifact in data/analysis/[CASE_ID_SLUG]_references.json
        (ANALYSIS_DIR / f"{file_path.stem}_references.json").write_text(
            json.dumps(per_case_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # If updating normalized file, inject canonical_references section
        if NORMALIZED_DIR.exists() and (NORMALIZED_DIR / file_path.name).exists():
            norm_file = NORMALIZED_DIR / file_path.name
            norm_data = json.loads(norm_file.read_text(encoding="utf-8"))
            norm_data["canonical_references"] = {
                "statutes": case_provisions,
                "legal_concepts": per_case_analysis["concepts"],
                "offences": per_case_analysis["offences"],
                "case_citations": per_case_analysis["case_citations"],
            }
            norm_file.write_text(json.dumps(norm_data, ensure_ascii=False, indent=2), encoding="utf-8")

    vocabulary = {
        "processing": {
            "source": "normalized case JSON",
            "case_count": len(cases),
            "external_sources_used": False,
        },
        "normalization": {
            "policy": "Canonical IDs merge equivalent formatting into human-readable Neo4j primary keys while preserving display names and parsed metadata properties.",
            "provision_identifier_examples": {
                "IPC 304A": "IPC:304A",
                "IPC 304(A)": "IPC:304A",
                "IPC 120B": "IPC:120B",
                "IPC 304 Part I": "IPC:304(Part I)",
                "IPC 304 (Part 1)": "IPC:304(Part I)",
                "IPC Section 300 Exception 1": "IPC:300(Exception 1)",
                "IPC Section 300 Exception IV": "IPC:300(Exception 4)",
                "CrPC 173(2)": "CrPC:173(2)",
                "Evidence Act 106": "EvidenceAct:106",
            },
        },
        "statutes": {
            key: {"name": official_name, "provisions": provision_entries(provisions[key])}
            for key, (_, official_name) in STATUTES.items()
        },
        "courts": [
            {
                "canonical_id": "COURT:" + re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_"),
                "name": name,
                "case_count": len(item["case_ids"]),
                "case_ids": sorted(item["case_ids"]),
                "raw_variants": sorted(item["raw_variants"]),
            }
            for name, item in sorted(courts.items())
        ],
        "legal_concepts": concept_entries(concepts),
        "offences": frequency_entries(offences),
        "case_citations": frequency_entries(citations),
        "cases": per_case,
    }

    (ANALYSIS_DIR / "legal_vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = ["NORMALIZED LEGAL REFERENCE REPORT", "=" * 40, f"Cases analysed: {len(cases)}", "", "STATUTORY PROVISIONS"]
    for key, payload in vocabulary["statutes"].items():
        report.append(f"\n{key} — {payload['name']}")
        report.extend(
            f"  {entry['display_name']} [{entry['canonical_id']}]: {entry['case_count']} case(s); source forms: {', '.join(entry['raw_variants'])}"
            for entry in payload["provisions"]
        )

    report.append("\nCOURTS")
    report.extend(
        f"  {entry['name']} [{entry['canonical_id']}]: {entry['case_count']} case(s); source forms: {', '.join(entry['raw_variants'])}"
        for entry in vocabulary["courts"]
    )

    report.append("\nLEGAL CONCEPTS")
    report.extend(
        f"  {entry['name']} [{entry['canonical_id']}]: {entry['case_count']} case(s); aliases: {', '.join(entry['aliases'])}"
        for entry in vocabulary["legal_concepts"]
    )

    (REPORTS_DIR / "legal_reference_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (REPORTS_DIR / "legal_reference_report.json").write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n".join(report))


if __name__ == "__main__":
    main()
