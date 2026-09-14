"""
query_parser.py

Parses user legal queries to extract explicit statutory references, Act names,
section numbers, and key concepts for metadata filtering and graph expansion.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

ACT_ALIASES = {
    "IPC": ["IPC", "INDIAN PENAL CODE", "PENAL CODE"],
    "BNS": ["BNS", "BHARATIYA NYAYA SANHITA", "NYAYA SANHITA"],
    "CRPC": ["CRPC", "CODE OF CRIMINAL PROCEDURE", "CRIMINAL PROCEDURE CODE"],
    "BNSS": ["BNSS", "BHARATIYA NAGARIK SURAKSHA SANHITA", "NAGARIK SURAKSHA SANHITA"],
    "IEA": ["IEA", "INDIAN EVIDENCE ACT", "EVIDENCE ACT"],
    "BSA": ["BSA", "BHARATIYA SAKSHYA ADHINIYAM", "SAKSHYA ADHINIYAM"],
}

# Mapping of section numbers commonly associated with specific acts when act is omitted
COMMON_SECTION_ACT_DEFAULT = {
    "300": "IPC",
    "302": "IPC",
    "304": "IPC",
    "304A": "IPC",
    "307": "IPC",
    "376": "IPC",
    "120A": "IPC",
    "313": "CRPC",
    "32": "IEA",
    "101": "BNS",
    "103": "BNS",
    "105": "BNS",
    "351": "BNSS",
}


class LegalQueryParser:
    """Parses natural language legal queries into structured intent & metadata filters."""

    def __init__(self):
        # Build regex pattern for acts
        all_act_terms = []
        for canonical, aliases in ACT_ALIASES.items():
            for alias in aliases:
                all_act_terms.append(re.escape(alias))
        self.act_pattern = re.compile(r"\b(" + "|".join(all_act_terms) + r")\b", re.IGNORECASE)

        # Regex for explicit statutory section identifiers like IPC 300, Section 300, Sec. 300, IPC:300, BNS 103, CrPC 313
        # Group 1: Optional Act, Group 2: Section Number (with sub-letters like 304A or numbers like 120A)
        self.explicit_sec_pattern = re.compile(
            r"\b(?:(IPC|BNS|CRPC|BNSS|IEA|BSA)\s*[:\-\s]?\s*)?(?:SECTION|SEC\.?|S\.)\s*([0-9]+[A-Z]?)\b",
            re.IGNORECASE,
        )

        # Direct Act Section pattern e.g. "IPC 300", "BNS 103", "CrPC 313"
        self.direct_act_sec_pattern = re.compile(
            r"\b(IPC|BNS|CRPC|BNSS|IEA|BSA)\s*[:\-\s]?\s*([0-9]+[A-Z]?)\b",
            re.IGNORECASE,
        )

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parses a query string and returns a structured object containing:
        - raw_query: original string
        - extracted_sections: List of full section IDs (e.g. ['IPC:300', 'BNS:103'])
        - extracted_acts: List of Acts detected (e.g. ['IPC', 'BNS'])
        - section_numbers: List of raw section numbers (e.g. ['300', '103'])
        - primary_target_section: The single section most explicitly requested if any
        """
        query_upper = query.upper()
        extracted_sections: Set[str] = set()
        extracted_acts: Set[str] = set()
        section_numbers: Set[str] = set()

        # Find all act occurrences
        for canonical, aliases in ACT_ALIASES.items():
            for alias in aliases:
                if re.search(r"\b" + re.escape(alias) + r"\b", query_upper):
                    extracted_acts.add(canonical)
                    break

        # 1. Direct explicit section pattern e.g. "Section 300", "IPC Section 300", "Sec 304A"
        for match in self.explicit_sec_pattern.finditer(query):
            act_prefix, sec_num = match.groups()
            sec_num = sec_num.upper()
            section_numbers.add(sec_num)

            if act_prefix:
                act_canonical = self._normalize_act(act_prefix)
                extracted_acts.add(act_canonical)
                extracted_sections.add(f"{act_canonical}:{sec_num}")
            else:
                # Act was not attached directly to "Section XXX"
                if sec_num in COMMON_SECTION_ACT_DEFAULT:
                    default_act = COMMON_SECTION_ACT_DEFAULT[sec_num]
                    extracted_sections.add(f"{default_act}:{sec_num}")
                elif extracted_acts:
                    for act in extracted_acts:
                        extracted_sections.add(f"{act}:{sec_num}")

        # 2. Direct Act Sec pattern e.g. "IPC 300", "BNS 103"
        for match in self.direct_act_sec_pattern.finditer(query):
            act_prefix, sec_num = match.groups()
            sec_num = sec_num.upper()
            act_canonical = self._normalize_act(act_prefix)
            extracted_acts.add(act_canonical)
            section_numbers.add(sec_num)
            extracted_sections.add(f"{act_canonical}:{sec_num}")

        # Primary section priority resolution
        primary_target_section = None
        if len(extracted_sections) == 1:
            primary_target_section = list(extracted_sections)[0]
        elif len(extracted_sections) > 1:
            sorted_secs = list(extracted_sections)
            primary_target_section = sorted_secs[0]

        return {
            "raw_query": query,
            "extracted_sections": sorted(list(extracted_sections)),
            "extracted_acts": sorted(list(extracted_acts)),
            "section_numbers": sorted(list(section_numbers)),
            "primary_target_section": primary_target_section,
        }

    def _normalize_act(self, act_str: str) -> str:
        act_upper = act_str.upper()
        for canonical, aliases in ACT_ALIASES.items():
            if act_upper in aliases:
                return canonical
        return act_upper


if __name__ == "__main__":
    parser = LegalQueryParser()
    test_queries = [
        "What are the exceptions to murder under Section 300?",
        "Punishment for culpable homicide not amounting to murder under Section 304 or BNS 105",
        "Procedure for statement examination under CrPC 313 or BNSS 351",
        "Dying declaration under section 32 exception requirement and evidentiary value",
        "IPC 302 vs BNS 103 penalties",
    ]
    for q in test_queries:
        res = parser.parse(q)
        print(f"\nQuery: '{q}'")
        print(f"  Parsed: {res}")
