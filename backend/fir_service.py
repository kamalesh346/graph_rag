"""
backend/fir_service.py

FIR-specific legal analysis — Phase 8.

Pipeline
--------
1. Receive raw FIR text.
2. Run GraphRAG retrieval using the FIR text as the query.
3. Single LLM call: extract entities + match provisions to FIR facts + flag gaps.
4. Validate all cited source IDs against the actual retrieved set.
5. Return FirResult with cautionary, allegation-only language.

Shared with Phase 7:
- LLM provider selection (Gemini primary → Groq fallback)
- Source ID validation (no hallucinated citations)
- GraphRAGHybridRetriever (unchanged)
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Re-use provider config from llm_service
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY",   "")
GEMINI_MODEL    = os.getenv("LLM_GEMINI_MODEL", "gemini-3.8-flash")
GROQ_MODEL      = os.getenv("LLM_GROQ_MODEL",  "openai/gpt-oss-120b")
MAX_RETRIES     = 2
RETRY_DELAY_SEC = 1.5

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class FirResult:
    entities: dict[str, Any]           # people, events, injury, weapon, location, time
    candidate_provisions: list[dict]   # validated provision cards
    is_out_of_scope: bool
    provisions: list[dict]             # full retrieved provision cards for UI
    judgments: list[dict]              # full retrieved judgment cards for UI
    provider_used: str
    error: str = ""


# ---------------------------------------------------------------------------
# System prompt — cautionary language, single-pass extraction + matching
# ---------------------------------------------------------------------------

_FIR_SYSTEM_PROMPT = textwrap.dedent("""
You are a cautious Indian legal analyst assisting with FIR (First Information Report) review.

You will receive:
  - A submitted FIR text (ALLEGATIONS ONLY — not established facts)
  - Retrieved legal provisions and judgments (CONTEXT)

Your tasks:
1. Extract factual entities from the FIR into "entities":
     people, events, injury, weapon, location, time.
   These are ALLEGED — mark them as such.

2. For each provision in CONTEXT that MAY be relevant, produce a candidate card.
   A provision is a candidate only if the FIR text describes facts that could
   conceivably overlap with one or more SUBSTANTIVE INGREDIENTS of that provision.

   STRICT REJECTION RULES — exclude a provision if it:
   a) Only describes INVESTIGATION PROCEDURE (e.g. how police must file a report,
      how witnesses are examined, timelines for completing investigation).
      Examples to ALWAYS reject: CrPC § 173, CrPC § 161, CrPC § 203,
      BNSS § 193, BNSS § 180, BNSS § 351, and any section whose title contains
      "Report of police officer", "Examination of witnesses by police",
      "Dismissal of complaint", "Inquiry", or "Investigation procedure".
   b) Defines a GENERAL CONCEPT without creating a substantive offence.
   c) Is about EVIDENCE RULES that apply only at trial (unless the FIR explicitly
      involves an evidence-tampering allegation).

   In short: only include sections that define or punish the SPECIFIC OFFENCE
   described in the FIR.

   LAW APPLICABILITY — mandatory rule:
   The Bharatiya Nyaya Sanhita (BNS) replaced the Indian Penal Code (IPC) on 1 July 2024.
   The BNSS replaced CrPC, and the BSA replaced IEA on the same date.
   - For any FIR dated ON OR AFTER 1 July 2024: BNS / BNSS / BSA are the applicable laws.
   - If CONTEXT contains BOTH a BNS section AND an IPC section covering the same offence,
     include ONLY the BNS section as a candidate provision.
   - You may mention the IPC predecessor only in the "why_relevant" text as background,
     but do NOT create a separate candidate card for it.
   - Include an IPC / CrPC / IEA section as a candidate ONLY if no BNS / BNSS / BSA
     equivalent is present anywhere in CONTEXT (i.e., a KB gap).

3. For each candidate, clearly separate:
   - facts_from_fir: specific phrases or facts from the FIR that raise this provision
   - unclear_points: ingredients of the provision that cannot be determined from
     the FIR alone (intent, knowledge, threshold, etc.)

Language rules — strictly enforced:
- NEVER write "the offence is established", "the accused is guilty", or anything definitive.
- ALWAYS use: "may be relevant", "alleged", "if the facts alleged are proved",
  "based on the FIR text", "cannot be determined from the FIR alone".
- If no SUBSTANTIVE provision is relevant, set is_out_of_scope to true and
  candidate_provisions to [].
- The candidate_provisions list must contain ONLY source IDs present in CONTEXT.

Return ONLY valid JSON — no markdown, no prose:
{
  "entities": {
    "people":   ["alleged victim: ...", "alleged accused: ..."],
    "events":   ["..."],
    "injury":   "...",
    "weapon":   "...",
    "location": "...",
    "time":     "..."
  },
  "candidate_provisions": [
    {
      "source_id":    "<act>:<section> exactly as in CONTEXT>",
      "act":          "...",
      "section":      "...",
      "title":        "...",
      "why_relevant": "One sentence using 'may be relevant' language.",
      "facts_from_fir":   ["Specific phrase or fact from the FIR", "..."],
      "unclear_points":   ["What cannot be determined from the FIR alone", "..."],
      "judgment_id":  "<judgment source_id from CONTEXT, or null>"
    }
  ],
  "is_out_of_scope": false
}
""").strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object in LLM response")
    return json.loads(match.group())


def _build_context(provisions: list[dict], judgments: list[dict]) -> tuple[str, set[str]]:
    lines: list[str] = []
    valid_ids: set[str] = set()

    for p in provisions:
        sid = f"{p.get('act', '')}:{p.get('section', '')}"
        valid_ids.add(sid)
        lines.append(
            f"[STATUTE id={sid}] {p.get('act','')} § {p.get('section','')} — {p.get('title','')}\n"
            f"{str(p.get('excerpt',''))[:500]}\n---"
        )

    for j in judgments:
        sid = str(j.get("source_id") or j.get("case_id") or "")
        valid_ids.add(sid)
        lines.append(
            f"[JUDGMENT id={sid}] Case: {j.get('case_id','')}\n"
            f"{str(j.get('excerpt',''))[:400]}\n---"
        )

    logger.debug("FIR context built: %d provisions, %d judgments, valid_ids=%s",
                 len(provisions), len(judgments), valid_ids)
    return "\n".join(lines), valid_ids


# ---------------------------------------------------------------------------
# Procedural section blocklist — never valid as a substantive offence provision
# ---------------------------------------------------------------------------

# These sections define investigation procedure, not offences.
# The LLM is instructed to reject them via the system prompt; this filter is
# a hard safety net in case the LLM still includes them.
_PROCEDURAL_SECTION_IDS: set[str] = {
    # CrPC — investigation procedure
    "CrPC:154", "CrPC:155", "CrPC:156", "CrPC:157", "CrPC:158",
    "CrPC:159", "CrPC:160", "CrPC:161", "CrPC:162", "CrPC:163",
    "CrPC:164", "CrPC:167", "CrPC:168", "CrPC:169", "CrPC:170",
    "CrPC:172", "CrPC:173", "CrPC:174", "CrPC:175", "CrPC:176",
    "CrPC:200", "CrPC:202", "CrPC:203", "CrPC:204",
    # BNSS — investigation procedure (new code)
    "BNSS:173", "BNSS:174", "BNSS:175", "BNSS:176", "BNSS:177",
    "BNSS:178", "BNSS:179", "BNSS:180", "BNSS:181", "BNSS:182",
    "BNSS:190", "BNSS:191", "BNSS:192", "BNSS:193", "BNSS:194",
    "BNSS:195", "BNSS:220", "BNSS:221", "BNSS:351",
}

_PROCEDURAL_TITLE_KEYWORDS: tuple[str, ...] = (
    "report of police officer",
    "examination of witnesses by police",
    "dismissal of complaint",
    "completion of investigation",
    "inquest",
    "inquiry by magistrate",
    "power to examine",
    "recording of confessions",
    "statements to police",
)


def _filter_procedural(candidates: list[dict], provisions: list[dict]) -> list[dict]:
    """
    Hard post-LLM filter: remove any candidate whose source_id or title
    identifies it as a pure investigation-procedure section.
    """
    prov_map = {f"{p.get('act','')}:{p.get('section','')}": p for p in provisions}
    kept: list[dict] = []
    for c in candidates:
        sid = str(c.get("source_id", ""))
        if sid in _PROCEDURAL_SECTION_IDS:
            logger.info("FIR: DROPPED procedural section %s (blocklist)", sid)
            continue
        title = (c.get("title") or prov_map.get(sid, {}).get("title", "")).lower()
        if any(kw in title for kw in _PROCEDURAL_TITLE_KEYWORDS):
            logger.info("FIR: DROPPED procedural section %s (title match: %r)", sid, title)
            continue
        kept.append(c)
    if len(kept) < len(candidates):
        logger.info(
            "FIR procedural filter: %d removed, %d kept",
            len(candidates) - len(kept), len(kept),
        )
    return kept


# Superseded acts — applicable before 1 July 2024 only
_SUPERSEDED_ACTS: frozenset[str] = frozenset({"IPC", "CrPC", "IEA"})
# Current acts — applicable on/after 1 July 2024
_CURRENT_ACTS: frozenset[str] = frozenset({"BNS", "BNSS", "BSA"})


def _prefer_bns_over_ipc(candidates: list[dict]) -> list[dict]:
    """
    Post-LLM law-applicability filter.

    Rule: If the candidate list contains at least one BNS / BNSS / BSA section,
    drop all IPC / CrPC / IEA sections — they are superseded for FIRs from 2024+.

    Fallback: if ONLY IPC / CrPC / IEA sections exist (no BNS equivalent was
    retrieved), keep them and log the KB gap.
    """
    has_current    = any(c.get("act", "").upper() in _CURRENT_ACTS    for c in candidates)
    has_superseded = any(c.get("act", "").upper() in _SUPERSEDED_ACTS for c in candidates)

    if not has_current:
        # No BNS/BNSS/BSA in candidates — KB gap; keep IPC as-is
        if has_superseded:
            logger.info(
                "FIR law-pref: no BNS/BNSS/BSA candidates — keeping %d IPC/CrPC/IEA "
                "as KB-gap fallback", len(candidates)
            )
        return candidates

    # BNS present — drop superseded acts
    kept    = [c for c in candidates if c.get("act", "").upper() not in _SUPERSEDED_ACTS]
    dropped = [c for c in candidates if c.get("act", "").upper() in _SUPERSEDED_ACTS]
    if dropped:
        logger.info(
            "FIR law-pref: dropped %d superseded IPC/CrPC/IEA candidate(s) %s; "
            "kept %d BNS/BNSS/BSA candidate(s)",
            len(dropped),
            [c.get("source_id") for c in dropped],
            len(kept),
        )
    return kept


def _validate_candidates(
    candidates: list[dict],
    valid_ids: set[str],
    provisions: list[dict],
    judgments: list[dict],
) -> list[dict]:
    """Strip hallucinated source IDs and enrich cards with full title."""
    prov_map = {f"{p.get('act','')}:{p.get('section','')}": p for p in provisions}
    judg_map = {str(j.get("source_id") or j.get("case_id") or ""): j for j in judgments}

    out = []
    for c in candidates:
        sid = str(c.get("source_id", ""))
        if sid not in valid_ids:
            logger.warning("FIR: REJECTED candidate source_id=%r (not in valid_ids %s)",
                           sid, valid_ids)
            continue
        # Enrich from retrieval data if LLM left fields blank
        if sid in prov_map:
            p = prov_map[sid]
            c.setdefault("act", p.get("act", ""))
            c.setdefault("section", p.get("section", ""))
            c.setdefault("title", p.get("title", ""))
        jid = c.get("judgment_id")
        if jid and jid not in judg_map:
            logger.warning("FIR: stripped hallucinated judgment_id=%r from candidate %s", jid, sid)
            c["judgment_id"] = None
        logger.debug("FIR: ACCEPTED candidate source_id=%r why=%s",
                     sid, str(c.get('why_relevant', ''))[:80])
        out.append(c)

    logger.info("FIR validation: %d/%d candidates accepted", len(out), len(candidates))
    return out


# ---------------------------------------------------------------------------
# Provider calls  (same pattern as llm_service.py)
# ---------------------------------------------------------------------------

def _call_gemini(user_msg: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
    model   = os.getenv("LLM_GEMINI_MODEL", GEMINI_MODEL)
    client  = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=_FIR_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return _extract_json(resp.text)


def _call_groq(user_msg: str) -> dict[str, Any]:
    from groq import Groq
    model  = os.getenv("LLM_GROQ_MODEL", GROQ_MODEL)
    client = Groq(api_key=os.getenv("GROQ_API_KEY", GROQ_API_KEY))
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _FIR_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return _extract_json(completion.choices[0].message.content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_fir(
    fir_text: str,
    provisions: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> FirResult:
    """
    Analyse FIR text against retrieved provisions/judgments.

    All source IDs in the returned candidate_provisions are validated against
    the actual retrieved set — the LLM cannot invent citations.
    """
    if not fir_text.strip():
        return FirResult(
            entities={},
            candidate_provisions=[],
            is_out_of_scope=True,
            provisions=provisions,
            judgments=judgments,
            provider_used="none",
            error="Empty FIR text.",
        )

    context, valid_ids = _build_context(provisions, judgments)

    if not valid_ids:
        return FirResult(
            entities={},
            candidate_provisions=[],
            is_out_of_scope=True,
            provisions=provisions,
            judgments=judgments,
            provider_used="none",
            error="No legal sources retrieved for this FIR.",
        )

    user_msg = (
        f"FIR TEXT (alleged facts only):\n{fir_text.strip()}\n\n"
        f"CONTEXT (retrieved legal sources):\n{context}\n\n"
        "Respond with JSON only."
    )

    # Provider order from env (same logic as llm_service.py)
    primary = os.getenv("LLM_PRIMARY", "gemini").lower()
    _all = [
        ("gemini", _call_gemini, os.getenv("GEMINI_API_KEY", "")),
        ("groq",   _call_groq,   os.getenv("GROQ_API_KEY",   "")),
    ]
    if primary == "groq":
        _all = list(reversed(_all))
    providers = [(name, fn) for name, fn, key in _all if key]

    last_error = "No LLM provider configured."
    for provider_name, call_fn in providers:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = call_fn(user_msg)

                entities   = raw.get("entities", {})
                candidates = raw.get("candidate_provisions", [])
                is_oos     = bool(raw.get("is_out_of_scope", False))

                logger.info(
                    "FIR LLM raw output via %s: %d candidates, is_out_of_scope=%s",
                    provider_name, len(candidates), is_oos,
                )
                for i, c in enumerate(candidates):
                    logger.debug("  raw candidate[%d]: source_id=%r why=%s",
                                 i, c.get('source_id'), str(c.get('why_relevant',''))[:80])

                # Validate all cited source IDs
                validated = _validate_candidates(candidates, valid_ids, provisions, judgments)

                # Hard filter: drop procedural sections the LLM should not have included
                validated = _filter_procedural(validated, provisions)

                # Law-applicability filter: prefer BNS/BNSS/BSA over superseded IPC/CrPC/IEA
                validated = _prefer_bns_over_ipc(validated)

                if not validated and not is_oos:
                    logger.warning(
                        "FIR: all %d candidates rejected (after procedural + law-pref filters) "
                        "and LLM said is_out_of_scope=False — overriding to is_out_of_scope=True "
                        "(retrieval_count=%d)",
                        len(candidates), len(provisions),
                    )
                    is_oos = True

                logger.info(
                    "FIR analysis via %s | candidates=%d | out_of_scope=%s",
                    provider_name, len(validated), is_oos,
                )

                return FirResult(
                    entities=entities,
                    candidate_provisions=validated,
                    is_out_of_scope=is_oos,
                    provisions=provisions,
                    judgments=judgments,
                    provider_used=provider_name,
                )

            except Exception as exc:
                last_error = f"{provider_name} attempt {attempt}: {exc}"
                logger.warning("FIR LLM call failed — %s", last_error)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)

    return FirResult(
        entities={},
        candidate_provisions=[],
        is_out_of_scope=False,
        provisions=provisions,
        judgments=judgments,
        provider_used="none",
        error=last_error,
    )
