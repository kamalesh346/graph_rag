"""
backend/llm_service.py

Source-grounded Q&A for Legal GraphRAG — Phase 7.

Pipeline:
  1. Build a context block from retrieved provisions + judgments (IDs as anchors).
  2. Ask the LLM to tag each source RELEVANT / IRRELEVANT, then compose an answer
     using ONLY relevant ones.
  3. Post-validation: strip any cited source ID not in the actual retrieved set.
     The LLM cannot invent citations — period.
  4. If no valid relevant sources survive → return out-of-scope response.

Primary : Google Gemini 1.5 Flash
Fallback : Groq (llama-3.3-70b-versatile)
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (loaded from .env via load_dotenv in main.py)
# ---------------------------------------------------------------------------
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY",   "")
GEMINI_MODEL    = os.getenv("LLM_GEMINI_MODEL", "gemini-2.0-flash")
GROQ_MODEL      = os.getenv("LLM_GROQ_MODEL",  "openai/gpt-oss-120b")
MAX_RETRIES     = 2
RETRY_DELAY_SEC = 1.5

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AskResult:
    answer: str
    sources: list[str]          # validated source IDs only
    limitations: list[str]
    is_out_of_scope: bool
    provider_used: str          # "gemini" | "groq" | "none"
    error: str = ""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""
You are a precise Indian legal research assistant.
Answer the user's question using ONLY the source documents in the CONTEXT block.

Strict rules:
1. NEVER cite a section, case, or ID that does not appear in CONTEXT.
2. NEVER invent legal provisions, case names, or statistics.
3. Mark every source as RELEVANT or IRRELEVANT in "relevance_tags" before writing
   the answer.  A source is relevant only if it directly addresses the query.
4. If no sources are directly relevant, set "is_out_of_scope" to true and write
   a polite explanation in "answer".
5. The "sources" array must contain only the IDs you used in the answer.
6. Answer in plain English. Be concise and factual.
7. Add any caveats or uncertainties to the "limitations" array.

Return ONLY valid JSON — no markdown, no extra text:
{
  "is_out_of_scope": false,
  "relevance_tags": {"<source_id>": "RELEVANT"},
  "answer": "...",
  "sources": ["<id>"],
  "limitations": ["..."]
}
""").strip()


def _build_context(provisions: list[dict], judgments: list[dict]) -> tuple[str, set[str]]:
    """Return (context_text, valid_ids) from retrieved provisions and judgments."""
    lines: list[str] = []
    valid_ids: set[str] = set()

    for p in provisions:
        sid = f"{p.get('act', '')}:{p.get('section', '')}"
        valid_ids.add(sid)
        lines.append(
            f"[STATUTE id={sid}] {p.get('act','')} § {p.get('section','')} — {p.get('title','')}\n"
            f"{str(p.get('excerpt',''))[:600]}\n---"
        )

    for j in judgments:
        sid = str(j.get("source_id") or j.get("case_id") or "")
        valid_ids.add(sid)
        lines.append(
            f"[JUDGMENT id={sid}] Case: {j.get('case_id','')}\n"
            f"{str(j.get('excerpt',''))[:600]}\n---"
        )

    return "\n".join(lines), valid_ids


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object in LLM response")
    return json.loads(match.group())


def _validate(raw: dict[str, Any], valid_ids: set[str]) -> AskResult:
    """Validate LLM output; strip hallucinated source IDs."""
    is_oos   = bool(raw.get("is_out_of_scope", False))
    answer   = str(raw.get("answer", "")).strip()
    raw_srcs = raw.get("sources", [])
    limits   = [str(l) for l in raw.get("limitations", [])]

    validated = [s for s in raw_srcs if str(s) in valid_ids]

    if not validated and not is_oos:
        is_oos = True
        answer = (
            "The legal sources available in this knowledge base do not appear to "
            "directly address your query. This may be outside the scope of the "
            "covered legislation and case law. Please rephrase your question or "
            "consult a qualified legal professional."
        )
        limits = ["No directly relevant provisions or judgments were found."]

    return AskResult(
        answer=answer,
        sources=validated,
        limitations=limits,
        is_out_of_scope=is_oos,
        provider_used="",
    )


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

def _call_gemini(user_msg: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return _extract_json(response.text)


def _call_groq(user_msg: str) -> dict[str, Any]:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return _extract_json(completion.choices[0].message.content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    provisions: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> AskResult:
    """
    Generate a source-grounded answer with hallucination guards.

    Every source ID in the returned `sources` list is guaranteed to exist in
    the provided `provisions` or `judgments` lists — the LLM cannot invent
    references.
    """
    context, valid_ids = _build_context(provisions, judgments)

    if not valid_ids:
        return AskResult(
            answer=(
                "No legal sources were retrieved for your query. "
                "This topic may not be covered in the current knowledge base."
            ),
            sources=[],
            limitations=["Knowledge base returned no results."],
            is_out_of_scope=True,
            provider_used="none",
        )

    user_msg = f"CONTEXT:\n{context}\n\nUSER QUERY:\n{query}\n\nRespond with JSON only."

    providers = []
    # Groq first (GPT-OSS 120B) — reliable fallback
    if GROQ_API_KEY:
        providers.append(("groq", _call_groq))
    # Gemini as second option
    if GEMINI_API_KEY:
        providers.append(("gemini", _call_gemini))

    last_error = "No LLM provider configured."
    for provider_name, call_fn in providers:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw    = call_fn(user_msg)
                result = _validate(raw, valid_ids)
                result.provider_used = provider_name
                logger.info(
                    "Answer via %s | out_of_scope=%s | sources=%s",
                    provider_name, result.is_out_of_scope, result.sources,
                )
                return result
            except Exception as exc:
                last_error = f"{provider_name} attempt {attempt}: {exc}"
                logger.warning("LLM call failed — %s", last_error)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC)

    return AskResult(
        answer="The answer generation service is temporarily unavailable. Please try again.",
        sources=[],
        limitations=["LLM service unavailable."],
        is_out_of_scope=False,
        provider_used="none",
        error=last_error,
    )
