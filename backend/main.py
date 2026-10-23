"""FastAPI adapter for the Legal GraphRAG system.

Endpoints
---------
GET  /api/health          – liveness + store availability check
POST /api/research        – GraphRAG retrieval  (Phase 3.5)
POST /api/research/plain  – Plain-RAG baseline  (Phase 5, vector-only)
POST /api/research/compare – Both modes in one call, with graph_additions diff
POST /api/ask             – Source-grounded Q&A via LLM  (Phase 7)
POST /api/fir             – FIR legal analysis  (Phase 8)
POST /api/trace           – Raw debug output (internal use only)
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrieval.graph_retriever import Neo4jGraphRetriever          # noqa: E402
from scripts.retrieval.hybrid_reranker import CHROMA_DB_DIR, GraphRAGHybridRetriever  # noqa: E402
from scripts.retrieval.plain_rag import PlainRAGRetriever                  # noqa: E402
from backend.llm_service import generate_answer                            # noqa: E402
from backend.fir_service import analyse_fir                                # noqa: E402
from backend import admin_service                                           # noqa: E402

logger = logging.getLogger(__name__)

_chroma_client: chromadb.PersistentClient | None = None
_statute_col = None

def _get_statute_col():
    """Lazy singleton for the ChromaDB statute collection."""
    global _chroma_client, _statute_col
    if _statute_col is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        _statute_col = _chroma_client.get_collection("legal_statute_chunks")
    return _statute_col


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Legal GraphRAG API",
    version="0.2.0",
    description=(
        "Retrieves relevant statutory provisions and judgment sources. "
        "Supports plain-RAG (vector-only) and GraphRAG (vector + graph) modes."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://10.10.119.57:3000",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://10.10.119.57:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=12_000,
        description="A legal research question or FIR-style narrative.",
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _split_references(value: Any) -> list[str]:
    """Convert comma-separated metadata fields into a clean list."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _title_from_text(text: str, fallback: str) -> str:
    """Derive a concise heading from the first line of the text."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if " - " in first_line:
        first_line = first_line.split(" - ", 1)[1].strip()
    first_line = re.split(
        r"\s+(?:Description:|\(|If,|Whoever|When|Where|In every)\b",
        first_line,
        maxsplit=1,
    )[0]
    return first_line[:160].strip(" -:") or fallback


def _fetch_related_sections(section_ids: list[str]) -> list[dict[str, Any]]:
    """
    Given a list of 'ACT:SECTION' strings (e.g. 'IPC:302'), fetch their text
    from ChromaDB by querying the 'section_id' metadata field.

    We cannot use col.get(ids=...) because ChromaDB doc IDs use a different
    format (e.g. 'IPC_302_SECTION_001') than the reference strings.
    """
    if not section_ids:
        return []

    col = _get_statute_col()
    out: list[dict[str, Any]] = []

    for sid in section_ids:
        try:
            res = col.get(
                where={"section_id": sid},
                include=["documents", "metadatas"],
            )
        except Exception:
            continue

        if not res or not res.get("ids"):
            continue

        # Take the first matching chunk
        meta = (res.get("metadatas") or [[]])[0] or {}
        text = ((res.get("documents") or [[]])[0]) or ""

        # section_id is "ACT:SECTION" — derive act from it if metadata missing
        raw_act, _, raw_sec = sid.partition(":")
        act = meta.get("act") or meta.get("act_id") or raw_act
        section = meta.get("section_number") or raw_sec
        title = meta.get("title") or _title_from_text(text, sid)

        out.append({
            "id": sid,
            "act": act,
            "section": section,
            "title": title,
            "excerpt": text[:400],
        })

    return out



def _provision_payload(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    section_id = str(item.get("section_id", ""))
    act, _, section_from_id = section_id.partition(":")
    text = str(item.get("text", ""))
    related_ids = _split_references(metadata.get("successor_references"))
    related_ids.extend(_split_references(metadata.get("predecessor_references")))
    related_ids = list(dict.fromkeys(related_ids))  # dedup, preserve order

    return {
        "act": item.get("act") or metadata.get("act") or metadata.get("act_id") or act,
        "section": item.get("section_number") or metadata.get("section_number") or section_from_id,
        "title": item.get("title") or metadata.get("title") or _title_from_text(text, section_id),
        "excerpt": text[:800],
        "related_provisions": related_ids,
        "related_provisions_detail": _fetch_related_sections(related_ids),
    }


def _judgment_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": item.get("id"),
        "case_id": item.get("case_id") or "Unknown case",
        "excerpt": str(item.get("text", ""))[:800],
        "related_sections": _split_references(item.get("legal_references")),
    }


def _section_ids(provisions: list[dict[str, Any]]) -> set[str]:
    """Return the set of 'ACT:SECTION' identifiers from a provisions list."""
    ids: set[str] = set()
    for p in provisions:
        act = p.get("act", "")
        sec = p.get("section", "")
        if act and sec:
            ids.add(f"{act}:{sec}")
    return ids


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness check — reports ChromaDB and Neo4j availability."""
    graph = Neo4jGraphRetriever()
    neo4j_available = graph.is_connected()
    graph.close()

    return {
        "status": "ok" if CHROMA_DB_DIR.exists() else "unavailable",
        "vector_store_available": CHROMA_DB_DIR.exists(),
        "neo4j_available": neo4j_available,
    }


@app.post("/api/research")
def research_graphrag(request: ResearchRequest) -> dict[str, Any]:
    """
    Phase 3.5 — Enhanced GraphRAG retrieval.

    Uses ChromaDB vector search + Neo4j graph traversal + RRF fusion.
    Falls back to vector-only if Neo4j is unavailable.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank.")

    retriever: GraphRAGHybridRetriever | None = None
    try:
        retriever = GraphRAGHybridRetriever()
        neo4j_available = retriever.graph_retriever.is_connected()
        result = retriever.search(query)
    except Exception as error:
        raise HTTPException(status_code=503, detail="GraphRAG retrieval is currently unavailable.") from error
    finally:
        if retriever:
            retriever.close()

    return {
        "query": query,
        "mode": "graphrag",
        "neo4j_available": neo4j_available,
        "provisions": [_provision_payload(item) for item in result["statute_results"]],
        "judgments": [_judgment_payload(item) for item in result["judgment_results"]],
    }


@app.post("/api/research/plain")
def research_plain_rag(request: ResearchRequest) -> dict[str, Any]:
    """
    Phase 5 — Plain-RAG baseline.

    Uses ChromaDB vector similarity only.
    No Neo4j, no query parsing, no graph expansion, no fusion boosts.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank.")

    retriever: PlainRAGRetriever | None = None
    try:
        retriever = PlainRAGRetriever()
        result = retriever.search(query)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Plain-RAG retrieval is currently unavailable.") from error
    finally:
        if retriever:
            retriever.close()

    return {
        "query": query,
        "mode": "rag",
        "neo4j_available": False,
        "provisions": [_provision_payload(item) for item in result["statute_results"]],
        "judgments": [_judgment_payload(item) for item in result["judgment_results"]],
    }


@app.post("/api/research/compare")
def research_compare(request: ResearchRequest) -> dict[str, Any]:
    """
    Phase 6 — Side-by-side comparison of Plain-RAG vs GraphRAG.

    Runs both retrievers on the same query and returns:
    - rag:  plain-RAG result
    - graphrag: GraphRAG result
    - graph_additions: provisions that GraphRAG found but plain-RAG did NOT
                       (these are the sections added by graph traversal)
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank.")

    # -- Plain RAG --
    plain_retriever: PlainRAGRetriever | None = None
    try:
        plain_retriever = PlainRAGRetriever()
        plain_result = plain_retriever.search(query)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Plain-RAG unavailable.") from error
    finally:
        if plain_retriever:
            plain_retriever.close()

    plain_provisions = [_provision_payload(item) for item in plain_result["statute_results"]]
    plain_judgments  = [_judgment_payload(item)  for item in plain_result["judgment_results"]]

    # -- GraphRAG --
    graph_retriever_inst: GraphRAGHybridRetriever | None = None
    try:
        graph_retriever_inst = GraphRAGHybridRetriever()
        neo4j_available = graph_retriever_inst.graph_retriever.is_connected()
        graph_result = graph_retriever_inst.search(query)
    except Exception as error:
        raise HTTPException(status_code=503, detail="GraphRAG unavailable.") from error
    finally:
        if graph_retriever_inst:
            graph_retriever_inst.close()

    graph_provisions = [_provision_payload(item) for item in graph_result["statute_results"]]
    graph_judgments  = [_judgment_payload(item)  for item in graph_result["judgment_results"]]

    # -- Compute graph_additions: sections in GraphRAG not returned by plain RAG --
    plain_ids = _section_ids(plain_provisions)
    graph_additions = [p for p in graph_provisions if f"{p['act']}:{p['section']}" not in plain_ids]

    return {
        "query": query,
        "neo4j_available": neo4j_available,
        "rag": {
            "provisions": plain_provisions,
            "judgments": plain_judgments,
        },
        "graphrag": {
            "provisions": graph_provisions,
            "judgments": graph_judgments,
        },
        # Sections that the graph found but vector similarity alone missed
        "graph_additions": graph_additions,
    }


@app.post("/api/trace")
def trace(request: ResearchRequest) -> dict[str, Any]:
    """Raw debug output from the GraphRAG retriever (internal use only)."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank.")
    retriever: GraphRAGHybridRetriever | None = None
    try:
        retriever = GraphRAGHybridRetriever()
        result = retriever.search(query)
    finally:
        if retriever:
            retriever.close()
    return {"query": query, "raw_result": result}


@app.post("/api/ask")
def ask(request: ResearchRequest) -> dict[str, Any]:
    """
    Phase 7 — Source-grounded Q&A.

    Runs GraphRAG retrieval then passes retrieved provisions and judgments
    to the LLM service.  Every cited source ID is validated against the
    actual retrieved set — the LLM cannot invent citations.

    Returns
    -------
    {
      query, mode, neo4j_available,
      answer, sources, limitations, is_out_of_scope,
      provider_used,
      provisions, judgments     ← full retrieved cards for display
    }
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank.")

    retriever: GraphRAGHybridRetriever | None = None
    try:
        retriever = GraphRAGHybridRetriever()
        neo4j_available = retriever.graph_retriever.is_connected()
        result = retriever.search(query)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Retrieval unavailable.") from error
    finally:
        if retriever:
            retriever.close()

    provisions = [_provision_payload(item) for item in result["statute_results"]]
    judgments  = [_judgment_payload(item)  for item in result["judgment_results"]]

    llm_result = generate_answer(query, provisions, judgments)

    return {
        "query":           query,
        "mode":            "graphrag+llm",
        "neo4j_available": neo4j_available,
        # LLM answer fields
        "answer":          llm_result.answer,
        "sources":         llm_result.sources,          # validated IDs
        "limitations":     llm_result.limitations,
        "is_out_of_scope": llm_result.is_out_of_scope,
        "provider_used":   llm_result.provider_used,
        # Full retrieval cards for UI display
        "provisions":      provisions,
        "judgments":       judgments,
    }


class FirRequest(BaseModel):
    fir_text: str = Field(..., min_length=10, description="Raw FIR text to analyse.")


@app.post("/api/fir")
def fir_analysis(request: FirRequest) -> dict[str, Any]:
    """
    Phase 8 — FIR Legal Analysis.

    Runs GraphRAG retrieval using the FIR text as the search query, then
    calls fir_service to extract entities and match candidate provisions.
    All cited source IDs are validated against the retrieved set.

    FIR text is long and narrative — we retrieve more candidates (top_k=6)
    than the default (3) so the LLM has a broader pool to match against.
    """
    fir_text = request.fir_text.strip()
    if not fir_text:
        raise HTTPException(status_code=422, detail="FIR text must not be blank.")

    retriever: GraphRAGHybridRetriever | None = None
    try:
        retriever = GraphRAGHybridRetriever()
        neo4j_available = retriever.graph_retriever.is_connected()
        # FIR text is narrative — retrieve more candidates than the default
        result = retriever.search(
            fir_text[:800],          # cap to avoid token overload
            top_k_statutes=12,       # wider net: BNS + IPC both need room in top-k
            top_k_judgments=4,
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail="Retrieval unavailable.") from error
    finally:
        if retriever:
            retriever.close()

    provisions = [_provision_payload(item) for item in result["statute_results"]]
    judgments  = [_judgment_payload(item)  for item in result["judgment_results"]]

    # Debug: log what retrieval returned before sending to LLM
    prov_ids = [f"{p['act']}:{p['section']}" for p in provisions]
    logger.info("FIR retrieval: %d provisions %s, %d judgments",
                len(provisions), prov_ids, len(judgments))

    fir_result = analyse_fir(fir_text, provisions, judgments)

    return {
        "neo4j_available":      neo4j_available,
        "entities":             fir_result.entities,
        "candidate_provisions": fir_result.candidate_provisions,
        "is_out_of_scope":      fir_result.is_out_of_scope,
        "provider_used":        fir_result.provider_used,
        "provisions":           provisions,
        "judgments":            judgments,
        "error":                fir_result.error,
        # Debug / transparency fields
        "retrieval_count":      len(provisions),
    }


# ===========================================================================
# Admin endpoints — knowledge-base management
# ===========================================================================

class StatuteUploadRequest(BaseModel):
    act:          str = Field(..., description="Act code, e.g. BNS, IPC")
    section:      str = Field(..., description="Section number, e.g. 115")
    title:        str = Field(..., description="Short title of the section")
    text:         str = Field(..., description="Full section text")
    predecessor:  str = Field("",  description="Old-law equivalent, e.g. IPC:323")
    successor:    str = Field("",  description="New-law equivalent, e.g. BNS:115")
    overwrite:    bool = Field(False, description="Replace if already indexed")


class JudgmentUploadRequest(BaseModel):
    case_id:   str  = Field(..., description="Unique case identifier, e.g. 2025 INSC 123")
    text:      str  = Field(..., description="Full judgment text")
    overwrite: bool = Field(False, description="Replace if already indexed")


@app.get("/api/admin/stats")
def admin_stats() -> dict[str, Any]:
    """Return current knowledge-base document counts."""
    return admin_service.get_stats()


@app.get("/api/admin/check/statute/{act}/{section}")
def admin_check_statute(act: str, section: str) -> dict[str, Any]:
    """Check whether a statute section is already indexed."""
    exists = admin_service.statute_exists(act.upper(), section)
    return {
        "section_id": f"{act.upper()}:{section}",
        "exists":     exists,
    }


@app.get("/api/admin/check/judgment/{case_id:path}")
def admin_check_judgment(case_id: str) -> dict[str, Any]:
    """Check whether a judgment is already indexed."""
    exists = admin_service.judgment_exists(case_id)
    return {"case_id": case_id, "exists": exists}


@app.post("/api/admin/statute")
def admin_upload_statute(request: StatuteUploadRequest) -> dict[str, Any]:
    """Index a statute section into ChromaDB + JSONL + Neo4j."""
    try:
        return admin_service.ingest_statute(
            act=request.act,
            section=request.section,
            title=request.title,
            text=request.text,
            predecessor=request.predecessor,
            successor=request.successor,
            overwrite=request.overwrite,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("admin statute upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/admin/judgment")
def admin_upload_judgment(request: JudgmentUploadRequest) -> dict[str, Any]:
    """Chunk, embed, and index a judgment into ChromaDB + JSONL + Neo4j."""
    try:
        return admin_service.ingest_judgment(
            case_id=request.case_id,
            text=request.text,
            overwrite=request.overwrite,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("admin judgment upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
