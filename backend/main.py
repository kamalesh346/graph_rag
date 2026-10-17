"""FastAPI adapter for the existing ChromaDB and Neo4j GraphRAG retriever."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_DIR = PROJECT_ROOT / "scripts" / "retrieval"

load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrieval.graph_retriever import Neo4jGraphRetriever  # noqa: E402
from scripts.retrieval.hybrid_reranker import CHROMA_DB_DIR, GraphRAGHybridRetriever  # noqa: E402


app = FastAPI(
    title="Legal GraphRAG API",
    version="0.1.0",
    description="Retrieves relevant statutory provisions and judgment sources.",
)

frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://10.10.119.57:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=12_000,
        description="A legal research question or FIR-style narrative.",
    )


def _split_references(value: Any) -> list[str]:
    """Convert the project's comma-separated metadata fields into a clean list."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _title_from_text(text: str, fallback: str) -> str:
    """Use the stored title when available; otherwise derive a concise heading."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if " - " in first_line:
        first_line = first_line.split(" - ", 1)[1].strip()
    first_line = re.split(r"\s+(?:Description:|\(|If,|Whoever|When|Where|In every)\b", first_line, maxsplit=1)[0]
    return first_line[:160].strip(" -:") or fallback


def _provision_payload(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    section_id = str(item.get("section_id", ""))
    act, _, section_from_id = section_id.partition(":")
    text = str(item.get("text", ""))
    related = _split_references(metadata.get("successor_references"))
    related.extend(_split_references(metadata.get("predecessor_references")))

    return {
        "act": item.get("act") or metadata.get("act") or metadata.get("act_id") or act,
        "section": item.get("section_number") or metadata.get("section_number") or section_from_id,
        "title": item.get("title") or metadata.get("title") or _title_from_text(text, section_id),
        "excerpt": text[:800],
        "related_provisions": list(dict.fromkeys(related)),
    }


def _judgment_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": item.get("id"),
        "case_id": item.get("case_id") or "Unknown case",
        "excerpt": str(item.get("text", ""))[:800],
        "related_sections": _split_references(item.get("legal_references")),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Report availability without requiring Neo4j for the API to be usable."""
    graph = Neo4jGraphRetriever()
    neo4j_available = graph.is_connected()
    graph.close()

    return {
        "status": "ok" if CHROMA_DB_DIR.exists() else "unavailable",
        "vector_store_available": CHROMA_DB_DIR.exists(),
        "neo4j_available": neo4j_available,
    }


@app.post("/api/research")
def research(request: ResearchRequest) -> dict[str, Any]:
    """Return source-ready legal material without exposing internal retrieval scores."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank.")

    # Instantiate the hybrid retriever – it internally handles Neo4j connectivity.
    retriever: GraphRAGHybridRetriever | None = None
    try:
        retriever = GraphRAGHybridRetriever()
        # If Neo4j is unavailable, the retriever will still function with vector‑only flow.
        neo4j_available = retriever.graph_retriever.is_connected()
        result = retriever.search(query)
    except Exception as error:
        # Unexpected failure – respond with a generic error.
        raise HTTPException(status_code=503, detail="Research retrieval is currently unavailable.") from error
    finally:
        if retriever:
            retriever.close()

    # Include Neo4j availability in the response for client awareness.
    response_payload = {
        "query": query,
        "provisions": [_provision_payload(item) for item in result["statute_results"]],
        "judgments": [_judgment_payload(item) for item in result["judgment_results"]],
        "neo4j_available": neo4j_available,
    }
    return response_payload


@app.post("/api/trace")
def trace(request: ResearchRequest) -> dict[str, Any]:
    """Return detailed internal retrieval information for debugging.
    The payload includes the raw hybrid retriever output (vector scores,
    graph anchors, fusion results, etc.) without any post‑processing.
    """
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
    return {
        "query": query,
        "raw_result": result,
    }

