"""
backend/admin_service.py

Admin knowledge-base ingestion service.

Supports:
  - Statute section upload  → ChromaDB + JSONL + Neo4j (mandatory)
  - Judgment upload         → ChromaDB + JSONL + Neo4j (mandatory)
  - Existence check         → used by frontend before upload (for duplicate dialog)
  - KB stats                → section + judgment counts

Embedding model: all-MiniLM-L6-v2  (MUST match the model used in initial indexing)
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
STATUTES_JSONL  = PROJECT_ROOT / "data" / "chunks" / "statutes" / "all_statute_chunks.jsonl"
JUDGMENTS_JSONL = PROJECT_ROOT / "data" / "chunks" / "all_judgment_chunks.jsonl"

STATUTE_COLLECTION  = "legal_statute_chunks"
JUDGMENT_COLLECTION = "legal_judgment_chunks"
EMBED_MODEL         = "all-MiniLM-L6-v2"

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


# ── Lazy singletons ────────────────────────────────────────────────────────

_qdrant_client = None
_embed_model   = None


def _get_qdrant():
    """Lazy Qdrant Cloud client."""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        url = os.getenv("QDRANT_URL", "")
        key = os.getenv("QDRANT_API_KEY", "")
        if not url or not key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set.")
        _qdrant_client = QdrantClient(url=url, port=443, api_key=key, prefer_grpc=False)
    return _qdrant_client


def _get_embed():
    """Lazy sentence-transformers model."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Deterministic UUID5 — matches migrate_chroma_to_qdrant.py for consistent IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"legal_graphrag.{chunk_id}"))


def _neo4j_driver():
    """Return a connected Neo4j driver or raise RuntimeError."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as exc:
        raise RuntimeError(
            f"Neo4j is required for admin uploads but is not reachable at {NEO4J_URI}: {exc}"
        ) from exc


# ── Public helpers ─────────────────────────────────────────────────────────

def statute_exists(act: str, section: str) -> bool:
    """Return True if a statute section is already indexed in Qdrant."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    section_id = f"{act.strip().upper()}:{section.strip()}"
    try:
        results, _ = _get_qdrant().scroll(
            collection_name=STATUTE_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="section_id", match=MatchValue(value=section_id))
            ]),
            limit=1, with_payload=False, with_vectors=False,
        )
        return len(results) > 0
    except Exception:
        return False


def judgment_exists(case_id: str) -> bool:
    """Return True if any chunk for this case_id exists in Qdrant."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        results, _ = _get_qdrant().scroll(
            collection_name=JUDGMENT_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="case_id", match=MatchValue(value=case_id))
            ]),
            limit=1, with_payload=False, with_vectors=False,
        )
        return len(results) > 0
    except Exception:
        return False


def get_stats() -> dict[str, int]:
    """Return counts of indexed statutes and judgment chunks."""
    try:
        qdrant = _get_qdrant()
        s_info = qdrant.get_collection(STATUTE_COLLECTION)
        j_info = qdrant.get_collection(JUDGMENT_COLLECTION)
        return {
            "statute_sections": s_info.points_count or 0,
            "judgment_chunks":  j_info.points_count or 0,
        }
    except Exception as exc:
        logger.warning("admin stats error: %s", exc)
        return {"statute_sections": 0, "judgment_chunks": 0}


# ── Statute ingestion ───────────────────────────────────────────────────────

def _statute_chunk_id(act: str, section: str) -> str:
    safe_section = section.replace(".", "_").replace(" ", "_")
    return f"{act.upper()}_{safe_section}_SECTION_001"


def ingest_statute(
    act: str,
    section: str,
    title: str,
    text: str,
    predecessor: str = "",
    successor: str = "",
    legal_concepts: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Index one statute section into ChromaDB + JSONL + Neo4j.
    Raises ValueError if duplicate and overwrite=False.
    """
    act      = act.strip().upper()
    section  = section.strip()
    title    = title.strip()
    text     = text.strip()

    chunk_id   = _statute_chunk_id(act, section)
    section_id = f"{act}:{section}"

    # Build JSONL record (unchanged)
    chunk = {
        "chunk_id":   chunk_id,
        "act_id":     act,
        "section_id": section_id,
        "document_type": "statute",
        "chunk_type":    "section",
        "text": text,
        "source": {
            "section_number": section,
            "has_main_text":  True,
            "subsections":    [],
            "explanations":   [],
            "illustrations":  [],
            "exceptions":     [],
            "provisos":       [],
        },
        "legal_references":       [],
        "legal_concepts":         legal_concepts or [],
        "successor_references":   [successor]   if successor   else [],
        "predecessor_references": [predecessor] if predecessor else [],
        "token_count": len(text.split()),
    }

    # 1. Qdrant — check existing, then upsert
    from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList, PointStruct
    qdrant = _get_qdrant()
    embed  = _get_embed()

    existing_pts, _ = qdrant.scroll(
        collection_name=STATUTE_COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="section_id", match=MatchValue(value=section_id))
        ]),
        limit=1, with_payload=False, with_vectors=False,
    )
    overwritten = False
    if existing_pts:
        if not overwrite:
            raise ValueError(f"{section_id} already exists. Pass overwrite=True to replace.")
        qdrant.delete(
            collection_name=STATUTE_COLLECTION,
            points_selector=PointIdsList(points=[pt.id for pt in existing_pts]),
        )
        overwritten = True

    # ── Pre-flight: verify Neo4j is reachable before writing anything ─────
    driver = _neo4j_driver()   # raises RuntimeError if unreachable

    payload = {
        "act_id":        act,
        "section_id":    section_id,
        "document_type": "statute",
        "chunk_type":    "section",
        "section_number": section,
        "title":         title,
        "has_main_text": True,
        "subsections":   "",
        "explanations":  "",
        "illustrations": "",
        "exceptions":    "",
        "provisos":      "",
        "token_count":   chunk["token_count"],
        "legal_references":       "",
        "legal_concepts":         ",".join(legal_concepts or []),
        "successor_references":   successor,
        "predecessor_references": predecessor,
        "text":    text,
        "chunk_id": chunk_id,
    }
    vector = embed.encode(text).tolist()
    qdrant.upsert(
        collection_name=STATUTE_COLLECTION,
        points=[PointStruct(
            id=_chunk_id_to_uuid(chunk_id),
            vector=vector,
            payload=payload,
        )],
    )
    logger.info("admin: statute %s indexed in Qdrant (overwrite=%s)", section_id, overwritten)

    # 2. JSONL
    _upsert_jsonl(STATUTES_JSONL, chunk, key_field="chunk_id", overwrite=overwrite)

    # 3. Neo4j
    try:
        with driver.session() as sess:
            for cql in [
                "CREATE CONSTRAINT act_id_unique IF NOT EXISTS FOR (a:Act) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT statute_section_unique IF NOT EXISTS FOR (s:StatuteSection) REQUIRE s.id IS UNIQUE",
            ]:
                try:
                    sess.run(cql)
                except Exception:
                    pass

            sess.run("""
                MERGE (a:Act {id: $act_id})
                MERGE (s:StatuteSection {id: $sec_id})
                SET s.title = $title,
                    s.section_number = $section,
                    s.act_id = $act_id
                MERGE (a)-[:HAS_SECTION]->(s)
            """, act_id=act, sec_id=section_id, title=title, section=section)

            sess.run("""
                MERGE (chk:Chunk {id: $chunk_id})
                SET chk.document_type = 'statute',
                    chk.chunk_type    = 'section',
                    chk.text          = $text,
                    chk.act_id        = $act_id,
                    chk.section_id    = $sec_id
                WITH chk
                MATCH (s:StatuteSection {id: $sec_id})
                MERGE (s)-[:HAS_CHUNK]->(chk)
            """, chunk_id=chunk_id, text=text[:2000], act_id=act, sec_id=section_id)

            if predecessor:
                sess.run("""
                    MERGE (s1:StatuteSection {id: $old_id})
                    MERGE (s2:StatuteSection {id: $new_id})
                    MERGE (s1)-[:CORRESPONDS_TO]->(s2)
                """, old_id=predecessor, new_id=section_id)

            if successor:
                sess.run("""
                    MERGE (s1:StatuteSection {id: $cur_id})
                    MERGE (s2:StatuteSection {id: $suc_id})
                    MERGE (s1)-[:CORRESPONDS_TO]->(s2)
                """, cur_id=section_id, suc_id=successor)

        logger.info("admin: statute %s ingested into Neo4j", section_id)
    finally:
        driver.close()

    return {
        "chunk_id":      chunk_id,
        "section_id":    section_id,
        "indexed":       True,
        "overwritten":   overwritten,
        "neo4j_updated": True,
    }


# ── Judgment ingestion ──────────────────────────────────────────────────────

_LEGAL_REF_RE = re.compile(
    r'\b(IPC|BNS|CrPC|BNSS|IEA|BSA|POCSO|CPC)\s*[§:s\.]*\s*(\d+[A-Za-z]?)',
    re.IGNORECASE,
)


def _extract_legal_refs(text: str) -> list[str]:
    refs: set[str] = set()
    for m in _LEGAL_REF_RE.finditer(text):
        refs.add(f"{m.group(1).upper()}:{m.group(2)}")
    return sorted(refs)


def _chunk_judgment_text(case_id: str, text: str) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks: list[dict] = []
    current: list[str] = []
    cur_tokens = 0
    idx = 1

    def flush():
        nonlocal idx, current, cur_tokens
        if not current:
            return
        body = "\n\n".join(current)
        refs = _extract_legal_refs(body)
        safe_id = re.sub(r'[^A-Za-z0-9_]', '_', case_id)
        chunks.append({
            "chunk_id":         f"{safe_id}_JUDGMENT_{idx:03d}",
            "case_id":          case_id,
            "document_type":    "judgment",
            "chunk_type":       "judgment",
            "text":             body,
            "source":           {"field": "judgment.paragraphs"},
            "legal_references": refs,
            "legal_concepts":   [],
            "cited_cases":      [],
            "token_count":      cur_tokens,
        })
        idx       += 1
        current    = []
        cur_tokens = 0

    for para in paragraphs:
        tokens = len(para.split())
        if cur_tokens + tokens > 600 and current:
            flush()
        current.append(para)
        cur_tokens += tokens
    flush()
    return chunks


def ingest_judgment(
    case_id: str,
    text: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Chunk, embed, and index a judgment into ChromaDB + JSONL + Neo4j.
    Raises ValueError if any chunk already exists and overwrite=False.
    """
    case_id = case_id.strip()
    text    = text.strip()
    if not case_id or not text:
        raise ValueError("case_id and text are required.")

    chunks    = _chunk_judgment_text(case_id, text)
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Check existing chunks in Qdrant
    from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList, PointStruct
    qdrant = _get_qdrant()
    embed  = _get_embed()

    existing_pts: list = []
    for cid in chunk_ids:
        pts, _ = qdrant.scroll(
            collection_name=JUDGMENT_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="chunk_id", match=MatchValue(value=cid))
            ]),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        existing_pts.extend(pts)

    overwritten = False
    if existing_pts:
        if not overwrite:
            raise ValueError(
                f"Case '{case_id}' already has {len(existing_pts)} chunk(s) indexed. "
                "Pass overwrite=True to replace."
            )
        qdrant.delete(
            collection_name=JUDGMENT_COLLECTION,
            points_selector=PointIdsList(points=[pt.id for pt in existing_pts]),
        )
        overwritten = True

    # ── Pre-flight: verify Neo4j is reachable before writing anything ─────
    driver = _neo4j_driver()   # raises RuntimeError if unreachable

    for c in chunks:
        meta = {
            "case_id":          c["case_id"],
            "document_type":    c["document_type"],
            "chunk_type":       c["chunk_type"],
            "source_field":     c["source"].get("field", ""),
            "page_start":       0,
            "page_end":         0,
            "token_count":      c["token_count"],
            "legal_references": ",".join(c["legal_references"]),
            "legal_concepts":   "",
            "cited_cases":      "",
            "text":             c["text"],
            "chunk_id":         c["chunk_id"],
        }
        vector = embed.encode(c["text"]).tolist()
        qdrant.upsert(
            collection_name=JUDGMENT_COLLECTION,
            points=[PointStruct(
                id=_chunk_id_to_uuid(c["chunk_id"]),
                vector=vector,
                payload=meta,
            )],
        )

    logger.info("admin: judgment %s — %d chunks indexed in Qdrant", case_id, len(chunks))

    for c in chunks:
        _upsert_jsonl(JUDGMENTS_JSONL, c, key_field="chunk_id", overwrite=overwrite)

    # Neo4j (mandatory)
    try:
        with driver.session() as sess:
            for cql in [
                "CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (chk:Chunk) REQUIRE chk.id IS UNIQUE",
                "CREATE CONSTRAINT statute_id_unique IF NOT EXISTS FOR (s:StatuteSection) REQUIRE s.id IS UNIQUE",
            ]:
                try:
                    sess.run(cql)
                except Exception:
                    pass

            sess.run("MERGE (c:Case {id: $case_id})", case_id=case_id)

            for c in chunks:
                sess.run("""
                    MATCH (cas:Case {id: $case_id})
                    MERGE (chk:Chunk {id: $chunk_id})
                    SET chk.document_type = 'judgment',
                        chk.chunk_type    = 'judgment',
                        chk.text          = $text,
                        chk.case_id       = $case_id
                    MERGE (cas)-[:HAS_CHUNK]->(chk)
                """, case_id=case_id, chunk_id=c["chunk_id"], text=c["text"][:2000])

                for ref in c["legal_references"]:
                    sess.run("""
                        MATCH (chk:Chunk {id: $chunk_id})
                        MERGE (s:StatuteSection {id: $ref_id})
                        MERGE (chk)-[:REFERS_TO]->(s)
                    """, chunk_id=c["chunk_id"], ref_id=ref)

        logger.info("admin: judgment %s ingested into Neo4j", case_id)
    finally:
        driver.close()

    return {
        "case_id":        case_id,
        "chunks_created": len(chunks),
        "indexed":        True,
        "overwritten":    overwritten,
        "neo4j_updated":  True,
    }


# ── JSONL helper ────────────────────────────────────────────────────────────

def _upsert_jsonl(path: Path, record: dict, key_field: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_val = record[key_field]

    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        new_lines: list[str] = []
        replaced = False
        for line in lines:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                new_lines.append(line)
                continue
            if obj.get(key_field) == key_val:
                if overwrite:
                    new_lines.append(json.dumps(record, ensure_ascii=False))
                    replaced = True
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        if not replaced:
            new_lines.append(json.dumps(record, ensure_ascii=False))
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    else:
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
