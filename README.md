# Legal GraphRAG

A Graph-Augmented Retrieval system for Indian legal research. Ask a question or paste an FIR — get relevant statute sections, supporting judgments, and a grounded plain-language answer. It does this without making up citations.

---

## What It Does

| Capability | Plain RAG | GraphRAG |
|---|---|---|
| Statute retrieval | ✅ | ✅ |
| Judgment retrieval | ✅ | ✅ |
| IPC ↔ BNS / CrPC ↔ BNSS cross-mapping | ❌ | ✅ |
| Section-to-section graph expansion | ❌ | ✅ |
| Concept-aware traversal | ❌ | ✅ |
| Source-grounded answer | ✅ | ✅ |
| FIR fact extraction + gap analysis | ❌ | ✅ |

GraphRAG answers: *"Which chunks are similar, which sections correspond across acts, which judgments explicitly cite those sections, and what precedents do those cases cite?"*

Plain RAG only answers: *"Which chunks are textually similar to this query?"*

---

## Tech Stack

**Backend** — Python / FastAPI, deployed on Render
**Vector Store** — Qdrant Cloud (cloud inference, no local model)
**Knowledge Graph** — Neo4j AuraDB
**LLM** — Gemini 2.5 Flash (primary) → Groq GPT-OSS 120B (fallback)
**Frontend** — Next.js 14 + TypeScript + Tailwind CSS + shadcn/ui
**Embedding** — `sentence-transformers/all-minilm-l6-v2` via Qdrant Cloud Inference (server-side, zero local RAM)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│         FastAPI Backend (Render)        │
│                                         │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ Query Parser │  │  FIR Extractor   │ │
│  └──────┬───────┘  └────────┬─────────┘ │
│         │                   │           │
│  ┌──────▼───────────────────▼─────────┐ │
│  │       GraphRAG Hybrid Retriever    │ │
│  │                                    │ │
│  │  Neo4j Graph Expansion             │ │
│  │  ├─ CORRESPONDS_TO (IPC ↔ BNS)    │ │
│  │  ├─ REFERS_TO (cross-references)  │ │
│  │  └─ DISCUSSES (legal concepts)    │ │
│  │                                    │ │
│  │  Qdrant Cloud Inference            │ │
│  │  ├─ legal_statute_chunks           │ │
│  │  └─ legal_judgment_chunks          │ │
│  │                                    │ │
│  │  RRF Fusion + Metadata Boosts      │ │
│  └────────────────────────────────────┘ │
│         │                               │
│  ┌──────▼───────┐                       │
│  │  LLM Service │ Gemini → Groq         │
│  │  (grounded   │ (source-validated,    │
│  │   answer)    │  no hallucination)    │
│  └──────────────┘                       │
└─────────────────────────────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness + Qdrant + Neo4j status |
| `POST` | `/api/research` | GraphRAG retrieval |
| `POST` | `/api/research/plain` | Plain-RAG baseline (vector-only) |
| `POST` | `/api/research/compare` | Both modes side-by-side + graph diff |
| `POST` | `/api/ask` | Source-grounded Q&A via LLM |
| `POST` | `/api/fir` | FIR legal analysis |

---

## What's Built

### 🔍 Research Page
Natural-language legal queries. Returns statute cards with section excerpts, related act mappings (IPC/BNS, CrPC/BNSS), and judgment source cards. No scores, no technical jargon shown to users.

### ⚖️ Compare Page
Run one query through both Plain RAG and GraphRAG side-by-side. Clearly shows what connections the graph discovered that pure vector search missed.

### 📋 FIR Analysis Page
Paste raw FIR text. The system:
1. Runs GraphRAG using the FIR as the query
2. Calls an LLM to extract entities (people, events, injury, weapon, location)
3. Matches FIR facts to candidate statute sections
4. Flags missing information needed for a complete legal picture
5. Uses cautionary language — *"may be relevant"*, never *"offence established"*

### 💬 Q&A (Answer Generation)
Every generated answer is validated against the retrieved source IDs before being returned. If sources are insufficient, the system says so. No invented section references.

### 🛠️ Admin Panel
Upload new statute sections or court judgments through the UI. Ingestion pipeline:
- Chunks the text
- Embeds via Qdrant Cloud Inference (server-side)
- Writes payload to Qdrant
- Creates corresponding Neo4j nodes and edges
- Saves to JSONL backup

Duplicate detection with overwrite confirmation.

### 🏗️ Knowledge Base
- **108 statute sections** across IPC, BNS, CrPC, BNSS, IEA, BSA
- **285 judgment chunks** from selected Supreme Court / High Court cases
- All stored in Qdrant Cloud; graph relationships in Neo4j AuraDB

---

## Highlights

- **No local embedding model** — embedding moved to Qdrant Cloud Inference, eliminating `sentence-transformers` + `torch` from the Render deployment. Significant RAM reduction.
- **Graceful degradation** — if Neo4j is unreachable, the system falls back to vector-only retrieval automatically. No crash.
- **Source validation** — every LLM answer is checked against the actual retrieved point IDs before returning. Hallucinated citations are blocked.
- **Deterministic IDs** — all Qdrant point IDs are UUID5 derived from chunk IDs, making upserts fully idempotent. Re-running ingestion never creates duplicates.
- **Dual LLM** — Gemini 2.5 Flash as primary (1M context), Groq GPT-OSS 120B as fallback. Groq has documented zero-data-retention settings, relevant for FIR content.
- **33 automated tests** — 28 unit tests (no network) + 5 integration tests against live Qdrant Cloud.

---

## Running Locally

```bash
# Backend
cd graph_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in QDRANT_URL, QDRANT_API_KEY, NEO4J_*, GEMINI_API_KEY
uvicorn backend.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Required env vars: `QDRANT_URL`, `QDRANT_API_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GEMINI_API_KEY`, `GROQ_API_KEY`

---

## Deployment

- **Backend**: Render (Python web service)
- **Frontend**: Vercel or any static host
- **Vector DB**: Qdrant Cloud (managed)
- **Graph DB**: Neo4j AuraDB (managed)

A `keep_alive.py` loop prevents Render's free-tier spin-down by pinging the service and checking connectivity on a schedule.

---

## Data Privacy Note

FIR text may contain PII. Gemini Flash and Groq both offer data retention controls — review your provider's policy before production use with real FIR content. PII redaction is planned.
