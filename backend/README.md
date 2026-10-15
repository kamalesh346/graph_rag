# Legal GraphRAG API

Start Neo4j, activate the project virtual environment, then run:

```bash
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive OpenAPI docs are at
`http://127.0.0.1:8000/docs`.

```bash
curl http://127.0.0.1:8000/api/health

curl -X POST http://127.0.0.1:8000/api/research \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the legal procedure for examining an accused person?"}'
```

Neo4j is optional for request handling. If it is unavailable, the existing
retriever falls back to ChromaDB vector retrieval and returns results without
graph-expanded links.

For a separate frontend development server, configure allowed browser origins
in `.env` (the default is `http://localhost:3000`):

```env
FRONTEND_ORIGINS=http://localhost:3000
```
