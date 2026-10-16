# Legal GraphRAG frontend

This is the Next.js frontend for the Legal GraphRAG project. The Research page
calls the FastAPI service at `http://127.0.0.1:8000` by default.

## Run locally

Install Node.js 20 or newer, then:

```bash
cd frontend
npm install
npm run dev
```

In another terminal, start the backend from the project root:

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

Open `http://localhost:3000`. The routes are `/research`, `/compare`,
`/history`, and `/login`. To use a backend at another address, copy
`.env.example` to `.env.local` and set `NEXT_PUBLIC_API_BASE_URL`.
