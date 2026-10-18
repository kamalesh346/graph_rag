// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export type Provision = {
  act: string;
  section: string;
  title: string;
  excerpt: string;
  related_provisions: string[];
};

export type Judgment = {
  source_id: string;
  case_id: string;
  excerpt: string;
  related_sections: string[];
};

// ---------------------------------------------------------------------------
// Single-mode response  (POST /api/research  or  POST /api/research/plain)
// ---------------------------------------------------------------------------

export type ResearchResponse = {
  query: string;
  mode: "graphrag" | "rag";
  neo4j_available: boolean;
  provisions: Provision[];
  judgments: Judgment[];
};

// ---------------------------------------------------------------------------
// Compare response  (POST /api/research/compare)
// ---------------------------------------------------------------------------

export type CompareResponse = {
  query: string;
  neo4j_available: boolean;
  rag: {
    provisions: Provision[];
    judgments: Judgment[];
  };
  graphrag: {
    provisions: Provision[];
    judgments: Judgment[];
  };
  /** Provisions that GraphRAG found through graph traversal but plain RAG missed */
  graph_additions: Provision[];
};

// ---------------------------------------------------------------------------
// API base
// ---------------------------------------------------------------------------

const apiBaseUrl = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

async function postJSON<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(
      detail?.detail || "Unable to retrieve legal material. Please try again."
    );
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/** GraphRAG retrieval (Phase 3.5) — used on the Research page */
export function research(query: string): Promise<ResearchResponse> {
  return postJSON<ResearchResponse>("/api/research", { query });
}

/** Plain-RAG baseline (Phase 5) — vector-only retrieval */
export function researchPlain(query: string): Promise<ResearchResponse> {
  return postJSON<ResearchResponse>("/api/research/plain", { query });
}

/** Side-by-side comparison (Phase 6) — both modes in one call */
export function researchCompare(query: string): Promise<CompareResponse> {
  return postJSON<CompareResponse>("/api/research/compare", { query });
}
