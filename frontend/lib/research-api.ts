// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export type RelatedProvision = {
  id: string;
  act: string;
  section: string;
  title: string;
  excerpt: string;
};

export type Provision = {
  act: string;
  section: string;
  title: string;
  excerpt: string;
  related_provisions: string[];
  related_provisions_detail: RelatedProvision[];
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
// Ask response  (POST /api/ask)  — Phase 7
// ---------------------------------------------------------------------------

export type AskResponse = {
  query: string;
  mode: string;
  neo4j_available: boolean;
  /** LLM-generated plain-language answer */
  answer: string;
  /** Validated source IDs — guaranteed to exist in provisions or judgments */
  sources: string[];
  /** Caveats and uncertainties noted by the LLM */
  limitations: string[];
  /** True when no retrieved sources were relevant to the query */
  is_out_of_scope: boolean;
  /** Which LLM provider generated this answer */
  provider_used: string;
  /** Full retrieved provision cards (for display) */
  provisions: Provision[];
  /** Full retrieved judgment cards (for display) */
  judgments: Judgment[];
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

async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail || "Request failed. Please try again.");
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

/** Source-grounded Q&A (Phase 7) — GraphRAG retrieval + LLM synthesis */
export function ask(query: string): Promise<AskResponse> {
  return postJSON<AskResponse>("/api/ask", { query });
}

// ---------------------------------------------------------------------------
// FIR Analysis  (POST /api/fir)  — Phase 8
// ---------------------------------------------------------------------------

export type FirEntities = {
  people: string[];
  events: string[];
  injury: string;
  weapon: string;
  location: string;
  time: string;
};

export type FirCandidateProvision = {
  /** e.g. "IPC:302" — guaranteed to exist in provisions[] */
  source_id: string;
  act: string;
  section: string;
  title: string;
  /** One sentence with cautionary "may be relevant" language */
  why_relevant: string;
  /** Specific phrases from the FIR that raise this provision */
  facts_from_fir: string[];
  /** Ingredients or thresholds that cannot be determined from the FIR alone */
  unclear_points: string[];
  /** Supporting judgment source_id if matched, or null */
  judgment_id: string | null;
};

export type FirResponse = {
  neo4j_available: boolean;
  entities: FirEntities;
  candidate_provisions: FirCandidateProvision[];
  is_out_of_scope: boolean;
  provider_used: string;
  /** Full retrieved provision cards — for statutory text display */
  provisions: Provision[];
  /** Full retrieved judgment cards — for supporting judgment display */
  judgments: Judgment[];
  error: string;
  /** Number of provisions retrieved — 0 means retrieval failure, >0 means genuine no-match */
  retrieval_count: number;
};

/** FIR legal analysis (Phase 8) — entity extraction + candidate provision matching */
export function analyseFir(firText: string): Promise<FirResponse> {
  return postJSON<FirResponse>("/api/fir", { fir_text: firText });
}

// ---------------------------------------------------------------------------
// Admin API — knowledge-base management
// ---------------------------------------------------------------------------

export type KbStats = {
  statute_sections: number;
  judgment_chunks:  number;
};

export type CheckResult = {
  section_id?: string;
  case_id?:    string;
  exists:      boolean;
};

export type StatuteUploadResult = {
  chunk_id:      string;
  section_id:    string;
  indexed:       boolean;
  overwritten:   boolean;
  neo4j_updated: boolean;
};

export type JudgmentUploadResult = {
  case_id:        string;
  chunks_created: number;
  indexed:        boolean;
  overwritten:    boolean;
  neo4j_updated:  boolean;
};

export function getAdminStats(): Promise<KbStats> {
  return getJSON<KbStats>("/api/admin/stats");
}

export function checkStatuteExists(act: string, section: string): Promise<CheckResult> {
  return getJSON<CheckResult>(`/api/admin/check/statute/${encodeURIComponent(act)}/${encodeURIComponent(section)}`);
}

export function checkJudgmentExists(caseId: string): Promise<CheckResult> {
  return getJSON<CheckResult>(`/api/admin/check/judgment/${encodeURIComponent(caseId)}`);
}

export function uploadStatute(payload: {
  act: string; section: string; title: string; text: string;
  predecessor?: string; successor?: string; overwrite?: boolean;
}): Promise<StatuteUploadResult> {
  return postJSON<StatuteUploadResult>("/api/admin/statute", {
    act:         payload.act,
    section:     payload.section,
    title:       payload.title,
    text:        payload.text,
    predecessor: payload.predecessor ?? "",
    successor:   payload.successor  ?? "",
    overwrite:   payload.overwrite  ?? false,
  });
}

export function uploadJudgment(payload: {
  case_id: string; text: string; overwrite?: boolean;
}): Promise<JudgmentUploadResult> {
  return postJSON<JudgmentUploadResult>("/api/admin/judgment", {
    case_id:  payload.case_id,
    text:     payload.text,
    overwrite: payload.overwrite ?? false,
  });
}
