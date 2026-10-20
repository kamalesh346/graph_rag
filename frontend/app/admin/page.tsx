"use client";

import { useState, useEffect, useCallback } from "react";
import {
  checkStatuteExists, checkJudgmentExists,
  uploadStatute, uploadJudgment, getAdminStats,
  type KbStats, type StatuteUploadResult, type JudgmentUploadResult,
} from "@/lib/research-api";

// ── Tiny helper components ─────────────────────────────────────────────────

function Badge({ children, variant = "default" }: { children: React.ReactNode; variant?: "default" | "success" | "warning" | "error" }) {
  const cls = {
    default: "bg-muted text-muted-foreground",
    success: "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30",
    warning: "bg-amber-500/15 text-amber-400 border border-amber-500/30",
    error:   "bg-rose-500/15 text-rose-400 border border-rose-500/30",
  }[variant];
  return <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${cls}`}>{children}</span>;
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
    </svg>
  );
}

// ── Duplicate-replace confirmation dialog ──────────────────────────────────

function ReplaceDialog({
  label, onReplace, onCancel,
}: { label: string; onReplace: () => void; onCancel: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in">
      <div className="bg-card border border-border rounded-2xl shadow-2xl max-w-md w-full mx-4 p-6 space-y-4 animate-in zoom-in-95">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-amber-500/15 flex items-center justify-center flex-shrink-0">
            <svg className="h-5 w-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
            </svg>
          </div>
          <div>
            <h3 className="font-semibold text-foreground">Already indexed</h3>
            <p className="text-sm text-muted-foreground mt-0.5">
              <span className="font-mono text-amber-400">{label}</span> is already in the knowledge base.
            </p>
          </div>
        </div>
        <p className="text-sm text-muted-foreground">
          Do you want to <strong className="text-foreground">replace</strong> the existing entry? This will update the ChromaDB vector, JSONL file, and Neo4j graph node.
        </p>
        <div className="flex gap-3 pt-1">
          <button
            onClick={onReplace}
            className="flex-1 rounded-lg bg-amber-500 hover:bg-amber-400 text-black font-semibold py-2 text-sm transition-colors"
          >
            Replace
          </button>
          <button
            onClick={onCancel}
            className="flex-1 rounded-lg border border-border hover:bg-muted text-foreground font-semibold py-2 text-sm transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Result banner ─────────────────────────────────────────────────────────

function ResultBanner({ result, type }: { result: StatuteUploadResult | JudgmentUploadResult; type: "statute" | "judgment" }) {
  return (
    <div className="mt-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4 space-y-2 animate-in fade-in slide-in-from-bottom-2">
      <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"/>
        </svg>
        {type === "statute"
          ? `Section ${(result as StatuteUploadResult).section_id} indexed successfully`
          : `${(result as JudgmentUploadResult).chunks_created} chunk(s) indexed for case ${(result as JudgmentUploadResult).case_id}`}
      </div>
      <div className="flex flex-wrap gap-2 text-xs">
        {result.overwritten && <Badge variant="warning">Replaced existing</Badge>}
        {result.neo4j_updated
          ? <Badge variant="success">Neo4j ✓</Badge>
          : <Badge variant="error">Neo4j ✗</Badge>}
        {type === "statute" && <Badge variant="default">{(result as StatuteUploadResult).chunk_id}</Badge>}
      </div>
    </div>
  );
}

// ── KB Stats bar ──────────────────────────────────────────────────────────

function StatsBar({ stats, loading }: { stats: KbStats | null; loading: boolean }) {
  return (
    <div className="grid grid-cols-2 gap-4 mb-8">
      {[
        { label: "Statute Sections", value: stats?.statute_sections },
        { label: "Judgment Chunks",  value: stats?.judgment_chunks  },
      ].map(({ label, value }) => (
        <div key={label} className="rounded-xl border border-border bg-card p-5 flex items-center gap-4">
          <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
            <svg className="h-5 w-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-3-3v6M4 6h16M4 18h16"/>
            </svg>
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">
              {loading ? "—" : (value ?? 0)}
            </p>
            <p className="text-xs text-muted-foreground mt-0.5">{label}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Statute tab ───────────────────────────────────────────────────────────

const ACT_OPTIONS = ["BNS", "IPC", "BNSS", "CrPC", "BSA", "IEA", "POCSO", "CPC"];

function StatuteTab({ onSuccess }: { onSuccess: () => void }) {
  const [form, setForm] = useState({ act: "BNS", section: "", title: "", text: "", predecessor: "", successor: "" });
  const [status, setStatus] = useState<"idle" | "checking" | "submitting" | "done" | "error">("idle");
  const [error, setError]   = useState<string | null>(null);
  const [result, setResult] = useState<StatuteUploadResult | null>(null);
  const [dialog, setDialog] = useState(false);

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
    setForm(prev => ({ ...prev, [k]: e.target.value }));

  const canSubmit = form.section.trim() && form.title.trim() && form.text.trim();

  async function doSubmit(overwrite: boolean) {
    setDialog(false);
    setStatus("submitting");
    setError(null);
    try {
      const res = await uploadStatute({ ...form, overwrite });
      setResult(res);
      setStatus("done");
      onSuccess();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setStatus("error");
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setStatus("checking");
    setError(null);
    setResult(null);
    try {
      const check = await checkStatuteExists(form.act, form.section);
      if (check.exists) {
        setStatus("idle");
        setDialog(true);
      } else {
        await doSubmit(false);
      }
    } catch {
      await doSubmit(false);
    }
  }

  return (
    <>
      {dialog && (
        <ReplaceDialog
          label={`${form.act}:${form.section}`}
          onReplace={() => doSubmit(true)}
          onCancel={() => { setDialog(false); setStatus("idle"); }}
        />
      )}

      <form onSubmit={handleSubmit} className="space-y-5">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1.5">Act *</label>
            <select
              value={form.act}
              onChange={set("act")}
              className="w-full rounded-lg border border-border bg-muted/50 px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            >
              {ACT_OPTIONS.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1.5">Section Number *</label>
            <input
              value={form.section}
              onChange={set("section")}
              placeholder="e.g. 115"
              className="w-full rounded-lg border border-border bg-muted/50 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1.5">Section Title *</label>
          <input
            value={form.title}
            onChange={set("title")}
            placeholder="e.g. Voluntarily causing hurt"
            className="w-full rounded-lg border border-border bg-muted/50 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1.5">Predecessor (old law)</label>
            <input
              value={form.predecessor}
              onChange={set("predecessor")}
              placeholder="e.g. IPC:323"
              className="w-full rounded-lg border border-border bg-muted/50 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1.5">Successor (new law)</label>
            <input
              value={form.successor}
              onChange={set("successor")}
              placeholder="e.g. BNS:115"
              className="w-full rounded-lg border border-border bg-muted/50 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1.5">Full Section Text *</label>
          <textarea
            value={form.text}
            onChange={set("text")}
            rows={10}
            placeholder="Paste the complete section text here, including subsections, explanations, illustrations, and exceptions…"
            className="w-full rounded-lg border border-border bg-muted/50 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/50 font-mono resize-y"
          />
        </div>

        {status === "error" && error && (
          <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-400">
            {error}
          </div>
        )}

        {status === "done" && result && <ResultBanner result={result} type="statute" />}

        <button
          type="submit"
          disabled={!canSubmit || status === "checking" || status === "submitting"}
          className="w-full rounded-xl bg-primary hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed text-primary-foreground font-semibold py-3 text-sm transition-all flex items-center justify-center gap-2"
        >
          {(status === "checking" || status === "submitting") && <Spinner />}
          {status === "checking"   ? "Checking…"
           : status === "submitting" ? "Indexing…"
           : "Index Section →"}
        </button>
      </form>
    </>
  );
}

// ── Judgment tab ──────────────────────────────────────────────────────────

function JudgmentTab({ onSuccess }: { onSuccess: () => void }) {
  const [caseId, setCaseId] = useState("");
  const [text,   setText]   = useState("");
  const [status, setStatus] = useState<"idle" | "checking" | "submitting" | "done" | "error">("idle");
  const [error,  setError]  = useState<string | null>(null);
  const [result, setResult] = useState<JudgmentUploadResult | null>(null);
  const [dialog, setDialog] = useState(false);

  const canSubmit = caseId.trim() && text.trim();

  async function doSubmit(overwrite: boolean) {
    setDialog(false);
    setStatus("submitting");
    setError(null);
    try {
      const res = await uploadJudgment({ case_id: caseId.trim(), text: text.trim(), overwrite });
      setResult(res);
      setStatus("done");
      onSuccess();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setStatus("error");
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setStatus("checking");
    setError(null);
    setResult(null);
    try {
      const check = await checkJudgmentExists(caseId.trim());
      if (check.exists) {
        setStatus("idle");
        setDialog(true);
      } else {
        await doSubmit(false);
      }
    } catch {
      await doSubmit(false);
    }
  }

  return (
    <>
      {dialog && (
        <ReplaceDialog
          label={caseId}
          onReplace={() => doSubmit(true)}
          onCancel={() => { setDialog(false); setStatus("idle"); }}
        />
      )}

      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1.5">Case ID *</label>
          <input
            value={caseId}
            onChange={e => setCaseId(e.target.value)}
            placeholder="e.g. 2025 INSC 123"
            className="w-full rounded-lg border border-border bg-muted/50 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
          <p className="mt-1 text-xs text-muted-foreground">Must be unique. Used as the case node ID in Neo4j and chunk prefix in ChromaDB.</p>
        </div>

        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1.5">Full Judgment Text *</label>
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            rows={14}
            placeholder="Paste the full judgment text here. It will be automatically chunked (~600 tokens each), embedded using all-MiniLM-L6-v2, and legal references like IPC:302 will be extracted automatically…"
            className="w-full rounded-lg border border-border bg-muted/50 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/50 font-mono resize-y"
          />
          <p className="mt-1 text-xs text-muted-foreground">
            Legal references (IPC:302, BNS:115, etc.) are extracted automatically and used to build Neo4j REFERS_TO edges.
          </p>
        </div>

        {status === "error" && error && (
          <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-400">
            {error}
          </div>
        )}

        {status === "done" && result && <ResultBanner result={result} type="judgment" />}

        <button
          type="submit"
          disabled={!canSubmit || status === "checking" || status === "submitting"}
          className="w-full rounded-xl bg-primary hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed text-primary-foreground font-semibold py-3 text-sm transition-all flex items-center justify-center gap-2"
        >
          {(status === "checking" || status === "submitting") && <Spinner />}
          {status === "checking"    ? "Checking…"
           : status === "submitting" ? "Indexing…"
           : "Index Judgment →"}
        </button>
      </form>
    </>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

type Tab = "statute" | "judgment";

export default function AdminPage() {
  const [tab,         setTab]         = useState<Tab>("statute");
  const [stats,       setStats]       = useState<KbStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);

  const refreshStats = useCallback(async () => {
    setStatsLoading(true);
    try {
      setStats(await getAdminStats());
    } finally {
      setStatsLoading(false);
    }
  }, []);

  useEffect(() => { refreshStats(); }, [refreshStats]);

  return (
    <div className="mx-auto max-w-3xl px-4 py-10 space-y-8">
      {/* Header */}
      <div className="space-y-1">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center">
            <svg className="h-5 w-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z"/>
            </svg>
          </div>
          <div>
            <h1 className="text-2xl font-bold text-foreground tracking-tight">Admin Panel</h1>
            <p className="text-sm text-muted-foreground">Expand the knowledge base — statutes and judgments</p>
          </div>
        </div>
      </div>

      {/* Stats */}
      <StatsBar stats={stats} loading={statsLoading} />

      {/* Tab selector */}
      <div className="flex rounded-xl border border-border bg-muted/40 p-1 gap-1">
        {(["statute", "judgment"] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 rounded-lg py-2.5 text-sm font-semibold transition-all ${
              tab === t
                ? "bg-card text-foreground shadow-sm border border-border"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {t === "statute" ? "📋 Statute Section" : "⚖️ Judgment"}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
        <div className="mb-5">
          <h2 className="font-semibold text-foreground">
            {tab === "statute" ? "Add Statute Section" : "Add Judgment"}
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            {tab === "statute"
              ? "Indexes into ChromaDB (legal_statute_chunks), JSONL, and Neo4j. Must use the same act/section format as existing entries."
              : "Text is chunked (~600 tokens), embedded with all-MiniLM-L6-v2, and legal references are auto-extracted for Neo4j edges."}
          </p>
        </div>

        {tab === "statute"
          ? <StatuteTab  onSuccess={refreshStats} />
          : <JudgmentTab onSuccess={refreshStats} />}
      </div>

      {/* Info footer */}
      <div className="rounded-xl border border-border bg-muted/30 p-4 text-xs text-muted-foreground space-y-1">
        <p><strong className="text-foreground">Embedding model:</strong> all-MiniLM-L6-v2 (must match existing index)</p>
        <p><strong className="text-foreground">Duplicate handling:</strong> A confirmation dialog appears before replacing any existing entry</p>
        <p><strong className="text-foreground">Neo4j:</strong> Mandatory — Act, StatuteSection, and Chunk nodes are created with CORRESPONDS_TO and HAS_CHUNK edges</p>
        <p><strong className="text-foreground">Immediate availability:</strong> New sections are available to the retriever on the next request with no restart required</p>
      </div>
    </div>
  );
}
