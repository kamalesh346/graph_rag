"use client";

import { FormEvent, useState } from "react";
import {
  AlertCircle,
  ChevronDown,
  GitCompareArrows,
  LoaderCircle,
  Network,
  Search,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  researchCompare,
  type CompareResponse,
  type Provision,
  type Judgment,
} from "@/lib/research-api";

/* ------------------------------------------------------------------ */
/*  Shared sub-components                                             */
/* ------------------------------------------------------------------ */

function ProvisionCard({ provision }: { provision: Provision }) {
  const [expanded, setExpanded] = useState(false);
  const preview = provision.excerpt.slice(0, 200);

  return (
    <Card>
      <CardContent className="space-y-3 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-medium">
            {provision.act} Section {provision.section}
          </span>
          <h3 className="text-sm font-semibold">{provision.title}</h3>
        </div>
        <p className="text-xs leading-5 text-muted-foreground">
          {expanded
            ? provision.excerpt
            : `${preview}${provision.excerpt.length > preview.length ? "…" : ""}`}
        </p>
        {provision.excerpt.length > preview.length && (
          <Button
            variant="ghost"
            size="sm"
            className="h-auto px-0 py-0 text-xs"
            onClick={() => setExpanded((v) => !v)}
          >
            <ChevronDown
              className={`mr-1 size-3 transition-transform ${expanded ? "rotate-180" : ""}`}
            />
            {expanded ? "Less" : "More"}
          </Button>
        )}
        {provision.related_provisions.length > 0 && (
          <p className="text-xs text-muted-foreground">
            <span className="font-medium">Related:</span>{" "}
            {provision.related_provisions.join(", ")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function JudgmentCard({ judgment }: { judgment: Judgment }) {
  return (
    <Card>
      <CardContent className="space-y-2 p-4">
        <h3 className="text-sm font-semibold">{judgment.case_id}</h3>
        <p className="text-xs leading-5 text-muted-foreground">
          {judgment.excerpt.slice(0, 300)}
          {judgment.excerpt.length > 300 ? "…" : ""}
        </p>
        {judgment.related_sections.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Sections: {judgment.related_sections.join(", ")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

/** Labelled panel used for each side */
function ResultPanel({
  label,
  icon,
  provisions,
  judgments,
  accentClass,
}: {
  label: string;
  icon: React.ReactNode;
  provisions: Provision[];
  judgments: Judgment[];
  accentClass: string;
}) {
  return (
    <div className="space-y-4">
      <div className={`flex items-center gap-2 ${accentClass}`}>
        {icon}
        <h2 className="text-lg font-semibold">{label}</h2>
      </div>

      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Provisions ({provisions.length})
        </p>
        {provisions.length === 0 ? (
          <p className="text-sm text-muted-foreground">No provisions returned.</p>
        ) : (
          <div className="grid gap-3">
            {provisions.map((p) => (
              <ProvisionCard key={`${p.act}-${p.section}`} provision={p} />
            ))}
          </div>
        )}
      </div>

      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Judgments ({judgments.length})
        </p>
        {judgments.length === 0 ? (
          <p className="text-sm text-muted-foreground">No judgments returned.</p>
        ) : (
          <div className="grid gap-3">
            {judgments.map((j) => (
              <JudgmentCard key={j.source_id} judgment={j} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Graph additions highlight                                         */
/* ------------------------------------------------------------------ */

function GraphAdditions({ additions }: { additions: Provision[] }) {
  if (additions.length === 0) return null;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-emerald-600">
        <Sparkles className="size-5" />
        <h2 className="text-lg font-semibold">Graph-added connections</h2>
      </div>
      <p className="text-sm text-muted-foreground">
        These provisions were found by GraphRAG through graph traversal
        (CORRESPONDS_TO, REFERS_TO, DISCUSSES, CITES_CASE) but were{" "}
        <strong>not returned</strong> by plain vector similarity.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        {additions.map((p) => (
          <Card
            key={`${p.act}-${p.section}`}
            className="border-emerald-200 bg-emerald-50/50 dark:border-emerald-800 dark:bg-emerald-950/30"
          >
            <CardContent className="space-y-2 p-4">
              <span className="rounded-full bg-emerald-100 px-2.5 py-1 text-xs font-medium text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200">
                {p.act} Section {p.section}
              </span>
              <h3 className="text-sm font-semibold">{p.title}</h3>
              <p className="text-xs leading-5 text-muted-foreground">
                {p.excerpt.slice(0, 200)}
                {p.excerpt.length > 200 ? "…" : ""}
              </p>
              {p.related_provisions.length > 0 && (
                <p className="text-xs text-muted-foreground">
                  Related: {p.related_provisions.join(", ")}
                </p>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export default function ComparePage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanQuery = query.trim();
    if (!cleanQuery) {
      setError("Enter a legal question to compare retrieval approaches.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      setResult(await researchCompare(cleanQuery));
    } catch (err) {
      setResult(null);
      setError(
        err instanceof Error ? err.message : "Comparison failed."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      {/* Header */}
      <section className="space-y-2">
        <p className="text-sm font-medium text-primary">Retrieval comparison</p>
        <h1 className="text-3xl font-semibold tracking-tight">
          RAG vs GraphRAG
        </h1>
        <p className="max-w-3xl text-muted-foreground">
          Run the same legal query through both retrieval approaches.
          Plain RAG uses vector similarity only. GraphRAG adds graph traversal
          (cross-act correspondence, case citations, legal concepts) on top.
        </p>
      </section>

      {/* Query input */}
      <Card>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-3">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter a legal question to compare retrieval approaches…"
              aria-label="Comparison query"
            />
            {error && (
              <p role="alert" className="flex items-center gap-2 text-sm text-red-700">
                <AlertCircle className="size-4" />
                {error}
              </p>
            )}
            <div className="flex justify-end">
              <Button type="submit" disabled={loading}>
                {loading ? (
                  <>
                    <LoaderCircle className="size-4 animate-spin" />
                    Comparing…
                  </>
                ) : (
                  <>
                    <GitCompareArrows className="size-4" />
                    Compare approaches
                  </>
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Empty state */}
      {!result && !loading && !error && (
        <Card className="border-dashed">
          <CardContent className="py-12 text-center text-muted-foreground">
            <GitCompareArrows className="mx-auto mb-3 size-8" />
            Side-by-side comparison results will appear here.
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-8">
          {/* Graph additions highlight — the key differentiator */}
          <GraphAdditions additions={result.graph_additions} />

          {/* Side-by-side panels */}
          <div className="grid gap-8 lg:grid-cols-2">
            <ResultPanel
              label="Plain RAG (vector only)"
              icon={<Search className="size-5" />}
              provisions={result.rag.provisions}
              judgments={result.rag.judgments}
              accentClass="text-blue-600"
            />
            <ResultPanel
              label="GraphRAG (vector + graph)"
              icon={<Network className="size-5" />}
              provisions={result.graphrag.provisions}
              judgments={result.graphrag.judgments}
              accentClass="text-violet-600"
            />
          </div>
        </div>
      )}
    </div>
  );
}
