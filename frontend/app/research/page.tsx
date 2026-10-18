"use client";

import { FormEvent, useRef, useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  FileText,
  Gavel,
  Info,
  LoaderCircle,
  MessageSquare,
  RotateCcw,
  Send,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  ask,
  type AskResponse,
  type Provision,
  type RelatedProvision,
  type Judgment,
} from "@/lib/research-api";

/* ================================================================
   Sub-components
   ================================================================ */

function RelatedProvisionCard({ rel }: { rel: RelatedProvision }) {
  const [open, setOpen] = useState(false);
  const preview = rel.excerpt.slice(0, 150);
  return (
    <div className="rounded-md border border-dashed bg-muted/40 p-3 text-xs space-y-1">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="rounded-full bg-muted px-2 py-0.5 font-medium">
          {rel.act} §{rel.section}
        </span>
        <span className="font-medium">{rel.title}</span>
      </div>
      <p className="leading-5 text-muted-foreground">
        {open ? rel.excerpt : `${preview}${rel.excerpt.length > preview.length ? "…" : ""}`}
      </p>
      {rel.excerpt.length > preview.length && (
        <button
          className="text-primary underline-offset-2 hover:underline text-xs"
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}

function ProvisionCard({
  provision,
  cited,
}: {
  provision: Provision;
  cited: boolean;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const detail = provision.related_provisions_detail ?? [];
  const preview = provision.excerpt.slice(0, 280);

  return (
    <Card
      className={cited ? "border-emerald-400 ring-1 ring-emerald-400/50" : ""}
    >
      <CardContent className="space-y-3 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-medium">
            {provision.act} § {provision.section}
          </span>
          {cited && (
            <span className="flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200">
              <CheckCircle2 className="size-3" /> cited
            </span>
          )}
          <h3 className="text-sm font-semibold">{provision.title}</h3>
        </div>
        <p className="text-xs leading-5 text-muted-foreground">
          {isExpanded
            ? provision.excerpt
            : `${preview}${provision.excerpt.length > preview.length ? "…" : ""}`}
        </p>
        {provision.excerpt.length > preview.length && (
          <Button
            variant="ghost"
            size="sm"
            className="px-0 h-auto py-0 text-xs"
            onClick={() => setIsExpanded((v) => !v)}
          >
            <ChevronDown
              className={`size-3 mr-1 transition-transform ${isExpanded ? "rotate-180" : ""}`}
            />
            {isExpanded ? "Hide text" : "Read full text"}
          </Button>
        )}
        {detail.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">
              Related sections
            </p>
            {detail.map((rel) => (
              <RelatedProvisionCard key={rel.id} rel={rel} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function JudgmentCard({
  judgment,
  cited,
}: {
  judgment: Judgment;
  cited: boolean;
}) {
  return (
    <Card
      className={cited ? "border-emerald-400 ring-1 ring-emerald-400/50" : ""}
    >
      <CardContent className="space-y-2 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <Gavel className="size-3.5 text-muted-foreground" />
          <h3 className="text-sm font-semibold">{judgment.case_id}</h3>
          {cited && (
            <span className="flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200">
              <CheckCircle2 className="size-3" /> cited
            </span>
          )}
        </div>
        <p className="text-xs leading-5 text-muted-foreground">
          {judgment.excerpt.slice(0, 350)}
          {judgment.excerpt.length > 350 ? "…" : ""}
        </p>
        {judgment.related_sections.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Referenced sections: {judgment.related_sections.join(", ")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

/* ================================================================
   Answer card
   ================================================================ */

function AnswerCard({ result }: { result: AskResponse }) {
  if (result.is_out_of_scope) {
    return (
      <Card className="border-amber-400 bg-amber-50/60 dark:bg-amber-950/30">
        <CardContent className="space-y-3 p-5">
          <div className="flex items-center gap-2 text-amber-700 dark:text-amber-400">
            <AlertTriangle className="size-5" />
            <h2 className="font-semibold">Outside the knowledge base</h2>
          </div>
          <p className="text-sm leading-6 text-foreground">{result.answer}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-primary/30 bg-gradient-to-br from-primary/5 to-transparent">
      <CardContent className="space-y-4 p-5">
        <div className="flex items-center gap-2 text-primary">
          <Sparkles className="size-5" />
          <h2 className="font-semibold">Legal Answer</h2>
          <span className="ml-auto text-xs text-muted-foreground">
            via {result.provider_used}
          </span>
        </div>
        <p className="text-sm leading-7 text-foreground whitespace-pre-wrap">
          {result.answer}
        </p>
        {result.limitations.length > 0 && (
          <div className="rounded-lg bg-muted/60 p-3 space-y-1">
            <p className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <Info className="size-3.5" /> Limitations
            </p>
            <ul className="list-disc list-inside space-y-0.5">
              {result.limitations.map((l, i) => (
                <li key={i} className="text-xs text-muted-foreground">
                  {l}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ================================================================
   Session turn (one Q&A pair shown in history)
   ================================================================ */

function TurnSummary({
  turn,
  onExpand,
}: {
  turn: AskResponse;
  onExpand: () => void;
}) {
  return (
    <div className="flex items-start gap-3 rounded-lg border bg-muted/30 p-3">
      <MessageSquare className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-muted-foreground truncate">
          {turn.query}
        </p>
        <p className="text-sm line-clamp-2 mt-0.5">
          {turn.is_out_of_scope ? "⚠ Outside knowledge base" : turn.answer}
        </p>
      </div>
      <Button variant="ghost" size="sm" className="shrink-0" onClick={onExpand}>
        View
      </Button>
    </div>
  );
}

/* ================================================================
   Page
   ================================================================ */

export default function ResearchPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [current, setCurrent] = useState<AskResponse | null>(null);
  const [history, setHistory] = useState<AskResponse[]>([]);
  const [expandedHistoryIdx, setExpandedHistoryIdx] = useState<number | null>(null);

  const inputRef = useRef<HTMLTextAreaElement>(null);

  const citedSet = new Set(current?.sources ?? []);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const q = query.trim();
    if (!q) {
      setError("Enter a legal question to begin.");
      return;
    }
    setError(null);
    setLoading(true);

    // Push current result to history before overwriting
    if (current) {
      setHistory((prev) => [current, ...prev]);
    }
    setCurrent(null);

    try {
      const res = await ask(q);
      setCurrent(res);
      setQuery("");
      setExpandedHistoryIdx(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleReset() {
    setCurrent(null);
    setHistory([]);
    setQuery("");
    setError(null);
    setExpandedHistoryIdx(null);
    inputRef.current?.focus();
  }

  /* Which result are we displaying in the main panel? */
  const displayed =
    expandedHistoryIdx !== null ? history[expandedHistoryIdx] : current;
  const displayedCited = new Set(displayed?.sources ?? []);

  return (
    <div className="mx-auto max-w-4xl space-y-6 pb-32">
      {/* Header */}
      <section className="space-y-1">
        <p className="text-sm font-medium text-primary">Legal research session</p>
        <h1 className="text-3xl font-semibold tracking-tight">
          Ask a legal question
        </h1>
        <p className="max-w-2xl text-muted-foreground text-sm">
          Powered by GraphRAG retrieval + LLM synthesis. Every cited source is
          validated against the retrieved knowledge base — no invented references.
        </p>
      </section>

      {/* Session history (collapsed) */}
      {history.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Previous questions
          </p>
          {history.map((turn, idx) => (
            <TurnSummary
              key={idx}
              turn={turn}
              onExpand={() =>
                setExpandedHistoryIdx((prev) => (prev === idx ? null : idx))
              }
            />
          ))}
        </div>
      )}

      {/* Main result panel */}
      {displayed && (
        <section className="space-y-6" id="result">
          {/* Query label */}
          <div className="flex items-center gap-2">
            <MessageSquare className="size-4 text-muted-foreground" />
            <p className="text-sm text-muted-foreground italic">
              &ldquo;{displayed.query}&rdquo;
            </p>
            {expandedHistoryIdx !== null && (
              <Button
                variant="ghost"
                size="sm"
                className="ml-auto text-xs"
                onClick={() => setExpandedHistoryIdx(null)}
              >
                Back to latest
              </Button>
            )}
          </div>

          {/* Answer */}
          <AnswerCard result={displayed} />

          {/* Provisions */}
          {displayed.provisions.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <BookOpen className="size-4 text-muted-foreground" />
                <h2 className="text-base font-semibold">Statutory provisions</h2>
                <span className="text-xs text-muted-foreground">
                  ({displayed.provisions.length} retrieved
                  {displayedCited.size > 0
                    ? `, ${[...displayedCited].filter((id) =>
                        displayed.provisions.some(
                          (p) => `${p.act}:${p.section}` === id
                        )
                      ).length} cited`
                    : ""}
                  )
                </span>
              </div>
              <div className="grid gap-3">
                {displayed.provisions.map((p) => (
                  <ProvisionCard
                    key={`${p.act}-${p.section}`}
                    provision={p}
                    cited={displayedCited.has(`${p.act}:${p.section}`)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Judgments */}
          {displayed.judgments.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Gavel className="size-4 text-muted-foreground" />
                <h2 className="text-base font-semibold">Supporting judgments</h2>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                {displayed.judgments.map((j) => (
                  <JudgmentCard
                    key={j.source_id}
                    judgment={j}
                    cited={displayedCited.has(j.source_id)}
                  />
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {/* Empty state */}
      {!displayed && !loading && (
        <Card className="border-dashed">
          <CardContent className="py-14 text-center text-muted-foreground space-y-2">
            <FileText className="mx-auto size-9 opacity-40" />
            <p className="text-sm">
              Ask a legal question to get a grounded, cited answer.
            </p>
            <p className="text-xs opacity-60">
              Example: &ldquo;What are the penalties for culpable homicide under
              BNS?&rdquo;
            </p>
          </CardContent>
        </Card>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="space-y-3 animate-pulse">
          <div className="h-28 rounded-xl bg-muted/60" />
          <div className="h-20 rounded-xl bg-muted/40" />
          <div className="h-20 rounded-xl bg-muted/40" />
        </div>
      )}

      {/* Sticky query input */}
      <div className="fixed bottom-0 left-0 right-0 z-50 border-t bg-background/95 backdrop-blur px-4 py-3">
        <div className="mx-auto max-w-4xl">
          <form onSubmit={handleSubmit} className="space-y-2">
            {error && (
              <p
                role="alert"
                className="flex items-center gap-1.5 text-xs text-red-700"
              >
                <AlertCircle className="size-3.5" />
                {error}
              </p>
            )}
            <div className="flex gap-2 items-end">
              <Textarea
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    e.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder="Ask a legal question… (Enter to submit, Shift+Enter for new line)"
                aria-label="Legal research query"
                className="min-h-[60px] max-h-36 resize-none"
                rows={2}
              />
              <div className="flex flex-col gap-1.5">
                <Button type="submit" disabled={loading} size="sm">
                  {loading ? (
                    <LoaderCircle className="size-4 animate-spin" />
                  ) : (
                    <Send className="size-4" />
                  )}
                </Button>
                {(current || history.length > 0) && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleReset}
                    title="Start new session"
                  >
                    <RotateCcw className="size-4" />
                  </Button>
                )}
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
