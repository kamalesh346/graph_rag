"use client";

import { FormEvent, useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  BookOpen,
  ChevronDown,
  ChevronRight,
  FileSearch,
  Gavel,
  HelpCircle,
  Info,
  LoaderCircle,
  MapPin,
  Scale,
  Shield,
  Sparkles,
  User,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  analyseFir,
  type FirCandidateProvision,
  type FirEntities,
  type FirResponse,
  type Provision,
  type Judgment,
} from "@/lib/research-api";

/* ================================================================
   Entities summary strip
   ================================================================ */

function EntitiesStrip({ entities }: { entities: FirEntities }) {
  const chips: { icon: React.ReactNode; label: string; value: string }[] = [
    { icon: <User className="size-3" />, label: "People", value: (entities.people ?? []).join("; ") },
    { icon: <MapPin className="size-3" />, label: "Location", value: entities.location ?? "" },
    { icon: <AlertTriangle className="size-3" />, label: "Injury", value: entities.injury ?? "" },
    { icon: <Shield className="size-3" />, label: "Weapon", value: entities.weapon ?? "" },
    { icon: <Info className="size-3" />, label: "Time", value: entities.time ?? "" },
  ].filter((c) => c.value);

  const events = entities.events ?? [];

  if (chips.length === 0 && events.length === 0) return null;

  return (
    <Card className="border-muted bg-muted/20">
      <CardContent className="p-4 space-y-3">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Entities extracted from FIR (alleged)
        </p>
        <div className="flex flex-wrap gap-2">
          {chips.map((c) => (
            <span
              key={c.label}
              className="inline-flex items-center gap-1.5 rounded-full border bg-background px-2.5 py-1 text-xs"
            >
              {c.icon}
              <span className="font-medium">{c.label}:</span> {c.value}
            </span>
          ))}
        </div>
        {events.length > 0 && (
          <div className="space-y-1">
            <p className="text-xs font-medium text-muted-foreground">Alleged events</p>
            <ul className="list-disc list-inside space-y-0.5">
              {events.map((e, i) => (
                <li key={i} className="text-xs text-muted-foreground">{e}</li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ================================================================
   Supporting source panels inside a candidate card
   ================================================================ */

function StatutePanel({ provision }: { provision: Provision | undefined }) {
  const [open, setOpen] = useState(false);
  if (!provision) return null;
  const preview = provision.excerpt.slice(0, 200);
  return (
    <div className="rounded-md border bg-muted/30 p-3 space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-xs font-medium">
          <BookOpen className="size-3.5 text-muted-foreground" />
          Statutory provision
        </div>
        <button
          className="text-xs text-primary underline-offset-2 hover:underline"
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "Hide text" : "Show text"}
        </button>
      </div>
      <p className="text-xs font-semibold">{provision.act} § {provision.section} — {provision.title}</p>
      {open && (
        <p className="text-xs leading-5 text-muted-foreground">
          {provision.excerpt}
        </p>
      )}
      {!open && (
        <p className="text-xs leading-5 text-muted-foreground">
          {preview}{provision.excerpt.length > preview.length ? "…" : ""}
        </p>
      )}
    </div>
  );
}

function JudgmentPanel({ judgment }: { judgment: Judgment | undefined }) {
  if (!judgment) return null;
  return (
    <div className="rounded-md border bg-muted/30 p-3 space-y-1.5">
      <div className="flex items-center gap-1.5 text-xs font-medium">
        <Gavel className="size-3.5 text-muted-foreground" />
        Supporting judgment
      </div>
      <p className="text-xs font-semibold">{judgment.case_id}</p>
      <p className="text-xs leading-5 text-muted-foreground">
        {judgment.excerpt.slice(0, 300)}
        {judgment.excerpt.length > 300 ? "…" : ""}
      </p>
    </div>
  );
}

/* ================================================================
   Candidate provision card
   ================================================================ */

function CandidateCard({
  candidate,
  provisions,
  judgments,
}: {
  candidate: FirCandidateProvision;
  provisions: Provision[];
  judgments: Judgment[];
}) {
  const [open, setOpen] = useState(true);
  const provision = provisions.find(
    (p) => `${p.act}:${p.section}` === candidate.source_id
  );
  const judgment = candidate.judgment_id
    ? judgments.find((j) => j.source_id === candidate.judgment_id)
    : undefined;

  return (
    <Card className="border-primary/20 shadow-sm">
      <CardContent className="p-0">
        {/* Header — always visible */}
        <button
          className="w-full flex items-center justify-between p-4 text-left"
          onClick={() => setOpen((v) => !v)}
        >
          <div className="flex items-center gap-2 flex-wrap">
            <Scale className="size-4 text-primary shrink-0" />
            <span className="font-semibold text-sm">
              {candidate.act} § {candidate.section}
            </span>
            <span className="text-sm text-muted-foreground">— {candidate.title}</span>
            <span className="rounded-full border border-amber-400 bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700 dark:bg-amber-950 dark:text-amber-300">
              may be relevant
            </span>
          </div>
          <ChevronDown
            className={`size-4 shrink-0 text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`}
          />
        </button>

        {open && (
          <div className="border-t px-4 pb-4 space-y-4 pt-3">
            {/* Why relevant */}
            <div className="space-y-1">
              <p className="text-xs font-semibold uppercase tracking-wide text-primary/80">
                Why it may be relevant
              </p>
              <p className="text-sm leading-6 text-foreground">
                {candidate.why_relevant}
              </p>
            </div>

            {/* Facts from FIR */}
            {candidate.facts_from_fir.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700 dark:text-emerald-400">
                  Facts identified in the FIR
                </p>
                <ul className="space-y-1">
                  {candidate.facts_from_fir.map((f, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                      <ChevronRight className="size-3 mt-0.5 shrink-0 text-emerald-600" />
                      {f}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* What remains unclear */}
            {candidate.unclear_points.length > 0 && (
              <div className="space-y-1.5">
                <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-400">
                  <HelpCircle className="size-3.5" />
                  What remains unclear
                </p>
                <ul className="space-y-1">
                  {candidate.unclear_points.map((u, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                      <ChevronRight className="size-3 mt-0.5 shrink-0 text-amber-500" />
                      {u}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Supporting sources */}
            {(provision || judgment) && (
              <div className="space-y-2">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Supporting sources
                </p>
                <StatutePanel provision={provision} />
                {judgment && <JudgmentPanel judgment={judgment} />}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ================================================================
   Page
   ================================================================ */

const DISCLAIMER =
  "This analysis identifies provisions that may be relevant based on the supplied FIR text. " +
  "It does not constitute legal advice, an assessment of guilt, or a determination of any offence. " +
  "Consult a qualified legal professional for advice.";

export default function FirAnalysisPage() {
  const [firText, setFirText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FirResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const text = firText.trim();
    if (text.length < 10) {
      setError("Please enter at least 10 characters of FIR text.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      setResult(await analyseFir(text));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6 pb-16">
      {/* Header */}
      <section className="space-y-1">
        <p className="text-sm font-medium text-primary">FIR Analysis</p>
        <h1 className="text-3xl font-semibold tracking-tight">
          Legal provision matching
        </h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Paste an FIR description to identify potentially applicable legal provisions.
          Results are based on alleged facts only and use cautionary language throughout.
        </p>
      </section>

      {/* Disclaimer */}
      <div className="rounded-lg border border-amber-300 bg-amber-50/60 p-3 dark:bg-amber-950/30">
        <p className="flex items-start gap-2 text-xs text-amber-800 dark:text-amber-300">
          <AlertTriangle className="size-4 mt-0.5 shrink-0" />
          {DISCLAIMER}
        </p>
      </div>

      {/* Input form */}
      <Card>
        <CardContent className="pt-4">
          <form onSubmit={handleSubmit} className="space-y-3">
            <Textarea
              value={firText}
              onChange={(e) => setFirText(e.target.value)}
              placeholder={
                "Paste the FIR text here…\n\nExample: On 12 October 2024 at approximately 9 PM, the complainant states that the accused entered the premises and caused grievous injury to the victim using a sharp object…"
              }
              aria-label="FIR text for analysis"
              className="min-h-[160px] resize-y font-mono text-sm"
              rows={8}
            />
            {error && (
              <p role="alert" className="flex items-center gap-1.5 text-sm text-red-700">
                <AlertCircle className="size-4 shrink-0" />
                {error}
              </p>
            )}
            <div className="flex items-center justify-between">
              <p className="text-xs text-muted-foreground">
                {firText.length} characters
              </p>
              <Button type="submit" disabled={loading}>
                {loading ? (
                  <>
                    <LoaderCircle className="size-4 animate-spin" />
                    Analysing…
                  </>
                ) : (
                  <>
                    <FileSearch className="size-4" />
                    Analyse FIR
                  </>
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Loading skeleton */}
      {loading && (
        <div className="space-y-4 animate-pulse">
          <div className="h-20 rounded-xl bg-muted/60" />
          <div className="h-40 rounded-xl bg-muted/40" />
          <div className="h-40 rounded-xl bg-muted/40" />
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <section className="space-y-5" id="fir-result">
          {/* Entities */}
          <EntitiesStrip entities={result.entities} />

          {/* Out of scope */}
          {result.is_out_of_scope && (
            <Card className="border-amber-400 bg-amber-50/60 dark:bg-amber-950/30">
              <CardContent className="p-5 space-y-2">
                <div className="flex items-center gap-2 text-amber-700 dark:text-amber-400">
                  <AlertTriangle className="size-5" />
                  <h2 className="font-semibold">No matching provisions found</h2>
                </div>
                <p className="text-sm text-muted-foreground">
                  {result.retrieval_count === 0
                    ? "The retrieval system did not return any legal provisions for this FIR text. This may indicate a knowledge base gap or connectivity issue."
                    : `${result.retrieval_count} provision(s) were retrieved from the knowledge base, but none appear to match the facts described in the FIR. The knowledge base may not contain the specific sections relevant to this type of offence.`}
                </p>
                {result.provisions.length > 0 && (
                  <details className="mt-2">
                    <summary className="text-xs font-medium text-muted-foreground cursor-pointer hover:text-foreground">
                      Retrieved provisions (not matched)
                    </summary>
                    <ul className="mt-1 space-y-1 text-xs text-muted-foreground">
                      {result.provisions.map((p) => (
                        <li key={`${p.act}:${p.section}`}>
                          {p.act} § {p.section} — {p.title}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </CardContent>
            </Card>
          )}

          {/* Candidate provisions */}
          {result.candidate_provisions.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Sparkles className="size-4 text-primary" />
                <h2 className="text-base font-semibold">
                  Possible relevant provisions
                </h2>
                <span className="text-xs text-muted-foreground">
                  ({result.candidate_provisions.length} identified ·{" "}
                  via {result.provider_used})
                </span>
              </div>
              <div className="space-y-3">
                {result.candidate_provisions.map((c) => (
                  <CandidateCard
                    key={c.source_id}
                    candidate={c}
                    provisions={result.provisions}
                    judgments={result.judgments}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Caveat footer */}
          <p className="text-xs text-muted-foreground border-t pt-4">
            <span className="font-medium">Note:</span> This is an illustrative analysis
            based on the submitted FIR text and the provisions available in the knowledge
            base. It is not an assessment of any actual case or an opinion on guilt.
          </p>
        </section>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <Card className="border-dashed">
          <CardContent className="py-14 text-center text-muted-foreground space-y-2">
            <FileSearch className="mx-auto size-9 opacity-40" />
            <p className="text-sm">Paste an FIR and click Analyse FIR to begin.</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
