"use client";

import { FormEvent, useState } from "react";
import { AlertCircle, ChevronDown, FileText, LoaderCircle, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { research, type Provision, type ResearchResponse } from "@/lib/research-api";

function ProvisionCard({ provision }: { provision: Provision }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const hasRelatedProvision = provision.related_provisions.length > 0;
  const preview = provision.excerpt.slice(0, 280);

  return (
    <Card>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-medium">{provision.act} Section {provision.section}</span>
          <h3 className="font-semibold">{provision.title}</h3>
        </div>
        <p className="text-sm leading-6 text-muted-foreground">{isExpanded ? provision.excerpt : `${preview}${provision.excerpt.length > preview.length ? "…" : ""}`}</p>
        {provision.excerpt.length > preview.length && (
          <Button variant="ghost" size="sm" className="px-0" onClick={() => setIsExpanded((value) => !value)}>
            <ChevronDown className={`size-4 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
            {isExpanded ? "Hide section text" : "Read section text"}
          </Button>
        )}
        {hasRelatedProvision && (
          <div className="rounded-lg bg-muted/70 p-3 text-sm">
            <p><span className="font-medium">Related provision:</span> {provision.related_provisions.join(", ")}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function ResearchPage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanQuery = query.trim();
    if (!cleanQuery) {
      setError("Enter a legal question or FIR details to begin.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      setResult(await research(cleanQuery));
    } catch (caughtError) {
      setResult(null);
      setError(caughtError instanceof Error ? caughtError.message : "Unable to retrieve legal material.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <section className="space-y-2">
        <p className="text-sm font-medium text-primary">Research workspace</p>
        <h1 className="text-3xl font-semibold tracking-tight">Find relevant legal provisions</h1>
        <p className="max-w-2xl text-muted-foreground">Ask a legal question or paste FIR details. The results highlight potentially relevant provisions and supporting sources.</p>
      </section>

      <Card>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-3">
            <Textarea value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask a legal question or paste FIR details…" aria-label="Legal research query" />
            {error && <p role="alert" className="flex items-center gap-2 text-sm text-red-700"><AlertCircle className="size-4" />{error}</p>}
            <div className="flex justify-end">
              <Button type="submit" disabled={loading}>{loading ? <><LoaderCircle className="size-4 animate-spin" />Searching</> : <><Search className="size-4" />Find sections</>}</Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {!result && !loading && !error && (
        <Card className="border-dashed"><CardContent className="py-12 text-center text-muted-foreground"><FileText className="mx-auto mb-3 size-8" />Your relevant provisions and supporting judgments will appear here.</CardContent></Card>
      )}

      {result && <section className="space-y-6">
        <div><h2 className="text-xl font-semibold">Relevant provisions</h2><p className="text-sm text-muted-foreground">Results for: {result.query}</p></div>
        <div className="grid gap-4">
          {result.provisions.map((provision) => <ProvisionCard key={`${provision.act}-${provision.section}`} provision={provision} />)}
        </div>
        <div><h2 className="text-xl font-semibold">Supporting judgments</h2><p className="text-sm text-muted-foreground">Retrieved case passages related to the query.</p></div>
        <div className="grid gap-4 md:grid-cols-2">{result.judgments.map((judgment) => <Card key={judgment.source_id}><CardContent className="space-y-3"><h3 className="font-semibold">{judgment.case_id}</h3><p className="text-sm leading-6 text-muted-foreground">{judgment.excerpt}</p>{judgment.related_sections.length > 0 && <p className="text-xs text-muted-foreground">Related sections: {judgment.related_sections.join(", ")}</p>}</CardContent></Card>)}</div>
      </section>}
    </div>
  );
}
