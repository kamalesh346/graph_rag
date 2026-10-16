import { Scale } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export default function ComparePage() {
  return <div className="mx-auto max-w-4xl space-y-6"><div><p className="text-sm font-medium text-primary">Retrieval comparison</p><h1 className="text-3xl font-semibold">RAG vs GraphRAG</h1><p className="mt-2 text-muted-foreground">Run the same legal query through both retrieval approaches in a later phase.</p></div><Card className="border-dashed"><CardContent className="py-16 text-center text-muted-foreground"><Scale className="mx-auto mb-3 size-8" />Comparison results will appear here.</CardContent></Card></div>;
}
