import { History } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export default function HistoryPage() {
  return <div className="mx-auto max-w-4xl space-y-6"><div><p className="text-sm font-medium text-primary">Saved research</p><h1 className="text-3xl font-semibold">History</h1><p className="mt-2 text-muted-foreground">Saved research sessions will be available after sign-in is added.</p></div><Card className="border-dashed"><CardContent className="py-16 text-center text-muted-foreground"><History className="mx-auto mb-3 size-8" />No saved research yet.</CardContent></Card></div>;
}
