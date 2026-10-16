import { LockKeyhole } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function LoginPage() {
  return <div className="mx-auto flex max-w-md justify-center pt-10"><Card className="w-full"><CardContent className="space-y-5 p-6 text-center"><span className="mx-auto grid size-10 place-items-center rounded-full bg-muted"><LockKeyhole className="size-5" /></span><div><h1 className="text-xl font-semibold">Sign in</h1><p className="mt-2 text-sm text-muted-foreground">Google OAuth will be enabled in the authentication phase.</p></div><Button className="w-full" variant="outline" disabled>Continue with Google</Button></CardContent></Card></div>;
}
