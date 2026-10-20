"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BookOpen, FileSearch, History, Scale, Search, ShieldCheck } from "lucide-react";

import { cn } from "@/lib/utils";

const navItems = [
  { href: "/research",  label: "Research",     icon: Search },
  { href: "/compare",   label: "Compare",      icon: Scale },
  { href: "/fir",       label: "FIR Analysis", icon: FileSearch },
  { href: "/history",   label: "History",      icon: History },
  { href: "/admin",     label: "Admin",        icon: ShieldCheck },
];


export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-muted/40">
      <header className="sticky top-0 z-20 border-b bg-background/95 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
          <Link href="/research" className="flex items-center gap-2 font-semibold text-foreground">
            <span className="grid size-8 place-items-center rounded-md bg-primary text-primary-foreground"><BookOpen className="size-4" /></span>
            Legal Research
          </Link>
          <Link href="/login" className="text-sm font-medium text-muted-foreground hover:text-foreground">Sign in</Link>
        </div>
      </header>
      <div className="mx-auto flex max-w-6xl flex-col md:flex-row">
        <nav className="flex gap-1 overflow-x-auto border-b bg-background p-3 md:min-h-[calc(100vh-4rem)] md:w-52 md:flex-col md:border-b-0 md:border-r md:p-4">
          {navItems.map(({ href, label, icon: Icon }) => (
            <Link key={href} href={href} className={cn("flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground", pathname === href && "bg-muted text-foreground")}>
              <Icon className="size-4" />{label}
            </Link>
          ))}
        </nav>
        <main className="min-w-0 flex-1 p-4 sm:p-6">{children}</main>
      </div>
    </div>
  );
}
