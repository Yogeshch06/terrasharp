import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";
import NavLinks from "@/components/ui/NavLinks";

export const metadata: Metadata = {
  title: "TerraSharp — Satellite Super-Resolution",
  description: "Turn 10-meter pixels into 2.5-meter intelligence with AI-powered Sentinel-2 super-resolution.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="sticky top-0 z-50 w-full border-b border-border bg-background">
          <div className="relative flex h-16 w-full items-center justify-center px-6">
            <Link href="/" className="absolute left-6 flex items-center gap-3 text-lg font-semibold tracking-tight">
              <span className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground text-xs font-bold">TS</span>
              <span>TerraSharp</span>
            </Link>
            <NavLinks />
          </div>
        </header>
        <main className="min-h-[calc(100vh-4rem)]">
          {children}
        </main>
        <footer className="border-t border-border py-8 text-center text-xs text-muted-foreground">
          TerraSharp · SIH 2026 · Sentinel-2 super-resolution research
        </footer>
      </body>
    </html>
  );
}
