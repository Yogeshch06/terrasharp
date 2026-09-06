import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";

export const metadata: Metadata = {
  title: "TerraSharp — Satellite Super-Resolution",
  description: "Turn 10-meter pixels into 2.5-meter intelligence with AI-powered Sentinel-2 super-resolution.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>
        <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2 font-bold text-lg text-primary">
              <span className="text-2xl">🛰️</span>
              <span>TerraSharp</span>
            </Link>
            <nav className="flex items-center gap-1">
              <Link href="/" className="px-3 py-1.5 rounded-md text-sm text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors">
                Home
              </Link>
              <Link href="/enhance" className="px-3 py-1.5 rounded-md text-sm text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors">
                Enhance
              </Link>
              <Link href="/change-detection" className="px-3 py-1.5 rounded-md text-sm text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors">
                Change Detection
              </Link>
            </nav>
          </div>
        </header>
        <main className="min-h-[calc(100vh-3.5rem)]">
          {children}
        </main>
        <footer className="border-t border-border/40 py-6 text-center text-xs text-muted-foreground">
          TerraSharp — SIH 2024 · Sentinel-2 4× Super-Resolution
        </footer>
      </body>
    </html>
  );
}
