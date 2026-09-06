"use client";

import dynamic from "next/dynamic";
import CopernicusFetcher from "@/components/upload/CopernicusFetcher";

const GeoTiffUploader = dynamic(() => import("@/components/upload/GeoTiffUploader"), { ssr: false });

export default function HomePage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-12 space-y-16">
      <section className="text-center space-y-4 pt-8">
        <div className="inline-flex items-center gap-2 bg-primary/10 border border-primary/20 text-primary text-xs px-3 py-1 rounded-full mb-4">
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
          SwinIR-Light · Sentinel-2 · 4× Super-Resolution
        </div>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
          Turn 10-meter pixels into{" "}
          <span className="text-primary">2.5-meter intelligence</span>
        </h1>
        <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
          Upload a 4-band Sentinel-2 GeoTIFF or fetch live data from Copernicus. TerraSharp enhances resolution 4× using a neural super-resolution model with per-pixel uncertainty maps.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Upload GeoTIFF</h2>
        <GeoTiffUploader />
      </section>

      <section className="space-y-4">
        <details className="group">
          <summary className="cursor-pointer list-none flex items-center gap-2 text-xl font-semibold select-none">
            <span className="text-muted-foreground group-open:rotate-90 transition-transform inline-block">▶</span>
            Fetch from Copernicus
          </summary>
          <div className="mt-4">
            <CopernicusFetcher />
          </div>
        </details>
      </section>

      <section className="space-y-6">
        <h2 className="text-xl font-semibold">How it works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { step: "01", title: "Upload or Fetch", desc: "Provide a 4-band Sentinel-2 GeoTIFF or let TerraSharp pull data from Copernicus Sentinel Hub." },
            { step: "02", title: "AI Enhancement", desc: "SwinIR-Light runs tiled inference with Monte Carlo uncertainty estimation on each 256×256 tile." },
            { step: "03", title: "Analyze & Download", desc: "Explore spectral metrics, edge preservation scores, NDVI fidelity, and download enhanced GeoTIFFs." },
          ].map(({ step, title, desc }) => (
            <div key={step} className="rounded-xl border border-border bg-card p-5 space-y-2">
              <div className="text-xs font-mono text-primary">{step}</div>
              <div className="font-semibold">{title}</div>
              <div className="text-sm text-muted-foreground">{desc}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
