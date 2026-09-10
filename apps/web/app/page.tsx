import Link from "next/link";
import { ArrowRight, Check, Database, ScanSearch } from "lucide-react";

export default function HomePage() {
  return (
    <div className="mx-auto max-w-6xl space-y-24 px-6 pb-16 pt-0 md:pb-24">
      <section className="relative left-1/2 isolate aspect-video w-screen -translate-x-1/2 overflow-hidden">
        <img src="/eleven.png" alt="Satellite view of the earth" className="absolute inset-0 -z-20 h-full w-full object-contain" />
        <div className="absolute inset-0 -z-10 bg-black/50" />
        <div className="absolute inset-x-6 top-1/2 -translate-y-1/2 space-y-7 text-white md:inset-x-16">
          <h1 className="max-w-5xl text-5xl font-semibold leading-[1.02] tracking-tight md:text-7xl">Sharper earth observation, without the premium imagery bill.</h1>
          <p className="max-w-2xl text-lg leading-8 text-white/80 md:text-xl">TerraSharp turns four-band Sentinel-2 imagery into a clearer 2.5-meter view for faster environmental analysis and repeatable insight.</p>
          <Link href="/enhance" className="inline-flex items-center gap-2 bg-primary px-5 py-3 text-sm font-semibold text-primary-foreground hover:bg-primary/90">Try it now <ArrowRight size={16} /></Link>
        </div>
      </section>

      <section className="grid md:grid-cols-[0.8fr_1.2fr] gap-12 border-t border-border pt-12">
        <div><p className="text-sm font-semibold text-primary mb-3">THE RESOLUTION GAP</p><h2 className="text-3xl font-semibold tracking-tight">Free and frequent data is not always detailed enough.</h2></div>
        <p className="text-lg leading-8 text-muted-foreground">Sentinel-2 is free, global, and refreshed often, but its 10-meter pixels can hide the boundaries and textures that matter. Commercial high-resolution imagery is sharper, but expensive and less frequent. TerraSharp makes the existing archive more useful.</p>
      </section>

      <section className="space-y-8"><div><p className="text-sm font-semibold text-primary mb-3">HOW IT WORKS</p><h2 className="text-3xl font-semibold tracking-tight">A short path from scene to signal.</h2></div>
        <div className="grid md:grid-cols-3 gap-6">{[
          { n: "01", title: "Bring a scene", desc: "Upload a four-band GeoTIFF or fetch a scene from Copernicus.", icon: Database },
          { n: "02", title: "Enhance resolution", desc: "A tiled neural model reconstructs detail while estimating uncertainty.", icon: ScanSearch },
          { n: "03", title: "Read the result", desc: "Compare imagery, inspect spectral metrics, and download the outputs.", icon: Check },
        ].map(({ n, title, desc, icon: Icon }) => <div key={n} className="border border-border bg-card p-6 space-y-5"><div className="flex items-center justify-between"><span className="text-sm font-semibold text-primary">{n}</span><Icon size={18} className="text-muted-foreground" /></div><h3 className="text-lg font-semibold">{title}</h3><p className="text-sm leading-6 text-muted-foreground">{desc}</p></div>)}</div>
      </section>

      <section className="space-y-8"><div><p className="text-sm font-semibold text-primary mb-3">A SAMPLE OUTPUT</p><h2 className="text-3xl font-semibold tracking-tight">From baseline pixels to a more legible scene.</h2></div><div className="grid md:grid-cols-2 gap-6"><figure className="space-y-3"><img src="/samples/preview_before.png" alt="Sentinel-2 baseline preview" className="w-full aspect-square object-cover border border-border bg-card" /><figcaption className="text-sm text-muted-foreground">Sentinel-2 baseline</figcaption></figure><figure className="space-y-3"><img src="/samples/preview_after.png" alt="TerraSharp enhanced preview" className="w-full aspect-square object-cover border border-border bg-card" /><figcaption className="text-sm text-muted-foreground">TerraSharp enhanced preview</figcaption></figure></div></section>

      <section className="border-t border-border pt-8 text-sm text-muted-foreground">TerraSharp is an SIH project exploring practical, uncertainty-aware super-resolution for open satellite imagery.</section>
    </div>
  );
}
