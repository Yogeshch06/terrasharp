"use client";

import { useState, useRef, useCallback } from "react";
import BandSelector from "./BandSelector";
import type { BandMode } from "@/types";
import { buildDownloadUrl } from "@/lib/api-client";

interface Props {
  jobId: string;
}

export default function SatelliteViewer({ jobId }: Props) {
  const [bandMode, setBandMode] = useState<BandMode>("RGB");
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const enhancedUrl = buildDownloadUrl(jobId, "enhanced.tif");

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pos = ((e.clientX - rect.left) / rect.width) * 100;
    setSliderPos(Math.max(0, Math.min(100, pos)));
  }, []);

  const onTouchMove = useCallback((e: React.TouchEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pos = ((e.touches[0].clientX - rect.left) / rect.width) * 100;
    setSliderPos(Math.max(0, Math.min(100, pos)));
  }, []);

  const bandLabel = bandMode === "RGB" ? "True Color" : bandMode === "NIR" ? "Near-Infrared" : bandMode;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <BandSelector activeMode={bandMode} onChange={setBandMode} />
        <a
          href={enhancedUrl}
          download="enhanced.tif"
          className="text-xs px-3 py-1.5 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          ↓ Download Enhanced GeoTIFF
        </a>
      </div>

      <div
        ref={containerRef}
        className="relative rounded-xl border border-border overflow-hidden cursor-col-resize bg-black select-none"
        style={{ height: "420px" }}
        onMouseMove={onMouseMove}
        onMouseDown={() => { dragging.current = true; }}
        onMouseUp={() => { dragging.current = false; }}
        onMouseLeave={() => { dragging.current = false; }}
        onTouchMove={onTouchMove}
      >
        <div className="absolute inset-0 flex items-center justify-center text-muted-foreground/40 text-sm">
          <div className="text-center space-y-2">
            <div className="text-5xl">🗺️</div>
            <div>Before / After viewer</div>
            <div className="text-xs">({bandLabel} · drag divider to compare)</div>
          </div>
        </div>

        <div
          className="absolute top-0 bottom-0 bg-white/20 w-px"
          style={{ left: `${sliderPos}%` }}
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full border-2 border-white bg-background flex items-center justify-center text-xs text-muted-foreground shadow-lg cursor-col-resize z-10"
          style={{ left: `${sliderPos}%` }}
          onMouseDown={(e) => { e.stopPropagation(); dragging.current = true; }}
        >
          ⇄
        </div>

        <div className="absolute top-2 left-3 text-xs bg-black/60 text-white px-2 py-0.5 rounded">
          Bicubic
        </div>
        <div className="absolute top-2 right-3 text-xs bg-black/60 text-primary px-2 py-0.5 rounded">
          SR Enhanced
        </div>
      </div>

      <div className="text-xs text-muted-foreground text-center">
        Rendering: <strong>{bandLabel}</strong> · Drag the divider to compare bicubic baseline vs SR output
      </div>
    </div>
  );
}
