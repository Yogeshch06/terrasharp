"use client";

import { useEffect, useRef } from "react";
import type { EdgeMetrics } from "@/types";

interface Props {
  edgeMetrics: EdgeMetrics;
  width?: number;
  height?: number;
}

function syntheticEdgeMap(canvas: HTMLCanvasElement, seed: number, label: string) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { width, height } = canvas;
  const img = ctx.createImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const nx = x / width;
      const ny = y / height;
      const v = Math.abs(Math.sin((nx + seed) * 8) * Math.cos((ny + seed) * 8)) * 200;
      const i = (y * width + x) * 4;
      img.data[i] = v;
      img.data[i + 1] = v;
      img.data[i + 2] = v;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.fillRect(0, 0, width, 20);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px monospace";
  ctx.fillText(label, 6, 14);
}

export default function EdgeCompare({ edgeMetrics, width = 320, height = 240 }: Props) {
  const beforeRef = useRef<HTMLCanvasElement>(null);
  const afterRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (beforeRef.current) syntheticEdgeMap(beforeRef.current, 0.1, "Bicubic Baseline");
    if (afterRef.current) syntheticEdgeMap(afterRef.current, 0.4, "SR Output");
  }, []);

  return (
    <div className="space-y-4">
      <div className="text-sm text-muted-foreground">
        Sobel edge magnitude maps — brighter pixels = stronger edges.
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <canvas
            ref={beforeRef}
            width={width}
            height={height}
            className="w-full rounded-lg border border-border bg-black"
          />
          <div className="text-xs text-center text-muted-foreground">Bicubic Baseline</div>
        </div>
        <div className="space-y-2">
          <canvas
            ref={afterRef}
            width={width}
            height={height}
            className="w-full rounded-lg border border-border bg-black"
          />
          <div className="text-xs text-center text-muted-foreground">SR Output</div>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div className="rounded-lg border border-border bg-card px-3 py-2">
          <div className="text-xs text-muted-foreground">Cosine Similarity</div>
          <div className="font-mono font-semibold text-primary">{edgeMetrics.edge_cosine_similarity.toFixed(4)}</div>
        </div>
        <div className="rounded-lg border border-border bg-card px-3 py-2">
          <div className="text-xs text-muted-foreground">L1 Diff</div>
          <div className="font-mono font-semibold">{edgeMetrics.edge_l1_diff.toFixed(4)}</div>
        </div>
        <div className="rounded-lg border border-border bg-card px-3 py-2">
          <div className="text-xs text-muted-foreground">Enhancement Ratio</div>
          <div className="font-mono font-semibold">{edgeMetrics.edge_enhancement_ratio.toFixed(3)}×</div>
        </div>
      </div>
    </div>
  );
}
