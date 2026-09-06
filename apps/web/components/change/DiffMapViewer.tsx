"use client";

import { useMemo } from "react";
import type { SpectralMetrics } from "@/types";
import MetricsCard from "@/components/metrics/MetricsCard";

interface Props {
  metrics1: SpectralMetrics;
  metrics2: SpectralMetrics;
  jobId1: string;
  jobId2: string;
}

function computeNdviChange(m1: SpectralMetrics, m2: SpectralMetrics): number {
  const ndvi1 = m1.ndvi_preservation ?? 0;
  const ndvi2 = m2.ndvi_preservation ?? 0;
  return Math.abs(ndvi2 - ndvi1);
}

export default function DiffMapViewer({ metrics1, metrics2, jobId1, jobId2 }: Props) {
  const ndviDelta = useMemo(() => computeNdviChange(metrics1, metrics2), [metrics1, metrics2]);
  const psnrDelta = useMemo(
    () => (metrics2.psnr.mean_psnr - metrics1.psnr.mean_psnr).toFixed(2),
    [metrics1, metrics2]
  );
  const edgeDelta = useMemo(
    () => (metrics2.edge_metrics.edge_cosine_similarity - metrics1.edge_metrics.edge_cosine_similarity).toFixed(4),
    [metrics1, metrics2]
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <MetricsCard label="ΔNDVI Preservation" value={ndviDelta} subtitle="Abs difference T1→T2" />
        <MetricsCard label="ΔPSNR" value={psnrDelta} unit="dB" subtitle="T2 – T1" />
        <MetricsCard label="ΔEdge Similarity" value={edgeDelta} subtitle="T2 – T1 cosine sim" />
      </div>

      <div className="grid grid-cols-3 gap-4">
        {["T1 (Before)", "T2 (After)", "Diff"].map((label, i) => (
          <div key={label} className="space-y-2">
            <div className="text-sm font-medium">{label}</div>
            <div
              className="rounded-xl border border-border bg-black flex items-center justify-center text-muted-foreground/40 text-xs"
              style={{ height: "220px" }}
            >
              {i === 2 ? (
                <div className="text-center space-y-2">
                  <div className="text-3xl">📊</div>
                  <div>Client-side NDVI diff</div>
                  <div className="text-xs text-muted-foreground">
                    ΔNDVI = <span className={ndviDelta > 0.05 ? "text-red-400" : "text-green-400"}>{ndviDelta.toFixed(4)}</span>
                  </div>
                </div>
              ) : (
                <div className="text-center space-y-1">
                  <div className="text-3xl">🛰️</div>
                  <div>{label}</div>
                  <div className="font-mono text-xs">{(i === 0 ? jobId1 : jobId2).slice(0, 8)}…</div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-xl border border-border bg-card p-4 text-xs text-muted-foreground">
        <strong>Note:</strong> Diff computed client-side from NDVI preservation metrics of each job.
        A dedicated <code>/change-detect</code> backend endpoint would provide pixel-level change maps.
        Red = increased NDVI, Green = decreased NDVI, Blue = stable.
      </div>
    </div>
  );
}
