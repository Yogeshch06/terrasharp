"use client";

import type { EdgeMetrics } from "@/types";
import { buildDownloadUrl } from "@/lib/api-client";

interface Props {
  jobId: string;
  edgeMetrics: EdgeMetrics;
  width?: number;
  height?: number;
}

export default function EdgeCompare({ jobId, edgeMetrics, width = 320, height = 240 }: Props) {

  return (
    <div className="space-y-4">
      <div className="text-sm text-muted-foreground">
        Sobel edge magnitude maps — brighter pixels = stronger edges.
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <img
            src={buildDownloadUrl(jobId, "edge_before.png")}
            alt="Bicubic baseline edge map"
            width={width}
            height={height}
            className="w-full border border-border bg-secondary"
          />
          <div className="text-xs text-center text-muted-foreground">Bicubic Baseline</div>
        </div>
        <div className="space-y-2">
          <img
            src={buildDownloadUrl(jobId, "edge_after.png")}
            alt="SR output edge map"
            width={width}
            height={height}
            className="w-full border border-border bg-secondary"
          />
          <div className="text-xs text-center text-muted-foreground">SR Output</div>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div className="border border-border bg-card px-3 py-2">
          <div className="text-xs text-muted-foreground">Cosine Similarity</div>
          <div className="font-mono font-semibold text-primary">{edgeMetrics.edge_cosine_similarity.toFixed(4)}</div>
        </div>
        <div className="border border-border bg-card px-3 py-2">
          <div className="text-xs text-muted-foreground">L1 Diff</div>
          <div className="font-mono font-semibold">{edgeMetrics.edge_l1_diff.toFixed(4)}</div>
        </div>
        <div className="border border-border bg-card px-3 py-2">
          <div className="text-xs text-muted-foreground">Enhancement Ratio</div>
          <div className="font-mono font-semibold">{edgeMetrics.edge_enhancement_ratio.toFixed(3)}×</div>
        </div>
      </div>
    </div>
  );
}
