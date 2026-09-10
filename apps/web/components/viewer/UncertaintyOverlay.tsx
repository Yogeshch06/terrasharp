"use client";

import { useState } from "react";
import { buildDownloadUrl } from "@/lib/api-client";

interface Props {
  jobId: string;
  uncertaintyUrl: string;
}

export default function UncertaintyOverlay({ jobId, uncertaintyUrl }: Props) {
  const [enabled, setEnabled] = useState(false);
  const [opacity, setOpacity] = useState(0.6);

  return (
    <div className="border border-border bg-card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-medium">Uncertainty Overlay</div>
          <div className="text-xs text-muted-foreground mt-0.5">
            Red regions are model-inferred, not directly observed.
          </div>
        </div>
        <button
          id="uncertainty-toggle"
          onClick={() => setEnabled((v) => !v)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors
            ${enabled ? "bg-primary" : "bg-secondary"}`}
          role="switch"
          aria-checked={enabled}
        >
          <span
            className={`inline-block h-4 w-4 transform bg-white transition-transform
              ${enabled ? "translate-x-6" : "translate-x-1"}`}
          />
        </button>
      </div>

      {enabled && (
        <div className="space-y-2">
          <label htmlFor="uncertainty-opacity" className="text-xs text-muted-foreground">
            Opacity: <span className="text-foreground font-medium">{Math.round(opacity * 100)}%</span>
          </label>
          <input
            id="uncertainty-opacity"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-full accent-primary"
          />
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <div className="flex gap-1 items-center">
              <div className="w-3 h-3 rounded-sm bg-blue-500" />
              Low
            </div>
            <div className="flex gap-1 items-center">
              <div className="w-3 h-3 rounded-sm bg-yellow-400" />
              Medium
            </div>
            <div className="flex gap-1 items-center">
              <div className="w-3 h-3 rounded-sm bg-red-500" />
              High
            </div>
          </div>
          <div className="text-xs text-muted-foreground italic">
            Heatmap rendered from <code>uncertainty.tif</code> using the model uncertainty range.
          </div>
          <img
            src={buildDownloadUrl(jobId, "uncertainty_heatmap.png")}
            alt="Uncertainty heatmap"
            className="w-full rounded-lg border border-border"
            style={{ opacity }}
          />
          <a
            href={uncertaintyUrl}
            download="uncertainty.tif"
            className="text-xs text-primary underline hover:no-underline"
          >
            Download uncertainty.tif
          </a>
        </div>
      )}
    </div>
  );
}
