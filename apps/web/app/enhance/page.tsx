"use client";

import { useEffect, useState, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { getJobStatus } from "@/lib/api-client";
import type { EnhanceJob } from "@/types";
import SpectralDashboard from "@/components/metrics/SpectralDashboard";
import EdgeCompare from "@/components/metrics/EdgeCompare";
import NdviPreservation from "@/components/metrics/NdviPreservation";
import UncertaintyOverlay from "@/components/viewer/UncertaintyOverlay";
import SatelliteViewer from "@/components/viewer/SatelliteViewer";
import { buildDownloadUrl } from "@/lib/api-client";
import { POLL_INTERVAL_MS } from "@/lib/constants";

function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-xl bg-secondary ${className}`} />;
}

export default function EnhancePage() {
  const searchParams = useSearchParams();
  const jobId = searchParams.get("job");

  const [job, setJob] = useState<EnhanceJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pollStatus = useCallback(async () => {
    if (!jobId) return;
    try {
      const data = await getJobStatus(jobId);
      setJob(data);
      if (data.status !== "completed" && !data.status.startsWith("failed")) {
        setTimeout(pollStatus, POLL_INTERVAL_MS);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to get job status.");
    }
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    pollStatus();
  }, [jobId, pollStatus]);

  if (!jobId) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-20 text-center">
        <div className="text-5xl mb-4">🛰️</div>
        <h1 className="text-2xl font-bold mb-2">No job selected</h1>
        <p className="text-muted-foreground mb-6">Upload a GeoTIFF or fetch from Copernicus to start.</p>
        <a href="/" className="inline-block bg-primary text-primary-foreground px-5 py-2 rounded-md hover:bg-primary/90 transition-colors text-sm">
          Go to Upload
        </a>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-20 text-center">
        <div className="text-red-400 text-lg font-semibold mb-2">Error</div>
        <div className="text-muted-foreground text-sm">{error}</div>
      </div>
    );
  }

  const isPending = !job || (job.status !== "completed" && !job.status.startsWith("failed"));
  const isFailed = job?.status.startsWith("failed");

  if (isPending) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-10 space-y-8">
        <div className="space-y-2">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-4 w-48" />
        </div>
        <div className="flex items-center gap-3 text-sm text-muted-foreground">
          <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          Processing job <code className="text-xs font-mono">{jobId}</code>…
        </div>
        <Skeleton className="h-96 w-full" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-24" />)}
        </div>
      </div>
    );
  }

  if (isFailed) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-20 text-center">
        <div className="text-red-400 text-lg font-semibold mb-2">Job Failed</div>
        <div className="text-muted-foreground text-sm">{job?.status}</div>
      </div>
    );
  }

  const metrics = job!.metrics!;
  const uncertaintyUrl = buildDownloadUrl(jobId, "uncertainty.tif");

  return (
    <div className="max-w-5xl mx-auto px-4 py-10 space-y-10">
      <div>
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-2xl font-bold">Enhancement Result</h1>
            <div className="text-xs text-muted-foreground font-mono mt-1">{jobId}</div>
          </div>
          <div className="flex gap-2">
            <a
              href={buildDownloadUrl(jobId, "enhanced.tif")}
              download="enhanced.tif"
              id="download-enhanced"
              className="text-sm px-4 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              ↓ Enhanced GeoTIFF
            </a>
            <a
              href={uncertaintyUrl}
              download="uncertainty.tif"
              id="download-uncertainty"
              className="text-sm px-4 py-2 rounded-md border border-border hover:bg-secondary transition-colors"
            >
              ↓ Uncertainty Map
            </a>
          </div>
        </div>
      </div>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Before / After Viewer</h2>
        <SatelliteViewer jobId={jobId} />
        <UncertaintyOverlay uncertaintyUrl={uncertaintyUrl} />
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Spectral Metrics</h2>
        <SpectralDashboard metrics={metrics} />
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Index Preservation</h2>
        <NdviPreservation ndviPreservation={metrics.ndvi_preservation} ndwiPreservation={metrics.ndwi_preservation} />
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Edge Analysis</h2>
        <EdgeCompare edgeMetrics={metrics.edge_metrics} />
      </section>
    </div>
  );
}
