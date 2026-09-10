"use client";

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import type { SpectralMetrics } from "@/types";
import MetricsCard from "./MetricsCard";

const BAND_COLORS = ["rgb(var(--primary))", "rgb(var(--foreground))", "rgb(var(--muted-foreground))", "rgb(var(--border))"];
const BAND_ORDER = ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"];

function bandChartData(metrics: Record<string, number>) {
  return BAND_ORDER.map((b) => ({
    name: b.replace("_", " "),
    value: parseFloat((metrics[b] ?? 0).toFixed(3)),
  }));
}

interface Props {
  metrics: SpectralMetrics;
}

export default function SpectralDashboard({ metrics }: Props) {
  const psnrData = bandChartData(metrics.psnr);
  const ssimData = bandChartData(metrics.ssim);

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricsCard label="Mean PSNR" value={metrics.psnr.mean_psnr} unit="dB" highlight />
        <MetricsCard label="Mean SSIM" value={metrics.ssim.mean_ssim} subtitle="Higher is better" highlight />
        <MetricsCard label="SAM" value={metrics.spectral_angle_mapper_deg} unit="°" subtitle="Lower is better" />
        <MetricsCard label="Edge Similarity" value={metrics.edge_metrics.edge_cosine_similarity} subtitle="Cosine similarity" />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <MetricsCard label="NDVI Preservation" value={metrics.ndvi_preservation} subtitle="Correlation vs bicubic baseline" />
        <MetricsCard label="NDWI Preservation" value={metrics.ndwi_preservation} subtitle="Correlation vs bicubic baseline" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="border border-border bg-card p-5">
          <h3 className="text-sm font-semibold mb-4">PSNR per Band (dB)</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={psnrData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: "rgb(var(--muted-foreground))" }} />
              <YAxis tick={{ fontSize: 11, fill: "rgb(var(--muted-foreground))" }} />
              <Tooltip
                contentStyle={{ background: "rgb(var(--card))", border: "1px solid rgb(var(--border))", borderRadius: "8px" }}
                labelStyle={{ color: "rgb(var(--foreground))" }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {psnrData.map((_, i) => <Cell key={i} fill={BAND_COLORS[i % BAND_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="border border-border bg-card p-5">
          <h3 className="text-sm font-semibold mb-4">SSIM per Band</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={ssimData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: "rgb(var(--muted-foreground))" }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "rgb(var(--muted-foreground))" }} />
              <Tooltip
                contentStyle={{ background: "rgb(var(--card))", border: "1px solid rgb(var(--border))", borderRadius: "8px" }}
                labelStyle={{ color: "rgb(var(--foreground))" }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {ssimData.map((_, i) => <Cell key={i} fill={BAND_COLORS[i % BAND_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
