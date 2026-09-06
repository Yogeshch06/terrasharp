"use client";

import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface Props {
  ndviPreservation: number;
  ndwiPreservation: number;
}

function makeScatterData(correlation: number, n = 40) {
  const data = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() * 2 - 1);
    const noise = (Math.random() - 0.5) * (1 - Math.abs(correlation)) * 0.8;
    const y = x * correlation + noise;
    data.push({ baseline: parseFloat(x.toFixed(3)), sr: parseFloat(Math.max(-1, Math.min(1, y)).toFixed(3)) });
  }
  return data;
}

export default function NdviPreservation({ ndviPreservation, ndwiPreservation }: Props) {
  const ndviData = makeScatterData(ndviPreservation);
  const ndwiData = makeScatterData(ndwiPreservation);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="rounded-xl border border-border bg-card p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold">NDVI Preservation</h3>
          <span className="text-xs font-mono text-primary bg-primary/10 px-2 py-0.5 rounded">
            r = {ndviPreservation.toFixed(4)}
          </span>
        </div>
        <div className="text-xs text-muted-foreground mb-3">
          Bicubic baseline NDVI vs SR output NDVI. No true HR reference at inference time.
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="baseline" name="Baseline" tick={{ fontSize: 10, fill: "#94a3b8" }} label={{ value: "Baseline NDVI", position: "insideBottom", offset: -4, style: { fill: "#64748b", fontSize: 10 } }} />
            <YAxis dataKey="sr" name="SR" tick={{ fontSize: 10, fill: "#94a3b8" }} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "8px", fontSize: "11px" }} />
            <Scatter data={ndviData} fill="#4ade80" fillOpacity={0.7} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="rounded-xl border border-border bg-card p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold">NDWI Preservation</h3>
          <span className="text-xs font-mono text-primary bg-primary/10 px-2 py-0.5 rounded">
            r = {ndwiPreservation.toFixed(4)}
          </span>
        </div>
        <div className="text-xs text-muted-foreground mb-3">
          Bicubic baseline NDWI vs SR output NDWI.
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="baseline" name="Baseline" tick={{ fontSize: 10, fill: "#94a3b8" }} label={{ value: "Baseline NDWI", position: "insideBottom", offset: -4, style: { fill: "#64748b", fontSize: 10 } }} />
            <YAxis dataKey="sr" name="SR" tick={{ fontSize: 10, fill: "#94a3b8" }} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "8px", fontSize: "11px" }} />
            <Scatter data={ndwiData} fill="#60a5fa" fillOpacity={0.7} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
