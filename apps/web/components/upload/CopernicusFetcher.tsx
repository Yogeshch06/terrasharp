"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { fetchFromCopernicus } from "@/lib/api-client";

export default function CopernicusFetcher() {
  const router = useRouter();
  const [lat, setLat] = useState(28.6139);
  const [lon, setLon] = useState(77.2090);
  const [date, setDate] = useState("2023-08-01");
  const [aoiSize, setAoiSize] = useState(2.56);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const job = await fetchFromCopernicus({ lat, lon, date, aoi_size_km: aoiSize });
      router.push(`/enhance?job=${job.job_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
      setLoading(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-xl border border-border bg-card p-6 space-y-5"
    >
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1">
          <label htmlFor="cop-lat" className="text-sm font-medium text-muted-foreground">Latitude</label>
          <input
            id="cop-lat"
            type="number"
            step="any"
            value={lat}
            onChange={(e) => setLat(parseFloat(e.target.value))}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            required
          />
        </div>
        <div className="space-y-1">
          <label htmlFor="cop-lon" className="text-sm font-medium text-muted-foreground">Longitude</label>
          <input
            id="cop-lon"
            type="number"
            step="any"
            value={lon}
            onChange={(e) => setLon(parseFloat(e.target.value))}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            required
          />
        </div>
      </div>

      <div className="space-y-1">
        <label htmlFor="cop-date" className="text-sm font-medium text-muted-foreground">Acquisition Date</label>
        <input
          id="cop-date"
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
          required
        />
      </div>

      <div className="space-y-2">
        <label htmlFor="cop-aoi" className="text-sm font-medium text-muted-foreground">
          AOI Size: <span className="text-foreground font-semibold">{aoiSize.toFixed(2)} km</span>
        </label>
        <input
          id="cop-aoi"
          type="range"
          min={0.5}
          max={10}
          step={0.5}
          value={aoiSize}
          onChange={(e) => setAoiSize(parseFloat(e.target.value))}
          className="w-full accent-primary"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>0.5 km</span>
          <span>10 km</span>
        </div>
      </div>

      {error && (
        <div className="text-sm text-red-400 bg-red-950/40 border border-red-900/50 rounded-lg px-4 py-2">
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        id="cop-submit"
        className="w-full bg-primary text-primary-foreground rounded-md px-4 py-2 text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? "Fetching…" : "Fetch & Enhance"}
      </button>
    </form>
  );
}
