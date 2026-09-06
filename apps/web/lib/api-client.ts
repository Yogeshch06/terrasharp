import { API_URL } from "./constants";
import type { EnhanceJob, CopernicusRequest } from "@/types";

export async function uploadForEnhance(file: File): Promise<EnhanceJob> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/enhance`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Upload failed (${res.status}): ${detail}`);
  }
  return res.json() as Promise<EnhanceJob>;
}

export async function getJobStatus(jobId: string): Promise<EnhanceJob> {
  const res = await fetch(`${API_URL}/status/${jobId}`);
  if (!res.ok) throw new Error(`Status check failed (${res.status})`);
  return res.json() as Promise<EnhanceJob>;
}

export async function fetchFromCopernicus(req: CopernicusRequest): Promise<EnhanceJob> {
  const res = await fetch(`${API_URL}/fetch-copernicus`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      lat: req.lat,
      lon: req.lon,
      date: req.date,
      aoi_size_km: req.aoi_size_km,
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Copernicus fetch failed (${res.status}): ${detail}`);
  }
  return res.json() as Promise<EnhanceJob>;
}

export function buildDownloadUrl(jobId: string, filename: string): string {
  return `${API_URL}/download/${jobId}/${filename}`;
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}
