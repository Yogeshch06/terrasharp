"use client";

import { useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { uploadForEnhance } from "@/lib/api-client";
import { MAX_UPLOAD_BYTES } from "@/lib/constants";

export default function GeoTiffUploader() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<"idle" | "validating" | "uploading" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);

  const processFile = useCallback(async (file: File) => {
    setError(null);
    setFileName(file.name);

    if (file.size > MAX_UPLOAD_BYTES) {
      setError("File exceeds 50 MB limit.");
      setStatus("error");
      return;
    }

    if (!file.name.match(/\.(tif|tiff)$/i)) {
      setError("Please upload a GeoTIFF (.tif / .tiff) file.");
      setStatus("error");
      return;
    }

    setStatus("validating");
    try {
      const { parseGeoTiff } = await import("@/lib/geotiff");
      const info = await parseGeoTiff(file);
      if (info.bandCount < 4) {
        setError(`Expected 4 bands (B02/B03/B04/B08). Found ${info.bandCount}.`);
        setStatus("error");
        return;
      }
    } catch {
      setError("Could not read file as GeoTIFF. Ensure it is a valid 4-band Sentinel-2 image.");
      setStatus("error");
      return;
    }

    setStatus("uploading");
    let p = 0;
    const interval = setInterval(() => {
      p = Math.min(p + 8, 85);
      setProgress(p);
    }, 200);

    try {
      const job = await uploadForEnhance(file);
      clearInterval(interval);
      setProgress(100);
      router.push(`/enhance?job=${job.job_id}`);
    } catch (err) {
      clearInterval(interval);
      setError(err instanceof Error ? err.message : "Upload failed.");
      setStatus("error");
    }
  }, [router]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const onInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  }, [processFile]);

  const isLoading = status === "validating" || status === "uploading";

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !isLoading && inputRef.current?.click()}
      className={`relative border border-dashed bg-card p-10 text-center cursor-pointer transition-all
        ${dragging ? "border-primary bg-primary/5" : "border-border hover:border-primary hover:bg-secondary"}
        ${isLoading ? "pointer-events-none opacity-80" : ""}
        ${status === "error" ? "border-destructive/60" : ""}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".tif,.tiff"
        className="hidden"
        onChange={onInputChange}
        id="geotiff-file-input"
      />

      <div className="space-y-3">
        <div className="mx-auto flex h-10 w-10 items-center justify-center border border-border text-sm font-semibold text-primary">TIFF</div>
        <div className="font-semibold text-lg">
          {isLoading ? (
            status === "validating" ? "Validating GeoTIFF…" : `Uploading ${fileName}…`
          ) : dragging ? (
            "Drop it!"
          ) : (
            "Drag & drop a 4-band Sentinel-2 GeoTIFF"
          )}
        </div>
        <div className="text-sm text-muted-foreground">
          {isLoading ? `${progress}%` : "or click to browse · .tif / .tiff · max 50 MB · bands: B02 B03 B04 B08"}
        </div>

        {isLoading && (
          <div className="w-full bg-secondary h-1.5 mt-4 overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-200"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}

        {status === "error" && error && (
          <div className="mt-3 text-sm text-destructive border-l-4 border-destructive px-4 py-2 text-left">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
