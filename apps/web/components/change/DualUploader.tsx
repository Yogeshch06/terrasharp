"use client";

import { useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { uploadForEnhance } from "@/lib/api-client";
import { MAX_UPLOAD_BYTES } from "@/lib/constants";

interface FileSlot {
  file: File | null;
  label: string;
  jobId: string | null;
  error: string | null;
  uploading: boolean;
}

function initSlot(label: string): FileSlot {
  return { file: null, label, jobId: null, error: null, uploading: false };
}

export default function DualUploader({ onBothUploaded }: { onBothUploaded: (id1: string, id2: string) => void }) {
  const [slots, setSlots] = useState<[FileSlot, FileSlot]>([initSlot("T1 (Before)"), initSlot("T2 (After)")]);

  const updateSlot = (idx: 0 | 1, patch: Partial<FileSlot>) => {
    setSlots((prev) => {
      const next = [...prev] as [FileSlot, FileSlot];
      next[idx] = { ...next[idx], ...patch };
      return next;
    });
  };

  const handleFile = async (idx: 0 | 1, file: File) => {
    if (file.size > MAX_UPLOAD_BYTES) {
      updateSlot(idx, { error: "File exceeds 50 MB limit." });
      return;
    }
    updateSlot(idx, { file, error: null, uploading: true, jobId: null });
    try {
      const job = await uploadForEnhance(file);
      updateSlot(idx, { jobId: job.job_id, uploading: false });

      setSlots((prev) => {
        const [a, b] = prev;
        const updated = [...prev] as [FileSlot, FileSlot];
        updated[idx] = { ...updated[idx], jobId: job.job_id, uploading: false };
        const j0 = updated[0].jobId;
        const j1 = updated[1].jobId;
        if (j0 && j1) onBothUploaded(j0, j1);
        return updated;
      });
    } catch (e) {
      updateSlot(idx, { error: e instanceof Error ? e.message : "Upload failed.", uploading: false });
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {slots.map((slot, i) => (
        <label
          key={i}
          htmlFor={`dual-upload-${i}`}
            className={`flex flex-col items-center justify-center gap-3 border border-dashed p-8 cursor-pointer transition-all
            ${slot.jobId ? "border-primary bg-primary/5" : "border-border hover:border-primary hover:bg-secondary"}
            ${slot.uploading ? "opacity-70 pointer-events-none" : ""}
          `}
        >
          <input
            id={`dual-upload-${i}`}
            type="file"
            accept=".tif,.tiff"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(i as 0 | 1, f); }}
          />
          <div className="text-sm font-semibold text-primary">{slot.jobId ? "READY" : slot.uploading ? "UPLOADING" : "TIF"}</div>
          <div className="text-sm font-semibold">{slot.label}</div>
          <div className="text-xs text-muted-foreground text-center">
            {slot.uploading ? "Uploading…" : slot.jobId ? `Job: ${slot.jobId.slice(0, 8)}…` : "Click to select .tif"}
          </div>
          {slot.error && <div className="text-xs text-destructive">{slot.error}</div>}
        </label>
      ))}
    </div>
  );
}
