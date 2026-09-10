"use client";

import { useState } from "react";
import type { BandMode } from "@/types";

interface Props {
  activeMode: BandMode;
  onChange: (mode: BandMode) => void;
}

const MODES: { mode: BandMode; label: string }[] = [
  { mode: "RGB", label: "RGB" },
  { mode: "NIR", label: "NIR" },
  { mode: "NDVI", label: "NDVI" },
  { mode: "NDWI", label: "NDWI" },
];

export default function BandSelector({ activeMode, onChange }: Props) {
  return (
    <div className="inline-flex border border-border bg-card p-1 gap-1">
      {MODES.map(({ mode, label }) => (
        <button
          key={mode}
          id={`band-mode-${mode.toLowerCase()}`}
          onClick={() => onChange(mode)}
          className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors
            ${activeMode === mode
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground hover:bg-secondary"
            }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
