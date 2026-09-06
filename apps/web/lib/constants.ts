export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:10000";
export const TILE_SIZE = 256;
export const UPSCALE_FACTOR = 4;

export const BAND_LABELS: Record<number, string> = {
  0: "B02 Blue",
  1: "B03 Green",
  2: "B04 Red",
  3: "B08 NIR",
};

export const BAND_NAMES = ["B02 Blue", "B03 Green", "B04 Red", "B08 NIR"];
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
export const POLL_INTERVAL_MS = 2000;
