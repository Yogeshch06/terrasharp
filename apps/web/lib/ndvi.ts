export function computeNdvi(nirBand: Float32Array, redBand: Float32Array): Float32Array {
  const out = new Float32Array(nirBand.length);
  for (let i = 0; i < nirBand.length; i++) {
    const nir = nirBand[i];
    const red = redBand[i];
    const denom = nir + red;
    out[i] = denom < 1e-6 ? 0 : (nir - red) / denom;
  }
  return out;
}

export function computeNdwi(greenBand: Float32Array, nirBand: Float32Array): Float32Array {
  const out = new Float32Array(greenBand.length);
  for (let i = 0; i < greenBand.length; i++) {
    const green = greenBand[i];
    const nir = nirBand[i];
    const denom = green + nir;
    out[i] = denom < 1e-6 ? 0 : (green - nir) / denom;
  }
  return out;
}

export function indexToRgb(index: Float32Array, colormap: "ndvi" | "ndwi" = "ndvi"): Uint8ClampedArray {
  const data = new Uint8ClampedArray(index.length * 4);
  for (let i = 0; i < index.length; i++) {
    const v = Math.max(-1, Math.min(1, index[i]));
    const t = (v + 1) / 2;
    let r = 0, g = 0, b = 0;
    if (colormap === "ndvi") {
      r = Math.round(255 * Math.max(0, 1 - 2 * t));
      g = Math.round(255 * Math.min(2 * t, 2 - 2 * t));
      b = 0;
    } else {
      r = 0;
      g = Math.round(255 * Math.max(0, 1 - 2 * t));
      b = Math.round(255 * Math.min(2 * t, 2 - 2 * t));
    }
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return data;
}
