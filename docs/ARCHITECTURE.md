# TerraSharp — Architecture

## Problem

Sentinel-2 imagery is free, global, and revisits every 5 days — but it's capped at 10m spatial
resolution. Commercial high-resolution satellite imagery (sub-3m) exists, but it's expensive and
not captured on a predictable schedule. This gap matters most exactly where frequent, current
imagery is needed: crop health monitoring, flood/disaster response, urban sprawl tracking. Basemap
services (Google Maps, etc.) are not a substitute — they show a single static, often outdated
snapshot, not a monitoring feed.

TerraSharp closes this gap with a 4x super-resolution model that upscales Sentinel-2's 10m imagery
to an effective 2.5m, using only the free, frequently-updated Sentinel-2 feed as input.

## System Overview

```
┌─────────────┐      ┌──────────────────┐      ┌───────────────────┐
│   Next.js  │────▶│   FastAPI       │─────▶│  ONNX Runtime   │
│   Frontend │◀────│   Inference API │◀─────│  (SwinIR-Light) │
│  (Vercel)  │     │   (Render)      │      │  fetched from   │
└─────────────┘      └──────────────────┘      │  Hugging Face Hub│
                              │             └───────────────────┘
                              ▼
                      ┌──────────────────┐
                      │ Copernicus      │
                      │ Data Space      │
                      │ (Sentinel Hub)  │
                      └──────────────────┘
```

## Components

### 1. Model — SwinIR-Light
A compact Swin Transformer-based super-resolution architecture, built from scratch (no external
Swin library dependency) for clean ONNX export:
- Input: 4-band Sentinel-2 tile (B02 Blue, B03 Green, B04 Red, B08 NIR), 256×256
- 3×3 conv stem → 4 shifted-window Swin Transformer blocks (embed_dim=64, heads=4, window=8) →
  residual skip → PixelShuffle 4x upsample → 3×3 conv output
- Output: 1024×1024, 4-band

**Loss function** (`TerraSharpLoss`): weighted combination of
- L1 pixel loss (1.0)
- VGG19 perceptual loss on RGB bands (0.1)
- Sobel edge L1 loss (0.5) — preserves field boundaries, roads, coastlines
- NDVI MAE loss (0.3) — preserves vegetation index fidelity, since agriculture is a primary use case

### 2. Training Data — Wald's Protocol
No public 2.5m ground-truth source exists for Sentinel-2 coverage areas. Training uses **Wald's
protocol**, the standard methodology in remote-sensing super-resolution literature: real Sentinel-2
10m imagery is treated as "HR," then degraded (Gaussian blur + bicubic downsample 4x) to produce
synthetic "LR" input. The model learns to reverse this degradation, and at inference time is applied
to real 10m imagery to produce a 2.5m-equivalent output.

- Source: Sentinel-2 L2A `.SAFE` products from Copernicus Data Space, 4 bands extracted at native
  10m resolution
- Chips: 256×256 non-overlapping tiles, filtered to exclude >5% nodata (scene-edge) tiles
- Split: 90/10 train/val

**Known limitation:** training/validation chips are drawn from the same scene(s), so validation
loss reflects unseen-pixel performance within a known scene rather than generalization to entirely
new geography. Noted here deliberately — see "Honest Limitations" below.

### 3. Backend — FastAPI Inference Service
- Loads the trained model as ONNX via `onnxruntime`, auto-fetched from Hugging Face Hub at startup
  (falls back to a randomly-initialized model if unavailable, logged loudly — never silently serves
  degraded output without a warning)
- Tiles large inputs to 256×256, runs inference per tile, reassembles with overlap feathering
- **Uncertainty estimation**: MC-dropout-style ensembling (multiple forward passes with input
  perturbation) produces a per-pixel uncertainty map — communicates model confidence rather than
  presenting output as ground truth
- **Metrics pipeline**: PSNR, SSIM per band; Spectral Angle Mapper (SAM); NDVI/NDWI preservation
  (correlation against bicubic baseline, since no true HR reference exists at inference time);
  Sobel edge preservation (cosine similarity of gradient maps)
- **GeoTIFF fidelity**: output preserves CRS and correctly rescales the geotransform (pixel size
  ÷4) so enhanced output aligns spatially with the original — verified against real Sentinel-2
  UTM-projected input (EPSG:32645 confirmed round-trip correct)
- **Copernicus integration**: OAuth2 client-credentials flow against Sentinel Hub Process API for
  direct fetch-and-enhance by coordinates, no manual download required
- SQLite job tracking; async-style job/status/download endpoint pattern

### 4. Frontend — Next.js 14
- Landing page: static marketing/explainer, pre-generated showcase imagery (no live inference
  dependency for the first impression)
- `/enhance`: upload or Copernicus fetch → polling job status → results view (before/after slider,
  band selector, uncertainty overlay, spectral/edge metrics dashboards)
- `/change-detection`: dual-scene comparison, NDVI delta analysis
- All comparison imagery served as backend-generated PNG previews (RGB-stretched, edge maps,
  uncertainty heatmap) rather than parsing multi-band GeoTIFFs client-side, for speed and
  consistency across browsers

## Performance

End-to-end `/enhance` latency (256×256 → 1024×1024, CPU inference, local benchmark):

| Stage | Time |
|---|---|
| Preprocessing | ~0.0s |
| Model inference | ~2.5–5s |
| Uncertainty (2-pass ensemble) | ~7–9s |
| Metrics computation | ~2–4s |
| Preview PNG generation | ~1.5–2s |
| GeoTIFF write | ~0.2s |
| **Total** | **~19s** (down from an initial unoptimized ~51s) |

Key optimizations applied: eliminated redundant bicubic-resize calls shared across metrics and
preview generation (was computed 2-3x independently), switched baseline interpolation from cubic
to linear (order=3 → order=1, ~0.35dB PSNR baseline shift, disclosed), replaced a hand-rolled PNG
encoder with Pillow, reduced MC-dropout passes from 5 to 2.

## Honest Limitations (know these before Q&A)

- **Single-scene training data**: the current model is trained on chips from one Sentinel-2 scene.
  Metrics reported (PSNR ~35dB, SAM ~2°) reflect strong reconstruction of *that scene's* held-out
  pixels, not proven generalization to unseen geography/land-cover types. Framed honestly as a
  scope decision made under hackathon time constraints, with a clear path to improve (train on
  3-5 diverse scenes — farmland, urban, coastal — the architecture requires no changes to do this).
- **No true HR ground truth at inference**: NDVI/NDWI preservation and SSIM are measured against
  a bicubic-upsampled baseline of the same input, not independent higher-resolution imagery,
  because no free 2.5m reference exists for arbitrary Sentinel-2 coverage. This is standard practice
  in the field for exactly this reason (Wald's protocol validates in training; inference-time
  metrics are relative-improvement indicators, not absolute accuracy claims).
- **Uncertainty is a proxy**: true MC-dropout requires dropout layers active at inference; the
  fallback (input-noise perturbation ensembling) approximates the same idea but is not identical
  to formal Bayesian uncertainty estimation.
- **CPU inference latency (~19s)**: acceptable for a demo/prototype; a production deployment would
  benefit from GPU inference or a smaller/quantized model for sub-5s response times.