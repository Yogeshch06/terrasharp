import os
import sys
import uuid
import json
import sqlite3
import datetime
import time
from typing import Optional
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import settings
from src.model import ONNXInferenceEngine
from src.preprocess import normalize_reflectance, create_tiling_plan
from src.postprocess import reassemble_tiles, compute_all_metrics, save_preview_pngs, apply_unsharp_mask
from src.spectral_metrics import match_spatial_dims
from src.edge_metrics import get_edge_maps
from src.geotiff_utils import read_geotiff, write_geotiff, write_uncertainty_geotiff
from src.copernicus import CopernicusClient

app = FastAPI(title="TerraSharp Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.ALLOWED_ORIGIN] if settings.ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[ONNXInferenceEngine] = None
copernicus_client: Optional[CopernicusClient] = None


def get_db_connection():
    db_path = os.path.abspath(settings.DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            metrics_json TEXT,
            created_at TIMESTAMP NOT NULL
        )
    """)
    conn.commit()
    conn.close()


@app.on_event("startup")
def startup_event():
    global engine, copernicus_client
    init_db()
    os.makedirs(settings.TMP_DIR, exist_ok=True)
    try:
        engine = ONNXInferenceEngine()
    except Exception as e:
        print(f"Error initializing inference engine on startup: {e}")
    copernicus_client = CopernicusClient()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": engine is not None and engine.session is not None
    }


class CopernicusRequest(BaseModel):
    lat: float
    lon: float
    date: str = "2023-08-01"
    aoi_size_km: float = 2.56


def run_super_resolution_pipeline(file_bytes: bytes, job_id: str, request_start: Optional[float] = None) -> dict:
    total_start = request_start if request_start is not None else time.time()
    job_dir = os.path.join(settings.TMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_data, profile = read_geotiff(file_bytes)
    preprocess_start = time.time()
    input_norm = normalize_reflectance(input_data)

    tiles, tiling_plan = create_tiling_plan(
        input_norm,
        tile_size=settings.TILE_SIZE,
        overlap=32
    )
    preprocess_time = time.time() - preprocess_start

    sr_tiles = []
    unc_tiles = []
    inference_timing = {"main_inference": 0.0, "uncertainty": 0.0, "uncertainty_passes": 0}
    for tile in tiles:
        sr_tile, unc_tile = engine.infer_with_uncertainty(
            tile,
            num_passes=settings.MC_DROPOUT_PASSES,
            timing=inference_timing
        )
        sr_tiles.append(sr_tile)
        unc_tiles.append(unc_tile)

    postprocess_start = time.time()
    sr_full = reassemble_tiles(
        sr_tiles,
        tiling_plan,
        input_norm.shape,
        upscale_factor=settings.UPSCALE_FACTOR
    )

    unc_full = reassemble_tiles(
        unc_tiles,
        tiling_plan,
        input_norm.shape,
        upscale_factor=settings.UPSCALE_FACTOR
    )
    postprocess_time = time.time() - postprocess_start

    enhanced_tif_path = os.path.join(job_dir, "enhanced.tif")
    uncertainty_tif_path = os.path.join(job_dir, "uncertainty.tif")

    geotiff_start = time.time()
    enhanced_sr_full = apply_unsharp_mask(sr_full)
    write_geotiff(
        enhanced_tif_path,
        enhanced_sr_full,
        profile,
        upscale_factor=settings.UPSCALE_FACTOR
    )

    write_uncertainty_geotiff(
        uncertainty_tif_path,
        unc_full,
        profile,
        upscale_factor=settings.UPSCALE_FACTOR
    )
    geotiff_time = time.time() - geotiff_start

    metrics_start = time.time()
    matched_lr_start = time.time()
    matched_lr = match_spatial_dims(input_norm, sr_full.shape)
    matched_lr_time = time.time() - matched_lr_start

    edge_maps_start = time.time()
    edge_maps = get_edge_maps(input_norm, sr_full, ref=matched_lr)
    edge_maps_time = time.time() - edge_maps_start

    metrics = compute_all_metrics(input_norm, sr_full, matched_lr=matched_lr, edge_maps=edge_maps)
    metrics_time = time.time() - metrics_start
    print(
        f"[METRICS DEBUG] match_spatial_dims: {matched_lr_time:.2f}s | "
        f"get_edge_maps: {edge_maps_time:.2f}s | "
        f"compute_all_metrics: {metrics_time - matched_lr_time - edge_maps_time:.2f}s"
    )

    png_start = time.time()
    save_preview_pngs(job_dir, input_norm, sr_full, unc_full, matched_lr=matched_lr, edge_maps=edge_maps)
    png_time = time.time() - png_start

    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.datetime.utcnow().isoformat()
    cursor.execute(
        "INSERT OR REPLACE INTO jobs (job_id, status, metrics_json, created_at) VALUES (?, ?, ?, ?)",
        (job_id, "completed", json.dumps(metrics), now_iso)
    )
    conn.commit()
    conn.close()

    total_time = time.time() - total_start
    print(
        f"[TIMING] Preprocess: {preprocess_time:.2f}s | "
        f"Inference: {inference_timing['main_inference']:.2f}s | "
        f"Uncertainty ({inference_timing['uncertainty_passes']} passes): {inference_timing['uncertainty']:.2f}s | "
        f"Postprocess: {postprocess_time:.2f}s | Metrics: {metrics_time:.2f}s | "
        f"PNG export: {png_time:.2f}s | GeoTIFF write: {geotiff_time:.2f}s | TOTAL: {total_time:.2f}s"
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "metrics": metrics,
        "download_urls": {
            "enhanced": f"/download/{job_id}/enhanced.tif",
            "uncertainty": f"/download/{job_id}/uncertainty.tif",
            "preview_before": f"/download/{job_id}/preview_before.png",
            "preview_after": f"/download/{job_id}/preview_after.png",
            "edge_before": f"/download/{job_id}/edge_before.png",
            "edge_after": f"/download/{job_id}/edge_after.png",
            "uncertainty_heatmap": f"/download/{job_id}/uncertainty_heatmap.png"
        }
    }


@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    if not engine:
        raise HTTPException(status_code=503, detail="Model inference engine is not ready.")

    request_start = time.time()
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_MB}MB.")

    job_id = str(uuid.uuid4())
    try:
        result = run_super_resolution_pipeline(content, job_id, request_start=request_start)
        return result
    except Exception as e:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO jobs (job_id, status, metrics_json, created_at) VALUES (?, ?, ?, ?)",
            (job_id, f"failed: {str(e)}", json.dumps({}), datetime.datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@app.post("/fetch-copernicus")
def fetch_copernicus(payload: CopernicusRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Model inference engine is not ready.")
    if not copernicus_client:
        raise HTTPException(status_code=503, detail="Copernicus client is not ready.")

    try:
        tif_bytes = copernicus_client.fetch_sentinel2(
            lat=payload.lat,
            lon=payload.lon,
            date=payload.date,
            aoi_size_km=payload.aoi_size_km
        )
        job_id = str(uuid.uuid4())
        result = run_super_resolution_pipeline(tif_bytes, job_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Copernicus processing failed: {str(e)}")


@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT job_id, status, metrics_json, created_at FROM jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")

    metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "metrics": metrics,
        "created_at": row["created_at"],
        "download_urls": {
            "enhanced": f"/download/{job_id}/enhanced.tif",
            "uncertainty": f"/download/{job_id}/uncertainty.tif",
            "preview_before": f"/download/{job_id}/preview_before.png",
            "preview_after": f"/download/{job_id}/preview_after.png",
            "edge_before": f"/download/{job_id}/edge_before.png",
            "edge_after": f"/download/{job_id}/edge_after.png",
            "uncertainty_heatmap": f"/download/{job_id}/uncertainty_heatmap.png"
        } if row["status"] == "completed" else {}
    }


@app.get("/download/{job_id}/{filename}")
def download_file(job_id: str, filename: str):
    file_path = os.path.join(settings.TMP_DIR, job_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Requested file not found.")
    media_type = "image/png" if filename.lower().endswith(".png") else "image/tiff"
    return FileResponse(file_path, filename=filename, media_type=media_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)
