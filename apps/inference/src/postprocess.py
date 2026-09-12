import os
import time
from typing import List, Dict, Any
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

from spectral_metrics import (
    psnr_per_band,
    ssim_per_band,
    spectral_angle_mapper,
    match_spatial_dims,
    ndvi_preservation_score,
    ndwi_preservation_score
)
from edge_metrics import compute_edge_preservation, get_edge_maps


CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))

def _read_postprocess_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as fp:
            data = yaml.safe_load(fp) or {}
        post = data.get('postprocess', {}) if isinstance(data, dict) else {}
        unsharp = post.get('unsharp', {}) if isinstance(post, dict) else {}
        return {
            'strength': float(unsharp.get('strength', 0.5)),
            'sigma': float(unsharp.get('sigma', 1.0)),
        }
    return {'strength': 0.5, 'sigma': 1.0}


def apply_unsharp_mask(sr_output: np.ndarray) -> np.ndarray:
    """Apply a lightweight per-band unsharp-mask sharpening to SR output.

    Formula:
        blurred = gaussian_filter(sr_output, sigma=1.0)
        sharpened = sr_output + 0.5 * (sr_output - blurred)
    """
    cfg = _read_postprocess_config()
    strength = float(cfg.get('strength', 0.5))
    sigma = float(cfg.get('sigma', 1.0))

    sr_output = np.clip(sr_output, 0.0, 1.0)
    if sr_output.ndim != 3:
        return sr_output

    sharpened = np.empty_like(sr_output, dtype=np.float32)
    for band in range(sr_output.shape[0]):
        blurred = gaussian_filter(sr_output[band], sigma=sigma)
        sharpened[band] = sr_output[band] + strength * (sr_output[band] - blurred)

    return np.clip(sharpened, 0.0, 1.0)


def _create_feather_mask(tile_h: int, tile_w: int, blend_width: int = 16) -> np.ndarray:
    y = np.ones((tile_h, tile_w), dtype=np.float32)
    blend_y = min(blend_width, tile_h // 2)
    blend_x = min(blend_width, tile_w // 2)

    if blend_y > 0:
        fade_y = np.linspace(0.0, 1.0, blend_y)
        for i in range(blend_y):
            y[i, :] *= fade_y[i]
            y[tile_h - 1 - i, :] *= fade_y[i]

    if blend_x > 0:
        fade_x = np.linspace(0.0, 1.0, blend_x)
        for j in range(blend_x):
            y[:, j] *= fade_x[j]
            y[:, tile_w - 1 - j] *= fade_x[j]

    return np.maximum(y, 1e-4)


def reassemble_tiles(
    tiles: List[np.ndarray],
    tiling_plan: List[Dict[str, Any]],
    orig_shape: tuple,
    upscale_factor: int = 4
) -> np.ndarray:
    c = tiles[0].shape[0] if tiles[0].ndim == 3 else 4
    _, orig_h, orig_w = orig_shape
    out_h = orig_h * upscale_factor
    out_w = orig_w * upscale_factor

    canvas = np.zeros((c, out_h, out_w), dtype=np.float32)
    weight_map = np.zeros((1, out_h, out_w), dtype=np.float32)

    for tile, plan in zip(tiles, tiling_plan):
        y_start = plan["y_start"] * upscale_factor
        y_end = plan["y_end"] * upscale_factor
        x_start = plan["x_start"] * upscale_factor
        x_end = plan["x_end"] * upscale_factor

        target_h = y_end - y_start
        target_w = x_end - x_start

        cropped_tile = tile[:, :target_h, :target_w]
        feather = _create_feather_mask(target_h, target_w, blend_width=16 * upscale_factor)[np.newaxis, :, :]

        canvas[:, y_start:y_end, x_start:x_end] += cropped_tile * feather
        weight_map[:, y_start:y_end, x_start:x_end] += feather

    weight_map = np.maximum(weight_map, 1e-6)
    result = canvas / weight_map
    return np.clip(result, 0.0, 1.0)


def denormalize_reflectance(data: np.ndarray, to_uint16: bool = True) -> np.ndarray:
    if to_uint16:
        return np.clip(data * 10000.0, 0, 65535).astype(np.uint16)
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def compute_all_metrics(
    lr_img: np.ndarray,
    sr_img: np.ndarray,
    matched_lr: np.ndarray = None,
    edge_maps: tuple[np.ndarray, np.ndarray] = None
) -> Dict[str, Any]:
    matched_lr = matched_lr if matched_lr is not None else match_spatial_dims(lr_img, sr_img.shape)
    psnr_dict = psnr_per_band(matched_lr, sr_img)
    ssim_dict = ssim_per_band(matched_lr, sr_img)
    sam_deg = spectral_angle_mapper(matched_lr, sr_img)
    ndvi_score = ndvi_preservation_score(matched_lr, sr_img)
    ndwi_score = ndwi_preservation_score(matched_lr, sr_img)
    edge_dict = compute_edge_preservation(lr_img, sr_img, edge_maps=edge_maps)

    return {
        "psnr": psnr_dict,
        "ssim": ssim_dict,
        "spectral_angle_mapper_deg": sam_deg,
        "ndvi_preservation": ndvi_score,
        "ndwi_preservation": ndwi_score,
        "edge_metrics": edge_dict
    }


def _to_uint8(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)
    return (np.clip((data - low) / (high - low), 0.0, 1.0) * 255).astype(np.uint8)


def _write_png(path: str, data: np.ndarray) -> None:
    if data.ndim == 2:
        pixels = data
    else:
        channels = data.shape[0]
        if channels != 3:
            raise ValueError("PNG output must have one or three channels")
        pixels = np.moveaxis(data, 0, -1)
    Image.fromarray(pixels).save(path, format="PNG", compress_level=1)


def _rgb_preview(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = target_shape
    if image.shape[1:] != target_shape:
        image = zoom(image, (1.0, height / image.shape[1], width / image.shape[2]), order=3)
    rgb = np.stack([image[2], image[1], image[0]], axis=0)
    return np.stack([_to_uint8(channel) for channel in rgb], axis=0)


def _uncertainty_heatmap(uncertainty: np.ndarray) -> np.ndarray:
    magnitude = np.mean(uncertainty, axis=0) if uncertainty.ndim == 3 else uncertainty
    normalized = _to_uint8(magnitude).astype(np.float32) / 255.0
    stops = np.array([[0, 0, 255], [255, 255, 0], [255, 0, 0]], dtype=np.float32)
    positions = normalized * (len(stops) - 1)
    lower = np.floor(positions).astype(np.int32)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = positions - lower
    return np.moveaxis(stops[lower] * (1 - fraction[..., None]) + stops[upper] * fraction[..., None], -1, 0).astype(np.uint8)


def save_preview_pngs(
    job_dir: str,
    input_norm: np.ndarray,
    sr_full: np.ndarray,
    unc_full: np.ndarray,
    matched_lr: np.ndarray = None,
    edge_maps: tuple[np.ndarray, np.ndarray] = None
) -> None:
    sr_shape = (sr_full.shape[1], sr_full.shape[2])
    timings = []

    matched_lr = matched_lr if matched_lr is not None else match_spatial_dims(input_norm, sr_full.shape)
    edge_before, edge_after = edge_maps if edge_maps is not None else get_edge_maps(input_norm, sr_full)

    prep_start = time.time()
    preview_before = _rgb_preview(matched_lr, sr_shape)
    prep_time = time.time() - prep_start
    save_start = time.time()
    _write_png(os.path.join(job_dir, "preview_before.png"), preview_before)
    timings.append(("preview_before", prep_time, time.time() - save_start))

    prep_start = time.time()
    preview_after = _rgb_preview(apply_unsharp_mask(sr_full), sr_shape)
    prep_time = time.time() - prep_start
    save_start = time.time()
    _write_png(os.path.join(job_dir, "preview_after.png"), preview_after)
    timings.append(("preview_after", prep_time, time.time() - save_start))

    prep_start = time.time()
    edge_before_uint8 = _to_uint8(edge_before)
    prep_time = time.time() - prep_start
    save_start = time.time()
    _write_png(os.path.join(job_dir, "edge_before.png"), edge_before_uint8)
    timings.append(("edge_before", prep_time, time.time() - save_start))

    prep_start = time.time()
    edge_after_uint8 = _to_uint8(edge_after)
    prep_time = time.time() - prep_start
    save_start = time.time()
    _write_png(os.path.join(job_dir, "edge_after.png"), edge_after_uint8)
    timings.append(("edge_after", prep_time, time.time() - save_start))

    prep_start = time.time()
    uncertainty_heatmap = _uncertainty_heatmap(unc_full)
    prep_time = time.time() - prep_start
    save_start = time.time()
    _write_png(os.path.join(job_dir, "uncertainty_heatmap.png"), uncertainty_heatmap)
    timings.append(("uncertainty_heatmap", prep_time, time.time() - save_start))

    print(
        "[PNG DEBUG] "
        + " | ".join(
            f"{name}: prep={prep:.2f}s save={save:.2f}s"
            for name, prep, save in timings
        )
    )
