from typing import List, Dict, Any
import numpy as np

from spectral_metrics import (
    psnr_per_band,
    ssim_per_band,
    spectral_angle_mapper,
    ndvi_preservation_score,
    ndwi_preservation_score
)
from edge_metrics import compute_edge_preservation


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


def compute_all_metrics(lr_img: np.ndarray, sr_img: np.ndarray) -> Dict[str, Any]:
    psnr_dict = psnr_per_band(lr_img, sr_img)
    ssim_dict = ssim_per_band(lr_img, sr_img)
    sam_deg = spectral_angle_mapper(lr_img, sr_img)
    ndvi_score = ndvi_preservation_score(lr_img, sr_img)
    ndwi_score = ndwi_preservation_score(lr_img, sr_img)
    edge_dict = compute_edge_preservation(lr_img, sr_img)

    return {
        "psnr": psnr_dict,
        "ssim": ssim_dict,
        "spectral_angle_mapper_deg": sam_deg,
        "ndvi_preservation": ndvi_score,
        "ndwi_preservation": ndwi_score,
        "edge_metrics": edge_dict
    }
