from typing import Tuple
import numpy as np


def compute_uncertainty_statistics(uncertainty_map: np.ndarray) -> dict:
    if uncertainty_map.ndim == 3:
        band_std = [float(np.mean(uncertainty_map[c])) for c in range(uncertainty_map.shape[0])]
        overall_mean = float(np.mean(uncertainty_map))
        overall_max = float(np.max(uncertainty_map))
        overall_min = float(np.min(uncertainty_map))
    else:
        band_std = [float(np.mean(uncertainty_map))]
        overall_mean = float(np.mean(uncertainty_map))
        overall_max = float(np.max(uncertainty_map))
        overall_min = float(np.min(uncertainty_map))

    return {
        "mean_uncertainty": overall_mean,
        "max_uncertainty": overall_max,
        "min_uncertainty": overall_min,
        "band_uncertainties": band_std
    }


def normalize_uncertainty_map(uncertainty_map: np.ndarray) -> np.ndarray:
    if uncertainty_map.ndim == 3 and uncertainty_map.shape[0] in [1, 3, 4]:
        single_channel = np.mean(uncertainty_map, axis=0)
    else:
        single_channel = uncertainty_map

    min_val = np.min(single_channel)
    max_val = np.max(single_channel)
    if max_val > min_val:
        normalized = (single_channel - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(single_channel)
    return normalized.astype(np.float32)
