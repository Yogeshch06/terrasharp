import io
import os
from typing import Tuple, Dict, Any, Union
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine


def read_geotiff(source: Union[str, bytes]) -> Tuple[np.ndarray, Dict[str, Any]]:
    if isinstance(source, bytes):
        with MemoryFile(source) as memfile:
            with memfile.open() as src:
                data = src.read()
                profile = src.profile.copy()
    else:
        with rasterio.open(source) as src:
            data = src.read()
            profile = src.profile.copy()

    orig_dtype = data.dtype
    profile["orig_dtype"] = str(orig_dtype)

    data = data.astype(np.float32)
    max_val = np.max(data)
    if max_val > 1.0:
        if max_val > 255.0:
            data = np.clip(data / 10000.0, 0.0, 1.0)
        else:
            data = np.clip(data / 255.0, 0.0, 1.0)

    return data, profile


def write_geotiff(
    path: str,
    array: np.ndarray,
    profile: Dict[str, Any],
    upscale_factor: int = 4
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    elif array.ndim == 3 and array.shape[2] in [1, 3, 4] and array.shape[0] not in [1, 3, 4]:
        array = np.transpose(array, (2, 0, 1))

    c, h, w = array.shape
    out_profile = profile.copy()
    out_profile.pop("orig_dtype", None)

    transform: Affine = profile.get("transform")
    if transform is not None:
        new_transform = transform * Affine.scale(1.0 / upscale_factor, 1.0 / upscale_factor)
        out_profile["transform"] = new_transform

    out_profile.update({
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": "uint16"
    })

    scaled_data = np.clip(array * 10000.0, 0, 65535).astype(np.uint16)

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(scaled_data)

    return path


def write_uncertainty_geotiff(
    path: str,
    array: np.ndarray,
    profile: Dict[str, Any],
    upscale_factor: int = 4
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    elif array.ndim == 3 and array.shape[2] in [1, 3, 4] and array.shape[0] not in [1, 3, 4]:
        array = np.transpose(array, (2, 0, 1))

    c, h, w = array.shape
    out_profile = profile.copy()
    out_profile.pop("orig_dtype", None)

    transform: Affine = profile.get("transform")
    if transform is not None:
        new_transform = transform * Affine.scale(1.0 / upscale_factor, 1.0 / upscale_factor)
        out_profile["transform"] = new_transform

    out_profile.update({
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": "float32"
    })

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(np.float32))

    return path
