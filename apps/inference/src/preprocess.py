from typing import List, Tuple, Dict, Any
import numpy as np


def normalize_reflectance(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32)
    max_val = np.max(data)
    if max_val > 1.0:
        if max_val > 255.0:
            data = np.clip(data / 10000.0, 0.0, 1.0)
        else:
            data = np.clip(data / 255.0, 0.0, 1.0)
    return np.clip(data, 0.0, 1.0)


def pad_to_multiple(image: np.ndarray, tile_size: int = 256) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim == 3 and image.shape[2] in [1, 3, 4] and image.shape[0] not in [1, 3, 4]:
        image = np.transpose(image, (2, 0, 1))

    _, h, w = image.shape
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size

    pad_top = 0
    pad_bottom = pad_h
    pad_left = 0
    pad_right = pad_w

    if pad_h > 0 or pad_w > 0:
        padded = np.pad(
            image,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="reflect"
        )
    else:
        padded = image

    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def create_tiling_plan(
    image: np.ndarray,
    tile_size: int = 256,
    overlap: int = 32
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim == 3 and image.shape[2] in [1, 3, 4] and image.shape[0] not in [1, 3, 4]:
        image = np.transpose(image, (2, 0, 1))

    c, h, w = image.shape
    stride = tile_size - overlap

    tiles = []
    plan = []

    y_indices = list(range(0, max(1, h - tile_size + 1), stride))
    if len(y_indices) == 0 or y_indices[-1] + tile_size < h:
        y_indices.append(max(0, h - tile_size))

    x_indices = list(range(0, max(1, w - tile_size + 1), stride))
    if len(x_indices) == 0 or x_indices[-1] + tile_size < w:
        x_indices.append(max(0, w - tile_size))

    for y in sorted(list(set(y_indices))):
        for x in sorted(list(set(x_indices))):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = image[:, y_start:y_end, x_start:x_end]

            if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                pad_y = tile_size - tile.shape[1]
                pad_x = tile_size - tile.shape[2]
                tile = np.pad(tile, ((0, 0), (0, pad_y), (0, pad_x)), mode="reflect")

            tiles.append(tile)
            plan.append({
                "y_start": y_start,
                "y_end": y_end,
                "x_start": x_start,
                "x_end": x_end,
                "tile_size": tile_size
            })

    return tiles, plan
