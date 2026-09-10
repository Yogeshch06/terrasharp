import numpy as np
from scipy.ndimage import sobel, zoom


def _ensure_chw(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[np.newaxis, :, :]
    if img.ndim == 3 and img.shape[2] in [1, 3, 4] and img.shape[0] not in [1, 3, 4]:
        return np.transpose(img, (2, 0, 1))
    return img


def _compute_gradient_magnitude(img_2d: np.ndarray) -> np.ndarray:
    gx = sobel(img_2d, axis=0, mode="reflect")
    gy = sobel(img_2d, axis=1, mode="reflect")
    return np.hypot(gx, gy)


def compute_multichannel_edge_map(img_chw: np.ndarray) -> np.ndarray:
    edge_maps = [_compute_gradient_magnitude(img_chw[c]) for c in range(img_chw.shape[0])]
    return np.mean(edge_maps, axis=0)


def get_edge_maps(
    lr_img: np.ndarray,
    sr_img: np.ndarray,
    ref: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    ref = _ensure_chw(lr_img) if ref is None else _ensure_chw(ref)
    sr = _ensure_chw(sr_img)
    if ref.shape[1:] != sr.shape[1:]:
        zoom_factors = (1.0, sr.shape[1] / ref.shape[1], sr.shape[2] / ref.shape[2])
        ref = zoom(ref, zoom_factors, order=1)
    lr_edges = compute_multichannel_edge_map(ref)
    sr_edges = compute_multichannel_edge_map(sr)
    return lr_edges, sr_edges


def compute_edge_preservation(
    lr_img: np.ndarray,
    sr_img: np.ndarray,
    edge_maps: tuple[np.ndarray, np.ndarray] = None
) -> dict:
    lr_edges, sr_edges = edge_maps if edge_maps is not None else get_edge_maps(lr_img, sr_img)
    lr_flat = lr_edges.flatten()
    sr_flat = sr_edges.flatten()

    dot = np.sum(lr_flat * sr_flat)
    norm_lr = np.linalg.norm(lr_flat)
    norm_sr = np.linalg.norm(sr_flat)
    cosine_sim = float(dot / ((norm_lr * norm_sr) + 1e-8))

    l1_diff = float(np.mean(np.abs(lr_edges - sr_edges)))
    lr_energy = float(np.mean(lr_edges ** 2))
    sr_energy = float(np.mean(sr_edges ** 2))
    edge_enhancement_ratio = float(sr_energy / (lr_energy + 1e-8))

    return {
        "edge_cosine_similarity": np.clip(cosine_sim, -1.0, 1.0),
        "edge_l1_diff": l1_diff,
        "edge_enhancement_ratio": edge_enhancement_ratio
    }
