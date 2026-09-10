import numpy as np
from scipy.ndimage import zoom
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def _ensure_chw(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[np.newaxis, :, :]
    if img.ndim == 3 and img.shape[2] in [1, 3, 4] and img.shape[0] not in [1, 3, 4]:
        return np.transpose(img, (2, 0, 1))
    return img


def match_spatial_dims(lr_img: np.ndarray, target_shape: tuple) -> np.ndarray:
    lr_chw = _ensure_chw(lr_img)
    _, h_t, w_t = target_shape
    c, h_l, w_l = lr_chw.shape
    if (h_l, w_l) == (h_t, w_t):
        return lr_chw
    zoom_factors = (1.0, h_t / h_l, w_t / w_l)
    return zoom(lr_chw, zoom_factors, order=1)


def psnr_per_band(img_ref: np.ndarray, img_sr: np.ndarray) -> dict:
    ref = _ensure_chw(img_ref)
    sr = _ensure_chw(img_sr)
    if ref.shape != sr.shape:
        ref = match_spatial_dims(ref, sr.shape)
    band_names = ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"]
    results = {}
    for i in range(min(ref.shape[0], sr.shape[0])):
        b_name = band_names[i] if i < len(band_names) else f"Band_{i}"
        r_b = np.clip(ref[i], 0.0, 1.0)
        s_b = np.clip(sr[i], 0.0, 1.0)
        val = compute_psnr(r_b, s_b, data_range=1.0)
        results[b_name] = float(val)
    results["mean_psnr"] = float(np.mean(list(results.values())))
    return results


def ssim_per_band(img_ref: np.ndarray, img_sr: np.ndarray) -> dict:
    ref = _ensure_chw(img_ref)
    sr = _ensure_chw(img_sr)
    if ref.shape != sr.shape:
        ref = match_spatial_dims(ref, sr.shape)
    band_names = ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"]
    results = {}
    for i in range(min(ref.shape[0], sr.shape[0])):
        b_name = band_names[i] if i < len(band_names) else f"Band_{i}"
        r_b = np.clip(ref[i], 0.0, 1.0)
        s_b = np.clip(sr[i], 0.0, 1.0)
        val = compute_ssim(r_b, s_b, win_size=7, data_range=1.0)
        results[b_name] = float(val)
    results["mean_ssim"] = float(np.mean(list(results.values())))
    return results


def spectral_angle_mapper(img_ref: np.ndarray, img_sr: np.ndarray) -> float:
    ref = _ensure_chw(img_ref)
    sr = _ensure_chw(img_sr)
    if ref.shape != sr.shape:
        ref = match_spatial_dims(ref, sr.shape)
    ref_vec = ref.reshape(ref.shape[0], -1)
    sr_vec = sr.reshape(sr.shape[0], -1)
    dot_product = np.sum(ref_vec * sr_vec, axis=0)
    norm_ref = np.linalg.norm(ref_vec, axis=0)
    norm_sr = np.linalg.norm(sr_vec, axis=0)
    denominator = (norm_ref * norm_sr) + 1e-8
    cosine = np.clip(dot_product / denominator, -1.0, 1.0)
    angles_rad = np.arccos(cosine)
    angles_deg = np.degrees(angles_rad)
    return float(np.mean(angles_deg))


def compute_ndvi(img: np.ndarray) -> np.ndarray:
    chw = _ensure_chw(img)
    red = chw[2] if chw.shape[0] > 2 else chw[0]
    nir = chw[3] if chw.shape[0] > 3 else chw[-1]
    denom = nir + red + 1e-6
    return (nir - red) / denom


def compute_ndwi(img: np.ndarray) -> np.ndarray:
    chw = _ensure_chw(img)
    green = chw[1] if chw.shape[0] > 1 else chw[0]
    nir = chw[3] if chw.shape[0] > 3 else chw[-1]
    denom = green + nir + 1e-6
    return (green - nir) / denom


def ndvi_preservation_score(lr_img: np.ndarray, sr_img: np.ndarray) -> float:
    # LR-upsampled-bicubic baseline vs SR output used since no true HR ground truth exists at inference time.
    ref = _ensure_chw(lr_img)
    sr = _ensure_chw(sr_img)
    if ref.shape[1:] != sr.shape[1:]:
        ref = match_spatial_dims(ref, sr.shape)
    ndvi_ref = compute_ndvi(ref).flatten()
    ndvi_sr = compute_ndvi(sr).flatten()
    corr = np.corrcoef(ndvi_ref, ndvi_sr)[0, 1]
    if np.isnan(corr):
        return 1.0 - float(np.mean(np.abs(ndvi_ref - ndvi_sr)))
    return float(corr)


def ndwi_preservation_score(lr_img: np.ndarray, sr_img: np.ndarray) -> float:
    # LR-upsampled-bicubic baseline vs SR output used since no true HR ground truth exists at inference time.
    ref = _ensure_chw(lr_img)
    sr = _ensure_chw(sr_img)
    if ref.shape[1:] != sr.shape[1:]:
        ref = match_spatial_dims(ref, sr.shape)
    ndwi_ref = compute_ndwi(ref).flatten()
    ndwi_sr = compute_ndwi(sr).flatten()
    corr = np.corrcoef(ndwi_ref, ndwi_sr)[0, 1]
    if np.isnan(corr):
        return 1.0 - float(np.mean(np.abs(ndwi_ref - ndwi_sr)))
    return float(corr)
