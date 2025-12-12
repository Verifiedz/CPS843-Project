import cv2
import numpy as np


# =========================
# Dark Channel Prior (DCP)
# =========================

def _dark_channel(img_bgr_float01: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Dark channel = min over BGR channels, then min-filter over a local patch.
    img_bgr_float01: float32 image in [0,1]
    """
    min_per_pixel = np.min(img_bgr_float01, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_per_pixel, kernel)
    return dark


def _estimate_atmospheric_light(
    img_bgr_float01: np.ndarray,
    dark: np.ndarray,
    top_percent: float = 0.001
) -> np.ndarray:
    """
    Estimate A by taking the brightest pixels in the dark channel,
    then picking the one with the highest intensity in the original image.
    Returns A as (3,) float32 in [0,1]
    """
    h, w = dark.shape
    n = h * w
    k = max(1, int(n * top_percent))

    flat_dark = dark.reshape(-1)
    idxs = np.argpartition(flat_dark, -k)[-k:]  # top-k

    flat_img = img_bgr_float01.reshape(-1, 3)
    candidates = flat_img[idxs]
    brightness = candidates.sum(axis=1)

    best_idx = idxs[np.argmax(brightness)]
    A = flat_img[best_idx]
    return A


def _estimate_transmission(
    img_bgr_float01: np.ndarray,
    A: np.ndarray,
    omega: float,
    patch_size: int
) -> np.ndarray:
    """
    Standard DCP transmission:
      t(x) = 1 - omega * dark_channel(I / A)
    """
    A_safe = np.maximum(A, 1e-6).reshape(1, 1, 3)
    norm = img_bgr_float01 / A_safe
    dark_norm = _dark_channel(norm, patch_size)

    t = 1.0 - omega * dark_norm
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def _refine_transmission(img_bgr_uint8: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Refinement:
    - Try guided filter (opencv-contrib)
    - Fallback to bilateral filter
    """
    try:
        import cv2.ximgproc as xip  # requires opencv-contrib-python
        gray = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t_refined = xip.guidedFilter(guide=gray, src=t, radius=40, eps=1e-3)
        return np.clip(t_refined, 0.0, 1.0).astype(np.float32)
    except Exception:
        t_refined = cv2.bilateralFilter(t, d=15, sigmaColor=0.1, sigmaSpace=15)
        return np.clip(t_refined, 0.0, 1.0).astype(np.float32)


def _recover_radiance(
    img_bgr_float01: np.ndarray,
    t: np.ndarray,
    A: np.ndarray,
    t0: float
) -> np.ndarray:
    """
    Reconstruction formula:
      J(x) = (I(x) - A) / max(t(x), t0) + A
    """
    t_clip = np.maximum(t, t0)[:, :, None]
    A_vec = A.reshape(1, 1, 3)
    J = (img_bgr_float01 - A_vec) / t_clip + A_vec
    return np.clip(J, 0.0, 1.0).astype(np.float32)


# =========================
# Color correction (fix blue cast)
# =========================

def _gray_world_white_balance(img_bgr_uint8: np.ndarray) -> np.ndarray:
    """
    Simple gray-world white balance to reduce strong color casts
    (common DCP failure mode on bluish haze/smoke).
    """
    img = img_bgr_uint8.astype(np.float32)
    mean = img.mean(axis=(0, 1))  # (B, G, R)
    gray = mean.mean()
    scale = gray / (mean + 1e-6)
    out = img * scale.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# CLAHE (Contrast Enhance)
# =========================

def _apply_clahe_lab(img_bgr_uint8: np.ndarray, clip_limit: float = 2.0, grid=(8, 8)) -> np.ndarray:
    """
    Apply CLAHE on L channel in LAB color space.
    """
    lab = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out


# =========================
# Public API (Dev 1 output)
# =========================

def dehaze_dcp_clahe(
    img_bgr_uint8: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    refine: bool = True,
    use_clahe: bool = True,
    white_balance: bool = True,
) -> np.ndarray:
    """
    Dev 1 baseline:
      - DCP transmission + atmospheric light
      - Reconstruction using J(x)
      - Optional white balance (reduces blue cast)
      - Optional CLAHE contrast enhancement
    """
    img_float = img_bgr_uint8.astype(np.float32) / 255.0

    dark = _dark_channel(img_float, patch_size)
    A = _estimate_atmospheric_light(img_float, dark)
    t = _estimate_transmission(img_float, A, omega, patch_size)

    if refine:
        t = _refine_transmission(img_bgr_uint8, t)

    J = _recover_radiance(img_float, t, A, t0)
    out = (J * 255.0).astype(np.uint8)

    if white_balance:
        out = _gray_world_white_balance(out)

    if use_clahe:
        out = _apply_clahe_lab(out)

    return out
