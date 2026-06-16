from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def apply_corrections(
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
) -> np.ndarray:
    corrected = image.astype(np.float64).copy()
    for correction in (dark, offset):
        if correction is not None:
            _require_same_shape(corrected, correction, "dark/offset")
            corrected -= correction.astype(np.float64)

    if flat is not None:
        _require_same_shape(corrected, flat, "flat")
        flat_data = flat.astype(np.float64)
        flat_bg = np.percentile(flat_data, 1)
        flat_signal = flat_data - flat_bg
        mean_flat = np.mean(flat_signal[flat_signal > 0])
        if not np.isfinite(mean_flat) or mean_flat <= 0:
            raise ValueError("Flat image does not contain a usable positive signal.")
        corrected *= mean_flat / np.maximum(flat_signal, mean_flat * 1e-6)

    return corrected


def denoise_image(image: np.ndarray, sigma_px: float = 0.0) -> np.ndarray:
    if sigma_px <= 0:
        return image.astype(np.float64)
    return gaussian_filter(image.astype(np.float64), sigma=float(sigma_px))


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1, 99.5])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return image.astype(np.float64)
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image.astype(np.float64),
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _require_same_shape(reference: np.ndarray, image: np.ndarray, name: str) -> None:
    if reference.shape != image.shape:
        raise ValueError(
            f"{name} correction shape {image.shape} does not match image shape {reference.shape}."
        )
