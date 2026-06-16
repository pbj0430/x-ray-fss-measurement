from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


Roi = Tuple[int, int, int, int]


@dataclass
class RoiDetectionResult:
    roi: Roi
    signal_bbox: Roi
    polarity: str
    confidence: float
    message: str


def crop_roi(image: np.ndarray, roi: Roi) -> np.ndarray:
    x, y, w, h = clamp_roi(roi, image.shape)
    return image[y : y + h, x : x + w]


def clamp_roi(roi: Roi, shape: tuple[int, int]) -> Roi:
    height, width = shape[:2]
    x, y, w, h = [int(round(v)) for v in roi]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def parse_roi(text: str) -> Optional[Roi]:
    if not text.strip():
        return None
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,width,height.")
    return tuple(int(float(p)) for p in parts)  # type: ignore[return-value]


def auto_detect_roi(
    image: np.ndarray,
    *,
    orientation: str,
    margin_px: int = 30,
    min_area_fraction: float = 0.00005,
) -> RoiDetectionResult:
    """Detect a bright or dark slit-like signal and return an expanded ROI."""
    if image.ndim != 2:
        raise ValueError("auto_detect_roi expects a 2D grayscale image.")

    smoothed = gaussian_filter(image.astype(np.float64), sigma=1.5)
    median = float(np.median(smoothed))
    high_delta = float(np.percentile(smoothed, 99.5) - median)
    low_delta = float(median - np.percentile(smoothed, 0.5))
    polarity = "bright" if high_delta >= low_delta else "dark"
    signal = smoothed - median if polarity == "bright" else median - smoothed
    signal = np.maximum(signal, 0)

    if np.max(signal) <= 0:
        return _fallback_roi(image, orientation, "No positive signal detected.")

    threshold = max(float(np.percentile(signal, 99.0)) * 0.25, float(np.max(signal)) * 0.08)
    mask = (signal >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = image.shape[0] * image.shape[1] * min_area_fraction
    candidates: list[tuple[float, int]] = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue
        aspect = w / max(h, 1)
        if orientation == "horizontal":
            shape_score = min(aspect / 3.0, 3.0)
        else:
            shape_score = min((1.0 / max(aspect, 1e-6)) / 3.0, 3.0)
        score = float(area) * (1.0 + shape_score)
        candidates.append((score, idx))

    if not candidates:
        return _fallback_roi(image, orientation, "No slit-like component found.")

    _, best_idx = max(candidates, key=lambda item: item[0])
    x, y, w, h, area = [int(v) for v in stats[best_idx]]
    rx = x - margin_px
    ry = y - margin_px
    rw = w + margin_px * 2
    rh = h + margin_px * 2

    if orientation == "horizontal":
        rw = max(rw, int(image.shape[1] * 0.5))
        rx = min(max(0, x + w // 2 - rw // 2), image.shape[1] - 1)
    else:
        rh = max(rh, int(image.shape[0] * 0.5))
        ry = min(max(0, y + h // 2 - rh // 2), image.shape[0] - 1)

    roi = clamp_roi((rx, ry, rw, rh), image.shape)
    confidence = min(1.0, float(area) / max(min_area * 20.0, 1.0))
    return RoiDetectionResult(
        roi=roi,
        signal_bbox=clamp_roi((x, y, w, h), image.shape),
        polarity=polarity,
        confidence=confidence,
        message="Auto ROI detected.",
    )


def _fallback_roi(image: np.ndarray, orientation: str, message: str) -> RoiDetectionResult:
    height, width = image.shape[:2]
    if orientation == "horizontal":
        roi = (0, height // 4, width, height // 2)
    else:
        roi = (width // 4, 0, width // 2, height)
    return RoiDetectionResult(
        roi=clamp_roi(roi, image.shape),
        signal_bbox=(0, 0, width, height),
        polarity="bright",
        confidence=0.0,
        message=message,
    )
