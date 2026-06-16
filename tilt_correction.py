from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from preprocessing import rotate_image


@dataclass
class TiltResult:
    angle_deg: float
    correction_angle_deg: float
    corrected_image: np.ndarray
    line_angle_deg: float
    confidence: float
    message: str


def detect_and_correct_tilt(
    roi_image: np.ndarray,
    *,
    orientation: str,
    apply_correction: bool = True,
) -> TiltResult:
    line_angle, confidence, message = detect_tilt_angle(roi_image, orientation=orientation)
    target = 0.0 if orientation == "horizontal" else 90.0
    tilt = _signed_angle_difference(line_angle, target)
    correction = tilt
    corrected = rotate_image(roi_image, correction) if apply_correction else roi_image.copy()
    return TiltResult(
        angle_deg=tilt,
        correction_angle_deg=correction if apply_correction else 0.0,
        corrected_image=corrected,
        line_angle_deg=line_angle,
        confidence=confidence,
        message=message,
    )


def detect_tilt_angle(roi_image: np.ndarray, *, orientation: str) -> tuple[float, float, str]:
    if roi_image.ndim != 2:
        raise ValueError("detect_tilt_angle expects a 2D grayscale image.")

    smoothed = gaussian_filter(roi_image.astype(np.float64), sigma=1.2)
    median = float(np.median(smoothed))
    bright = smoothed - median
    dark = median - smoothed
    signal = bright if np.max(bright) >= np.max(dark) else dark
    signal = np.maximum(signal, 0)
    peak = float(np.max(signal))
    if peak <= 0:
        return (0.0 if orientation == "horizontal" else 90.0), 0.0, "No signal for tilt detection."

    mask = signal >= peak * 0.20
    y_coords, x_coords = np.nonzero(mask)
    if len(x_coords) < 10:
        return (0.0 if orientation == "horizontal" else 90.0), 0.0, "Too few signal pixels for tilt detection."

    weights = signal[y_coords, x_coords]
    weights = weights / np.sum(weights)
    x_mean = float(np.sum(x_coords * weights))
    y_mean = float(np.sum(y_coords * weights))
    x_centered = x_coords - x_mean
    y_centered = y_coords - y_mean
    cov_xx = float(np.sum(weights * x_centered * x_centered))
    cov_xy = float(np.sum(weights * x_centered * y_centered))
    cov_yy = float(np.sum(weights * y_centered * y_centered))
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvec = eigvecs[:, order[0]]
    line_angle = float(np.degrees(np.arctan2(eigvec[1], eigvec[0])))
    line_angle = _axis_angle_0_180(line_angle)
    confidence = float((eigvals[0] - eigvals[1]) / max(eigvals[0], 1e-12))
    return line_angle, confidence, "Tilt estimated by weighted PCA."


def _axis_angle_0_180(angle: float) -> float:
    angle = angle % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def _signed_angle_difference(angle: float, target: float) -> float:
    return ((angle - target + 90.0) % 180.0) - 90.0
