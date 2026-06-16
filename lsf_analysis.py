from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from config import MeasurementConfig


@dataclass
class LsfResult:
    orientation: str
    raw_profile: np.ndarray
    profile: np.ndarray
    positions_px: np.ndarray
    background: float
    polarity: str
    peak_value: float
    peak_index: int
    threshold_value: float
    left_crossing_px: Optional[float]
    right_crossing_px: Optional[float]
    measured_width_px: Optional[float]
    measured_width_detector_mm: Optional[float]
    focal_spot_size_mm: Optional[float]
    effective_pixel_size_mm: float
    snr: float
    crossing_ok: bool
    projection_method: str
    auto_exclude_non_slit_area: bool
    projection_used_fraction: float
    projection_used_count: int
    projection_total_count: int
    projection_mask: np.ndarray
    warnings: list[str] = field(default_factory=list)


@dataclass
class SlitLengthTrimResult:
    image: np.ndarray
    crop: tuple[int, int, int, int]
    applied: bool
    used_count: int
    total_count: int
    used_fraction: float
    message: str


def analyze_lsf(
    roi_image: np.ndarray,
    *,
    orientation: str,
    config: MeasurementConfig,
) -> LsfResult:
    if roi_image.ndim != 2:
        raise ValueError("analyze_lsf expects a 2D grayscale image.")

    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be 'horizontal' or 'vertical'.")

    raw_profile, projection_mask, projection_warnings = _project_lsf(
        roi_image,
        orientation=orientation,
        config=config,
    )
    positions = np.arange(raw_profile.size, dtype=np.float64)
    edge_n = max(3, int(raw_profile.size * 0.10))
    edges = np.concatenate([raw_profile[:edge_n], raw_profile[-edge_n:]])
    background = float(np.median(edges))
    bright_amp = float(np.max(raw_profile) - background)
    dark_amp = float(background - np.min(raw_profile))
    if dark_amp > bright_amp:
        polarity = "dark"
        profile = background - raw_profile
    else:
        polarity = "bright"
        profile = raw_profile - background

    if config.smoothing_sigma_px > 0:
        profile = gaussian_filter1d(profile.astype(np.float64), sigma=config.smoothing_sigma_px)
    else:
        profile = profile.astype(np.float64)

    profile = profile - min(float(np.percentile(profile, 1.0)), 0.0)
    peak_index = int(np.argmax(profile))
    peak_value = float(profile[peak_index])
    threshold_value = float(peak_value * config.threshold_level)
    noise = float(np.std(_edge_signal(profile, edge_n)))
    snr = float(peak_value / noise) if noise > 1e-12 else float("inf")

    warnings: list[str] = list(projection_warnings)
    left = _find_left_crossing(profile, peak_index, threshold_value)
    right = _find_right_crossing(profile, peak_index, threshold_value)
    crossing_ok = left is not None and right is not None and right > left

    measured_width_px = None
    measured_width_detector_mm = None
    focal_spot_size_mm = None
    if crossing_ok:
        measured_width_px = float(right - left)
        measured_width_detector_mm = measured_width_px * config.detector_pixel_size_mm
        focal_spot_size_mm = measured_width_detector_mm / config.magnification
        if config.apply_slit_width_correction:
            focal_spot_size_mm = max(0.0, focal_spot_size_mm - config.slit_width_mm)
    else:
        warnings.append("15% crossing point could not be detected on both sides.")

    if peak_value <= 0:
        warnings.append("No positive LSF peak was detected after background subtraction.")
    if not np.isfinite(snr) or snr < config.warning_snr:
        warnings.append("SNR is low.")

    return LsfResult(
        orientation=orientation,
        raw_profile=raw_profile,
        profile=profile,
        positions_px=positions,
        background=background,
        polarity=polarity,
        peak_value=peak_value,
        peak_index=peak_index,
        threshold_value=threshold_value,
        left_crossing_px=left,
        right_crossing_px=right,
        measured_width_px=measured_width_px,
        measured_width_detector_mm=measured_width_detector_mm,
        focal_spot_size_mm=focal_spot_size_mm,
        effective_pixel_size_mm=config.effective_pixel_size_mm,
        snr=snr,
        crossing_ok=crossing_ok,
        projection_method=config.projection_method,
        auto_exclude_non_slit_area=config.auto_exclude_non_slit_area,
        projection_used_fraction=float(np.mean(projection_mask)) if projection_mask.size else 1.0,
        projection_used_count=int(np.count_nonzero(projection_mask)),
        projection_total_count=int(projection_mask.size),
        projection_mask=projection_mask,
        warnings=warnings,
    )


def _project_lsf(
    roi_image: np.ndarray,
    *,
    orientation: str,
    config: MeasurementConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    if orientation == "horizontal":
        total_count = roi_image.shape[1]
    else:
        total_count = roi_image.shape[0]

    mask = np.ones(total_count, dtype=bool)

    if orientation == "horizontal":
        samples = roi_image
        axis = 1
    else:
        samples = roi_image
        axis = 0

    method = config.projection_method.strip().lower().replace("-", "_").replace(" ", "_")
    if method == "mean":
        profile = np.mean(samples, axis=axis)
    elif method == "median":
        profile = np.median(samples, axis=axis)
    elif method == "sum":
        profile = np.sum(samples, axis=axis)
    elif method in {"trimmed_mean", "trim"}:
        profile = _trimmed_mean(samples, axis=axis, trim_fraction=0.10)
    else:
        warnings.append(f"Unknown projection method '{config.projection_method}'; mean projection was used.")
        profile = np.mean(samples, axis=axis)

    return profile.astype(np.float64), mask, warnings


def trim_slit_length_direction(
    roi_image: np.ndarray,
    *,
    orientation: str,
    config: MeasurementConfig,
) -> SlitLengthTrimResult:
    if roi_image.ndim != 2:
        raise ValueError("trim_slit_length_direction expects a 2D grayscale image.")
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be 'horizontal' or 'vertical'.")

    height, width = roi_image.shape
    total_count = width if orientation == "horizontal" else height
    full_crop = (0, 0, width, height)
    if not config.auto_exclude_non_slit_area:
        return SlitLengthTrimResult(
            image=roi_image,
            crop=full_crop,
            applied=False,
            used_count=total_count,
            total_count=total_count,
            used_fraction=1.0,
            message="Length trim disabled.",
        )

    mask, message = _detect_slit_presence_mask(
        roi_image,
        orientation=orientation,
        threshold_fraction=config.slit_presence_threshold_fraction,
        min_fraction=config.min_slit_presence_fraction,
    )
    if message or not np.any(mask):
        return SlitLengthTrimResult(
            image=roi_image,
            crop=full_crop,
            applied=False,
            used_count=total_count,
            total_count=total_count,
            used_fraction=1.0,
            message=message or "Slit length trim mask was empty; full ROI was used.",
        )

    indices = np.flatnonzero(mask)
    start = int(indices[0])
    stop = int(indices[-1]) + 1
    if orientation == "horizontal":
        trimmed = roi_image[:, start:stop]
        crop = (start, 0, stop - start, height)
    else:
        trimmed = roi_image[start:stop, :]
        crop = (0, start, width, stop - start)

    used_count = stop - start
    return SlitLengthTrimResult(
        image=trimmed,
        crop=crop,
        applied=used_count < total_count,
        used_count=used_count,
        total_count=total_count,
        used_fraction=used_count / max(total_count, 1),
        message="Non-slit length region was cropped before LSF projection."
        if used_count < total_count
        else "Detected slit length covers the full ROI.",
    )


def _detect_slit_presence_mask(
    roi_image: np.ndarray,
    *,
    orientation: str,
    threshold_fraction: float,
    min_fraction: float,
) -> tuple[np.ndarray, str]:
    smoothed = gaussian_filter(roi_image.astype(np.float64), sigma=1.0)
    median = float(np.median(smoothed))
    bright = smoothed - median
    dark = median - smoothed
    signal = bright if float(np.max(bright)) >= float(np.max(dark)) else dark
    signal = np.maximum(signal, 0.0)

    if orientation == "horizontal":
        along_signal = np.percentile(signal, 99.0, axis=0)
    else:
        along_signal = np.percentile(signal, 99.0, axis=1)

    max_signal = float(np.max(along_signal)) if along_signal.size else 0.0
    if max_signal <= 0:
        return np.ones_like(along_signal, dtype=bool), "No slit signal found for presence mask; full ROI was used."

    threshold = max_signal * max(0.0, min(float(threshold_fraction), 1.0))
    candidate = along_signal >= threshold
    candidate = _fill_small_gaps(candidate, max_gap=max(2, int(candidate.size * 0.01)))
    mask = _largest_true_run(candidate)
    min_count = max(3, int(candidate.size * max(0.0, min(float(min_fraction), 1.0))))

    if int(np.count_nonzero(mask)) < min_count:
        return np.ones_like(along_signal, dtype=bool), "Detected slit extent was too small; full ROI was used."

    return mask, ""


def _fill_small_gaps(mask: np.ndarray, *, max_gap: int) -> np.ndarray:
    result = mask.astype(bool).copy()
    true_indices = np.flatnonzero(result)
    if true_indices.size < 2:
        return result
    start = int(true_indices[0])
    prev = start
    for idx in true_indices[1:]:
        idx = int(idx)
        gap = idx - prev - 1
        if 0 < gap <= max_gap:
            result[prev + 1 : idx] = True
        prev = idx
    return result


def _largest_true_run(mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    best_start = None
    best_length = 0
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            length = idx - start
            if length > best_length:
                best_start = start
                best_length = length
            start = None
    if start is not None:
        length = len(mask) - start
        if length > best_length:
            best_start = start
            best_length = length
    if best_start is not None:
        result[best_start : best_start + best_length] = True
    return result


def _trimmed_mean(samples: np.ndarray, *, axis: int, trim_fraction: float) -> np.ndarray:
    if samples.shape[axis] < 3:
        return np.mean(samples, axis=axis)
    sorted_samples = np.sort(samples, axis=axis)
    trim = int(samples.shape[axis] * trim_fraction)
    if trim <= 0 or trim * 2 >= samples.shape[axis]:
        return np.mean(samples, axis=axis)
    if axis == 0:
        return np.mean(sorted_samples[trim:-trim, :], axis=0)
    return np.mean(sorted_samples[:, trim:-trim], axis=1)


def _edge_signal(profile: np.ndarray, edge_n: int) -> np.ndarray:
    return np.concatenate([profile[:edge_n], profile[-edge_n:]])


def _find_left_crossing(
    profile: np.ndarray, peak_index: int, threshold: float
) -> Optional[float]:
    if threshold <= 0:
        return None
    idx = peak_index
    while idx > 0 and profile[idx] >= threshold:
        idx -= 1
    if idx == 0 and profile[idx] >= threshold:
        return None
    return _linear_crossing(idx, idx + 1, profile, threshold)


def _find_right_crossing(
    profile: np.ndarray, peak_index: int, threshold: float
) -> Optional[float]:
    if threshold <= 0:
        return None
    idx = peak_index
    last = profile.size - 1
    while idx < last and profile[idx] >= threshold:
        idx += 1
    if idx == last and profile[idx] >= threshold:
        return None
    return _linear_crossing(idx - 1, idx, profile, threshold)


def _linear_crossing(i0: int, i1: int, profile: np.ndarray, threshold: float) -> float:
    y0 = float(profile[i0])
    y1 = float(profile[i1])
    if abs(y1 - y0) < 1e-12:
        return float(i0)
    fraction = (threshold - y0) / (y1 - y0)
    return float(i0 + fraction * (i1 - i0))
