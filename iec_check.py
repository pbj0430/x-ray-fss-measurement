from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import MeasurementConfig
from lsf_analysis import LsfResult


@dataclass
class CheckItem:
    name: str
    status: str
    value: str
    message: str


@dataclass
class QualitySummary:
    items: list[CheckItem]
    overall_status: str
    peak_fraction: float
    max_fraction: float
    recommended_mas: Optional[float]


def build_quality_summary(
    image: np.ndarray,
    lsf: LsfResult,
    *,
    tilt_angle_deg: float,
    roi_shape: tuple[int, int],
    config: MeasurementConfig,
) -> QualitySummary:
    full_scale = float(config.full_scale)
    image_max = float(np.max(image))
    max_fraction = image_max / full_scale if full_scale > 0 else float("nan")
    peak_detector_value = lsf.peak_value + lsf.background
    peak_fraction = peak_detector_value / full_scale if full_scale > 0 else float("nan")

    items: list[CheckItem] = []
    saturated_pixels = int(np.count_nonzero(image >= full_scale))
    if saturated_pixels > 0:
        items.append(
            CheckItem(
                "Saturation",
                "FAIL",
                f"{saturated_pixels} pixels >= full scale",
                "Saturated pixels invalidate reliable focal spot measurement; reacquire image.",
            )
        )
    elif max_fraction >= config.saturation_warning_fraction:
        items.append(
            CheckItem(
                "Saturation",
                "WARNING",
                f"max {max_fraction * 100:.1f}% FS",
                "Peak is close to the configured saturation warning level.",
            )
        )
    else:
        items.append(
            CheckItem(
                "Saturation",
                "PASS",
                f"max {max_fraction * 100:.1f}% FS",
                "No saturated pixels detected.",
            )
        )

    mag_error = abs(config.magnification - config.recommended_magnification)
    if mag_error <= config.magnification_tolerance:
        mag_status = "PASS"
        mag_message = "Magnification is within the configured IEC-based acceptance band."
    else:
        mag_status = "WARNING"
        mag_message = "Magnification is outside the configured IEC-based acceptance band."
    items.append(
        CheckItem(
            "Magnification",
            mag_status,
            f"{config.magnification:.3f}x",
            mag_message,
        )
    )

    if abs(tilt_angle_deg) <= config.tilt_warning_deg:
        tilt_status = "PASS"
        tilt_message = "Tilt is within configured tolerance."
    else:
        tilt_status = "WARNING"
        tilt_message = "Tilt exceeds configured tolerance; review alignment and correction."
    items.append(
        CheckItem(
            "Tilt angle",
            tilt_status,
            f"{tilt_angle_deg:+.3f} deg",
            tilt_message,
        )
    )

    if lsf.measured_width_px is None:
        sampling_status = "FAIL"
        sampling_value = "n/a"
        sampling_message = "Sampling cannot be evaluated because crossings were not found."
    elif lsf.measured_width_px >= config.min_sampling_pixels:
        sampling_status = "PASS"
        sampling_value = f"{lsf.measured_width_px:.2f} px"
        sampling_message = "Measured 15% width spans enough detector pixels."
    else:
        sampling_status = "WARNING"
        sampling_value = f"{lsf.measured_width_px:.2f} px"
        sampling_message = "Measured width is sampled by few pixels; uncertainty may be high."
    items.append(CheckItem("Pixel sampling", sampling_status, sampling_value, sampling_message))

    if lsf.crossing_ok:
        left_margin = float(lsf.left_crossing_px or 0.0)
        right_margin = float((roi_shape[1] if lsf.orientation == "vertical" else roi_shape[0]) - (lsf.right_crossing_px or 0.0))
        if left_margin >= 2 and right_margin >= 2:
            roi_status = "PASS"
            roi_message = "Signal and crossings are inside the ROI."
        else:
            roi_status = "WARNING"
            roi_message = "15% crossing is close to the ROI boundary."
        roi_value = f"margins {left_margin:.1f}px / {right_margin:.1f}px"
    else:
        roi_status = "FAIL"
        roi_value = "crossing missing"
        roi_message = "Signal is not sufficiently contained or not measurable in ROI."
    items.append(CheckItem("ROI signal containment", roi_status, roi_value, roi_message))

    if lsf.snr >= config.min_snr:
        snr_status = "PASS"
        snr_message = "SNR is sufficient."
    elif lsf.snr >= config.warning_snr:
        snr_status = "WARNING"
        snr_message = "SNR is marginal; consider reacquisition or more averaging."
    else:
        snr_status = "FAIL"
        snr_message = "SNR is too low for reliable 15% crossing detection."
    snr_value = "inf" if np.isinf(lsf.snr) else f"{lsf.snr:.1f}"
    items.append(CheckItem("SNR", snr_status, snr_value, snr_message))

    items.append(
        CheckItem(
            "15% crossing",
            "PASS" if lsf.crossing_ok else "FAIL",
            "detected" if lsf.crossing_ok else "missing",
            "Left and right 15% crossing points were evaluated."
            if lsf.crossing_ok
            else "Reacquire or adjust ROI; crossings must be visible on both sides.",
        )
    )

    if peak_fraction < 0.10:
        brightness_status = "WARNING"
        brightness_message = "Image is dark; increasing exposure may improve SNR."
    elif peak_fraction >= config.saturation_warning_fraction:
        brightness_status = "WARNING"
        brightness_message = "Image is bright; reduce exposure to avoid saturation."
    else:
        brightness_status = "PASS"
        brightness_message = "Peak signal is inside the configured operating range."
    items.append(
        CheckItem(
            "Peak signal range",
            brightness_status,
            f"{peak_fraction * 100:.1f}% FS",
            brightness_message,
        )
    )

    recommended_mas = exposure_recommendation(config, peak_fraction)
    overall = _overall_status(items)
    return QualitySummary(
        items=items,
        overall_status=overall,
        peak_fraction=peak_fraction,
        max_fraction=max_fraction,
        recommended_mas=recommended_mas,
    )


def exposure_recommendation(
    config: MeasurementConfig, current_peak_fraction: float
) -> Optional[float]:
    current_mas = config.current_mas
    if current_mas is None or current_peak_fraction <= 0 or not np.isfinite(current_peak_fraction):
        return None
    return current_mas * config.target_peak_fraction / current_peak_fraction


def _overall_status(items: list[CheckItem]) -> str:
    if any(item.status == "FAIL" for item in items):
        return "FAIL"
    if any(item.status == "WARNING" for item in items):
        return "WARNING"
    return "PASS"
