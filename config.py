from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MeasurementConfig:
    detector_pixel_size_mm: float = 0.140
    focal_spot_to_slit_distance_mm: float = 100.0
    slit_to_detector_distance_mm: float = 400.0
    slit_width_mm: float = 0.010
    threshold_level: float = 0.15
    detector_bit_depth: int = 16
    saturation_warning_fraction: float = 0.80
    target_peak_fraction: float = 0.70
    nominal_focal_spot_size_mm: Optional[float] = None
    tube_voltage_kv: Optional[float] = None
    lens_voltage_kv: Optional[float] = None
    tube_current_ma: Optional[float] = None
    exposure_time_ms: Optional[float] = None
    smoothing_sigma_px: float = 1.0
    projection_method: str = "mean"
    auto_exclude_non_slit_area: bool = True
    slit_presence_threshold_fraction: float = 0.20
    min_slit_presence_fraction: float = 0.03
    auto_rotate: bool = True
    apply_slit_width_correction: bool = False
    min_snr: float = 20.0
    warning_snr: float = 10.0
    min_sampling_pixels: float = 5.0
    tilt_warning_deg: float = 2.0
    recommended_magnification: float = 4.0
    magnification_tolerance: float = 0.5

    @property
    def magnification(self) -> float:
        if self.focal_spot_to_slit_distance_mm <= 0:
            raise ValueError("Focal spot to slit distance must be greater than zero.")
        return self.slit_to_detector_distance_mm / self.focal_spot_to_slit_distance_mm

    @property
    def effective_pixel_size_mm(self) -> float:
        return self.detector_pixel_size_mm / self.magnification

    @property
    def full_scale(self) -> int:
        return (2**self.detector_bit_depth) - 1

    @property
    def current_mas(self) -> Optional[float]:
        if self.tube_current_ma is None or self.exposure_time_ms is None:
            return None
        return self.tube_current_ma * self.exposure_time_ms / 1000.0


DEFAULT_CONFIG = MeasurementConfig()
