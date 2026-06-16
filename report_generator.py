from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from config import MeasurementConfig
from iec_check import QualitySummary
from lsf_analysis import LsfResult
from roi_detection import Roi


@dataclass
class ImageAnalysisReport:
    label: str
    image_path: str
    original_image: np.ndarray
    roi: Roi
    roi_image: np.ndarray
    corrected_image: np.ndarray
    tilt_angle_deg: float
    lsf: LsfResult
    quality: QualitySummary
    roi_message: str
    length_trim_crop: Roi
    length_trim_applied: bool
    length_trim_used_count: int
    length_trim_total_count: int
    length_trim_used_fraction: float
    length_trim_message: str


def generate_reports(
    analyses: Iterable[ImageAnalysisReport],
    *,
    config: MeasurementConfig,
    output_dir: str | Path,
    input_files: Mapping[str, str | None] | None = None,
    base_name: str = "iec60336_focal_spot_report",
) -> tuple[Path, Path, list[Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analyses = list(analyses)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = output_dir / f"{base_name}_{timestamp}.pdf"
    csv_path = output_dir / f"{base_name}_{timestamp}.csv"
    png_paths: list[Path] = []

    with PdfPages(pdf_path) as pdf:
        fig = _summary_page(analyses, config, input_files=input_files)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for analysis in analyses:
            fig = _analysis_page(analysis, config)
            png_path = output_dir / f"{base_name}_{timestamp}_{analysis.label.lower()}.png"
            fig.savefig(png_path, dpi=160, bbox_inches="tight")
            png_paths.append(png_path)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    _write_csv(csv_path, analyses, config, input_files=input_files)
    return pdf_path, csv_path, png_paths


def _summary_page(
    analyses: list[ImageAnalysisReport],
    config: MeasurementConfig,
    *,
    input_files: Mapping[str, str | None] | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(111)
    ax.axis("off")
    footer_text = (
        "This report states that the analysis is based on IEC 60336 method; "
        "it does not imply official IEC certification."
    )
    y = 0.985
    body_bottom = 0.075
    lines = [
        ("IEC 60336 Focal Spot Measurement Report", 14, "bold"),
        (f"Measurement date/time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 8.5, "normal"),
        (
            "The focal spot dimensions were evaluated using the slit-camera method based on IEC 60336.",
            8.5,
            "normal",
        ),
        (
            "Warning: saturation-free image required for reliable IEC 60336-based analysis.",
            8.5,
            "bold",
        ),
        ("", 7, "normal"),
        ("Input files", 10, "bold"),
        *_input_file_lines(analyses, input_files),
        ("", 7, "normal"),
        ("Input conditions", 10, "bold"),
        (f"Detector pixel size: {config.detector_pixel_size_mm:.6g} mm", 8.3, "normal"),
        (
            f"Focal spot to slit distance: {config.focal_spot_to_slit_distance_mm:.6g} mm",
            8.3,
            "normal",
        ),
        (
            f"Slit to detector distance: {config.slit_to_detector_distance_mm:.6g} mm",
            8.3,
            "normal",
        ),
        (f"Magnification: {config.magnification:.6g}x", 8.3, "normal"),
        (
            f"Effective pixel size at focal spot plane: {config.effective_pixel_size_mm:.6g} mm",
            8.3,
            "normal",
        ),
        (f"Slit width: {config.slit_width_mm:.6g} mm", 8.3, "normal"),
        (f"Threshold level: {config.threshold_level * 100:.1f}%", 8.3, "normal"),
        (f"Projection method: {config.projection_method}", 8.3, "normal"),
        (
            f"Auto trim non-slit length in ROI: {'enabled' if config.auto_exclude_non_slit_area else 'disabled'}",
            8.3,
            "normal",
        ),
        (f"Detector bit depth: {config.detector_bit_depth} bit", 8.3, "normal"),
        (
            f"Saturation warning level: {config.saturation_warning_fraction * 100:.1f}% FS",
            8.3,
            "normal",
        ),
        ("", 7, "normal"),
        ("Additional measurement information", 10, "bold"),
        (
            f"Nominal focal spot size: {_fmt_optional(config.nominal_focal_spot_size_mm, 'mm')}",
            8.3,
            "normal",
        ),
        (f"Tube voltage: {_fmt_optional(config.tube_voltage_kv, 'kV')}", 8.3, "normal"),
        (f"Lens voltage: {_fmt_optional(config.lens_voltage_kv, 'kV')}", 8.3, "normal"),
        (f"Tube current: {_fmt_optional(config.tube_current_ma, 'mA')}", 8.3, "normal"),
        (f"Exposure time: {_fmt_optional(config.exposure_time_ms, 'ms')}", 8.3, "normal"),
        ("", 7, "normal"),
        ("Final results", 10, "bold"),
    ]
    for text, size, weight in lines:
        ax.text(0.04, y, text, fontsize=size, fontweight=weight, va="top", family="DejaVu Sans")
        y -= 0.020 if text else 0.010

    for analysis in analyses:
        lsf = analysis.lsf
        ax.text(
            0.04,
            y,
            f"{analysis.label}: focal spot size {_fmt_number(lsf.focal_spot_size_mm)} mm, "
            f"detector width {_fmt_number(lsf.measured_width_detector_mm)} mm, "
            f"tilt {analysis.tilt_angle_deg:+.3f} deg, quality {analysis.quality.overall_status}",
            fontsize=8.2,
            va="top",
            family="DejaVu Sans",
        )
        y -= 0.020

    y -= 0.006
    ax.text(0.04, y, "IEC condition check results", fontsize=10, fontweight="bold", va="top")
    checks_top = y - 0.022
    column_x = [0.04, 0.52]
    for idx, analysis in enumerate(analyses):
        x = column_x[idx % len(column_x)]
        column_y = checks_top
        ax.text(x, column_y, analysis.label, fontsize=8.5, fontweight="bold", va="top")
        column_y -= 0.017
        for item in analysis.quality.items:
            if column_y < body_bottom:
                break
            ax.text(
                x + 0.015,
                column_y,
                f"{item.name}: {item.status} | {item.value}",
                fontsize=6.4,
                va="top",
                family="DejaVu Sans",
            )
            column_y -= 0.015
        if analysis.quality.recommended_mas is not None:
            if column_y < body_bottom:
                break
            ax.text(
                x + 0.015,
                column_y,
                f"Exposure recommendation: {analysis.quality.recommended_mas:.4g} mAs",
                fontsize=6.4,
                va="top",
                family="DejaVu Sans",
            )

    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            0.058,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="none",
            zorder=2,
        )
    )
    ax.text(
        0.04,
        0.022,
        footer_text,
        fontsize=7.0,
        va="bottom",
        family="DejaVu Sans",
        zorder=3,
    )
    return fig


def _analysis_page(analysis: ImageAnalysisReport, config: MeasurementConfig) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle(f"{analysis.label} Image Analysis", fontsize=14, fontweight="bold", y=0.975)
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.875,
        bottom=0.105,
        wspace=0.20,
        hspace=0.42,
    )

    ax = axes[0, 0]
    ax.imshow(analysis.original_image, cmap="gray")
    x, y, w, h = analysis.roi
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="yellow", linewidth=1.5))
    ax.set_title("Original image + ROI", fontsize=12, pad=8)
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(analysis.roi_image, cmap="gray")
    ax.set_title(f"ROI crop ({analysis.roi_message})", fontsize=12, pad=8)
    ax.axis("off")

    ax = axes[1, 0]
    ax.imshow(analysis.corrected_image, cmap="gray")
    tx, ty, tw, th = analysis.length_trim_crop
    ax.add_patch(
        Rectangle(
            (tx, ty),
            tw,
            th,
            fill=False,
            edgecolor="cyan",
            linewidth=1.8,
            linestyle="--",
        )
    )
    trim_title = "Tilt corrected image + calculation region"
    ax.set_title(f"{trim_title}\nTilt {analysis.tilt_angle_deg:+.3f} deg", fontsize=11, pad=8)
    ax.axis("off")

    ax = axes[1, 1]
    lsf = analysis.lsf
    ax.plot(lsf.positions_px, lsf.profile, color="tab:blue", label="LSF")
    ax.axhline(lsf.threshold_value, color="tab:red", linestyle="--", label=f"{config.threshold_level * 100:.1f}%")
    ax.axvline(lsf.peak_index, color="tab:green", linestyle=":", label="Peak")
    if lsf.left_crossing_px is not None:
        ax.axvline(lsf.left_crossing_px, color="tab:orange", linestyle="-.", label="Left crossing")
    if lsf.right_crossing_px is not None:
        ax.axvline(lsf.right_crossing_px, color="tab:purple", linestyle="-.", label="Right crossing")
    if lsf.measured_width_px is not None:
        ax.annotate(
            f"{lsf.measured_width_detector_mm:.4g} mm detector\n{lsf.focal_spot_size_mm:.4g} mm focal plane",
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75},
        )
    projection_note = (
        f"Proj: {lsf.projection_method} | length used {analysis.length_trim_used_fraction * 100:.1f}%"
    )
    ax.annotate(
        projection_note,
        xy=(0.0, -0.20),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7.5,
        annotation_clip=False,
    )
    ax.set_title("LSF profile and 15% crossings", fontsize=12, pad=8)
    ax.set_xlabel("Profile position [pixel]")
    ax.set_ylabel("Background-subtracted signal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    fig.text(
        0.02,
        0.018,
        f"Quality: {analysis.quality.overall_status} | SNR: "
        f"{'inf' if np.isinf(lsf.snr) else f'{lsf.snr:.1f}'} | "
        f"Peak: {analysis.quality.peak_fraction * 100:.1f}% FS | "
        "saturation-free image required for reliable IEC 60336-based analysis",
        fontsize=9,
    )
    return fig


def _write_csv(
    path: Path,
    analyses: list[ImageAnalysisReport],
    config: MeasurementConfig,
    *,
    input_files: Mapping[str, str | None] | None = None,
) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "item", "value"])
        for label, value in _normalized_input_files(analyses, input_files).items():
            writer.writerow(["input_file", label, value])
        writer.writerow(["input", "detector_pixel_size_mm", config.detector_pixel_size_mm])
        writer.writerow(["input", "focal_spot_to_slit_distance_mm", config.focal_spot_to_slit_distance_mm])
        writer.writerow(["input", "slit_to_detector_distance_mm", config.slit_to_detector_distance_mm])
        writer.writerow(["input", "magnification", config.magnification])
        writer.writerow(["input", "effective_pixel_size_mm", config.effective_pixel_size_mm])
        writer.writerow(["input", "slit_width_mm", config.slit_width_mm])
        writer.writerow(["input", "threshold_level", config.threshold_level])
        writer.writerow(["input", "projection_method", config.projection_method])
        writer.writerow(["input", "auto_trim_non_slit_length", config.auto_exclude_non_slit_area])
        writer.writerow(["input", "detector_bit_depth", config.detector_bit_depth])
        writer.writerow(["input", "tube_voltage_kv", config.tube_voltage_kv])
        writer.writerow(["input", "lens_voltage_kv", config.lens_voltage_kv])
        writer.writerow(["input", "tube_current_ma", config.tube_current_ma])
        writer.writerow(["input", "exposure_time_ms", config.exposure_time_ms])
        for analysis in analyses:
            prefix = analysis.label.lower()
            lsf = analysis.lsf
            writer.writerow([prefix, "image_path", analysis.image_path])
            writer.writerow([prefix, "roi_x_y_w_h", analysis.roi])
            writer.writerow([prefix, "tilt_angle_deg", analysis.tilt_angle_deg])
            writer.writerow([prefix, "measured_width_px", lsf.measured_width_px])
            writer.writerow([prefix, "measured_width_detector_mm", lsf.measured_width_detector_mm])
            writer.writerow([prefix, "focal_spot_size_mm", lsf.focal_spot_size_mm])
            writer.writerow([prefix, "projection_method", lsf.projection_method])
            writer.writerow([prefix, "length_trim_crop_x_y_w_h", analysis.length_trim_crop])
            writer.writerow([prefix, "length_trim_applied", analysis.length_trim_applied])
            writer.writerow([prefix, "length_trim_used_count", analysis.length_trim_used_count])
            writer.writerow([prefix, "length_trim_total_count", analysis.length_trim_total_count])
            writer.writerow([prefix, "length_trim_used_fraction", analysis.length_trim_used_fraction])
            writer.writerow([prefix, "length_trim_message", analysis.length_trim_message])
            writer.writerow([prefix, "snr", lsf.snr])
            writer.writerow([prefix, "peak_fraction_full_scale", analysis.quality.peak_fraction])
            writer.writerow([prefix, "max_fraction_full_scale", analysis.quality.max_fraction])
            writer.writerow([prefix, "overall_quality", analysis.quality.overall_status])
            writer.writerow([prefix, "recommended_mas", analysis.quality.recommended_mas])
            for item in analysis.quality.items:
                writer.writerow([prefix, f"check_{item.name}", f"{item.status}: {item.value} - {item.message}"])


def _fmt_optional(value: float | None, unit: str) -> str:
    if value is None:
        return "not entered"
    return f"{value:.6g} {unit}"


def _fmt_number(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.6g}"


def _input_file_lines(
    analyses: list[ImageAnalysisReport],
    input_files: Mapping[str, str | None] | None,
) -> list[tuple[str, float, str]]:
    return [
        (f"{label}: {_short_path(value)}", 7.2, "normal")
        for label, value in _normalized_input_files(analyses, input_files).items()
    ]


def _normalized_input_files(
    analyses: list[ImageAnalysisReport],
    input_files: Mapping[str, str | None] | None,
) -> dict[str, str]:
    files: dict[str, str] = {analysis.label: analysis.image_path for analysis in analyses}
    if input_files:
        for key, value in input_files.items():
            key = str(key)
            if key in files:
                continue
            files[key] = str(value) if value else "not used"
    return files


def _short_path(value: str) -> str:
    if value == "not used":
        return value
    path = Path(value)
    text = f"{path.parent.name}/{path.name}" if path.parent.name else path.name
    if len(text) <= 88:
        return text
    return f"...{text[-85:]}"
