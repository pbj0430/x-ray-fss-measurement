from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np

from config import MeasurementConfig
from iec_check import build_quality_summary
from image_loader import load_image, load_optional_image
from lsf_analysis import analyze_lsf, trim_slit_length_direction
from preprocessing import apply_corrections
from report_generator import ImageAnalysisReport, generate_reports
from roi_detection import Roi, auto_detect_roi, clamp_roi, crop_roi, parse_roi
from roi_viewer import ImagePickerDialog, RoiSelectionDialog
from tilt_correction import detect_and_correct_tilt


APP_NAME = "FSS_Measurement"


def _ui_state_path() -> Path:
    if getattr(sys, "frozen", False):
        if sys.platform.startswith("win"):
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return base / APP_NAME / ".fss_ui_state.json"
    return Path(__file__).with_name(".fss_ui_state.json")


UI_STATE_PATH = _ui_state_path()


@dataclass
class AnalysisInputs:
    horizontal_path: str
    vertical_path: str
    output_dir: str
    dark_path: Optional[str] = None
    offset_path: Optional[str] = None
    flat_path: Optional[str] = None
    horizontal_roi: Optional[Roi] = None
    vertical_roi: Optional[Roi] = None
    raw_width: Optional[int] = None
    raw_height: Optional[int] = None
    raw_dtype: str = "uint16"


def run_analysis(inputs: AnalysisInputs, config: MeasurementConfig) -> tuple[Path, Path, list[Path], list[ImageAnalysisReport]]:
    raw_shape = _raw_shape(inputs)
    dark = load_optional_image(inputs.dark_path, raw_shape=raw_shape, raw_dtype=inputs.raw_dtype)
    offset = load_optional_image(inputs.offset_path, raw_shape=raw_shape, raw_dtype=inputs.raw_dtype)
    flat = load_optional_image(inputs.flat_path, raw_shape=raw_shape, raw_dtype=inputs.raw_dtype)

    analyses: list[ImageAnalysisReport] = []
    for label, path, orientation, manual_roi in (
        ("Horizontal", inputs.horizontal_path, "horizontal", inputs.horizontal_roi),
        ("Vertical", inputs.vertical_path, "vertical", inputs.vertical_roi),
    ):
        image = load_image(path, raw_shape=raw_shape, raw_dtype=inputs.raw_dtype)
        corrected = apply_corrections(image, dark=dark, offset=offset, flat=flat)

        if manual_roi is None:
            roi_detection = auto_detect_roi(corrected, orientation=orientation)
            roi = roi_detection.roi
            roi_message = roi_detection.message
        else:
            roi = clamp_roi(manual_roi, corrected.shape)
            roi_message = "Manual ROI"

        roi_image = crop_roi(corrected, roi)
        tilt = detect_and_correct_tilt(
            roi_image,
            orientation=orientation,
            apply_correction=config.auto_rotate,
        )
        trim = trim_slit_length_direction(
            tilt.corrected_image,
            orientation=orientation,
            config=config,
        )
        lsf = analyze_lsf(trim.image, orientation=orientation, config=config)
        quality = build_quality_summary(
            corrected,
            lsf,
            tilt_angle_deg=tilt.angle_deg,
            roi_shape=trim.image.shape,
            config=config,
        )
        analyses.append(
            ImageAnalysisReport(
                label=label,
                image_path=path,
                original_image=corrected,
                roi=roi,
                roi_image=roi_image,
                corrected_image=tilt.corrected_image,
                tilt_angle_deg=tilt.angle_deg,
                lsf=lsf,
                quality=quality,
                roi_message=roi_message,
                length_trim_crop=trim.crop,
                length_trim_applied=trim.applied,
                length_trim_used_count=trim.used_count,
                length_trim_total_count=trim.total_count,
                length_trim_used_fraction=trim.used_fraction,
                length_trim_message=trim.message,
            )
        )

    pdf_path, csv_path, png_paths = generate_reports(
        analyses,
        config=config,
        output_dir=inputs.output_dir,
        input_files={
            "Dark image": inputs.dark_path,
            "Offset image": inputs.offset_path,
            "Flat image": inputs.flat_path,
        },
    )
    return pdf_path, csv_path, png_paths, analyses


def _raw_shape(inputs: AnalysisInputs) -> Optional[tuple[int, int]]:
    if inputs.raw_width is None and inputs.raw_height is None:
        return None
    if inputs.raw_width is None or inputs.raw_height is None:
        raise ValueError("Both RAW width and RAW height are required for RAW files.")
    return (inputs.raw_height, inputs.raw_width)


class FocalSpotApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("IEC 60336 Slit Camera Focal Spot Measurement")
        self.geometry("1040x840")
        self.minsize(940, 760)
        self.ui_state = _load_ui_state()
        self.last_image_folder = str(self.ui_state.get("last_image_folder", ""))
        self._build_ui()
        self._restore_ui_values()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)

        file_frame = ttk.LabelFrame(container, text="Image files")
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        file_frame.columnconfigure(1, weight=1)
        self.horizontal_roi = tk.StringVar()
        self.vertical_roi = tk.StringVar()
        self.horizontal_path = self._file_row(
            file_frame,
            0,
            "Horizontal slit image",
            orientation="horizontal",
            roi_variable=self.horizontal_roi,
        )
        self.vertical_path = self._file_row(
            file_frame,
            1,
            "Vertical slit image",
            orientation="vertical",
            roi_variable=self.vertical_roi,
        )
        self.dark_path = self._file_row(file_frame, 2, "Dark image (optional)")
        self.offset_path = self._file_row(file_frame, 3, "Offset image (optional)")
        self.flat_path = self._file_row(file_frame, 4, "Flat image (optional)")

        output_frame = ttk.Frame(file_frame)
        output_frame.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        output_frame.columnconfigure(1, weight=1)
        ttk.Label(output_frame, text="Output folder").grid(row=0, column=0, sticky="w")
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "reports"))
        ttk.Entry(output_frame, textvariable=self.output_dir).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(output_frame, text="Browse", command=self._browse_output).grid(row=0, column=2)

        params = ttk.LabelFrame(container, text="Measurement parameters")
        params.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for col in range(4):
            params.columnconfigure(col, weight=1)

        self.pixel_size = self._param(params, 0, 0, "Detector pixel size [mm]", "0.140")
        self.fs_to_slit = self._param(params, 0, 2, "Focal spot to slit [mm]", "100")
        self.slit_to_detector = self._param(params, 1, 0, "Slit to detector [mm]", "400")
        self.slit_width = self._param(params, 1, 2, "Slit width [mm]", "0.010")
        self.threshold = self._param(params, 2, 0, "Threshold level [%]", "15")
        self.bit_depth = self._param(params, 2, 2, "Detector bit depth", "16")
        self.sat_warning = self._param(params, 3, 0, "Saturation warning [% FS]", "80")
        self.target_peak = self._param(params, 3, 2, "Target peak [% FS]", "70")
        self.smoothing = self._param(params, 4, 0, "Smoothing sigma [px]", "1.0")
        ttk.Label(params, text="Projection method").grid(row=4, column=2, sticky="w", padx=6, pady=4)
        self.projection_method = tk.StringVar(value="mean")
        ttk.Combobox(
            params,
            textvariable=self.projection_method,
            values=("mean", "median", "trimmed_mean", "sum"),
            state="readonly",
            width=16,
        ).grid(row=4, column=3, sticky="ew", padx=6, pady=4)
        self.raw_width = self._param(params, 5, 0, "RAW width [px]", "3072")
        self.raw_height = self._param(params, 5, 2, "RAW height [px]", "3072")
        self.raw_dtype = self._param(params, 6, 0, "RAW dtype", "uint16")

        flags = ttk.Frame(params)
        flags.grid(row=6, column=2, columnspan=2, sticky="w", padx=6, pady=4)
        self.auto_rotate = tk.BooleanVar(value=True)
        self.slit_correction = tk.BooleanVar(value=False)
        self.auto_exclude_non_slit_area = tk.BooleanVar(value=True)
        ttk.Checkbutton(flags, text="Tilt rotation correction", variable=self.auto_rotate).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Checkbutton(flags, text="Subtract slit width from focal size", variable=self.slit_correction).pack(side=tk.LEFT)
        ttk.Checkbutton(
            params,
            text="Auto trim non-slit length in ROI",
            variable=self.auto_exclude_non_slit_area,
        ).grid(row=7, column=0, columnspan=2, sticky="w", padx=6, pady=4)

        metadata_frame = ttk.LabelFrame(container, text="Additional measurement information")
        metadata_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for col in range(4):
            metadata_frame.columnconfigure(col, weight=1)
        self.nominal_size = self._param(metadata_frame, 0, 0, "Nominal focal spot [mm]", "")
        self.kv = self._param(metadata_frame, 0, 2, "Tube voltage [kV]", "")
        self.lens_voltage = self._param(metadata_frame, 1, 0, "Lens voltage [kV]", "")
        self.ma = self._param(metadata_frame, 1, 2, "Tube current [mA]", "")
        self.exposure_ms = self._param(metadata_frame, 2, 0, "Exposure time [ms]", "")

        roi_frame = ttk.LabelFrame(container, text="ROI")
        roi_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        roi_frame.columnconfigure(1, weight=1)
        ttk.Label(roi_frame, text="Leave ROI blank for automatic detection. Manual format: x,y,width,height").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(6, 2)
        )
        ttk.Label(roi_frame, text="Horizontal ROI").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(roi_frame, textvariable=self.horizontal_roi).grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(
            roi_frame,
            text="Select on image",
            command=lambda: self._open_roi_selector(self.horizontal_path, self.horizontal_roi, "horizontal"),
        ).grid(row=1, column=2, sticky="e", padx=6, pady=3)
        ttk.Label(roi_frame, text="Vertical ROI").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(roi_frame, textvariable=self.vertical_roi).grid(row=2, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(
            roi_frame,
            text="Select on image",
            command=lambda: self._open_roi_selector(self.vertical_path, self.vertical_roi, "vertical"),
        ).grid(row=2, column=2, sticky="e", padx=6, pady=3)

        status_frame = ttk.Frame(container)
        status_frame.grid(row=4, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        self.status_text = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_text).grid(row=0, column=0, sticky="w")
        ttk.Button(status_frame, text="Run analysis and generate report", command=self._run_clicked).grid(
            row=0, column=1, sticky="e"
        )

    def _file_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        *,
        orientation: Optional[str] = None,
        roi_variable: Optional[tk.StringVar] = None,
    ) -> tk.StringVar:
        variable = tk.StringVar()
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(parent, text="Select...", command=lambda: self._browse_file(variable)).grid(
            row=row, column=2, padx=6, pady=3
        )
        if roi_variable is not None and orientation is not None:
            ttk.Button(
                parent,
                text="ROI...",
                command=lambda: self._open_roi_selector(variable, roi_variable, orientation),
            ).grid(row=row, column=3, padx=6, pady=3)
        return variable

    def _param(self, parent: ttk.Frame, row: int, col: int, label: str, default: str) -> tk.StringVar:
        variable = tk.StringVar(value=default)
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=col + 1, sticky="ew", padx=6, pady=4)
        return variable

    def _browse_file(self, variable: tk.StringVar) -> None:
        try:
            dialog = ImagePickerDialog(
                self,
                initial_path=variable.get().strip(),
                initial_folder=self.last_image_folder,
                raw_shape=self._current_raw_shape(),
                raw_dtype=self.raw_dtype.get().strip() or "uint16",
            )
            if dialog.last_folder:
                self._remember_image_folder(dialog.last_folder)
            if dialog.result:
                variable.set(dialog.result)
        except Exception as exc:
            messagebox.showerror("Image selection failed", str(exc), parent=self)

    def _remember_image_folder(self, folder: str) -> None:
        path = Path(folder)
        if not path.exists() or not path.is_dir():
            return
        self.last_image_folder = str(path)
        self.ui_state["last_image_folder"] = self.last_image_folder
        _save_ui_state(self.ui_state)

    def _open_roi_selector(
        self,
        path_variable: tk.StringVar,
        roi_variable: tk.StringVar,
        orientation: str,
    ) -> None:
        path = path_variable.get().strip()
        if not path:
            messagebox.showwarning("No image selected", "Select the image file first.", parent=self)
            return
        try:
            RoiSelectionDialog(
                self,
                path=path,
                roi_variable=roi_variable,
                orientation=orientation,
                raw_shape=self._current_raw_shape(),
                raw_dtype=self.raw_dtype.get().strip() or "uint16",
            )
        except Exception as exc:
            messagebox.showerror("ROI selection failed", str(exc), parent=self)

    def _current_raw_shape(self) -> Optional[tuple[int, int]]:
        width = _optional_int(self.raw_width.get())
        height = _optional_int(self.raw_height.get())
        if width is None and height is None:
            return None
        if width is None or height is None:
            raise ValueError("Both RAW width and RAW height are required for RAW preview.")
        return (height, width)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def _run_clicked(self) -> None:
        try:
            config = self._read_config()
            horizontal_path = self.horizontal_path.get().strip()
            vertical_path = self.vertical_path.get().strip()
            if not horizontal_path or not vertical_path:
                raise ValueError("Horizontal and vertical slit images are required.")
            inputs = AnalysisInputs(
                horizontal_path=horizontal_path,
                vertical_path=vertical_path,
                output_dir=self.output_dir.get().strip() or str(Path.cwd() / "reports"),
                dark_path=self.dark_path.get().strip() or None,
                offset_path=self.offset_path.get().strip() or None,
                flat_path=self.flat_path.get().strip() or None,
                horizontal_roi=parse_roi(self.horizontal_roi.get()),
                vertical_roi=parse_roi(self.vertical_roi.get()),
                raw_width=_optional_int(self.raw_width.get()),
                raw_height=_optional_int(self.raw_height.get()),
                raw_dtype=self.raw_dtype.get().strip() or "uint16",
            )
            self._save_current_ui_state()
            self.status_text.set("Running analysis...")
            self.update_idletasks()
            pdf_path, csv_path, png_paths, analyses = run_analysis(inputs, config)
            summary = "\n".join(
                f"{a.label}: {a.quality.overall_status}, focal spot "
                f"{_fmt(a.lsf.focal_spot_size_mm)} mm, tilt {a.tilt_angle_deg:+.3f} deg"
                for a in analyses
            )
            self.status_text.set(f"Done: {pdf_path}")
            open_error = _open_with_system_viewer(pdf_path)
            open_note = "" if open_error is None else f"\n\nPDF open failed:\n{open_error}"
            messagebox.showinfo(
                "Analysis complete",
                f"{summary}\n\nPDF report:\n{pdf_path}\n\nCSV:\n{csv_path}\n\nPNG figures:\n"
                + "\n".join(str(p) for p in png_paths)
                + open_note,
            )
        except Exception as exc:
            self.status_text.set("Error")
            messagebox.showerror("Analysis failed", str(exc))

    def _restore_ui_values(self) -> None:
        text_fields = {
            "horizontal_path": self.horizontal_path,
            "vertical_path": self.vertical_path,
            "dark_path": self.dark_path,
            "offset_path": self.offset_path,
            "flat_path": self.flat_path,
            "output_dir": self.output_dir,
            "detector_pixel_size_mm": self.pixel_size,
            "focal_spot_to_slit_distance_mm": self.fs_to_slit,
            "slit_to_detector_distance_mm": self.slit_to_detector,
            "slit_width_mm": self.slit_width,
            "threshold_percent": self.threshold,
            "detector_bit_depth": self.bit_depth,
            "saturation_warning_percent": self.sat_warning,
            "target_peak_percent": self.target_peak,
            "nominal_focal_spot_size_mm": self.nominal_size,
            "tube_voltage_kv": self.kv,
            "lens_voltage_kv": self.lens_voltage,
            "tube_current_ma": self.ma,
            "exposure_time_ms": self.exposure_ms,
            "smoothing_sigma_px": self.smoothing,
            "projection_method": self.projection_method,
            "raw_width": self.raw_width,
            "raw_height": self.raw_height,
            "raw_dtype": self.raw_dtype,
            "horizontal_roi": self.horizontal_roi,
            "vertical_roi": self.vertical_roi,
        }
        for key, variable in text_fields.items():
            if key in self.ui_state:
                variable.set(self.ui_state[key])

        if self.projection_method.get() not in {"mean", "median", "trimmed_mean", "sum"}:
            self.projection_method.set("mean")

        bool_fields = {
            "auto_rotate": self.auto_rotate,
            "slit_width_correction": self.slit_correction,
            "auto_trim_non_slit_length": self.auto_exclude_non_slit_area,
        }
        for key, variable in bool_fields.items():
            if key in self.ui_state:
                variable.set(_state_bool(self.ui_state[key], default=variable.get()))

    def _save_current_ui_state(self) -> None:
        self.ui_state.update(
            {
                "horizontal_path": self.horizontal_path.get(),
                "vertical_path": self.vertical_path.get(),
                "dark_path": self.dark_path.get(),
                "offset_path": self.offset_path.get(),
                "flat_path": self.flat_path.get(),
                "output_dir": self.output_dir.get(),
                "detector_pixel_size_mm": self.pixel_size.get(),
                "focal_spot_to_slit_distance_mm": self.fs_to_slit.get(),
                "slit_to_detector_distance_mm": self.slit_to_detector.get(),
                "slit_width_mm": self.slit_width.get(),
                "threshold_percent": self.threshold.get(),
                "detector_bit_depth": self.bit_depth.get(),
                "saturation_warning_percent": self.sat_warning.get(),
                "target_peak_percent": self.target_peak.get(),
                "nominal_focal_spot_size_mm": self.nominal_size.get(),
                "tube_voltage_kv": self.kv.get(),
                "lens_voltage_kv": self.lens_voltage.get(),
                "tube_current_ma": self.ma.get(),
                "exposure_time_ms": self.exposure_ms.get(),
                "smoothing_sigma_px": self.smoothing.get(),
                "projection_method": self.projection_method.get(),
                "raw_width": self.raw_width.get(),
                "raw_height": self.raw_height.get(),
                "raw_dtype": self.raw_dtype.get(),
                "horizontal_roi": self.horizontal_roi.get(),
                "vertical_roi": self.vertical_roi.get(),
                "auto_rotate": _state_bool_text(self.auto_rotate.get()),
                "slit_width_correction": _state_bool_text(self.slit_correction.get()),
                "auto_trim_non_slit_length": _state_bool_text(self.auto_exclude_non_slit_area.get()),
                "last_image_folder": self.last_image_folder,
            }
        )
        _save_ui_state(self.ui_state)

    def _on_close(self) -> None:
        self._save_current_ui_state()
        self.destroy()

    def _read_config(self) -> MeasurementConfig:
        return MeasurementConfig(
            detector_pixel_size_mm=_float(self.pixel_size.get(), "Detector pixel size"),
            focal_spot_to_slit_distance_mm=_float(self.fs_to_slit.get(), "Focal spot to slit distance"),
            slit_to_detector_distance_mm=_float(self.slit_to_detector.get(), "Slit to detector distance"),
            slit_width_mm=_float(self.slit_width.get(), "Slit width"),
            threshold_level=_float(self.threshold.get(), "Threshold level") / 100.0,
            detector_bit_depth=int(_float(self.bit_depth.get(), "Detector bit depth")),
            saturation_warning_fraction=_float(self.sat_warning.get(), "Saturation warning") / 100.0,
            target_peak_fraction=_float(self.target_peak.get(), "Target peak") / 100.0,
            nominal_focal_spot_size_mm=_optional_float(self.nominal_size.get()),
            tube_voltage_kv=_optional_float(self.kv.get()),
            lens_voltage_kv=_optional_float(self.lens_voltage.get()),
            tube_current_ma=_optional_float(self.ma.get()),
            exposure_time_ms=_optional_float(self.exposure_ms.get()),
            smoothing_sigma_px=_float(self.smoothing.get(), "Smoothing sigma"),
            projection_method=self.projection_method.get().strip() or "mean",
            auto_exclude_non_slit_area=self.auto_exclude_non_slit_area.get(),
            auto_rotate=self.auto_rotate.get(),
            apply_slit_width_correction=self.slit_correction.get(),
        )


def _float(text: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _optional_float(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    return _float(text, "Optional parameter")


def _optional_int(text: str) -> Optional[int]:
    text = text.strip()
    if not text:
        return None
    value = _float(text, "Optional integer parameter")
    if int(value) != value:
        raise ValueError("Optional integer parameter must be an integer.")
    return int(value)


def _fmt(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.4g}"


def _open_with_system_viewer(path: Path) -> Optional[str]:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        return str(exc)
    return None


def _load_ui_state() -> dict[str, str]:
    if not UI_STATE_PATH.exists():
        return {}
    try:
        with UI_STATE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _save_ui_state(state: dict[str, str]) -> None:
    try:
        UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with UI_STATE_PATH.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _state_bool(value: str, *, default: bool) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _state_bool_text(value: bool) -> str:
    return "true" if value else "false"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="IEC 60336-based slit camera focal spot measurement")
    parser.add_argument("--horizontal", help="Horizontal slit image path")
    parser.add_argument("--vertical", help="Vertical slit image path")
    parser.add_argument("--output", default=str(Path.cwd() / "reports"), help="Output directory")
    parser.add_argument("--raw-width", type=int, help="RAW image width in pixels")
    parser.add_argument("--raw-height", type=int, help="RAW image height in pixels")
    parser.add_argument("--raw-dtype", default="uint16", help="RAW numpy dtype, default uint16")
    parser.add_argument("--no-gui", action="store_true", help="Run from CLI instead of opening Tkinter UI")
    args = parser.parse_args(argv)

    if args.no_gui:
        if not args.horizontal or not args.vertical:
            parser.error("--horizontal and --vertical are required with --no-gui")
        inputs = AnalysisInputs(
            horizontal_path=args.horizontal,
            vertical_path=args.vertical,
            output_dir=args.output,
            raw_width=args.raw_width,
            raw_height=args.raw_height,
            raw_dtype=args.raw_dtype,
        )
        pdf_path, csv_path, png_paths, analyses = run_analysis(inputs, MeasurementConfig())
        print(f"PDF report: {pdf_path}")
        print(f"CSV results: {csv_path}")
        for path in png_paths:
            print(f"PNG figure: {path}")
        for analysis in analyses:
            print(
                f"{analysis.label}: {analysis.quality.overall_status}, "
                f"focal spot {_fmt(analysis.lsf.focal_spot_size_mm)} mm"
            )
        return 0

    app = FocalSpotApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
