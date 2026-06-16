from __future__ import annotations

from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from image_loader import DICOM_EXTENSIONS, IMAGE_EXTENSIONS, RAW_EXTENSIONS, load_image
from preprocessing import normalize_for_display
from roi_detection import Roi, auto_detect_roi, clamp_roi, parse_roi


SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | DICOM_EXTENSIONS | RAW_EXTENSIONS


def apply_display_brightness(base_display: np.ndarray, brightness: float) -> np.ndarray:
    """Apply a display-only brightness offset to an already normalized uint8 image."""
    offset = float(brightness) / 200.0
    adjusted = np.clip(base_display.astype(np.float32) / 255.0 + offset, 0.0, 1.0)
    return (adjusted * 255.0).astype(np.uint8)


class ImagePickerDialog:
    def __init__(
        self,
        parent: tk.Misc,
        *,
        initial_path: str = "",
        initial_folder: str = "",
        raw_shape: Optional[tuple[int, int]] = None,
        raw_dtype: str = "uint16",
    ) -> None:
        self.parent = parent
        self.raw_shape = raw_shape
        self.raw_dtype = raw_dtype
        self.result: Optional[str] = None
        self.last_folder: Optional[str] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.preview_base_display: Optional[np.ndarray] = None
        self.brightness_var = tk.DoubleVar(value=0.0)
        self.brightness_label_var = tk.StringVar(value="Brightness 0")

        self.window = tk.Toplevel(parent)
        self.window.title("Select image with preview")
        self.window.geometry("980x680")
        self.window.minsize(820, 560)
        self.window.transient(parent)
        self.window.grab_set()

        initial = Path(initial_path) if initial_path else None
        fallback_folder = Path(initial_folder) if initial_folder else Path.cwd()
        if initial is not None and initial.is_file():
            initial_dir = initial.parent
            select_path = initial
        elif initial is not None and initial.is_dir():
            initial_dir = initial
            select_path = None
        else:
            initial_dir = fallback_folder
            select_path = None
        if not initial_dir.exists():
            initial_dir = Path.cwd()
        self.folder_var = tk.StringVar(value=str(initial_dir))
        self.last_folder = str(initial_dir)
        self.selected_file: Optional[Path] = None

        self._build_ui()
        self._populate_files(select_path=select_path)

        parent.wait_window(self.window)

    def _build_ui(self) -> None:
        root = ttk.Frame(self.window, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.folder_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Open folder", command=self._choose_folder).grid(row=0, column=2)
        ttk.Button(top, text="Refresh", command=lambda: self._populate_files()).grid(row=0, column=3, padx=(6, 0))

        left = ttk.Frame(root)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.file_list = tk.Listbox(left, width=42, exportselection=False)
        self.file_list.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.file_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.bind("<<ListboxSelect>>", self._on_select)
        self.file_list.bind("<Double-Button-1>", lambda _event: self._accept())

        preview = ttk.Frame(root)
        preview.grid(row=1, column=1, sticky="nsew")
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.preview_canvas = tk.Canvas(preview, background="#111111", highlightthickness=0)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", lambda _event: self._draw_preview())
        self.info_var = tk.StringVar(value="Select a file to preview it.")
        ttk.Label(preview, textvariable=self.info_var).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        display_controls = ttk.Frame(preview)
        display_controls.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        display_controls.columnconfigure(1, weight=1)
        ttk.Label(display_controls, textvariable=self.brightness_label_var, width=14).grid(row=0, column=0, sticky="w")
        ttk.Scale(
            display_controls,
            from_=-100,
            to=100,
            variable=self.brightness_var,
            command=self._brightness_changed,
        ).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(display_controls, text="Reset", command=self._reset_brightness).grid(row=0, column=2)

        bottom = ttk.Frame(root)
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        bottom.columnconfigure(0, weight=1)
        ttk.Label(
            bottom,
            text="RAW preview uses the current RAW width, height, and dtype from the main window.",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(bottom, text="Standard file dialog", command=self._standard_dialog).grid(row=0, column=1, padx=6)
        ttk.Button(bottom, text="Select", command=self._accept).grid(row=0, column=2, padx=6)
        ttk.Button(bottom, text="Cancel", command=self.window.destroy).grid(row=0, column=3)

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.folder_var.get() or str(Path.cwd()))
        if folder:
            self.folder_var.set(folder)
            self.last_folder = folder
            self._populate_files()

    def _standard_dialog(self) -> None:
        path = filedialog.askopenfilename(initialdir=self.folder_var.get() or str(Path.cwd()))
        if path:
            self.result = path
            self.last_folder = str(Path(path).parent)
            self.window.destroy()

    def _populate_files(self, select_path: Optional[Path] = None) -> None:
        folder = Path(self.folder_var.get())
        self.file_list.delete(0, tk.END)
        self.files: list[Path] = []
        if not folder.exists() or not folder.is_dir():
            self.info_var.set("Folder does not exist.")
            return
        self.last_folder = str(folder)

        for path in sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.files.append(path)
                self.file_list.insert(tk.END, path.name)

        if not self.files:
            self.info_var.set("No supported image files found in this folder.")
            self.preview_canvas.delete("all")
            return

        index = 0
        if select_path is not None:
            for idx, path in enumerate(self.files):
                if path.resolve() == select_path.resolve():
                    index = idx
                    break
        self.file_list.selection_set(index)
        self.file_list.see(index)
        self._preview_path(self.files[index])

    def _on_select(self, _event: tk.Event) -> None:
        selection = self.file_list.curselection()
        if not selection:
            return
        self._preview_path(self.files[int(selection[0])])

    def _preview_path(self, path: Path) -> None:
        self.selected_file = path
        try:
            image = load_image(path, raw_shape=self.raw_shape, raw_dtype=self.raw_dtype)
            self.preview_base_display = normalize_for_display(image)
            self._draw_preview()
            self.info_var.set(
                f"{path.name} | shape {image.shape[1]} x {image.shape[0]} | "
                f"min {np.nanmin(image):.4g} | max {np.nanmax(image):.4g}"
            )
        except Exception as exc:
            self.preview_base_display = None
            self.preview_canvas.delete("all")
            self.info_var.set(f"{path.name}: {exc}")

    def _draw_preview(self) -> None:
        if self.preview_base_display is None:
            return
        display = apply_display_brightness(self.preview_base_display, self.brightness_var.get())
        canvas_w = max(self.preview_canvas.winfo_width(), 640)
        canvas_h = max(self.preview_canvas.winfo_height(), 460)
        h, w = display.shape
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        out_w = max(1, int(w * scale))
        out_h = max(1, int(h * scale))
        pil_image = Image.fromarray(display).resize((out_w, out_h), Image.Resampling.BILINEAR)
        self.preview_photo = ImageTk.PhotoImage(pil_image)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.preview_photo, anchor=tk.CENTER)

    def _brightness_changed(self, value: str) -> None:
        self.brightness_label_var.set(f"Brightness {float(value):+.0f}")
        self._draw_preview()

    def _reset_brightness(self) -> None:
        self.brightness_var.set(0.0)
        self.brightness_label_var.set("Brightness 0")
        self._draw_preview()

    def _accept(self) -> None:
        if self.selected_file is None:
            messagebox.showwarning("No file selected", "Select an image file first.", parent=self.window)
            return
        self.result = str(self.selected_file)
        self.last_folder = str(self.selected_file.parent)
        self.window.destroy()


class RoiSelectionDialog:
    def __init__(
        self,
        parent: tk.Misc,
        *,
        path: str,
        roi_variable: tk.StringVar,
        orientation: str,
        raw_shape: Optional[tuple[int, int]] = None,
        raw_dtype: str = "uint16",
    ) -> None:
        self.parent = parent
        self.path = path
        self.roi_variable = roi_variable
        self.orientation = orientation
        self.raw_shape = raw_shape
        self.raw_dtype = raw_dtype
        self.scale = 1.0
        self.fit_scale = 1.0
        self.image: Optional[np.ndarray] = None
        self.base_display: Optional[np.ndarray] = None
        self.display_source: Optional[Image.Image] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.selection: Optional[Roi] = None
        self.drag_start: Optional[tuple[float, float]] = None
        self.brightness_var = tk.DoubleVar(value=0.0)
        self.brightness_label_var = tk.StringVar(value="Brightness 0")

        self.window = tk.Toplevel(parent)
        self.window.title(f"ROI selection - {Path(path).name}")
        self.window.geometry("1120x780")
        self.window.minsize(900, 620)
        self.window.transient(parent)
        self.window.grab_set()

        self._build_ui()
        self._load()
        parent.wait_window(self.window)

    def _build_ui(self) -> None:
        root = ttk.Frame(self.window, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text=Path(self.path).name).grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="Drag on the image to set ROI.")
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=1, columnspan=5, sticky="e", padx=12)
        ttk.Label(top, textvariable=self.brightness_label_var, width=14).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(
            top,
            from_=-100,
            to=100,
            variable=self.brightness_var,
            command=self._brightness_changed,
        ).grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(top, text="Reset", command=self._reset_brightness).grid(row=1, column=2, padx=(0, 12), pady=(6, 0))
        ttk.Label(top, text="Zoom").grid(row=1, column=3, sticky="e", pady=(6, 0))
        self.zoom_var = tk.StringVar(value="Fit")
        zoom = ttk.Combobox(
            top,
            textvariable=self.zoom_var,
            values=("Fit", "25%", "50%", "100%", "200%"),
            state="readonly",
            width=8,
        )
        zoom.grid(row=1, column=4, padx=(4, 0), pady=(6, 0))
        zoom.bind("<<ComboboxSelected>>", lambda _event: self._set_zoom())

        frame = ttk.Frame(root)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(frame, background="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        xbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        ybar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        xbar.grid(row=1, column=0, sticky="ew")
        ybar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._end_drag)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        bottom = ttk.Frame(root)
        bottom.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        bottom.columnconfigure(0, weight=1)
        ttk.Label(
            bottom,
            text="ROI format: x,y,width,height. Use separate ROI values for horizontal and vertical images.",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(bottom, text="Auto ROI", command=self._auto_roi).grid(row=0, column=1, padx=5)
        ttk.Button(bottom, text="Clear ROI", command=self._clear_roi).grid(row=0, column=2, padx=5)
        ttk.Button(bottom, text="Use ROI", command=self._use_roi).grid(row=0, column=3, padx=5)
        ttk.Button(bottom, text="Close", command=self.window.destroy).grid(row=0, column=4)

    def _load(self) -> None:
        self.image = load_image(self.path, raw_shape=self.raw_shape, raw_dtype=self.raw_dtype)
        self.base_display = normalize_for_display(self.image)
        self._refresh_display_source()
        h, w = self.base_display.shape
        self.fit_scale = min(1000 / w, 620 / h)
        self.scale = self.fit_scale
        existing = parse_roi(self.roi_variable.get()) if self.roi_variable.get().strip() else None
        if existing is not None:
            self.selection = clamp_roi(existing, self.image.shape)
        self._redraw()
        self._update_status()

    def _refresh_display_source(self) -> None:
        if self.base_display is None:
            return
        display = apply_display_brightness(self.base_display, self.brightness_var.get())
        self.display_source = Image.fromarray(display)

    def _set_zoom(self) -> None:
        value = self.zoom_var.get()
        if value == "Fit":
            self.fit_scale = self._compute_fit_scale()
            self.scale = self.fit_scale
        else:
            self.scale = float(value.rstrip("%")) / 100.0
        self._redraw()

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        if self.zoom_var.get() == "Fit":
            self.fit_scale = self._compute_fit_scale()
            self.scale = self.fit_scale
        self._redraw()

    def _compute_fit_scale(self) -> float:
        if self.display_source is None:
            return self.fit_scale
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        return max(0.01, min(canvas_w / self.display_source.width, canvas_h / self.display_source.height))

    def _redraw(self) -> None:
        if self.display_source is None:
            return
        width = max(1, int(self.display_source.width * self.scale))
        height = max(1, int(self.display_source.height * self.scale))
        resized = self.display_source.resize((width, height), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW, tags=("image",))
        self.canvas.configure(scrollregion=(0, 0, width, height))
        self._draw_selection()

    def _brightness_changed(self, value: str) -> None:
        self.brightness_label_var.set(f"Brightness {float(value):+.0f}")
        self._refresh_display_source()
        self._redraw()

    def _reset_brightness(self) -> None:
        self.brightness_var.set(0.0)
        self.brightness_label_var.set("Brightness 0")
        self._refresh_display_source()
        self._redraw()

    def _draw_selection(self) -> None:
        self.canvas.delete("roi")
        if self.selection is None:
            return
        x, y, w, h = self.selection
        self.canvas.create_rectangle(
            x * self.scale,
            y * self.scale,
            (x + w) * self.scale,
            (y + h) * self.scale,
            outline="yellow",
            width=2,
            tags=("roi",),
        )

    def _start_drag(self, event: tk.Event) -> None:
        self.drag_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def _drag(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1 = self.canvas.canvasx(event.x)
        y1 = self.canvas.canvasy(event.y)
        self.canvas.delete("roi")
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="yellow", width=2, tags=("roi",))

    def _end_drag(self, event: tk.Event) -> None:
        if self.drag_start is None or self.image is None:
            return
        x0, y0 = self.drag_start
        x1 = self.canvas.canvasx(event.x)
        y1 = self.canvas.canvasy(event.y)
        self.drag_start = None
        ox0, ox1 = sorted((x0 / self.scale, x1 / self.scale))
        oy0, oy1 = sorted((y0 / self.scale, y1 / self.scale))
        roi = (
            int(round(ox0)),
            int(round(oy0)),
            max(1, int(round(ox1 - ox0))),
            max(1, int(round(oy1 - oy0))),
        )
        self.selection = clamp_roi(roi, self.image.shape)
        self._draw_selection()
        self._update_status()

    def _auto_roi(self) -> None:
        if self.image is None:
            return
        try:
            result = auto_detect_roi(self.image, orientation=self.orientation)
            self.selection = result.roi
            self._draw_selection()
            self._update_status(extra=result.message)
        except Exception as exc:
            messagebox.showerror("Auto ROI failed", str(exc), parent=self.window)

    def _clear_roi(self) -> None:
        self.selection = None
        self.roi_variable.set("")
        self.canvas.delete("roi")
        self.status_var.set("ROI cleared. Blank ROI will use automatic detection during analysis.")

    def _use_roi(self) -> None:
        if self.selection is None:
            messagebox.showwarning("No ROI selected", "Drag an ROI or use Auto ROI first.", parent=self.window)
            return
        x, y, w, h = self.selection
        self.roi_variable.set(f"{x},{y},{w},{h}")
        self.window.destroy()

    def _update_status(self, extra: str = "") -> None:
        if self.image is None:
            return
        h, w = self.image.shape
        if self.selection is None:
            text = f"Image {w} x {h}. Drag to set ROI."
        else:
            x, y, rw, rh = self.selection
            text = f"Image {w} x {h}. ROI x={x}, y={y}, w={rw}, h={rh}."
        if extra:
            text = f"{text} {extra}"
        self.status_var.set(text)
