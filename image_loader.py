from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}
RAW_EXTENSIONS = {".raw", ".bin"}


class ImageLoadError(RuntimeError):
    pass


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ImageLoadError(f"Unsupported image shape: {image.shape}")


def load_image(
    path: str | Path,
    *,
    raw_shape: Optional[Tuple[int, int]] = None,
    raw_dtype: str = "uint16",
) -> np.ndarray:
    """Load an image as float64 grayscale data without intensity normalization."""
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"File does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ImageLoadError(f"OpenCV could not read image: {path}")
        image = _to_grayscale(image)
        return image.astype(np.float64)

    if suffix in DICOM_EXTENSIONS:
        try:
            import pydicom
        except ImportError as exc:
            raise ImageLoadError(
                "DICOM loading requires pydicom. Install requirements.txt first."
            ) from exc
        ds = pydicom.dcmread(str(path))
        image = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        return image * slope + intercept

    if suffix in RAW_EXTENSIONS:
        if raw_shape is None:
            raise ImageLoadError(
                "RAW image loading requires a raw_shape=(height, width) argument."
            )
        dtype = np.dtype(raw_dtype)
        data = np.fromfile(str(path), dtype=dtype)
        expected = int(raw_shape[0]) * int(raw_shape[1])
        if data.size != expected:
            raise ImageLoadError(
                f"RAW size mismatch: got {data.size} pixels, expected {expected}."
            )
        return data.reshape(raw_shape).astype(np.float64)

    raise ImageLoadError(f"Unsupported file extension: {suffix}")


def load_optional_image(
    path: str | Path | None,
    *,
    raw_shape: Optional[Tuple[int, int]] = None,
    raw_dtype: str = "uint16",
) -> Optional[np.ndarray]:
    if path is None:
        return None
    path_str = str(path).strip()
    if not path_str:
        return None
    return load_image(path_str, raw_shape=raw_shape, raw_dtype=raw_dtype)


def supported_filetypes() -> Sequence[tuple[str, str]]:
    return (
        ("Supported images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.dcm *.dicom *.raw *.bin"),
        ("TIFF", "*.tif *.tiff"),
        ("PNG", "*.png"),
        ("JPEG", "*.jpg *.jpeg"),
        ("BMP", "*.bmp"),
        ("DICOM", "*.dcm *.dicom"),
        ("RAW", "*.raw *.bin"),
        ("All files", "*.*"),
    )
