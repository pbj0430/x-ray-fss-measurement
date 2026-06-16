# IEC 60336 Slit Camera Focal Spot Measurement

Python software for slit-camera focal spot size measurement. The workflow accepts one horizontal slit image and one vertical slit image, evaluates the LSF at the configured 15% level, and saves a PDF report, PNG analysis figures, and CSV results.

The report wording is intentionally limited to **based on IEC 60336 method**. It does not claim official IEC certification.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Open the GUI:

```powershell
python main.py
```

Batch mode with defaults:

```powershell
python main.py --no-gui --horizontal path\to\horizontal.tif --vertical path\to\vertical.tif --output reports
```

Batch mode for RAW files:

```powershell
python main.py --no-gui --horizontal "sample_data\FSS_width_lens 0kV.raw" --vertical "sample_data\FSS_length_lens 0kV.raw" --raw-width 3072 --raw-height 3072 --raw-dtype uint16 --output reports
```

## Build Windows EXE

To create a single-file Windows executable for users who do not run Python directly:

```powershell
.\build_exe.ps1
```

The generated file is:

```text
dist\FSS_Measurement.exe
```

Share `FSS_Measurement.exe` with the operator. The first launch can be slow because PyInstaller extracts the bundled application to a temporary folder. Generated reports are still written to the selected output folder.

Build outputs under `build/` and `dist/` are ignored by Git. Rebuild the exe locally when the source changes.

Use the **Select...** buttons in the GUI to load the horizontal and vertical slit images. The selector shows a preview panel, including RAW images, using the current RAW width, height, and dtype fields.

Use **ROI...** or **Select on image** to open the image viewer and drag a rectangular ROI. Horizontal and vertical images keep separate ROI values, because their slit positions may differ. Leave an ROI blank to use automatic ROI detection during analysis.

The preview and ROI viewer include a brightness slider. This changes only the displayed image for easier visual inspection; it does not modify the RAW data or affect the measurement calculation.

The GUI remembers the last-used image paths, output folder, measurement parameters, RAW settings, ROI values, checkboxes, projection method, and image folder. Source runs save this in `.fss_ui_state.json` beside `main.py`; the Windows exe saves it under `%APPDATA%\FSS_Measurement\.fss_ui_state.json`. If this file does not exist, the built-in defaults are used. In the ROI viewer, `Fit` zoom follows the current window size when the window is resized.

## Supported Images

Supported file formats:

- TIFF / TIF
- PNG
- JPG / JPEG
- BMP
- DICOM, when `pydicom` is installed
- RAW / BIN, when width, height, and dtype are supplied

For RAW files, enter `RAW width`, `RAW height`, and `RAW dtype` in the GUI. The local sample RAW files used during development were 18,874,368 bytes, which matches `3072 x 3072 x uint16`, so the GUI defaults to `3072`, `3072`, `uint16`.

The RAW preview and ROI viewer also use these values. If the preview reports a RAW size mismatch, correct the RAW width, height, or dtype before selecting ROI.

## Default Parameters

The GUI starts with these defaults:

- Detector pixel size = `0.140 mm`
- Focal spot to slit distance = `100 mm`
- Slit to detector distance = `400 mm`
- Magnification = `400 / 100 = 4.0`
- Slit width = `0.010 mm`
- Threshold = `15%`
- Effective pixel size at focal spot plane = `0.140 / 4 = 0.035 mm`

After the first run or normal window close, the GUI restores the saved values. Delete the saved UI state file to return to the built-in defaults.

Additional analysis inputs:

- Detector bit depth
- Saturation warning level
- Smoothing sigma
- Projection method: `mean`, `median`, `trimmed_mean`, or `sum`
- Auto trim non-slit length in ROI
- Target peak level for exposure recommendation

Additional measurement information, grouped separately in the GUI:

- Nominal focal spot size
- Tube voltage
- Lens voltage
- Tube current
- Exposure time

These fields are saved into the report/CSV. Tube current and exposure time are also used for the exposure recommendation calculation when available, but they do not change the focal spot size calculation itself.

## Calculation

Magnification:

```text
M = slit_to_detector_distance / focal_spot_to_slit_distance
```

Effective focal-plane pixel size:

```text
effective_pixel_size = detector_pixel_size / M
```

The ROI is averaged along the slit direction to produce the Line Spread Function:

- Horizontal slit image: average along image columns, profile across rows
- Vertical slit image: average along image rows, profile across columns

When `Auto trim non-slit length in ROI` is enabled, the software detects where the slit signal exists along the slit length, then uses only that length region for LSF projection. For a vertical slit, rows above/below the actual slit are excluded from calculation. For a horizontal slit, columns left/right of the actual slit are excluded from calculation. The report still shows the full tilt-corrected ROI and overlays the calculation region as a dashed rectangle.

Projection method options:

- `mean`: arithmetic mean projection
- `median`: median projection, more robust to defective pixels
- `trimmed_mean`: discards the top and bottom 10% of projection samples before averaging
- `sum`: summed projection

The background is estimated from both profile edges and subtracted. The peak value is detected, the 15% threshold is calculated, and the left/right crossing points are found by linear interpolation.

Detector-plane width:

```text
measured_width_on_detector = (right_crossing_px - left_crossing_px) * detector_pixel_size
```

Focal-plane size:

```text
focal_spot_size = measured_width_on_detector / M
```

Slit width correction is optional in the GUI. By default it is not subtracted and should be treated as a measurement uncertainty term unless your local procedure requires correction.

## IEC 60336-Based Workflow

1. Load horizontal and vertical slit images.
2. Apply optional dark, offset, and flat correction.
3. Compute magnification from the entered geometry.
4. Check saturation, peak range, magnification, tilt, ROI containment, SNR, sampling, and 15% crossing detection.
5. Detect ROI automatically, or use a manual `x,y,width,height` ROI.
6. Detect tilt using weighted PCA of the slit signal.
7. Rotate the ROI for tilt correction when enabled.
8. Average the ROI along the slit direction to generate the LSF.
9. Measure the 15% LSF width and convert it to focal spot size.
10. Save PDF, PNG, and CSV results.

## Report Outputs

The generated PDF includes:

- `IEC 60336 Focal Spot Measurement Report`
- Measurement date/time
- Input file list
- Input geometry and exposure conditions
- Magnification and effective pixel size
- IEC condition check results
- Horizontal and vertical image analysis results
- Original image with ROI
- ROI crop
- Tilt-corrected image
- LSF curve with peak, 15% threshold, crossing points, and measured width

PNG copies of each analysis page and a CSV summary are saved in the same output folder.

## Quality Status

The software reports `PASS`, `WARNING`, or `FAIL` for:

- Saturation
- Peak signal range
- Magnification
- Tilt angle
- ROI signal containment
- SNR
- Pixel sampling
- 15% crossing detection

If the current peak is too high or low and tube current plus exposure time were entered, the software estimates:

```text
recommended_mAs = current_mAs * target_peak_fraction / current_peak_fraction
```

Example: if the current peak is 95% of full scale and the target is 70%, then:

```text
recommended_mAs = current_mAs * 70 / 95
```

## Important Notes

Saturated images cannot be recovered by post-processing. If saturated pixels are detected, the result is marked as unreliable and the image should be reacquired.

The PDF includes this warning:

```text
saturation-free image required for reliable IEC 60336-based analysis
```

For production or accreditation work, verify the configured magnification tolerance, tilt tolerance, sampling rule, and slit-width handling against your controlled IEC 60336 procedure.
