# Local Suite2p Functional Modifications

This file records functional changes made in this local fork/branch of Suite2p.

## 2026-06-17

### Main GUI left/right pane synchronization

- Kept the main ROI GUI left and right image panes synchronized for pan and zoom changes.
- Removed the behavior where choosing left-only or right-only pane sizes unlinked the hidden pane.
- Added explicit two-way range synchronization so changes made from either pane are mirrored to the other pane.
- Reset the main image panes to the full frame after loading an experiment and use padded pan limits so aspect-locked zoom-out can still show the full image.
- Re-applied the full-frame reset after the load-time activity-mode redraw, so loading starts with the whole frame visible.
- Fixed the `cells` / `both` / `not cells` buttons so the selected mode is highlighted and switching modes preserves the current pan/zoom range.

### Optional signal-area and ROI-size filtering during detection

- Added detection setting `bright_area_percentile`, shown in the GUI as `Bright area`, defaulting to `30.0`.
- Added detection setting `include_area_percentile`, shown in the GUI as `Include area`, defaulting to `5.0`.
- Added detection setting `lateral_exclude_percent`, defaulting to `5.0`, to exclude symmetric left/right frame edges from peak candidate selection only.
- Added detection setting `include_min_roi_fraction`, defaulting to `0.5`.
- Added detection settings `min_roi_pixels`, `max_roi_pixels`, `min_roi_width`, `max_roi_width`, `min_roi_height`, and `max_roi_height`, all defaulting to `0` so each filter is disabled unless explicitly set.
- Added sparsery setting `max_peaks_to_check`, defaulting to `20000`, to cap how many candidate peaks are tested while trying to reach `max_ROIs`.
- Removed the Otsu dark-mask mode and the separate `Ignore dark areas` switch. `Bright area` or `Include area` set to `0` uses the whole frame for that specific step.
- Suite2p now constructs coherent signal masks from the mean image:
  - the mean image is robustly normalized,
  - spatially smoothed using the existing ROI `diameter`/structure-size setting,
  - thresholded to keep the brightest `bright_area_percentile` of pixels for peak-threshold estimation,
  - thresholded at `include_area_percentile` for ROI validation.
- In sparsery, the full correlation/activity map is still built and searched for peaks, but the spatial-scale and peak-stop threshold are estimated from maps masked to the brightest mean-image pixels.
- Left/right edge pixels excluded by `lateral_exclude_percent` cannot seed detected peaks but still contribute to `Bright area` threshold estimation.
- After ROI detection, ROIs are removed if less than `include_min_roi_fraction` of their footprint lies inside the `Include area` mask.
- Sparsery applies the raw footprint pixel-count and width/height limits during candidate detection, so rejected-size candidates do not count toward `max_ROIs`.
- Sparsery stops once either `max_ROIs` accepted ROIs or `max_peaks_to_check` tested candidate peaks is reached.
- Sparsery progress logging now reports running rejection counts for each ROI size-limit reason.
- ROIs can also be removed after detection if their raw footprint pixel count or footprint width/height is outside the configured min/max limits.
- Detection outputs save removal counts for the Include-area, pixel-count, width, and height filters, shown in Mask diagnostics.
- Detection outputs now save:
  - `meanImg_signal_mask`: full-frame boolean threshold-estimation mask,
  - `meanImg_roi_signal_mask`: full-frame boolean Include-area ROI-valid mask,
  - `meanImg_signal_map`: smoothed mean-intensity map used to make the mask.
- Mask diagnostics show the Bright-area and Include-area thresholds, valid-pixel fractions, and ROI filter removal counts.
- Detection outputs now also save diagnostic maps:
  - `meanImg_signal_masked`: mean image multiplied by the signal mask,
  - `Vcorr_scales`: full-resolution correlation maps at each sparsery scale,
  - `Vcorr_signal_scales`: those scale maps masked by the signal mask used for threshold estimation,
  - `Vmap_signal`: native-resolution peak maps masked by the signal mask.
- The GUI background view list now has separate `Corr` and `Mask` buttons:
  - `Corr` shows the existing correlation map,
  - `Mask` shows the saved low-signal exclusion mask and opens a diagnostic window with mean-image, signal-mask, scale-correlation, masked-correlation, and peak-map views.
- Diagnostic correlation and peak maps use a shared raw color scale, with color scale bars anchored to the saved peak threshold where available.
- Sparsery peak threshold uses the original Suite2p formula; the unscaled base threshold is saved as `signal_peak_threshold_unscaled`.
- Result loading now merges `reg_outputs.npy` and `detect_outputs.npy` over `ops.npy` when present, so newer saved background maps are visible when reopening older-style result folders.
- Combined-plane output now carries `meanImg_signal_mask` into the combined view.

## 2026-06-16

### Bidirectional phase estimation from central frame region

- Changed bidirectional phase estimation to use the central fraction of each frame rather than the whole frame.
- Added `central_fraction` to bidirectional phase calculation, defaulting to `0.7`.
- This makes the bidirectional scanning line-offset estimate less sensitive to edge/cropped regions.

### Optional registration metrics

- Added configurable `do_regmetrics` behavior under registration settings.
- The GUI Run Suite2p registration settings can now control whether registration metrics are calculated.
- Existing user settings missing `do_regmetrics` are tolerated by the run-window default-setting merge.

### Registration metrics logging and GPU device propagation

- Added an explicit log message before registration metrics calculation starts.
- Added timing output for registration metrics calculation.
- Passed the selected processing device into registration metrics calculation so forced GPU settings are respected when supported.

### Sparsery spatial-scale experiment reverted

- An experimental change to force sparsery correlation-map/threshold behavior to the requested spatial scale was made and then reverted because it did not improve the observed small dark-region ROI detections.
- Current behavior for sparsery scale selection is therefore restored to the upstream-style multi-scale behavior.
