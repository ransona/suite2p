"""Fast manual ROI editor for adding many fixed circular or freehand ROIs."""
import json
import os
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from matplotlib.path import Path as MplPath
from scipy import ndimage
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import (
    QApplication, QAbstractItemView,
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressDialog, QPushButton, QSpinBox, QTreeWidget,
    QTreeWidgetItem, QTreeWidgetItemIterator, QVBoxLayout, QWidget,
)

from . import drawroi, io


TREE_FILENAME = "rapid_rois.json"
PIXEL_CENTER_OFFSET = 0.5


def circle_pixels(center_y, center_x, diameter, ly, lx):
    radius = max(float(diameter) / 2.0, 0.5)
    y0 = max(0, int(np.floor(center_y - radius)))
    y1 = min(int(ly), int(np.ceil(center_y + radius)) + 1)
    x0 = max(0, int(np.floor(center_x - radius)))
    x1 = min(int(lx), int(np.ceil(center_x + radius)) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    include = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= radius ** 2
    return yy[include].astype(np.int32), xx[include].astype(np.int32)


def polygon_pixels(vertices_yx, ly, lx):
    vertices = np.asarray(vertices_yx, dtype=float)
    if vertices.shape[0] < 3:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    y0 = max(0, int(np.floor(vertices[:, 0].min())))
    y1 = min(int(ly), int(np.ceil(vertices[:, 0].max())) + 1)
    x0 = max(0, int(np.floor(vertices[:, 1].min())))
    x1 = min(int(lx), int(np.ceil(vertices[:, 1].max())) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    path = MplPath(np.column_stack((vertices[:, 1], vertices[:, 0])))
    include = path.contains_points(points, radius=0.5).reshape(yy.shape)
    return yy[include].astype(np.int32), xx[include].astype(np.int32)


def find_peak_candidates(image, smoothing_sigma, amplitude_threshold, min_distance, excluded=None,
                         refine_radius=0):
    """Find separated peaks in the same Gaussian-smoothed image shown to users."""
    image = np.asarray(image, dtype=float)
    if image.ndim == 3:
        image = image.mean(axis=-1)
    if not np.isfinite(image).any():
        return np.empty((0, 2), dtype=int)
    image = np.nan_to_num(image, nan=np.nanmedian(image))
    smooth = ndimage.gaussian_filter(image, smoothing_sigma) if smoothing_sigma else image
    # "Peak amplitude" is deliberately measured directly from the displayed
    # smoothed map, not a hidden difference-of-Gaussians contrast image.  This
    # makes a candidate's position and threshold visually verifiable.
    low, high = np.nanpercentile(smooth, (1, 99.9))
    if high <= low:
        return np.empty((0, 2), dtype=int)
    amplitude_cutoff = low + amplitude_threshold * (high - low)
    window = max(3, int(2 * round(min_distance) + 1))
    peaks = (smooth == ndimage.maximum_filter(smooth, size=window))
    peaks &= smooth >= amplitude_cutoff
    if excluded is not None:
        peaks &= ~excluded
    candidates = np.column_stack(np.nonzero(peaks))
    if not len(candidates):
        return candidates
    if refine_radius:
        refined = []
        for y, x in candidates:
            y0, y1 = max(0, y - refine_radius), min(image.shape[0], y + refine_radius + 1)
            x0, x1 = max(0, x - refine_radius), min(image.shape[1], x + refine_radius + 1)
            dy, dx = np.unravel_index(np.argmax(image[y0:y1, x0:x1]), (y1 - y0, x1 - x0))
            refined.append((y0 + dy, x0 + dx))
        candidates = np.unique(np.asarray(refined, dtype=int), axis=0)

    # Refinement can move two formerly separate maxima toward one another.
    # Keep the stronger one whenever that would make the requested circles
    # overlap (or violate a larger user-selected spacing).
    scores = smooth[candidates[:, 0], candidates[:, 1]]
    kept = []
    for index in np.argsort(scores)[::-1]:
        candidate = candidates[index]
        if not kept or np.all(np.sum((np.asarray(kept) - candidate) ** 2, axis=1) >= min_distance ** 2):
            kept.append(candidate)
    return np.asarray(kept, dtype=int)


class RapidROIViewBox(pg.ViewBox):
    def __init__(self, editor):
        super().__init__(lockAspect=True, invertY=True)
        self.editor = editor
        # Be explicit: the rapid editor uses left-drag for panning except when
        # the user has selected Freehand shape mode.
        self.setMouseMode(self.PanMode)
        self.setMouseEnabled(x=True, y=True)

    def wheelEvent(self, event, axis=None):
        """Zoom about the mouse position, independent of pyqtgraph defaults."""
        delta = event.delta()
        if not delta:
            event.ignore()
            return
        point = self.mapSceneToView(event.scenePos())
        self.editor.set_mouse_position(point.y(), point.x())
        self.editor.zoom_at_mouse(0.8 if delta > 0 else 1.25)
        event.accept()


    def mouseClickEvent(self, event):
        point = self.mapSceneToView(event.scenePos())
        if event.button() == QtCore.Qt.RightButton:
            self.editor.remove_roi_at(point.y(), point.x())
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton:
            candidate_index = self.editor.candidate_index_at(point.y(), point.x())
            if candidate_index is not None:
                self.editor.candidate_index = candidate_index
                if event.double():
                    self.editor.accept_candidate()
                else:
                    self.editor._show_current_candidate()
                event.accept()
                return
            if self.editor.select_roi_at(point.y(), point.x()):
                event.accept()
                return
            if self.editor.draw_mode() == "circle":
                self.editor.add_circle(point.y(), point.x())
                event.accept()
                return
        super().mouseClickEvent(event)

    def mouseDragEvent(self, event, axis=None):
        if event.button() == QtCore.Qt.LeftButton and self.editor.draw_mode() == "freehand":
            point = self.mapSceneToView(event.pos())
            if event.isStart():
                self.editor.start_freehand(point.y(), point.x())
            elif event.isFinish():
                self.editor.finish_freehand(point.y(), point.x())
            else:
                self.editor.extend_freehand(point.y(), point.x())
            event.accept()
            return
        # In circle mode ViewBox's PanMode supplies conventional image panning.
        if event.button() == QtCore.Qt.LeftButton:
            point = self.mapSceneToView(event.pos())
            self.editor.set_mouse_position(point.y(), point.x())
        super().mouseDragEvent(event, axis=axis)

    def mouseMoveEvent(self, event):
        point = self.mapSceneToView(event.scenePos())
        self.editor.set_mouse_position(point.y(), point.x())
        super().mouseMoveEvent(event)

    def hoverEvent(self, event):
        if event.isExit():
            self.editor.mouse_position = None
        super().hoverEvent(event)


class RapidROIWindow(QMainWindow):
    VIEW_SPECS = (("W", "Mean", 1), ("E", "Enhanced", 2), ("R", "Corr.", 3),
                  ("M", "Mask", 4), ("T", "Max", 5),
                  ("Y", "Ch2 corr.", 6), ("U", "Ch2", 7),
                  ("S", "Smoothed", 8))

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.ly, self.lx = int(parent.ops["Ly"]), int(parent.ops["Lx"])
        self.records = []
        self.new_records = []
        self.deleted_existing_indices = set()
        self.roi_hit_map = np.full((self.ly, self.lx), -1, dtype=np.int32)
        self.hit_record_ids = []
        self.selected_id = None
        self.selected_ids = []
        self.current_parent_id = None
        self.current_segment = None
        self._suppress_selection_zoom = False
        self.mouse_position = None
        self.freehand_points = []
        self.current_view_index = 1
        self.peak_source_view = 2
        self.candidate_peaks = np.empty((0, 2), dtype=int)
        self.candidate_index = None
        self.rejected_peaks = set()
        self.extracted = False
        self.save_gui = False
        self.save_started = False
        self.setWindowTitle("Suite2p Rapid ROIs")
        self.resize(1300, 900)
        self._load_saved_tree()
        self._add_missing_existing_records()
        self._build_ui()
        self._install_zoom_shortcuts()
        QApplication.instance().installEventFilter(self)
        self._rebuild_roi_hit_map()
        self._refresh_tree()
        self.set_view(1)

    @property
    def tree_path(self):
        return Path(self.parent.basename) / TREE_FILENAME

    def _load_saved_tree(self):
        if not self.tree_path.exists():
            return
        try:
            payload = json.loads(self.tree_path.read_text(encoding="utf-8"))
            for record in payload.get("rois", []):
                record = dict(record)
                record["existing"] = True
                record["save_in_tree"] = True
                self.records.append(record)
        except Exception as error:
            print(f"Could not load {self.tree_path}: {error}")

    def _add_missing_existing_records(self):
        """Represent every Suite2p ROI, including automatically detected ones.

        rapid_rois.json only describes ROIs made through this editor.  The
        canonical ROI masks are in parent.stat, so use those masks for the
        outlines and hit testing of both automatic and previously saved rapid
        ROIs.  Automatically detected records are deliberately not persisted
        to rapid_rois.json.
        """
        stats = self.parent.stat
        saved_by_index = {
            int(record["roi_index"]): record
            for record in self.records
            if record.get("existing") and record.get("roi_index") is not None
            and 0 <= int(record["roi_index"]) < len(stats)
        }
        for roi_index, stat in enumerate(stats):
            record = saved_by_index.get(roi_index)
            if record is None:
                record = {
                    "id": uuid.uuid4().hex,
                    "shape": "existing_mask",
                    "roi_index": roi_index,
                    "parent_id": None,
                    "existing": True,
                    "save_in_tree": False,
                }
                self.records.append(record)
            record["mask_ypix"] = np.asarray(stat["ypix"], dtype=np.int32).tolist()
            record["mask_xpix"] = np.asarray(stat["xpix"], dtype=np.int32).tolist()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.hierarchy_enabled = QCheckBox("Enable ROI hierarchy")
        self.hierarchy_enabled.toggled.connect(self._refresh_tree)
        left_layout.addWidget(self.hierarchy_enabled)
        left_layout.addWidget(QLabel("ROIs / hierarchy"))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["ROI"])
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self.tree_selection_changed)
        left_layout.addWidget(self.tree, 1)
        self.make_root_button = QPushButton("Set selected as root")
        self.make_child_button = QPushButton("Add children to selected")
        self.make_root_button.clicked.connect(self.set_selected_as_root)
        self.make_child_button.clicked.connect(self.set_selected_as_parent)
        left_layout.addWidget(self.make_root_button)
        left_layout.addWidget(self.make_child_button)
        self.delete_button = QPushButton("Delete selected ROI(s) [Backspace]")
        self.delete_button.clicked.connect(self.delete_selected)
        left_layout.addWidget(self.delete_button)
        self.remove_detected_button = QPushButton("Remove all existing ROIs")
        self.remove_detected_button.clicked.connect(self.remove_existing_non_manual_rois)
        left_layout.addWidget(self.remove_detected_button)
        self.save_button = QPushButton("Save ROIs")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_rois)
        left_layout.addWidget(self.save_button)
        layout.addWidget(left, 1)

        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        controls = QVBoxLayout()
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Click circles", "circle")
        self.mode_combo.addItem("Freehand shape", "freehand")
        self.mode_combo.currentIndexChanged.connect(self._update_mode_status)
        top_row.addWidget(self.mode_combo)
        top_row.addWidget(QLabel("View"))
        self.view_group = QButtonGroup(self)
        view_box = QWidget()
        view_layout = QHBoxLayout(view_box)
        view_layout.setContentsMargins(0, 0, 0, 0)
        # Keep the common backgrounds together while retaining their standard
        # main-GUI keyboard shortcuts.  Max projection is useful for dim ROIs.
        for key, label, index in (
            self.VIEW_SPECS[0], self.VIEW_SPECS[1], self.VIEW_SPECS[2],
            self.VIEW_SPECS[4], self.VIEW_SPECS[-1],
        ):
            button = QPushButton(f"{key}: {label}")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, idx=index: self.set_view(idx))
            self.view_group.addButton(button, index)
            view_layout.addWidget(button)
        top_row.addWidget(view_box)
        top_row.addWidget(QLabel("Zoom"))
        zoom_box = QWidget()
        zoom_layout = QGridLayout(zoom_box)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.setSpacing(2)
        for row in range(3):
            for col in range(3):
                button = QPushButton(str(row * 3 + col + 1))
                button.setFixedSize(24, 22)
                button.clicked.connect(lambda _checked=False, r=row, c=col: self.zoom_segment(r, c))
                zoom_layout.addWidget(button, row, col)
        zoom_out = QPushButton("Out")
        zoom_out.setFixedHeight(22)
        zoom_out.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out, 3, 0, 1, 3)
        top_row.addWidget(zoom_box)
        top_row.addStretch()
        controls.addLayout(top_row)

        peak_row = QHBoxLayout()
        peak_row.addWidget(QLabel("Circle diameter (pixels)"))
        self.diameter = QSpinBox()
        self.diameter.setRange(1, max(self.ly, self.lx))
        self.diameter.setValue(12)
        self.diameter.valueChanged.connect(self._refresh_peak_candidates)
        peak_row.addWidget(self.diameter)
        self.peak_candidates_enabled = QCheckBox("Peak candidates")
        self.peak_candidates_enabled.toggled.connect(self._set_peak_candidate_enabled)
        peak_row.addWidget(self.peak_candidates_enabled)
        peak_row.addWidget(QLabel("Smooth (px)"))
        self.peak_smoothing = QSpinBox()
        self.peak_smoothing.setRange(0, 20)
        self.peak_smoothing.setValue(1)
        self.peak_smoothing.valueChanged.connect(self._refresh_peak_candidates)
        peak_row.addWidget(self.peak_smoothing)
        peak_row.addWidget(QLabel("Peak spacing (px; min. diameter)"))
        self.peak_spacing = QSpinBox()
        self.peak_spacing.setRange(1, max(self.ly, self.lx))
        self.peak_spacing.setValue(5)
        self.peak_spacing.valueChanged.connect(self._refresh_peak_candidates)
        peak_row.addWidget(self.peak_spacing)
        peak_row.addWidget(QLabel("Snap to source peak (px)"))
        self.peak_refine_radius = QSpinBox()
        self.peak_refine_radius.setRange(0, 20)
        # Refining against the raw image can pull a candidate toward an adjacent
        # bright structure, so leave it opt-in rather than shifting centres by
        # default.
        self.peak_refine_radius.setValue(0)
        self.peak_refine_radius.valueChanged.connect(self._refresh_peak_candidates)
        peak_row.addWidget(self.peak_refine_radius)
        peak_row.addStretch()
        controls.addLayout(peak_row)

        selection_row = QHBoxLayout()
        selection_row.addWidget(QLabel("Peak amplitude"))
        self.peak_threshold = QDoubleSpinBox()
        self.peak_threshold.setRange(0.001, 1.0)
        self.peak_threshold.setDecimals(3)
        self.peak_threshold.setSingleStep(0.02)
        self.peak_threshold.setValue(0.5)
        self.peak_threshold.valueChanged.connect(self._refresh_peak_candidates)
        selection_row.addWidget(self.peak_threshold)
        self.more_candidates_button = QPushButton("More candidates")
        self.more_candidates_button.clicked.connect(self.reduce_peak_threshold)
        selection_row.addWidget(self.more_candidates_button)
        self.peak_count_label = QLabel("")
        selection_row.addWidget(self.peak_count_label)
        selection_row.addWidget(QLabel("Detect peaks from"))
        self.peak_source = QComboBox()
        self.peak_source.addItem("Mean", 1)
        self.peak_source.addItem("Enhanced mean", 2)
        self.peak_source.setCurrentIndex(1)
        self.peak_source.currentIndexChanged.connect(self._set_peak_source)
        selection_row.addWidget(self.peak_source)
        self.include_all_button = QPushButton("Include all")
        self.include_all_button.clicked.connect(self.include_all_candidates)
        selection_row.addWidget(self.include_all_button)
        selection_row.addStretch()
        controls.addLayout(selection_row)

        border_row = QHBoxLayout()
        border_row.addWidget(QLabel("Exclude border pixels:"))
        border_row.addWidget(QLabel("Left"))
        self.exclude_left = QSpinBox()
        self.exclude_left.setRange(0, self.lx)
        border_row.addWidget(self.exclude_left)
        border_row.addWidget(QLabel("Right"))
        self.exclude_right = QSpinBox()
        self.exclude_right.setRange(0, self.lx)
        border_row.addWidget(self.exclude_right)
        border_row.addWidget(QLabel("Top"))
        self.exclude_top = QSpinBox()
        self.exclude_top.setRange(0, self.ly)
        border_row.addWidget(self.exclude_top)
        border_row.addWidget(QLabel("Bottom"))
        self.exclude_bottom = QSpinBox()
        self.exclude_bottom.setRange(0, self.ly)
        border_row.addWidget(self.exclude_bottom)
        self.apply_border_button = QPushButton("Apply border exclusion")
        self.apply_border_button.clicked.connect(self._refresh_peak_candidates)
        border_row.addWidget(self.apply_border_button)
        border_row.addStretch()
        controls.addLayout(border_row)
        self._set_peak_candidate_enabled(False)
        middle_layout.addLayout(controls)
        self.status = QLabel("Click the image to add circular ROIs. Freehand mode: drag on the image.")
        self.status.setWordWrap(True)
        middle_layout.addWidget(self.status)
        self.graphics = pg.GraphicsLayoutWidget()
        middle_layout.addWidget(self.graphics, 1)
        self.graphics.scene().sigMouseMoved.connect(self._scene_mouse_moved)
        self.viewbox = RapidROIViewBox(self)
        self.viewbox.setMouseEnabled(x=False, y=False)
        self.graphics.addItem(self.viewbox)
        # Use conventional image coordinates locally: x is the image column and
        # y is the image row. The main Suite2p GUI uses PyQtGraph's historic
        # col-major convention, but this editor draws and stores ROI pixels in
        # row-major NumPy coordinates.
        self.image = pg.ImageItem(axisOrder="row-major")
        self.viewbox.addItem(self.image)
        # Purple is reserved for pre-existing Suite2p masks. It is distinct
        # from cyan new ROIs, pink peak candidates, green candidate selection,
        # yellow selected ROIs, and salmon freehand drawing.
        self.existing_preview = pg.PlotCurveItem(
            pen=pg.mkPen((170, 90, 255), width=1.5), connect="finite"
        )
        self.preview = pg.PlotCurveItem(pen=pg.mkPen((0, 220, 255), width=1.5), connect="finite")
        self.selected_preview = pg.PlotCurveItem(pen=pg.mkPen((255, 220, 0), width=3), connect="finite")
        self.peak_markers = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(255, 80, 200))
        self.peak_markers.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self.peak_outlines = pg.PlotCurveItem(pen=pg.mkPen((255, 80, 200), width=1.5), connect="finite")
        self.rejected_peak_outlines = pg.PlotCurveItem(pen=pg.mkPen((130, 130, 130), width=1), connect="finite")
        self.selected_peak_outline = pg.PlotCurveItem(pen=pg.mkPen((0, 255, 80), width=3), connect="finite")
        self.drawing_preview = pg.PlotCurveItem(pen=pg.mkPen((255, 120, 100), width=2))
        self.viewbox.addItem(self.existing_preview)
        self.viewbox.addItem(self.preview)
        self.viewbox.addItem(self.selected_preview)
        self.viewbox.addItem(self.peak_markers)
        self.viewbox.addItem(self.peak_outlines)
        self.viewbox.addItem(self.rejected_peak_outlines)
        self.viewbox.addItem(self.selected_peak_outline)
        self.viewbox.addItem(self.drawing_preview)
        layout.addWidget(middle, 4)

    def draw_mode(self):
        return self.mode_combo.currentData()

    def _update_mode_status(self):
        messages = {
            "circle": "Click the image to add circular ROIs.",
            "freehand": "Freehand mode: drag on the image to draw an ROI shape.",
        }
        self.status.setText(messages[self.draw_mode()])

    def set_view(self, index):
        self.current_view_index = index
        image = self._smoothed_peak_image() if index == 8 else self.parent.views[index]
        self.image.setImage(image)
        button = self.view_group.button(index)
        if button is not None:
            button.setChecked(True)
        self._refresh_preview()
        self._refresh_peak_candidates()

    def _peak_source_image(self):
        """Return the original floating-point image, not its 8-bit GUI copy."""
        source_index = self.peak_source.currentData()
        source_key = {1: "meanImg", 2: "meanImgE"}[source_index]
        image = self.parent.ops.get(source_key)
        if image is None:
            # A safe fallback for unusual / legacy Suite2p output.
            image = self.parent.views[source_index]
        image = np.asarray(image, dtype=float)
        if image.ndim == 3:
            image = image.mean(axis=-1)
        if image.shape == (self.ly, self.lx):
            return image
        # Registration summaries can be cropped to the valid registration
        # region; put them back into full-frame coordinates like Suite2p Views.
        full = np.zeros((self.ly, self.lx), dtype=float)
        yrange, xrange = self.parent.ops.get("yrange"), self.parent.ops.get("xrange")
        if yrange is not None and xrange is not None:
            target = full[yrange[0]:yrange[1], xrange[0]:xrange[1]]
            if image.shape == target.shape:
                target[:] = image
                return full
        # Do not silently produce a coordinate mismatch for an unknown layout.
        raise ValueError(f"{source_key} has shape {image.shape}; expected {(self.ly, self.lx)}")

    def _set_peak_source(self):
        """Choose the fixed image used for candidates, independently of display."""
        self.peak_source_view = self.peak_source.currentData()
        self._refresh_peak_candidates()

    def _smoothed_peak_image(self):
        image = self._peak_source_image()
        sigma = self.peak_smoothing.value() if hasattr(self, "peak_smoothing") else 0
        return ndimage.gaussian_filter(image, sigma) if sigma else image

    def reduce_peak_threshold(self):
        self.peak_threshold.setValue(max(self.peak_threshold.minimum(), self.peak_threshold.value() * 0.9))

    def _set_peak_candidate_enabled(self, enabled):
        for control in (self.peak_source, self.peak_smoothing, self.peak_spacing, self.peak_refine_radius,
                        self.peak_threshold, self.more_candidates_button, self.include_all_button):
            control.setEnabled(enabled)
        self._refresh_peak_candidates()

    def _refresh_peak_candidates(self, *_args):
        if not hasattr(self, "peak_markers"):
            return
        if not self.peak_candidates_enabled.isChecked():
            self.candidate_peaks = np.empty((0, 2), dtype=int)
            self.candidate_index = None
            self.peak_markers.setData([], [])
            self.peak_outlines.setData([], [])
            self.rejected_peak_outlines.setData([], [])
            self.selected_peak_outline.setData([], [])
            self.peak_count_label.setText("")
            return
        occupied = self.roi_hit_map >= 0
        if occupied.any():
            excluded = ndimage.distance_transform_edt(~occupied) <= self.diameter.value() / 2
        else:
            excluded = None
        # Two circles overlap when their centres are closer than their diameter.
        # Honour a larger user spacing, but never offer overlapping candidates.
        minimum_separation = max(self.peak_spacing.value(), self.diameter.value())
        self.candidate_peaks = find_peak_candidates(
            self._peak_source_image(), self.peak_smoothing.value(), self.peak_threshold.value(),
            minimum_separation, excluded, self.peak_refine_radius.value(),
        )
        self.candidate_peaks = self._remove_border_candidates(self.candidate_peaks)
        # Review an entire 3x3 segment at a time: top-to-bottom, then
        # left-to-right.  Within each segment retain a stable spatial order.
        if len(self.candidate_peaks):
            rows = np.minimum(2, (self.candidate_peaks[:, 0] * 3 / self.ly).astype(int))
            cols = np.minimum(2, (self.candidate_peaks[:, 1] * 3 / self.lx).astype(int))
            order = np.lexsort((self.candidate_peaks[:, 1], self.candidate_peaks[:, 0], cols, rows))
            self.candidate_peaks = self.candidate_peaks[order]
        if self.current_view_index == 8:
            self.image.setImage(self._smoothed_peak_image())
        # ImageItem maps array pixel (row, column) to the square beginning at
        # (column, row), so draw the candidate at that square's centre.
        self.peak_markers.setData(
            self.candidate_peaks[:, 1] + PIXEL_CENTER_OFFSET,
            self.candidate_peaks[:, 0] + PIXEL_CENTER_OFFSET,
        )
        self.peak_count_label.setText(f"{len(self.candidate_peaks)} peaks")
        if self.candidate_index is not None and self.candidate_index >= len(self.candidate_peaks):
            self.candidate_index = len(self.candidate_peaks) - 1 if len(self.candidate_peaks) else None
        self._refresh_peak_outlines()

    def _remove_border_candidates(self, candidates):
        """Remove a candidate if any pixel in its circle lies in an excluded border."""
        left, right = self.exclude_left.value(), self.exclude_right.value()
        top, bottom = self.exclude_top.value(), self.exclude_bottom.value()
        if not any((left, right, top, bottom)):
            return candidates
        kept = []
        for y, x in candidates:
            ypix, xpix = circle_pixels(y, x, self.diameter.value(), self.ly, self.lx)
            touches_border = (
                np.any(xpix < left) or np.any(xpix >= self.lx - right)
                or np.any(ypix < top) or np.any(ypix >= self.ly - bottom)
            )
            if not touches_border:
                kept.append((y, x))
        return np.asarray(kept, dtype=int).reshape((-1, 2))

    def _refresh_peak_outlines(self):
        xs, ys, rejected_xs, rejected_ys = [], [], [], []
        for y, x in self.candidate_peaks:
            y, x = y + PIXEL_CENTER_OFFSET, x + PIXEL_CENTER_OFFSET
            radius = self.diameter.value() / 2
            angles = np.linspace(0, 2 * np.pi, 25)
            target_xs, target_ys = (rejected_xs, rejected_ys) if tuple((y, x)) in self.rejected_peaks else (xs, ys)
            target_xs.extend(x + radius * np.cos(angles)); target_xs.append(np.nan)
            target_ys.extend(y + radius * np.sin(angles)); target_ys.append(np.nan)
        self.peak_outlines.setData(xs, ys)
        self.rejected_peak_outlines.setData(rejected_xs, rejected_ys)
        if self.candidate_index is None or not len(self.candidate_peaks):
            self.selected_peak_outline.setData([], [])
            return
        y, x = self.candidate_peaks[self.candidate_index] + PIXEL_CENTER_OFFSET
        radius = self.diameter.value() / 2 + 2
        angles = np.linspace(0, 2 * np.pi, 25)
        self.selected_peak_outline.setData(x + radius * np.cos(angles), y + radius * np.sin(angles))

    def candidate_index_at(self, y, x):
        if not self.peak_candidates_enabled.isChecked() or not len(self.candidate_peaks):
            return None
        distances = ((self.candidate_peaks[:, 0] + PIXEL_CENTER_OFFSET - y) ** 2
                     + (self.candidate_peaks[:, 1] + PIXEL_CENTER_OFFSET - x) ** 2)
        index = int(np.argmin(distances))
        click_radius = min(8.0, max(3.0, self.diameter.value() / 2))
        if distances[index] <= click_radius ** 2:
            return index
        return None

    def _candidate_at(self, y, x):
        index = self.candidate_index_at(y, x)
        return self.candidate_peaks[index] if index is not None else None

    def select_candidate(self, direction):
        if not self.peak_candidates_enabled.isChecked() or not len(self.candidate_peaks):
            return
        if self.candidate_index is None:
            self.candidate_index = 0 if direction > 0 else len(self.candidate_peaks) - 1
        else:
            self.candidate_index = (self.candidate_index + direction) % len(self.candidate_peaks)
        self._show_current_candidate()

    def _show_current_candidate(self):
        """Bring the selected candidate's 3x3 segment into view."""
        if self.candidate_index is None or not len(self.candidate_peaks):
            return
        y, x = self.candidate_peaks[self.candidate_index]
        segment = (min(2, int(y * 3 / self.ly)), min(2, int(x * 3 / self.lx)))
        self.zoom_segment(*segment)
        self._refresh_peak_outlines()
        self.status.setText(
            f"Candidate {self.candidate_index + 1}/{len(self.candidate_peaks)} — "
            "comma: previous, full stop: next, N: reject, M: keep"
        )

    def accept_candidate(self):
        if self.candidate_index is None or not len(self.candidate_peaks):
            return
        # Adding the ROI removes this candidate from the recomputed list.  The
        # same list position is consequently the following candidate (or wraps
        # to the first), which keeps M/M and comma/full-stop predictable.
        next_index = self.candidate_index
        y, x = self.candidate_peaks[self.candidate_index]
        self.rejected_peaks.discard(tuple((y, x)))
        self.add_circle(float(y), float(x))
        if len(self.candidate_peaks):
            self.candidate_index = next_index % len(self.candidate_peaks)
            self._show_current_candidate()
        else:
            self.candidate_index = None

    def reject_candidate(self):
        if self.candidate_index is None or not len(self.candidate_peaks):
            return
        self.rejected_peaks.add(tuple(self.candidate_peaks[self.candidate_index]))
        self._refresh_peak_outlines()
        self.select_candidate(1)
        self._refresh_peak_candidates()

    def _handle_candidate_key(self, key):
        """Handle keys reserved for reviewing peak candidates.

        These are deliberately handled before Suite2p's window shortcuts: ``M``
        normally selects the mask image in the main GUI, but means *keep* while
        candidate review is enabled.
        """
        if not self.peak_candidates_enabled.isChecked():
            return False
        actions = {
            QtCore.Qt.Key_Comma: lambda: self.select_candidate(-1),
            QtCore.Qt.Key_Period: lambda: self.select_candidate(1),
            QtCore.Qt.Key_N: self.reject_candidate,
            QtCore.Qt.Key_M: self.accept_candidate,
        }
        action = actions.get(key)
        if action is None:
            return False
        action()
        return True

    def eventFilter(self, watched, event):
        # The main Suite2p window installs application-level image shortcuts.
        # Consume review keys here, before those shortcuts can switch views.
        if (event.type() == QtCore.QEvent.KeyPress and self.isVisible()
                and self.isActiveWindow() and self._handle_candidate_key(event.key())):
            return True
        return super().eventFilter(watched, event)

    def set_mouse_position(self, y, x):
        if 0 <= y < self.ly and 0 <= x < self.lx:
            self.mouse_position = (y, x)
        else:
            self.mouse_position = None

    def _scene_mouse_moved(self, scene_position):
        point = self.viewbox.mapSceneToView(scene_position)
        self.set_mouse_position(point.y(), point.x())

    def _install_zoom_shortcuts(self):
        shortcuts = (("X", lambda: self.zoom_at_mouse(0.8)),
                     ("K", lambda: self.zoom_at_mouse(0.8)),
                     ("Z", lambda: self.zoom_at_mouse(1.25)),
                     ("C", self.zoom_out))
        self.zoom_shortcuts = []
        for key, action in shortcuts:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.WindowShortcut)
            shortcut.activated.connect(action)
            self.zoom_shortcuts.append(shortcut)

    def zoom_at_mouse(self, factor):
        """Zoom by *factor* about the most recent in-image mouse position."""
        if self.mouse_position is None:
            return
        y, x = self.mouse_position
        (x0, x1), (y0, y1) = self.viewbox.viewRange()
        width, height = (x1 - x0) * factor, (y1 - y0) * factor
        width, height = min(width, self.lx), min(height, self.ly)
        new_x0 = np.clip(x - (x - x0) * factor, 0, self.lx - width)
        new_y0 = np.clip(y - (y - y0) * factor, 0, self.ly - height)
        self.viewbox.setRange(xRange=(new_x0, new_x0 + width), yRange=(new_y0, new_y0 + height), padding=0)
        # This is a free zoom rather than one of the fixed 3x3 segments.
        self.current_segment = None

    def _record_name(self, record):
        index = record.get("roi_index")
        if record.get("existing"):
            return f"ROI #{index} (existing)"
        suffix = f"#{index}" if index is not None else f"new {self.records.index(record) + 1}"
        return f"ROI {suffix} ({record['shape']})"

    def _refresh_tree(self):
        selected_ids = set(self.selected_ids)
        selected_id = self.selected_id
        self.tree.clear()
        by_parent = {}
        for record in self.records:
            parent_id = record.get("parent_id") if self.hierarchy_enabled.isChecked() else None
            by_parent.setdefault(parent_id, []).append(record)

        def add_records(parent_item, parent_id):
            for record in by_parent.get(parent_id, []):
                item = QTreeWidgetItem([self._record_name(record)])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, record["id"])
                parent_item.addChild(item)
                add_records(item, record["id"])
                item.setExpanded(True)
                if record["id"] in selected_ids:
                    item.setSelected(True)
                if record["id"] == selected_id:
                    self.tree.setCurrentItem(item)

        add_records(self.tree.invisibleRootItem(), None)
    def _record(self, record_id):
        return next((record for record in self.records if record["id"] == record_id), None)

    def tree_selection_changed(self):
        items = self.tree.selectedItems()
        self.selected_ids = [item.data(0, QtCore.Qt.ItemDataRole.UserRole) for item in items]
        self.selected_id = self.tree.currentItem().data(0, QtCore.Qt.ItemDataRole.UserRole) if self.tree.currentItem() else None
        if not self._suppress_selection_zoom:
            self._ensure_selected_visible()
        self._refresh_preview()

    def set_selected_as_root(self):
        record = self._record(self.selected_id)
        if record is None or record.get("existing"):
            return
        record["parent_id"] = None
        self.current_parent_id = None
        self._refresh_tree()

    def set_selected_as_parent(self):
        record = self._record(self.selected_id)
        if record is None:
            return
        self.hierarchy_enabled.setChecked(True)
        self.current_parent_id = record["id"]

    def add_circle(self, center_y, center_x):
        candidate = self._candidate_at(center_y, center_x)
        if candidate is not None:
            center_y, center_x = candidate
        self._add_record(self._new_circle_record(center_y, center_x))

    def _new_circle_record(self, center_y, center_x):
        return {
            "id": uuid.uuid4().hex, "shape": "circle", "diameter_px": int(self.diameter.value()),
            "center_yx": [float(center_y), float(center_x)],
            "parent_id": self.current_parent_id if self.hierarchy_enabled.isChecked() else None,
            "existing": False,
        }

    def include_all_candidates(self):
        """Convert all currently eligible (non-rejected) candidates into ROIs."""
        candidates = [
            (y, x) for y, x in self.candidate_peaks
            if tuple((y, x)) not in self.rejected_peaks
        ]
        if not candidates:
            self.status.setText("There are no eligible peak candidates to include.")
            return
        records = [self._new_circle_record(y, x) for y, x in candidates]
        self.records.extend(records)
        self.new_records.extend(records)
        self.selected_id = records[-1]["id"]
        self.selected_ids = [self.selected_id]
        self.extracted = False
        self.save_button.setEnabled(True)
        self._suppress_selection_zoom = True
        self._rebuild_roi_hit_map()
        self._refresh_peak_candidates()
        self._refresh_tree()
        self._suppress_selection_zoom = False
        self._refresh_preview()
        self.status.setText(f"Included {len(records)} peak candidates as ROIs. Right-click an ROI to remove it.")

    def start_freehand(self, y, x):
        self.freehand_points = [[float(y), float(x)]]
        self._refresh_drawing_preview()

    def extend_freehand(self, y, x):
        if self.freehand_points:
            self.freehand_points.append([float(y), float(x)])
            self._refresh_drawing_preview()

    def finish_freehand(self, y, x):
        self.extend_freehand(y, x)
        if len(self.freehand_points) >= 3:
            record = {
                "id": uuid.uuid4().hex, "shape": "polygon", "vertices_yx": self.freehand_points,
                "parent_id": self.current_parent_id if self.hierarchy_enabled.isChecked() else None,
                "existing": False,
            }
            self._add_record(record)
        self.freehand_points = []
        self._refresh_drawing_preview()

    def _add_record(self, record):
        self.records.append(record)
        self.new_records.append(record)
        self.selected_id = record["id"]
        self.selected_ids = [record["id"]]
        self.extracted = False
        self.save_button.setEnabled(True)
        # A new ROI was placed in the currently visible image region. Re-select
        # it in the list without triggering the list-selection segment jump.
        self._suppress_selection_zoom = True
        self._rebuild_roi_hit_map()
        self._refresh_peak_candidates()
        self._refresh_tree()
        self._suppress_selection_zoom = False
        self._refresh_preview()

    def _outline(self, record):
        if record.get("existing") and "mask_ypix" in record:
            ypix, xpix = self._record_pixels(record)
            if not ypix.size:
                return np.array([]), np.array([])
            y0, y1 = max(0, ypix.min() - 1), min(self.ly, ypix.max() + 2)
            x0, x1 = max(0, xpix.min() - 1), min(self.lx, xpix.max() + 2)
            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            mask[ypix - y0, xpix - x0] = 1
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outline_x, outline_y = [], []
            for contour in contours:
                points = contour[:, 0]
                outline_x.extend(points[:, 0] + x0)
                outline_x.append(np.nan)
                outline_y.extend(points[:, 1] + y0)
                outline_y.append(np.nan)
            return np.asarray(outline_x), np.asarray(outline_y)
        if record["shape"] == "circle":
            y, x = record["center_yx"]
            radius = record["diameter_px"] / 2.0
            angles = np.linspace(0, 2 * np.pi, 25)
            return x + radius * np.cos(angles), y + radius * np.sin(angles)
        points = np.asarray(record["vertices_yx"], dtype=float)
        points = np.vstack((points, points[0]))
        return points[:, 1], points[:, 0]

    def _record_pixels(self, record):
        if record.get("existing") and "mask_ypix" in record:
            return (
                np.asarray(record["mask_ypix"], dtype=np.int32),
                np.asarray(record["mask_xpix"], dtype=np.int32),
            )
        if record["shape"] == "circle":
            return circle_pixels(*record["center_yx"], record["diameter_px"], self.ly, self.lx)
        return polygon_pixels(record["vertices_yx"], self.ly, self.lx)

    def _rebuild_roi_hit_map(self):
        self.roi_hit_map.fill(-1)
        self.hit_record_ids = []
        for record in self.records:
            ypix, xpix = self._record_pixels(record)
            index = len(self.hit_record_ids)
            self.hit_record_ids.append(record["id"])
            self.roi_hit_map[ypix, xpix] = index

    def _record_at(self, y, x):
        y, x = int(round(y)), int(round(x))
        if not (0 <= y < self.ly and 0 <= x < self.lx):
            return None
        index = self.roi_hit_map[y, x]
        return self._record(self.hit_record_ids[index]) if index >= 0 else None

    def select_roi_at(self, y, x):
        record = self._record_at(y, x)
        if record is None:
            return False
        self.selected_id = record["id"]
        self.selected_ids = [record["id"]]
        self.tree.clearSelection()
        for item in self._tree_items():
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == record["id"]:
                item.setSelected(True)
                self.tree.setCurrentItem(item)
                break
        self.selected_id = record["id"]
        self.selected_ids = [record["id"]]
        self._refresh_preview()
        return True

    def remove_roi_at(self, y, x):
        record = self._record_at(y, x)
        if record is None:
            return False
        self.selected_id = record["id"]
        self.selected_ids = [record["id"]]
        self.delete_selected()
        return True

    def _refresh_preview(self):
        existing_xs, existing_ys = [], []
        new_xs, new_ys = [], []
        for record in self.records:
            x, y = self._outline(record)
            if record.get("existing"):
                existing_xs.extend(x); existing_xs.append(np.nan)
                existing_ys.extend(y); existing_ys.append(np.nan)
            else:
                new_xs.extend(x); new_xs.append(np.nan)
                new_ys.extend(y); new_ys.append(np.nan)
        self.existing_preview.setData(existing_xs, existing_ys)
        self.preview.setData(new_xs, new_ys)
        xs, ys = [], []
        for record_id in self.selected_ids:
            selected = self._record(record_id)
            if selected is not None:
                x, y = self._outline(selected)
                xs.extend(x); xs.append(np.nan)
                ys.extend(y); ys.append(np.nan)
        self.selected_preview.setData(xs, ys)

    def _refresh_drawing_preview(self):
        if not self.freehand_points:
            self.drawing_preview.setData([], [])
            return
        points = np.asarray(self.freehand_points)
        self.drawing_preview.setData(points[:, 1], points[:, 0])

    def zoom_segment(self, row, col):
        y0, y1 = row * self.ly / 3, (row + 1) * self.ly / 3
        x0, x1 = col * self.lx / 3, (col + 1) * self.lx / 3
        self.viewbox.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
        self.current_segment = (row, col)

    def zoom_out(self):
        self.viewbox.setRange(xRange=(0, self.lx), yRange=(0, self.ly), padding=0)
        self.current_segment = None

    def _record_center(self, record):
        if record["shape"] == "circle":
            return record["center_yx"]
        points = np.asarray(record["vertices_yx"], dtype=float)
        return points.mean(axis=0)

    def _ensure_selected_visible(self):
        record = self._record(self.selected_id)
        if record is None or self.current_segment is None:
            return
        y, x = self._record_center(record)
        segment = (min(2, int(y * 3 / self.ly)), min(2, int(x * 3 / self.lx)))
        if segment != self.current_segment:
            self.zoom_segment(*segment)

    def delete_selected(self):
        items = self.tree.selectedItems()
        if not items:
            return
        ordered_ids = [item.data(0, QtCore.Qt.ItemDataRole.UserRole) for item in self._tree_items()]
        selected_ids = {item.data(0, QtCore.Qt.ItemDataRole.UserRole) for item in items}
        first_index = min(ordered_ids.index(record_id) for record_id in selected_ids)
        deleted = 0
        for record_id in selected_ids:
            record = self._record(record_id)
            if record is None:
                continue
            for child in self.records:
                if child.get("parent_id") == record["id"]:
                    child["parent_id"] = None
            self.records.remove(record)
            if record.get("existing"):
                self.deleted_existing_indices.add(int(record["roi_index"]))
            else:
                self.new_records.remove(record)
                self.extracted = False
            deleted += 1
        if not deleted:
            return
        self.save_button.setEnabled(bool(self.new_records or self.deleted_existing_indices))
        remaining_ids = [record_id for record_id in ordered_ids if record_id not in selected_ids]
        self.selected_id = remaining_ids[max(0, first_index - 1)] if remaining_ids else None
        self.selected_ids = [self.selected_id] if self.selected_id else []
        if self.current_parent_id in selected_ids:
            self.current_parent_id = None
        self._rebuild_roi_hit_map()
        self._refresh_peak_candidates()
        self._refresh_tree()
        self._refresh_preview()

    def _remaining_roi_count(self):
        return len(self.parent.stat) - len(self.deleted_existing_indices) + len(self.new_records)

    def remove_existing_non_manual_rois(self):
        """Stage a clean-slate removal of every ROI present on opening."""
        removable_indices = [
            index for index in range(len(self.parent.stat))
            if index not in self.deleted_existing_indices
        ]
        if not removable_indices:
            QMessageBox.information(
                self, "Rapid ROIs", "There are no existing ROIs left to remove."
            )
            return

        remaining = self._remaining_roi_count() - len(removable_indices)
        warning = (
            f"Remove all {len(removable_indices)} existing ROI(s)?\n\n"
            "This includes ROIs listed in rapid_rois.json from earlier Rapid ROI sessions. "
            "Only ROIs added during this currently open Rapid ROI session will be retained.\n\n"
            "This is staged until Save ROIs is clicked. Suite2p cannot load an output "
            "with zero ROIs, so saving will be blocked unless at least one existing or "
            "new Rapid ROI remains."
        )
        if QMessageBox.question(
            self,
            "Remove all existing ROIs",
            warning,
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        self.deleted_existing_indices.update(removable_indices)
        removed_record_ids = {
            record["id"] for record in self.records if record.get("existing")
        }
        self.records = [record for record in self.records if not record.get("existing")]
        for record in self.records:
            if record.get("parent_id") in removed_record_ids:
                record["parent_id"] = None
        self.selected_ids = [
            record_id for record_id in self.selected_ids if record_id not in removed_record_ids
        ]
        self.selected_id = self.selected_ids[-1] if self.selected_ids else None
        if self.current_parent_id in removed_record_ids:
            self.current_parent_id = None
        self.save_button.setEnabled(True)
        self._rebuild_roi_hit_map()
        self._refresh_peak_candidates()
        self._refresh_tree()
        self._refresh_preview()
        if remaining:
            self.status.setText(
                f"Staged removal of all {len(removable_indices)} existing ROIs. "
                f"{remaining} ROI(s) will remain after saving."
            )
        else:
            self.status.setText(
                "Staged removal of all existing ROIs. Add and extract at least one Rapid ROI before saving."
            )

    def _tree_items(self):
        items = []
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value() is not None:
            items.append(iterator.value())
            iterator += 1
        return items

    def keyPressEvent(self, event):
        if self._handle_candidate_key(event.key()):
            return
        if event.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
            self.delete_selected()
            return
        key_to_view = {QtCore.Qt.Key_W: 1, QtCore.Qt.Key_E: 2, QtCore.Qt.Key_R: 3,
                       QtCore.Qt.Key_M: 4, QtCore.Qt.Key_T: 5, QtCore.Qt.Key_Y: 6,
                       QtCore.Qt.Key_U: 7, QtCore.Qt.Key_S: 8}
        if event.key() in key_to_view:
            self.set_view(key_to_view[event.key()])
            return
        super().keyPressEvent(event)

    def _stats_for_new_records(self):
        stats = []
        for record in self.new_records:
            if record["shape"] == "circle":
                ypix, xpix = circle_pixels(*record["center_yx"], record["diameter_px"], self.ly, self.lx)
            else:
                ypix, xpix = polygon_pixels(record["vertices_yx"], self.ly, self.lx)
            if ypix.size == 0:
                raise ValueError("An ROI has no pixels inside the image.")
            stats.append({"ypix": ypix, "xpix": xpix, "lam": np.ones(ypix.size), "npix": ypix.size,
                          "med": [float(ypix.mean()), float(xpix.mean())]})
        return stats

    def extract_rois(self):
        if not self.new_records:
            QMessageBox.information(self, "Rapid ROIs", "Add at least one new ROI before extracting.")
            return False
        progress = QProgressDialog("Preparing ROIs…", None, 0, 100, self)
        progress.setWindowTitle("Extracting rapid ROIs")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setAutoClose(False)
        progress.setMinimumDuration(0)
        progress.show()

        def report(value, message):
            progress.setLabelText(message)
            progress.setValue(int(round(value)))
            QApplication.processEvents()

        try:
            report(0, "Preparing ROI definitions…")
            stat = self._stats_for_new_records()
            if not os.path.isfile(self.parent.ops["reg_file"]):
                self.parent.ops["reg_file"] = os.path.join(self.parent.basename, "data.bin")
            result = drawroi.masks_and_traces(self.parent.ops, stat, self.parent.stat, progress_callback=report)
        except Exception as error:
            progress.close()
            traceback.print_exc()
            message = f"Rapid ROI extraction failed: {error}"
            self.status.setText(message)
            QMessageBox.critical(self, "Rapid ROI extraction failed", message)
            return False
        progress.setValue(100)
        progress.close()
        self.Fcell, self.Fneu, self.F_chan2, self.Fneu_chan2, self.Spks, _settings, self.new_stat = result
        self.extracted = True
        self.status.setText(f"Extracted and saving {len(self.new_records)} rapid ROIs…")
        return True

    def save_rois(self):
        # Extraction followed by saving is one indivisible action.  Guard
        # against a queued second click writing the newly appended rows twice.
        if self.save_started or self.save_gui:
            return
        if self._remaining_roi_count() == 0:
            QMessageBox.warning(
                self,
                "Rapid ROIs",
                "Suite2p cannot load an output with zero ROIs. Add and extract at least one Rapid ROI before saving.",
            )
            return
        self.save_started = True
        self.save_button.setEnabled(False)
        if self.new_records and not self.extracted and not self.extract_rois():
            self.save_started = False
            self.save_button.setEnabled(True)
            return
        self.save_and_quit()

    def _save_tree(self):
        kept_indices = [index for index in range(len(self.parent.stat)) if index not in self.deleted_existing_indices]
        existing_index_map = {old: new for new, old in enumerate(kept_indices)}
        existing_count = len(kept_indices)
        saved = []
        for record in self.records:
            if record.get("existing") and not record.get("save_in_tree", False):
                continue
            output = {
                key: value
                for key, value in record.items()
                if key not in {"existing", "save_in_tree", "mask_ypix", "mask_xpix"}
            }
            if record.get("existing"):
                output["roi_index"] = existing_index_map[int(output["roi_index"])]
            else:
                output["roi_index"] = existing_count + self.new_records.index(record)
            saved.append(output)
        self.tree_path.write_text(json.dumps({"schema_version": 1, "rois": saved}, indent=2), encoding="utf-8")

    def save_and_quit(self):
        if not self.extracted and self.new_records:
            return
        if self._remaining_roi_count() == 0:
            # This guard also protects against direct calls during GUI shutdown.
            QMessageBox.warning(
                self,
                "Rapid ROIs",
                "Suite2p cannot load an output with zero ROIs. Nothing was saved.",
            )
            self.save_started = False
            self.save_button.setEnabled(True)
            return
        basename = self.parent.basename
        np.save(os.path.join(basename, "stat_orig.npy"), self.parent.stat)
        keep = np.array(
            [index for index in range(len(self.parent.stat)) if index not in self.deleted_existing_indices],
            dtype=int,
        )
        stat_all = self.parent.stat[keep]
        if self.new_records:
            stat_all = np.concatenate((stat_all, self.new_stat))
        np.save(os.path.join(basename, "stat.npy"), stat_all)
        old_iscell = np.column_stack((self.parent.iscell, self.parent.probcell))[keep]
        F = self.parent.Fcell[keep]
        Fneu = self.parent.Fneu[keep]
        spks = self.parent.Spks[keep]
        if self.new_records:
            old_iscell = np.concatenate((old_iscell, np.ones((len(self.new_records), 2))))
            F = np.concatenate((F, self.Fcell))
            Fneu = np.concatenate((Fneu, self.Fneu))
            spks = np.concatenate((spks, self.Spks))
        np.save(os.path.join(basename, "iscell.npy"), old_iscell)
        np.save(os.path.join(basename, "F.npy"), F)
        np.save(os.path.join(basename, "Fneu.npy"), Fneu)
        np.save(os.path.join(basename, "spks.npy"), spks)
        if "reg_file_chan2" in self.parent.ops:
            F_chan2 = np.load(os.path.join(basename, "F_chan2.npy"))[keep]
            Fneu_chan2 = np.load(os.path.join(basename, "Fneu_chan2.npy"))[keep]
            redcell = np.load(os.path.join(basename, "redcell.npy"))[keep]
            if self.new_records:
                F_chan2 = np.concatenate((F_chan2, self.F_chan2))
                Fneu_chan2 = np.concatenate((Fneu_chan2, self.Fneu_chan2))
                redcell = np.concatenate((redcell, np.zeros((len(self.new_records), 2))))
            np.save(os.path.join(basename, "F_chan2.npy"), F_chan2)
            np.save(os.path.join(basename, "Fneu_chan2.npy"), Fneu_chan2)
            np.save(os.path.join(basename, "redcell.npy"), redcell)
        self._save_tree()
        io.load_proc(self.parent)
        self.save_gui = True
        self.close()

    def closeEvent(self, event):
        if not self.save_gui and (self.new_records or self.deleted_existing_indices):
            answer = QMessageBox.question(self, "Rapid ROIs", "Discard unextracted rapid ROIs?", QMessageBox.Yes | QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        QApplication.instance().removeEventFilter(self)
        event.accept()
