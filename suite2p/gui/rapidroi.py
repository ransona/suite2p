"""Fast manual ROI editor for adding many fixed circular or freehand ROIs."""
import json
import os
import traceback
import uuid
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from matplotlib.path import Path as MplPath
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import (
    QApplication, QAbstractItemView,
    QButtonGroup, QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressDialog, QPushButton, QSpinBox, QTreeWidget,
    QTreeWidgetItem, QTreeWidgetItemIterator, QVBoxLayout, QWidget,
)

from . import drawroi, io


TREE_FILENAME = "rapid_rois.json"


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


class RapidROIViewBox(pg.ViewBox):
    def __init__(self, editor):
        super().__init__(lockAspect=True, invertY=True)
        self.editor = editor

    def mouseClickEvent(self, event):
        point = self.mapSceneToView(event.scenePos())
        if event.button() == QtCore.Qt.RightButton:
            self.editor.remove_roi_at(point.y(), point.x())
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton:
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
    VIEW_SPECS = (("W", "Mean", 1), ("E", "Enhanced mean", 2), ("R", "Correlation", 3),
                  ("M", "Mask", 4), ("T", "Max projection", 5),
                  ("Y", "Channel 2 corrected", 6), ("U", "Channel 2", 7))

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
        self.extracted = False
        self.save_gui = False
        self.setWindowTitle("Suite2p Rapid ROIs")
        self.resize(1300, 900)
        self._load_saved_tree()
        self._build_ui()
        self._install_zoom_shortcuts()
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
                self.records.append(record)
        except Exception as error:
            print(f"Could not load {self.tree_path}: {error}")

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
        self.save_button = QPushButton("Save ROIs")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_rois)
        left_layout.addWidget(self.save_button)
        layout.addWidget(left, 1)

        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        controls = QGridLayout()
        controls.addWidget(QLabel("Mode"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Click circles", "circle")
        self.mode_combo.addItem("Freehand shape", "freehand")
        self.mode_combo.currentIndexChanged.connect(self._update_mode_status)
        controls.addWidget(self.mode_combo, 0, 1)
        controls.addWidget(QLabel("Circle diameter (pixels)"), 1, 0)
        self.diameter = QSpinBox()
        self.diameter.setRange(1, max(self.ly, self.lx))
        self.diameter.setValue(12)
        controls.addWidget(self.diameter, 1, 1)
        controls.addWidget(QLabel("View"), 0, 2)
        self.view_group = QButtonGroup(self)
        view_box = QWidget()
        view_layout = QHBoxLayout(view_box)
        view_layout.setContentsMargins(0, 0, 0, 0)
        for key, label, index in self.VIEW_SPECS[:3]:
            button = QPushButton(f"{key}: {label}")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, idx=index: self.set_view(idx))
            self.view_group.addButton(button, index)
            view_layout.addWidget(button)
        controls.addWidget(view_box, 0, 3)
        controls.addWidget(QLabel("Zoom"), 1, 2)
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
        controls.addWidget(zoom_box, 1, 3)
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
        self.preview = pg.PlotCurveItem(pen=pg.mkPen((0, 220, 255), width=1.5), connect="finite")
        self.selected_preview = pg.PlotCurveItem(pen=pg.mkPen((255, 220, 0), width=3), connect="finite")
        self.drawing_preview = pg.PlotCurveItem(pen=pg.mkPen((255, 120, 100), width=2))
        self.viewbox.addItem(self.preview)
        self.viewbox.addItem(self.selected_preview)
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
        image = self.parent.views[index]
        self.image.setImage(image)
        button = self.view_group.button(index)
        if button is not None:
            button.setChecked(True)
        self._refresh_preview()

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
        record = {
            "id": uuid.uuid4().hex, "shape": "circle", "diameter_px": int(self.diameter.value()),
            "center_yx": [float(center_y), float(center_x)],
            "parent_id": self.current_parent_id if self.hierarchy_enabled.isChecked() else None,
            "existing": False,
        }
        self._add_record(record)

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
        self._refresh_tree()
        self._suppress_selection_zoom = False
        self._refresh_preview()

    def _outline(self, record):
        if record["shape"] == "circle":
            y, x = record["center_yx"]
            radius = record["diameter_px"] / 2.0
            angles = np.linspace(0, 2 * np.pi, 25)
            return x + radius * np.cos(angles), y + radius * np.sin(angles)
        points = np.asarray(record["vertices_yx"], dtype=float)
        points = np.vstack((points, points[0]))
        return points[:, 1], points[:, 0]

    def _record_pixels(self, record):
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
        xs, ys = [], []
        for record in self.records:
            x, y = self._outline(record)
            xs.extend(x); xs.append(np.nan)
            ys.extend(y); ys.append(np.nan)
        self.preview.setData(xs, ys)
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
        self._refresh_tree()
        self._refresh_preview()

    def _tree_items(self):
        items = []
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value() is not None:
            items.append(iterator.value())
            iterator += 1
        return items

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
            self.delete_selected()
            return
        key_to_view = {QtCore.Qt.Key_W: 1, QtCore.Qt.Key_E: 2, QtCore.Qt.Key_R: 3,
                       QtCore.Qt.Key_M: 4, QtCore.Qt.Key_T: 5, QtCore.Qt.Key_Y: 6,
                       QtCore.Qt.Key_U: 7}
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
        if self.new_records and not self.extracted and not self.extract_rois():
            return
        self.save_and_quit()

    def _save_tree(self):
        kept_indices = [index for index in range(len(self.parent.stat)) if index not in self.deleted_existing_indices]
        existing_index_map = {old: new for new, old in enumerate(kept_indices)}
        existing_count = len(kept_indices)
        saved = []
        for record in self.records:
            output = {key: value for key, value in record.items() if key != "existing"}
            if record.get("existing"):
                output["roi_index"] = existing_index_map[int(output["roi_index"])]
            else:
                output["roi_index"] = existing_count + self.new_records.index(record)
            saved.append(output)
        self.tree_path.write_text(json.dumps({"schema_version": 1, "rois": saved}, indent=2), encoding="utf-8")

    def save_and_quit(self):
        if not self.extracted and self.new_records:
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
