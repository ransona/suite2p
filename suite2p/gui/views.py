"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
import pyqtgraph as pg
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import (QPushButton, QSlider, QButtonGroup, QLabel, QStyle,
                            QStyleOptionSlider, QApplication, QMainWindow)
from qtpy.QtGui import QPainter

from .. import registration


def _normalize_img(img):
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros(img.shape, dtype=np.float32)
    p1, p99 = np.percentile(img[finite], [1, 99])
    if p99 <= p1:
        return np.zeros(img.shape, dtype=np.float32)
    img = (img - p1) / (p99 - p1)
    return np.clip(img, 0, 1).astype(np.float32)


def _to_full_frame(parent, img):
    img = np.asarray(img)
    if img.shape == (parent.Ly, parent.Lx):
        return img
    full = np.zeros((parent.Ly, parent.Lx), dtype=img.dtype)
    yr = slice(parent.ops["yrange"][0], parent.ops["yrange"][1])
    xr = slice(parent.ops["xrange"][0], parent.ops["xrange"][1])
    if img.shape == full[yr, xr].shape:
        full[yr, xr] = img
        return full
    return img


def _as_scale_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value)
    value = np.asarray(value)
    if value.ndim == 3:
        return [value[k] for k in range(value.shape[0])]
    return []


def _shared_levels(images, threshold=None):
    vals = []
    for img in images:
        arr = np.asarray(img, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return (0, 1)
    vals = np.concatenate(vals)
    lo = 0 if vals.min() >= 0 else float(np.percentile(vals, 1))
    hi = float(threshold) if threshold is not None and np.isfinite(float(threshold)) else float(np.percentile(vals, 99.5))
    if hi <= lo:
        hi = float(np.percentile(vals, 99.5))
    if hi <= lo:
        hi = lo + 1
    return (lo, hi)


class MaskDiagnosticWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MaskDiagnosticWindow, self).__init__(parent)
        self.parent = parent
        self.setWindowTitle("Suite2p detection mask diagnostics")
        self.win = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.win)
        self.resize(1500, 900)
        self._build()

    def _add_image(self, row, col, title, img, full_frame=True, levels=None):
        if img is None:
            return
        img = _to_full_frame(self.parent, img) if full_frame else np.asarray(img)
        view = self.win.addViewBox(row=row, col=col, lockAspect=True, invertY=True)
        view.setMenuEnabled(False)
        item = pg.ImageItem()
        view.addItem(item)
        if levels is None:
            item.setImage(_normalize_img(img), levels=(0, 1))
        else:
            item.setImage(np.asarray(img, dtype=np.float32), levels=levels)
        self.win.addLabel(title, row=row + 1, col=col)

    def _add_colorbar(self, row, col, title, levels):
        bar = np.linspace(levels[0], levels[1], 256, dtype=np.float32)[:, np.newaxis]
        view = self.win.addViewBox(row=row, col=col, lockAspect=False, invertY=False)
        view.setMenuEnabled(False)
        item = pg.ImageItem()
        view.addItem(item)
        item.setImage(bar, levels=levels)
        self.win.addLabel(f"{title}: {levels[0]:0.2f} to {levels[1]:0.2f}", row=row + 1, col=col)

    def _build(self):
        ops = self.parent.ops
        threshold = ops.get("signal_peak_threshold", None)
        corr_scales = [_to_full_frame(self.parent, img) for img in _as_scale_list(ops.get("Vcorr_scales"))]
        corr_signal_scales = [
            _to_full_frame(self.parent, img) for img in _as_scale_list(ops.get("Vcorr_signal_scales"))
        ]
        peak_maps = _as_scale_list(ops.get("Vmap"))
        peak_signal_maps = _as_scale_list(ops.get("Vmap_signal"))
        activity_levels = _shared_levels(
            [ops.get("Vcorr")] + corr_scales + corr_signal_scales + peak_maps + peak_signal_maps,
            threshold=threshold,
        )
        bright_area = ops.get("bright_area_percentile", ops.get("signal_top_percent", "n/a"))
        include_area = ops.get("include_area_percentile", ops.get("dark_percentile", "n/a"))
        include_threshold = ops.get("include_area_threshold", ops.get("dark_mask_threshold", "n/a"))
        include_valid = ops.get("include_area_valid_fraction", ops.get("dark_mask_valid_fraction", "n/a"))
        include_fraction = ops.get("include_min_roi_fraction", ops.get("dark_min_roi_fraction", "n/a"))
        sparsery_rejected = ops.get("n_sparsery_rejected_size", {})
        title = (
            "Mask diagnostics"
            f" | Bright area {bright_area}%"
            f" valid {ops.get('threshold_mask_valid_fraction', 'n/a')}"
            f" threshold {ops.get('bright_area_threshold', 'n/a')}"
            f" | Include area {include_area}%"
            f" threshold {include_threshold}"
            f" valid {include_valid}"
            f" | peak threshold {ops.get('signal_peak_threshold', 'n/a')}"
            f" | unscaled {ops.get('signal_peak_threshold_unscaled', 'n/a')}"
            f" | lateral exclude {ops.get('lateral_exclude_percent', 'n/a')}%"
            f" | include ROI fraction {include_fraction}"
            f" | ROI pixels {ops.get('min_roi_pixels', 'n/a')}-{ops.get('max_roi_pixels', 'n/a')}"
            f" removed {ops.get('n_removed_min_roi_pixels', 'n/a')}/{ops.get('n_removed_max_roi_pixels', 'n/a')}"
            f" | ROI width {ops.get('min_roi_width', 'n/a')}-{ops.get('max_roi_width', 'n/a')}"
            f" removed {ops.get('n_removed_min_roi_width', 'n/a')}/{ops.get('n_removed_max_roi_width', 'n/a')}"
            f" | ROI height {ops.get('min_roi_height', 'n/a')}-{ops.get('max_roi_height', 'n/a')}"
            f" removed {ops.get('n_removed_min_roi_height', 'n/a')}/{ops.get('n_removed_max_roi_height', 'n/a')}"
            f" | include removed {ops.get('n_removed_include_area', 'n/a')}"
            f" | peaks checked {ops.get('n_sparsery_candidates', 'n/a')}/{ops.get('max_peaks_to_check', 'n/a')}"
            f" | sparsery size rejected {sparsery_rejected}"
        )
        self.win.addLabel(title, row=0, col=0, colspan=6)

        self._add_image(1, 0, "Mean image", ops.get("meanImg"), full_frame=True)
        self._add_image(1, 1, "Smoothed mean signal", ops.get("meanImg_signal_map"), full_frame=True)
        self._add_image(1, 2, "Bright area mask", ops.get("meanImg_signal_mask"), full_frame=True)
        self._add_image(1, 3, "Include area mask", ops.get("meanImg_roi_signal_mask"), full_frame=True)
        self._add_image(1, 4, "Max correlation", ops.get("Vcorr"), full_frame=True,
                        levels=activity_levels)
        self._add_colorbar(1, 5, "Activity scale", activity_levels)

        self._add_image(11, 0, "Mean x bright area", ops.get("meanImg_signal_masked"), full_frame=True)
        self._add_image(11, 1, "Mean x include area", ops.get("meanImg_roi_signal_masked"), full_frame=True)
        if "meanImg_signal_mask" in ops and "meanImg_roi_signal_mask" in ops:
            extra_roi_mask = (
                ops["meanImg_roi_signal_mask"].astype(np.float32) -
                ops["meanImg_signal_mask"].astype(np.float32)
            ) > 0
            self._add_image(11, 2, "Include-only pixels", extra_roi_mask, full_frame=True)

        for k, img in enumerate(corr_scales):
            self._add_image(3, k, f"Corr scale {k}", img, full_frame=True,
                            levels=activity_levels)

        self._add_colorbar(3, 5, "Shared corr scale", activity_levels)

        for k, img in enumerate(corr_signal_scales):
            self._add_image(5, k, f"Masked corr scale {k}", img, full_frame=True,
                            levels=activity_levels)

        self._add_colorbar(5, 5, "Shared masked scale", activity_levels)

        for k, img in enumerate(peak_maps):
            self._add_image(7, k, f"Peak map scale {k}", img, full_frame=False,
                            levels=activity_levels)

        self._add_colorbar(7, 5, "Peak threshold scale", activity_levels)

        for k, img in enumerate(peak_signal_maps):
            self._add_image(9, k, f"Masked peak map scale {k}", img, full_frame=False,
                            levels=activity_levels)

        self._add_colorbar(9, 5, "Masked peak scale", activity_levels)


def open_mask_diagnostics(parent):
    if "meanImg_signal_mask" not in parent.ops:
        return
    parent.mask_diagnostic_window = MaskDiagnosticWindow(parent)
    parent.mask_diagnostic_window.show()


def make_buttons(parent):
    """ view buttons"""
    # view buttons
    parent.view_names = [
        "Q: ROIs",
        "W: mean img",
        "E: mean img (enhanced)",
        "R: Corr",
        "Mask",
        "T: max projection",
        "Y: mean img chan2, corr",
        "U: mean img chan2",
    ]
    b = 0
    parent.viewbtns = QButtonGroup(parent)
    vlabel = QLabel(parent)
    vlabel.setText("<font color='white'>Background</font>")
    vlabel.setFont(parent.boldfont)
    vlabel.resize(vlabel.minimumSizeHint())
    parent.l0.addWidget(vlabel, 1, 0, 1, 1)
    for names in parent.view_names:
        btn = ViewButton(b, "&" + names, parent)
        parent.viewbtns.addButton(btn, b)
        if b > 0:
            parent.l0.addWidget(btn, b + 2, 0, 1, 1)
        else:
            parent.l0.addWidget(btn, b + 2, 0, 1, 1)
            label = QLabel("sat: ")
            label.setStyleSheet("color: white;")
            parent.l0.addWidget(label, b + 2, 1, 1, 1)
        btn.setEnabled(False)
        b += 1
    parent.viewbtns.setExclusive(True)
    slider = RangeSlider(parent)
    slider.setMinimum(0)
    slider.setMaximum(255)
    slider.setLow(0)
    slider.setHigh(255)
    slider.setTickPosition(QSlider.TicksBelow)
    parent.l0.addWidget(slider, 3, 1, len(parent.view_names) - 2, 1)

    b += 2
    return b


def init_views(parent):
    """ make views using parent.ops

    views in order:
        "Q: ROIs",
        "W: mean img",
        "E: mean img (enhanced)",
        "R: Corr",
        "Mask",
        "T: max projection",
        "Y: mean img chan2, corr",
        "U: mean img chan2",

    assigns parent.views

    """
    parent.Ly, parent.Lx = parent.ops["Ly"], parent.ops["Lx"]
    parent.views = np.zeros((8, parent.Ly, parent.Lx, 3), np.float32)
    for k in range(8):
        if k == 2:
            if "meanImgE" not in parent.ops:
                meanImgE = registration.highpass_mean_image(parent.ops["meanImg"], 
                                                              parent.ops.get("aspect", 1))
                parent.ops["meanImgE"] = meanImgE
            mimg = parent.ops["meanImgE"]
        elif k == 1:
            img = parent.ops["meanImg"]
            mimg1 = np.percentile(img, 1)
            mimg99 = np.percentile(img, 99)
            img = (img - mimg1) / (mimg99 - mimg1)
            img = np.clip(img, 0, 1)
            mimg = np.zeros((parent.Ly, parent.Lx), np.float32)
            if img.shape[0] != parent.Ly or img.shape[1] != parent.Lx:
                mimg[parent.ops["yrange"][0]:parent.ops["yrange"][1],
                     parent.ops["xrange"][0]:parent.ops["xrange"][1]] = img
            else:
                mimg = img
        elif k == 3:
            if "Vcorr" in parent.ops:
                vcorr = parent.ops["Vcorr"]
                mimg1 = np.percentile(vcorr, 1)
                mimg99 = np.percentile(vcorr, 99)
                vcorr = (vcorr - mimg1) / (mimg99 - mimg1)
                vcorr = np.clip(vcorr, 0, 1)
                mimg = np.zeros((parent.Ly, parent.Lx), np.float32)
                mimg[parent.ops["yrange"][0]:parent.ops["yrange"][1],
                     parent.ops["xrange"][0]:parent.ops["xrange"][1]] = vcorr
                mimg = np.maximum(0, np.minimum(1, mimg))
            else:
                mimg = np.zeros((parent.Ly, parent.Lx), np.float32)
        elif k == 4:
            mask_key = "meanImg_roi_signal_mask" if "meanImg_roi_signal_mask" in parent.ops else "meanImg_signal_mask"
            if mask_key in parent.ops:
                signal_mask = parent.ops[mask_key].astype(np.float32)
                mimg = np.zeros((parent.Ly, parent.Lx), np.float32)
                if signal_mask.shape[0] == parent.Ly and signal_mask.shape[1] == parent.Lx:
                    mimg = signal_mask
                else:
                    mimg[parent.ops["yrange"][0]:parent.ops["yrange"][1],
                         parent.ops["xrange"][0]:parent.ops["xrange"][1]] = signal_mask
            else:
                mimg = np.zeros((parent.Ly, parent.Lx), np.float32)
        elif k == 5:
            if "max_proj" in parent.ops:
                mproj = parent.ops["max_proj"]
                mimg1 = np.percentile(mproj, 1)
                mimg99 = np.percentile(mproj, 99)
                mproj = (mproj - mimg1) / (mimg99 - mimg1)
                mimg = np.zeros((parent.Ly, parent.Lx), np.float32)
                try:
                    mimg[parent.ops["yrange"][0]:parent.ops["yrange"][1],
                         parent.ops["xrange"][0]:parent.ops["xrange"][1]] = mproj
                except:
                    print("maxproj not in combined view")
                mimg = np.maximum(0, np.minimum(1, mimg))
            else:
                mimg = 0.5 * np.ones((parent.Ly, parent.Lx), np.float32)
        elif k == 6:
            if "meanImg_chan2_corrected" in parent.ops:
                mimg = parent.ops["meanImg_chan2_corrected"]
                mimg1 = np.percentile(mimg, 1)
                mimg99 = np.percentile(mimg, 99)
                mimg = (mimg - mimg1) / (mimg99 - mimg1)
                mimg = np.maximum(0, np.minimum(1, mimg))
        elif k == 7:
            if "meanImg_chan2" in parent.ops:
                mimg = parent.ops["meanImg_chan2"]
                mimg1 = np.percentile(mimg, 1)
                mimg99 = np.percentile(mimg, 99)
                mimg = (mimg - mimg1) / (mimg99 - mimg1)
                mimg = np.maximum(0, np.minimum(1, mimg))
        else:
            mimg = np.zeros((parent.Ly, parent.Lx), np.float32)

        mimg *= 255
        mimg = mimg.astype(np.uint8)
        parent.views[k] = np.tile(mimg[:, :, np.newaxis], (1, 1, 3))


def plot_views(parent):
    """ set parent.view1 and parent.view2 image based on parent.ops_plot["view"]"""
    k = parent.ops_plot["view"]
    parent.view1.setImage(parent.views[k], levels=parent.ops_plot["saturation"])
    parent.view2.setImage(parent.views[k], levels=parent.ops_plot["saturation"])
    parent.view1.show()
    parent.view2.show()


class ViewButton(QPushButton):
    """ custom QPushButton class for quadrant plotting
        requires buttons to put into a QButtonGroup (parent.viewbtns)
         allows only 1 button to pressed at a time
    """

    def __init__(self, bid, Text, parent=None):
        super(ViewButton, self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid):
        for b in range(len(parent.views)):
            if parent.viewbtns.button(b).isEnabled():
                parent.viewbtns.button(b).setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        parent.ops_plot["view"] = bid
        parent.update_plot()
        if bid == 4:
            open_mask_diagnostics(parent)


class RangeSlider(QSlider):
    """ A slider for ranges.

        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.

        This class emits the same signals as the QSlider base class, with the
        exception of valueChanged

        Found this slider here: https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
        and modified it
    """

    def __init__(self, parent=None, *args):
        super(RangeSlider, self).__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()

        self.pressed_control = QStyle.SC_None
        self.hover_control = QStyle.SC_None
        self.click_offset = 0

        self.setOrientation(QtCore.Qt.Vertical)
        self.setTickPosition(QSlider.TicksRight)
        self.setStyleSheet(\
                "QSlider::handle:horizontal {\
                background-color: white;\
                border: 1px solid #5c5c5c;\
                border-radius: 0px;\
                border-color: black;\
                height: 8px;\
                width: 6px;\
                margin: -8px 2; \
                }"                                                                        )

        #self.opt = QStyleOptionSlider()
        #self.opt.orientation=QtCore.Qt.Vertical
        #self.initStyleOption(self.opt)
        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0
        self.parent = parent

    def level_change(self):
        if self.parent is not None:
            if self.parent.loaded:
                self.parent.ops_plot["saturation"] = [self._low, self._high]
                self.parent.update_plot()

    def low(self):
        return self._low

    def setLow(self, low):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()

    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp
        painter = QPainter(self)
        style = QApplication.style()

        for i, value in enumerate([self._low, self._high]):
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            # Only draw the groove for the first slider so it doesn"t get drawn
            # on top of the existing ones every time
            if i == 0:
                opt.subControls = QStyle.SC_SliderHandle  #QStyle.SC_SliderGroove | QStyle.SC_SliderHandle
            else:
                opt.subControls = QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QStyle.SC_SliderTickmarks

            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
                opt.state |= QStyle.State_Sunken
            else:
                opt.activeSubControls = self.hover_control

            opt.sliderPosition = value
            opt.sliderValue = value
            style.drawComplexControl(QStyle.CC_Slider, opt, painter, self)

    def mousePressEvent(self, event):
        event.accept()

        style = QApplication.style()
        button = event.button()
        # In a normal slider control, when the user clicks on a point in the
        # slider"s total range, but not on the slider part of the control the
        # control would jump the slider value to where the user clicked.
        # For this control, clicks which are not direct hits will slide both
        # slider parts
        if button:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = -1

            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(),
                                                  self)
                if hit == style.SC_SliderHandle:
                    self.active_slider = i
                    self.pressed_control = hit

                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)

                    break

            if self.active_slider < 0:
                self.pressed_control = QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(self.__pick(
                    event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.pressed_control != QStyle.SC_SliderHandle:
            event.ignore()
            return

        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos

        self.click_offset = new_pos
        self.update()

    def mouseReleaseEvent(self, event):
        self.level_change()

    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()

    def __pixelPosToRangeValue(self, pos):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QApplication.style()

        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(self.minimum(), self.maximum(),
                                             pos - slider_min, slider_max - slider_min,
                                             opt.upsideDown)
