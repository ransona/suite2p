"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore
from pyqtgraph import Point
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu

from . import masks


class TraceBox(pg.PlotItem):

    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True,
                 invertY=False, enableMenu=True, name=None, invertX=False):
        super(TraceBox, self).__init__()
        self.parent = parent

    def mouseDoubleClickEvent(self, ev):
        self.zoom_plot()

    def zoom_plot(self):
        self.setXRange(0, self.parent.Fcell.shape[1])
        self.setYRange(self.parent.fmin, self.parent.fmax)
        self.parent.show()


class ViewBox(pg.ViewBox):

    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True,
                 invertY=False, enableMenu=True, name=None, invertX=False):
        #pg.ViewBox.__init__(self, border, lockAspect, enableMouse,
        #invertY, enableMenu, name, invertX)
        super(ViewBox, self).__init__()
        self.border = fn.mkPen(border)
        if enableMenu:
            self.menu = ViewBoxMenu(self)
        self.name = name
        self.parent = parent
        if self.name == "plot2":
            self.setXLink(parent.p1)
            self.setYLink(parent.p1)

        # set state
        self.state["enableMenu"] = enableMenu
        self.state["yInverted"] = invertY

    def mouseDoubleClickEvent(self, ev):
        if self.parent.loaded:
            self.zoom_plot()

    def mouseClickEvent(self, ev):
        if self.parent.loaded:
            pos = self.mapSceneToView(ev.scenePos())
            posy = int(pos.x())
            posx = int(pos.y())
            if self.name == "plot1":
                iplot = 0
            else:
                iplot = 1
            if posy >= 0 and posx >= 0 and posy <= self.parent.Lx and posx <= self.parent.Ly:
                ichosen = int(self.parent.rois["iROI"][iplot, 0, posx, posy])
                if ichosen < 0:
                    if ev.button() == QtCore.Qt.RightButton and self.menuEnabled():
                        self.raiseContextMenu(ev)
                    return
                else:
                    if ev.button() == QtCore.Qt.RightButton:
                        if ichosen not in self.parent.imerge:
                            self.parent.imerge = [ichosen]
                            self.parent.ichosen = ichosen
                        masks.flip_plot(self.parent)
                    else:
                        merged = False
                        if ev.modifiers() == QtCore.Qt.ShiftModifier or ev.modifiers(
                        ) == QtCore.Qt.ControlModifier:
                            if self.parent.iscell[self.parent.imerge[
                                    0]] == self.parent.iscell[ichosen]:
                                if ichosen not in self.parent.imerge:
                                    self.parent.imerge.append(ichosen)
                                    self.parent.ichosen = ichosen
                                    merged = True
                                elif ichosen in self.parent.imerge and len(
                                        self.parent.imerge) > 1:
                                    self.parent.imerge.remove(ichosen)
                                    self.parent.ichosen = self.parent.imerge[0]
                                    merged = True
                        if not merged:
                            self.parent.imerge = [ichosen]
                            self.parent.ichosen = ichosen

                    if self.parent.isROI:
                        self.parent.ROI_remove()
                    if not self.parent.sizebtns.button(1).isChecked():
                        for btn in self.parent.topbtns.buttons():
                            if btn.isChecked():
                                btn.setStyleSheet(self.parent.styleUnpressed)
                    self.parent.update_plot()

    def zoom_plot(self):
        reset_image_view(self.parent)
        self.parent.show()


def synchronize_views(parent):
    """Keep the two main image panes at the same pan/zoom range."""
    parent.p2.setXLink(parent.p1)
    parent.p2.setYLink(parent.p1)
    parent._view_sync_connected = True
    parent._syncing_view_range = False


def configure_image_view_aspect(parent, mode=None):
    """Lock aspect only for panes that are visible in the selected mode."""
    if mode is None:
        mode = parent.sizebtns.checkedId()
    if mode == -1:
        parent.p1.setAspectLocked(lock=False)
        parent.p2.setAspectLocked(lock=False)
        return
    ratio = getattr(parent, "xyrat", 1.0)
    parent.p1.setAspectLocked(lock=mode != 2, ratio=ratio)
    parent.p2.setAspectLocked(lock=mode != 0, ratio=ratio)


def set_shared_image_range(parent, x_range, y_range, mode=None):
    if mode is None:
        mode = parent.sizebtns.checkedId()
    configure_image_view_aspect(parent, mode)
    parent._syncing_view_range = True
    try:
        parent.p2.setXLink(None)
        parent.p2.setYLink(None)
        source = parent.p2 if mode == 2 else parent.p1
        target = parent.p1 if mode == 2 else parent.p2
        source.setRange(xRange=x_range, yRange=y_range, padding=0)
        fitted_range = source.viewRange()
        target.setRange(xRange=fitted_range[0], yRange=fitted_range[1], padding=0)
    finally:
        parent.p2.setXLink(parent.p1)
        parent.p2.setYLink(parent.p1)
        parent._syncing_view_range = False


def restore_image_range_after_layout(parent, x_range, y_range, mode):
    """Restore a range after pane visibility changes have resized the layout."""
    parent.p1.setAspectLocked(lock=False)
    parent.p2.setAspectLocked(lock=False)
    set_shared_image_range(parent, x_range, y_range, mode=-1)

    def restore():
        if getattr(parent, "loaded", False) and parent.sizebtns.checkedId() == mode:
            set_shared_image_range(parent, x_range, y_range, mode=mode)

    QtCore.QTimer.singleShot(0, restore)


def reset_image_view(parent):
    """Reset both main image panes to the full frame and prevent blank panning."""
    synchronize_views(parent)
    xpad = parent.ops["Lx"] * 0.5
    ypad = parent.ops["Ly"] * 0.5
    for viewbox in (parent.p1, parent.p2):
        viewbox.setLimits(xMin=-xpad, xMax=parent.ops["Lx"] + xpad,
                          yMin=-ypad, yMax=parent.ops["Ly"] + ypad)

    set_shared_image_range(parent, (0, parent.ops["Lx"]),
                           (0, parent.ops["Ly"]))
    synchronize_views(parent)


def reset_image_view_after_layout(parent):
    QtCore.QTimer.singleShot(
        0,
        lambda: reset_image_view(parent) if getattr(parent, "loaded", False) else None,
    )


def init_range(parent):
    reset_image_view(parent)
    parent.p3.setLimits(xMin=0, xMax=parent.Fcell.shape[1])
    parent.trange = np.arange(0, parent.Fcell.shape[1])


def ROI_index(settings, stat):
    """matrix Ly x Lx where each pixel is an ROI index (-1 if no ROI present)"""
    ncells = len(stat) - 1
    Ly = settings["Ly"]
    Lx = settings["Lx"]
    iROI = -1 * np.ones((Ly, Lx), dtype=np.int32)
    for n in range(ncells):
        ypix = stat[n]["ypix"][~stat[n]["overlap"]]
        if ypix is not None:
            xpix = stat[n]["xpix"][~stat[n]["overlap"]]
            iROI[ypix, xpix] = n
    return iROI
