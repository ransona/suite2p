"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import re
from pathlib import Path

import numpy as np
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QPushButton, QButtonGroup, QLabel, QLineEdit

from . import graphics


def plane_navigation_paths(basename):
    """Return adjacent-plane and paired-channel stat paths for a loaded plane."""
    plane_dir = Path(basename)
    match = re.fullmatch(r"plane(\d+)", plane_dir.name)
    if not match or plane_dir.parent.name != "suite2p":
        return None, None, None

    plane_number = int(match.group(1))
    plane_dirs = sorted(
        (
            path for path in plane_dir.parent.iterdir()
            if path.is_dir() and re.fullmatch(r"plane\d+", path.name)
            and (path / "stat.npy").is_file()
        ),
        key=lambda path: int(path.name[5:]),
    )
    try:
        index = plane_dirs.index(plane_dir)
    except ValueError:
        return None, None, None

    previous = plane_dirs[index - 1] / "stat.npy" if index else None
    following = plane_dirs[index + 1] / "stat.npy" if index + 1 < len(plane_dirs) else None

    # Lab outputs store channel 2 under <experiment>/ch2/suite2p/planeN,
    # alongside the canonical <experiment>/suite2p/planeN output tree.
    if plane_dir.parent.parent.name == "ch2":
        counterpart = plane_dir.parent.parent.parent / "suite2p" / plane_dir.name / "stat.npy"
    else:
        counterpart = plane_dir.parent.parent / "ch2" / "suite2p" / plane_dir.name / "stat.npy"
    return previous, following, counterpart if counterpart.is_file() else None


def make_selection(parent):
    """ buttons to draw a square on view """
    parent.topbtns = QButtonGroup()
    ql = QLabel("select cells")
    ql.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
    parent.l0.addWidget(ql, 0, 2, 1, 2)
    pos = [2, 3, 4]
    for b in range(3):
        btn = TopButton(b, parent)
        btn.setFont(QtGui.QFont("Arial", 8))
        parent.topbtns.addButton(btn, b)
        parent.l0.addWidget(btn, 0, (pos[b]) * 2, 1, 2)
        btn.setEnabled(False)
    parent.topbtns.setExclusive(True)
    parent.isROI = False
    parent.ROIplot = 0
    ql = QLabel("n=")
    ql.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    ql.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
    parent.l0.addWidget(ql, 0, 10, 1, 1)
    parent.topedit = QLineEdit(parent)
    parent.topedit.setValidator(QtGui.QIntValidator(0, 500))
    parent.topedit.setText("40")
    parent.ntop = 40
    parent.topedit.setFixedWidth(35)
    parent.topedit.setAlignment(QtCore.Qt.AlignRight)
    parent.topedit.returnPressed.connect(parent.top_number_chosen)
    parent.l0.addWidget(parent.topedit, 0, 11, 1, 1)


# minimize view
def make_cellnotcell(parent):
    """ buttons for cell / not cell views at top """
    # number of ROIs in each image
    parent.lcell0 = QLabel("")
    parent.lcell0.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    parent.l0.addWidget(parent.lcell0, 0, 12, 1, 2)
    parent.lcell1 = QLabel("")
    parent.l0.addWidget(parent.lcell1, 0, 20, 1, 2)

    parent.prev_plane_button = QPushButton("Prev")
    parent.prev_plane_button.clicked.connect(lambda: load_adjacent_plane(parent, -1))
    parent.l0.addWidget(parent.prev_plane_button, 0, 22, 1, 1)
    parent.plane_label = QLabel("Plane —")
    parent.plane_label.setAlignment(QtCore.Qt.AlignCenter)
    parent.l0.addWidget(parent.plane_label, 0, 23, 1, 2)
    parent.next_plane_button = QPushButton("Next")
    parent.next_plane_button.clicked.connect(lambda: load_adjacent_plane(parent, 1))
    parent.l0.addWidget(parent.next_plane_button, 0, 25, 1, 1)
    parent.channel_button = QPushButton("Channel")
    parent.channel_button.clicked.connect(lambda: load_paired_channel(parent))
    parent.l0.addWidget(parent.channel_button, 0, 26, 1, 2)
    for button in (parent.prev_plane_button, parent.next_plane_button, parent.channel_button):
        button.setEnabled(False)

    parent.sizebtns = QButtonGroup(parent)
    b = 0
    labels = [" cells", " both", " not cells"]
    for l in labels:
        btn = SizeButton(b, l, parent)
        parent.sizebtns.addButton(btn, b)
        parent.l0.addWidget(btn, 0, 14 + 2 * b, 1, 2)
        btn.setEnabled(False)
        if b == 1:
            btn.setChecked(True)
        b += 1
    parent.sizebtns.setExclusive(True)


def update_plane_navigation(parent):
    """Update plane/channel controls for the currently loaded stat.npy."""
    if not hasattr(parent, "plane_label"):
        return
    previous, following, counterpart = plane_navigation_paths(getattr(parent, "basename", ""))
    plane_name = Path(getattr(parent, "basename", "")).name
    if re.fullmatch(r"plane\d+", plane_name):
        parent.plane_label.setText(f"Plane {int(plane_name[5:])}")
    else:
        parent.plane_label.setText("Plane —")
    parent.prev_plane_button.setEnabled(previous is not None)
    parent.next_plane_button.setEnabled(following is not None)
    parent.channel_button.setEnabled(counterpart is not None)
    parent.channel_button.setToolTip(
        str(counterpart) if counterpart is not None else "No matching channel output for this plane"
    )


def _load_stat(parent, stat_path):
    if stat_path is None:
        return
    parent.fname = str(stat_path)
    from . import io
    io.load_proc(parent)


def load_adjacent_plane(parent, direction):
    previous, following, _counterpart = plane_navigation_paths(parent.basename)
    _load_stat(parent, previous if direction < 0 else following)


def load_paired_channel(parent):
    _previous, _following, counterpart = plane_navigation_paths(parent.basename)
    _load_stat(parent, counterpart)


def make_quadrants(parent):
    """ make quadrant buttons """
    parent.quadbtns = QButtonGroup(parent)
    for b in range(9):
        btn = QuadButton(b, " " + str(b + 1), parent)
        parent.quadbtns.addButton(btn, b)
        parent.l0.addWidget(btn, 0 + parent.quadbtns.button(b).ypos,
                            29 + parent.quadbtns.button(b).xpos, 1, 1)
        btn.setEnabled(False)
        b += 1
    parent.quadbtns.setExclusive(True)


class QuadButton(QPushButton):
    """ custom QPushButton class for quadrant plotting
        requires buttons to put into a QButtonGroup (parent.quadbtns)
         allows only 1 button to pressed at a time
    """

    def __init__(self, bid, Text, parent=None):
        super(QuadButton, self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.setMaximumWidth(22)
        self.xpos = bid % 3
        self.ypos = int(np.floor(bid / 3))
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid):
        self.xrange = np.array([self.xpos - .15, self.xpos + 1.15
                               ]) * parent.ops["Lx"] / 3
        self.yrange = np.array([self.ypos - .15, self.ypos + 1.15
                               ]) * parent.ops["Ly"] / 3
        # change the zoom
        parent.p1.setXRange(self.xrange[0], self.xrange[1])
        parent.p1.setYRange(self.yrange[0], self.yrange[1])
        parent.p2.setXRange(self.xrange[0], self.xrange[1])
        parent.p2.setYRange(self.yrange[0], self.yrange[1])
        parent.p2.setXLink("plot1")
        parent.p2.setYLink("plot1")
        if hasattr(parent, "_view_sync_connected"):
            parent._syncing_view_range = False
        parent.show()


# size of view
class SizeButton(QPushButton):
    """ buttons to make trace box bigger or smaller """

    def __init__(self, bid, Text, parent=None):
        super(SizeButton, self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent))
        self.bid = bid
        self.show()

    def press(self, parent):
        bid = self.bid
        previous_bid = getattr(parent, "_last_size_bid", parent.sizebtns.checkedId())
        source_view = parent.p2 if previous_bid == 2 else parent.p1
        view_range = source_view.viewRange()

        for btn in parent.sizebtns.buttons():
            btn.setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        self.setChecked(True)
        parent._last_size_bid = bid

        ts = 100
        if bid == 0:
            parent.win.ci.layout.setColumnStretchFactor(0, ts)
            parent.win.ci.layout.setColumnStretchFactor(1, 0)
        elif bid == 1:
            parent.win.ci.layout.setColumnStretchFactor(0, ts)
            parent.win.ci.layout.setColumnStretchFactor(1, ts)
            parent.p2.setXLink("plot1")
            parent.p2.setYLink("plot1")
        elif bid == 2:
            parent.win.ci.layout.setColumnStretchFactor(0, 0)
            parent.win.ci.layout.setColumnStretchFactor(1, ts)
        parent.p2.setXLink("plot1")
        parent.p2.setYLink("plot1")
        if parent.loaded:
            contains_full_frame = (
                view_range[0][0] <= 0 <= parent.ops["Lx"] <= view_range[0][1]
                and view_range[1][0] <= 0 <= parent.ops["Ly"] <= view_range[1][1]
            )
            if contains_full_frame:
                view_range = ((0, parent.ops["Lx"]), (0, parent.ops["Ly"]))
            graphics.restore_image_range_after_layout(
                parent, view_range[0], view_range[1], bid)
        else:
            graphics.configure_image_view_aspect(parent, bid)
        # only enable selection buttons when not in "both" view
        if bid != 1:
            if parent.ops_plot["color"] != 0:
                for btn in parent.topbtns.buttons():
                    btn.setEnabled(True)
            else:
                parent.topbtns.button(0).setEnabled(True)
        else:
            parent.ROI_remove()
            for btn in parent.topbtns.buttons():
                btn.setEnabled(False)
        parent.win.show()
        parent.show()


#
class TopButton(QPushButton):
    """ selection of top neurons"""

    def __init__(self, bid, parent=None):
        super(TopButton, self).__init__(parent)
        text = [" draw selection", " select top n", " select bottom n"]
        self.bid = bid
        self.setText(text[bid])
        self.setCheckable(True)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent))
        self.show()

    def press(self, parent):
        bid = self.bid
        if not parent.sizebtns.button(1).isChecked():
            if parent.ops_plot["color"] == 0:
                for b in [1, 2]:
                    parent.topbtns.button(b).setEnabled(False)
            else:
                for b in [1, 2]:
                    parent.topbtns.button(b).setEnabled(True)
        else:
            for b in range(3):
                parent.topbtns.button(b).setEnabled(False)
        if bid == 0:
            parent.ROI_selection()
        else:
            self.top_selection(parent)

    def top_selection(self, parent):
        bid = self.bid
        parent.ROI_remove()
        draw = False
        ncells = len(parent.stat)
        icells = np.minimum(ncells, parent.ntop)
        if bid == 1:
            top = True
        elif bid == 2:
            top = False
        if parent.sizebtns.button(0).isChecked():
            wplot = 0
            draw = True
        elif parent.sizebtns.button(2).isChecked():
            wplot = 1
            draw = True
        if draw:
            if parent.ops_plot["color"] != 0:
                c = parent.ops_plot["color"]
                istat = parent.colors["istat"][c]
                if wplot == 0:
                    icell = np.array(parent.iscell.nonzero()).flatten()
                    istat = istat[parent.iscell]
                else:
                    icell = np.array((~parent.iscell).nonzero()).flatten()
                    istat = istat[~parent.iscell]
                inds = istat.argsort()
                if top:
                    inds = inds[-icells:]
                    parent.ichosen = icell[inds[-1]]
                else:
                    inds = inds[:icells]
                    parent.ichosen = icell[inds[0]]
                parent.imerge = []
                for n in inds:
                    parent.imerge.append(icell[n])
                # draw choices
                parent.update_plot()
                parent.show()
