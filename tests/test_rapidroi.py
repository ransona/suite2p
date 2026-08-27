import numpy as np

from suite2p.gui.rapidroi import circle_pixels, polygon_pixels


def test_circle_pixels_is_centered_and_clipped():
    ypix, xpix = circle_pixels(0, 0, 8, 20, 20)
    assert ypix.size == xpix.size
    assert ypix.size > 0
    assert ypix.min() == 0 and xpix.min() == 0
    assert ypix.max() < 20 and xpix.max() < 20


def test_polygon_pixels_returns_interior():
    ypix, xpix = polygon_pixels([[2, 2], [2, 8], [8, 2]], 12, 12)
    assert ypix.size == xpix.size
    assert ypix.size > 0
    assert np.all((ypix >= 0) & (ypix < 12))
    assert np.all((xpix >= 0) & (xpix < 12))
