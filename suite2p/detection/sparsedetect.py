"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from copy import deepcopy
from enum import Enum
from warnings import warn
import time
import logging 
logger = logging.getLogger(__name__)


import numpy as np
from numpy.linalg import norm

from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.stats import mode

from . import utils

def downsample(mov, taper_edge=True):
    """
    Downsample a movie by 2x in both spatial dimensions.

    Averages adjacent pixels along Y then X. If a dimension has odd size,
    the last pixel is halved when ``taper_edge`` is True.

    Parameters
    ----------
    mov : numpy.ndarray
        Movie of shape (n_frames, Ly, Lx).
    taper_edge : bool, optional (default True)
        If True, halve the edge pixel when the dimension is odd.

    Returns
    -------
    mov2 : numpy.ndarray
        Downsampled movie of shape (n_frames, ceil(Ly/2), ceil(Lx/2)).
    """
    n_frames, Ly, Lx = mov.shape

    # bin along Y
    movd = np.zeros((n_frames, int(np.ceil(Ly / 2)), Lx), "float32")
    movd[:, :Ly // 2, :] = np.mean([mov[:, 0:-1:2, :], mov[:, 1::2, :]], axis=0)
    if Ly % 2 == 1:
        movd[:, -1, :] = mov[:, -1, :] / 2 if taper_edge else mov[:, -1, :]

    # bin along X
    mov2 = np.zeros((n_frames, int(np.ceil(Ly / 2)), int(np.ceil(Lx / 2))), "float32")
    mov2[:, :, :Lx // 2] = np.mean([movd[:, :, 0:-1:2], movd[:, :, 1::2]], axis=0)
    if Lx % 2 == 1:
        mov2[:, :, -1] = movd[:, :, -1] / 2 if taper_edge else movd[:, :, -1]

    return mov2


def threshold_reduce(mov, intensity_threshold):
    """
    Compute thresholded standard deviation map across frames.

    For each pixel, sums the squared values of frames exceeding
    the threshold, then takes the square root. Iterates frame-by-frame
    to reduce memory usage.

    Parameters
    ----------
    mov : numpy.ndarray
        Movie of shape (nbinned, Ly, Lx).
    intensity_threshold : float
        Only frames where the pixel value exceeds this are included.

    Returns
    -------
    Vt : numpy.ndarray
        Thresholded standard deviation map of shape (Ly, Lx), dtype float32.
    """
    nbinned, Lyp, Lxp = mov.shape
    Vt = np.zeros((Lyp, Lxp), "float32")
    for t in range(nbinned):
        Vt += mov[t]**2 * (mov[t] > intensity_threshold)
    Vt = Vt**.5
    return Vt


def neuropil_subtraction(mov, filter_size):
    """
    Subtract a spatially low-pass filtered version of the movie to remove neuropil.

    Parameters
    ----------
    mov : numpy.ndarray
        Movie of shape (nbinned, Ly, Lx).
    filter_size : int
        Width of the uniform spatial filter in pixels.

    Returns
    -------
    movt : numpy.ndarray
        High-pass filtered movie of shape (nbinned, Ly, Lx).
    """
    nbinned, Ly, Lx = mov.shape
    c1 = uniform_filter(np.ones((Ly, Lx)), size=filter_size, mode="constant")
    movt = np.zeros_like(mov)
    for frame, framet in zip(mov, movt):
        framet[:] = frame - (uniform_filter(frame, size=filter_size, mode="constant") /
                             c1)
    return movt


def square_convolution_2d(mov, filter_size):
    """
    Convolve each frame with a square uniform kernel.

    Parameters
    ----------
    mov : numpy.ndarray
        Movie of shape (nbinned, Ly, Lx).
    filter_size : int
        Width of the square uniform kernel in pixels.

    Returns
    -------
    movt : numpy.ndarray
        Convolved movie of shape (nbinned, Ly, Lx), dtype float32.
    """
    movt = np.zeros_like(mov, dtype=np.float32)
    for frame, framet in zip(mov, movt):
        framet[:] = filter_size * uniform_filter(frame, size=filter_size,
                                                 mode="constant")
    return movt


def multiscale_mask(ypix0, xpix0, lam0, Lyp, Lxp):
    """
    Downsample a mask to all spatial scales used in sparse detection.

    Given pixel coordinates and weights at the original resolution, creates
    downsampled versions by successively halving coordinates and accumulating
    weights, then extends each mask into surrounding pixels.

    Parameters
    ----------
    ypix0 : numpy.ndarray
        Y-coordinates of the mask pixels at full resolution.
    xpix0 : numpy.ndarray
        X-coordinates of the mask pixels at full resolution.
    lam0 : numpy.ndarray
        Pixel weights at full resolution.
    Lyp : numpy.ndarray
        Heights of the downsampled images at each scale, shape (n_scales,).
    Lxp : numpy.ndarray
        Widths of the downsampled images at each scale, shape (n_scales,).

    Returns
    -------
    ys : list of numpy.ndarray
        Y-coordinates of the mask at each spatial scale.
    xs : list of numpy.ndarray
        X-coordinates of the mask at each spatial scale.
    lms : list of numpy.ndarray
        Pixel weights at each spatial scale.
    """
    xs = [xpix0]
    ys = [ypix0]
    lms = [lam0]
    for j in range(1, len(Lyp)):
        ipix, ind = np.unique(
            np.int32(xs[j - 1] / 2) + np.int32(ys[j - 1] / 2) * Lxp[j],
            return_inverse=True)
        LAM = np.zeros(len(ipix))
        for i in range(len(xs[j - 1])):
            LAM[ind[i]] += lms[j - 1][i] / 2
        lms.append(LAM)
        ys.append(np.int32(ipix / Lxp[j]))
        xs.append(np.int32(ipix % Lxp[j]))
    for j in range(len(Lyp)):
        ys[j], xs[j], lms[j] = extend_mask(ys[j], xs[j], lms[j], Lyp[j], Lxp[j])
    return ys, xs, lms


def add_square(yi, xi, lx, Ly, Lx):
    """
    Create a square mask of pixels around a peak, normalized to unit norm.

    Parameters
    ----------
    yi : int
        Y-coordinate of the center pixel.
    xi : int
        X-coordinate of the center pixel.
    lx : int
        Side length of the square in pixels.
    Ly : int
        Height of the full image.
    Lx : int
        Width of the full image.

    Returns
    -------
    y0 : numpy.ndarray
        Y-coordinates of the square pixels, flattened.
    x0 : numpy.ndarray
        X-coordinates of the square pixels, flattened.
    mask : numpy.ndarray
        Pixel weights normalized to unit L2 norm, flattened.
    """
    lhf = int((lx - 1) / 2)
    ipix = np.tile(np.arange(-lhf, -lhf + lx, dtype=np.int32), reps=(lx, 1))
    x0 = xi + ipix
    y0 = yi + ipix.T
    mask = np.ones_like(ipix, dtype=np.float32)
    ix = np.all((y0 >= 0, y0 < Ly, x0 >= 0, x0 < Lx), axis=0)
    x0 = x0[ix]
    y0 = y0[ix]
    mask = mask[ix]
    mask = mask / norm(mask)
    return y0.flatten(), x0.flatten(), mask.flatten()


def iter_extend(ypix, xpix, mov, Lyc, Lxc, active_frames):
    """
    Iteratively extend a mask based on pixel activity on active frames.

    Repeatedly grows the ROI by one pixel on each side, keeping only pixels
    whose mean activity on active frames exceeds 1/5 of the maximum pixel. 
    Stops when the mask stops growing or reaches 10000 pixels.

    Parameters
    ----------
    ypix : numpy.ndarray
        Y-coordinates of the initial mask pixels.
    xpix : numpy.ndarray
        X-coordinates of the initial mask pixels.
    mov : numpy.ndarray
        Binned residual movie of shape (nbinned, Lyc * Lxc).
    Lyc : int
        Height of the movie frame.
    Lxc : int
        Width of the movie frame.
    active_frames : numpy.ndarray
        Indices of frames used to compute activity.

    Returns
    -------
    ypix : numpy.ndarray
        Y-coordinates of the extended mask.
    xpix : numpy.ndarray
        X-coordinates of the extended mask.
    lam : numpy.ndarray
        Pixel weights normalized to unit L2 norm.
    """
    npix = 0
    iter = 0
    while npix < 10000:
        npix = ypix.size
        # extend ROI by 1 pixel on each side
        ypix, xpix = extendROI(ypix, xpix, Lyc, Lxc, 1)
        # activity in proposed ROI on ACTIVE frames
        usub = mov[np.ix_(active_frames, ypix * Lxc + xpix)]
        lam = usub.mean(axis=0)
        ix = lam > max(0, lam.max() / 5.0)
        if ix.sum() == 0:
            break
        ypix, xpix, lam = ypix[ix], xpix[ix], lam[ix]
        if iter == 0:
            sgn = 1.
        if np.sign(sgn * (ix.sum() - npix)) <= 0:
            break
        else:
            npix = ypix.size
        iter += 1
    lam = lam / np.sum(lam**2)**.5
    return ypix, xpix, lam


def extendROI(ypix, xpix, Ly, Lx, niter=1):
    """
    Extend ROI pixel coordinates by growing the mask in 4-connected directions.

    Parameters
    ----------
    ypix : numpy.ndarray
        Y-coordinates of the mask pixels.
    xpix : numpy.ndarray
        X-coordinates of the mask pixels.
    Ly : int
        Height of the image.
    Lx : int
        Width of the image.
    niter : int, optional (default 1)
        Number of dilation iterations.

    Returns
    -------
    ypix : numpy.ndarray
        Extended Y-coordinates.
    xpix : numpy.ndarray
        Extended X-coordinates.
    """
    for k in range(niter):
        yx = ((ypix, ypix, ypix, ypix - 1, ypix + 1), (xpix, xpix + 1, xpix - 1, xpix,
                                                       xpix))
        yx = np.array(yx)
        yx = yx.reshape((2, -1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
        ypix, xpix = yu[:, ix]
    return ypix, xpix


def two_comps(mpix0, lam, Th2):
    """
    Check if splitting an ROI into two components increases variance explained.

    Projects the ROI movie onto the current mask, then tests whether a
    two-component split captures more variance. Returns the variance ratio
    and the better component.

    Parameters
    ----------
    mpix0 : numpy.ndarray
        Binned movie for pixels in the ROI, shape (nbinned, npix).
    lam : numpy.ndarray
        Pixel weights for the current ROI.
    Th2 : float
        Intensity threshold for determining active frames.

    Returns
    -------
    vrat : float
        Ratio of variance explained by two components to one component.
        Values above 1.25 suggest the ROI should be split.
    ipick : tuple
        Tuple of (mu, xproj, goodframe) for the better component, where
        mu is the pixel weights, xproj is the temporal projection on active
        frames, and goodframe is the boolean active-frame mask.
    """
    mpix = mpix0.copy()
    xproj = mpix @ lam
    gf0 = xproj > Th2

    mpix[gf0, :] -= np.outer(xproj[gf0], lam)
    vexp0 = np.sum(mpix0**2) - np.sum(mpix**2)

    k = np.argmax(np.sum(mpix * np.float32(mpix > 0), axis=1))
    mu = [lam * np.float32(mpix[k] < 0), lam * np.float32(mpix[k] > 0)]

    mpix = mpix0.copy()
    goodframe = []
    xproj = []
    for mu0 in mu:
        mu0[:] /= norm(mu0) + 1e-6
        xp = mpix @ mu0
        mpix[gf0, :] -= np.outer(xp[gf0], mu0)
        goodframe.append(gf0)
        xproj.append(xp[gf0])

    flag = [False, False]
    V = np.zeros(2)
    for t in range(3):
        for k in range(2):
            if flag[k]:
                continue
            mpix[goodframe[k], :] += np.outer(xproj[k], mu[k])
            xp = mpix @ mu[k]
            goodframe[k] = xp > Th2
            V[k] = np.sum(xp**2)
            if np.sum(goodframe[k]) == 0:
                flag[k] = True
                V[k] = -1
                continue
            xproj[k] = xp[goodframe[k]]
            mu[k] = np.mean(mpix[goodframe[k], :] * xproj[k][:, np.newaxis], axis=0)
            mu[k][mu[k] < 0] = 0
            mu[k] /= (1e-6 + np.sum(mu[k]**2)**.5)
            mpix[goodframe[k], :] -= np.outer(xproj[k], mu[k])
    k = np.argmax(V)
    vexp = np.sum(mpix0**2) - np.sum(mpix**2)
    vrat = vexp / vexp0
    return vrat, (mu[k], xproj[k], goodframe[k])


def extend_mask(ypix, xpix, lam, Ly, Lx):
    """
    Extend a mask into the 8 surrounding pixels of each pixel.

    Each pixel spreads its weight equally to itself and its 8 neighbors.
    Overlapping contributions are summed.

    Parameters
    ----------
    ypix : numpy.ndarray
        Y-coordinates of the mask pixels.
    xpix : numpy.ndarray
        X-coordinates of the mask pixels.
    lam : numpy.ndarray
        Pixel weights.
    Ly : int
        Height of the image.
    Lx : int
        Width of the image.

    Returns
    -------
    ypix1 : numpy.ndarray
        Y-coordinates of the extended mask.
    xpix1 : numpy.ndarray
        X-coordinates of the extended mask.
    lam1 : numpy.ndarray
        Accumulated pixel weights for the extended mask.
    """
    nel = len(xpix)
    yx = ((ypix, ypix, ypix, ypix - 1, ypix - 1, ypix - 1, ypix + 1, ypix + 1,
           ypix + 1), (xpix, xpix + 1, xpix - 1, xpix, xpix + 1, xpix - 1, xpix,
                       xpix + 1, xpix - 1))
    yx = np.array(yx)
    yx = yx.reshape((2, -1))
    yu, ind = np.unique(yx, axis=1, return_inverse=True)
    LAM = np.zeros(yu.shape[1])
    for j in range(len(ind)):
        LAM[ind[j]] += lam[j % nel] / 3
    ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
    ypix1, xpix1 = yu[:, ix]
    lam1 = LAM[ix]
    return ypix1, xpix1, lam1


class EstimateMode(Enum):
    Forced = "FORCED"
    Estimated = "estimated"


def estimate_spatial_scale(I):
    """
    Estimate the dominant spatial scale from multi-scale correlation maps.

    Finds the scale index that appears most frequently among the top 50
    brightest local maxima in the max-projected correlation image.

    Parameters
    ----------
    I : numpy.ndarray
        Multi-scale correlation maps of shape (n_scales, Ly, Lx).

    Returns
    -------
    im : int
        Index of the estimated best spatial scale.
    """
    I0 = I.max(axis=0)
    imap = np.argmax(I, axis=0).flatten()
    ipk = np.abs(I0 - maximum_filter(I0, size=(11, 11))).flatten() < 1e-4
    isort = np.argsort(I0.flatten()[ipk])[::-1]
    im, _ = mode(imap[ipk][isort[:50]], keepdims=True)
    return im.item()


def find_best_scale(I, spatial_scale):
    """
    Determine the best spatial scale, either forced or estimated from data.

    If ``spatial_scale`` is positive, clamps it to [1, 4] and returns it as
    forced. Otherwise estimates the scale from the multi-scale correlation
    maps.

    Parameters
    ----------
    I : numpy.ndarray
        Multi-scale correlation maps of shape (n_scales, Ly, Lx).
    spatial_scale : int
        User-specified spatial scale. If positive, used directly (forced).
        If zero or negative, the scale is estimated from the data.

    Returns
    -------
    scale : int
        Best spatial scale index.
    estimate_mode : EstimateMode
        Whether the scale was ``EstimateMode.Forced`` or
        ``EstimateMode.Estimated``.
    """
    if spatial_scale > 0:
        return max(1, min(4, spatial_scale)), EstimateMode.Forced
    else:
        scale = estimate_spatial_scale(I=I)
        if scale > 0:
            return scale, EstimateMode.Estimated
        else:
            warn(
                "Spatial scale estimation failed.  Setting spatial scale to 1 in order to continue."
            )
            return 1, EstimateMode.Forced


def sparsery(mov, sdmov, highpass_neuropil,
             spatial_scale, threshold_scaling, max_ROIs,
             active_percentile=0, max_peaks_to_check=20000, signal_mask=None,
             peak_candidate_mask=None, min_roi_pixels=0, max_roi_pixels=0,
             min_roi_width=0, max_roi_width=0, min_roi_height=0,
             max_roi_height=0):
    """
    Detect ROIs in a movie using doubly-sparse matrix decomposition.

    Subtracts neuropil, builds multi-scale correlation maps, then
    greedily chooses top activity peak, extends mask, optionally splits
    the ROI, and subtracts the detected signal from the residual movie, then 
    extracts the highest peak from the residual and continues.

    Parameters
    ----------
    mov : numpy.ndarray
        Binned movie of shape (nbinned, Ly, Lx).
    sdmov : numpy.ndarray
        Per-pixel standard deviation of shape (Ly, Lx), used for
        normalization.
    highpass_neuropil : int
        Filter size for spatial high-pass neuropil subtraction.
    spatial_scale : int
        Spatial scale setting. If positive, forced; if zero or negative,
        estimated from the data.
    threshold_scaling : float
        Multiplier for the activity threshold used to accept peaks.
    max_ROIs : int
        Maximum number of ROIs to detect.
    active_percentile : float, optional (default 0)
        If positive, use this percentile of the temporal projection as an
        alternative activity threshold.
    max_peaks_to_check : int, optional (default 20000)
        Maximum number of candidate peaks to test while looking for accepted ROIs.
    min_roi_pixels, max_roi_pixels : int, optional
        Raw ROI footprint pixel-count limits. A value of 0 disables that limit.
    min_roi_width, max_roi_width, min_roi_height, max_roi_height : int, optional
        ROI footprint extent limits in pixels. A value of 0 disables that limit.

    Returns
    -------
    new_settings : dict
        Dictionary with detection metadata including "Vmax", "ihop",
        "Vsplit", "Vcorr", "Vmap", and "spatscale_pix".
    stats : list of dict
        List of ROI statistics dictionaries, each containing "ypix",
        "xpix", "lam", "med", and "footprint".
    """

    mov = neuropil_subtraction(
        mov=mov / sdmov,
        filter_size=highpass_neuropil)  # subtract low-pass filtered movie

    _, Lyc, Lxc = mov.shape
    LL = np.meshgrid(np.arange(Lxc), np.arange(Lyc))
    gxy = [np.array(LL).astype("float32")]
    dmov = mov
    movu = []

    # downsample movie at various spatial scales
    Lyp, Lxp = np.zeros(5, "int32"), np.zeros(5, "int32")  # downsampled sizes
    for j in range(5):
        movu0 = square_convolution_2d(dmov, 3)
        dmov = 2 * downsample(dmov)
        gxy0 = downsample(gxy[j], False)
        gxy.append(gxy0)
        _, Lyp[j], Lxp[j] = movu0.shape
        movu.append(movu0)

    # spline over scales
    I = np.zeros((len(movu), gxy[0].shape[1], gxy[0].shape[2]))
    for movu0, gxy0, I0 in zip(movu, gxy, I):
        gmodel = RectBivariateSpline(gxy0[1, :, 0], gxy0[0, 0, :], movu0.max(axis=0),
                                     kx=min(3, gxy0.shape[1] - 1),
                                     ky=min(3, gxy0.shape[2] - 1))
        I0[:] = gmodel(gxy[0][1, :, 0], gxy[0][0, 0, :])
    v_corr = I.max(axis=0)

    if signal_mask is not None:
        I_for_threshold = I.copy()
        I_for_threshold[:, ~signal_mask] = 0
    else:
        I_for_threshold = I

    scale, estimate_mode = find_best_scale(I=I_for_threshold, spatial_scale=spatial_scale)

    spatscale_pix = 3 * 2**scale
    if isinstance(spatscale_pix, np.ndarray):
        spatscale_pix = spatscale_pix.item()
    mask_window = int(((spatscale_pix * 1.5) // 2) * 2)
    base_Th2 = 5 * max(1, scale)
    Th2 = threshold_scaling * base_Th2  # threshold for active frames
    vmultiplier = max(1, mov.shape[0] / 1200)
    logger.info("NOTE: %s spatial scale ~%d pixels, time epochs %2.2f, threshold %2.2f " %
          (estimate_mode.value, spatscale_pix, vmultiplier, vmultiplier * Th2))

    # get standard deviation maps from an unscaled base cutoff, then apply
    # threshold_scaling to the peak acceptance threshold.
    v_map = [threshold_reduce(movu0, base_Th2) for movu0 in movu]
    peak_threshold_unscaled = vmultiplier * base_Th2
    peak_threshold = threshold_scaling * peak_threshold_unscaled
    signal_masks = None
    candidate_masks = None
    v_map_masked = None
    if signal_mask is not None:
        signal_masks = []
        mask0 = signal_mask.astype(np.float32)[np.newaxis, :, :]
        for j in range(5):
            signal_masks.append(mask0[0] > 0.5)
            mask0 = downsample(mask0, taper_edge=False)
        v_map_masked = [v_map[j] * signal_masks[j] for j in range(5)]
        masked_values = np.concatenate([
            v_map[j][signal_masks[j]].ravel() for j in range(5)
            if signal_masks[j].any()
        ])
        masked_values = masked_values[np.isfinite(masked_values) & (masked_values > 0)]
        logger.info(
            "Using original sparsery peak threshold %2.2f "
            "(base %2.2f * threshold_scaling %2.2f); %d masked map pixels available for diagnostics" %
            (peak_threshold, peak_threshold_unscaled, threshold_scaling, masked_values.size)
        )
    if peak_candidate_mask is not None:
        candidate_masks = []
        mask0 = peak_candidate_mask.astype(np.float32)[np.newaxis, :, :]
        for j in range(5):
            candidate_masks.append(mask0[0] > 0.5)
            mask0 = downsample(mask0, taper_edge=False)
    movu = [movu0.reshape(movu0.shape[0], -1) for movu0 in movu]

    mov = np.reshape(mov, (-1, Lyc * Lxc))
    lxs = 3 * 2**np.arange(5)
    nscales = len(lxs)

    v_max = np.zeros(max_ROIs)
    ihop = np.zeros(max_ROIs)
    v_split = np.zeros(max_ROIs)
    V1 = deepcopy(v_map)
    stats = []
    patches = []
    seeds = []
    extract_patches = False
    min_roi_pixels = int(min_roi_pixels or 0)
    max_roi_pixels = int(max_roi_pixels or 0)
    min_roi_width = int(min_roi_width or 0)
    max_roi_width = int(max_roi_width or 0)
    min_roi_height = int(min_roi_height or 0)
    max_roi_height = int(max_roi_height or 0)
    n_rejected_size = {
        "min_pixels": 0,
        "max_pixels": 0,
        "min_width": 0,
        "max_width": 0,
        "min_height": 0,
        "max_height": 0,
    }

    def roi_size_rejection_reason(ypix, xpix):
        npix = ypix.size
        width = int(np.max(xpix) - np.min(xpix) + 1)
        height = int(np.max(ypix) - np.min(ypix) + 1)
        if min_roi_pixels > 0 and npix < min_roi_pixels:
            return "min_pixels"
        if max_roi_pixels > 0 and npix > max_roi_pixels:
            return "max_pixels"
        if min_roi_width > 0 and width < min_roi_width:
            return "min_width"
        if max_roi_width > 0 and width > max_roi_width:
            return "max_width"
        if min_roi_height > 0 and height < min_roi_height:
            return "min_height"
        if max_roi_height > 0 and height > max_roi_height:
            return "max_height"
        return None

    def rejection_summary():
        return ", ".join(f"{key}={value}" for key, value in n_rejected_size.items())

    max_peaks_to_check = int(max_peaks_to_check or 20000)
    logger.info(
        f"max_ROIs set to {max_ROIs}; max_peaks_to_check set to {max_peaks_to_check} "
        "- will stop when either limit is reached or no peaks remain above threshold."
    )
    t0 = time.time()
    tj = 0
    stopped_by_peak_limit = False
    while len(stats) < max_ROIs and tj < max_peaks_to_check:
        # find peaks in stddev"s
        if candidate_masks is None:
            V1_peak = V1
        else:
            V1_peak = [V1[j] * candidate_masks[j] for j in range(5)]
        v0max = np.array([V1_peak[j].max() for j in range(5)])
        imap = np.argmax(v0max)
        imax = np.argmax(V1_peak[imap])
        yi, xi = np.unravel_index(imax, (Lyp[imap], Lxp[imap]))
        # position of peak
        yi, xi = gxy[imap][1, yi, xi], gxy[imap][0, yi, xi]
        med = [int(yi), int(xi)]

        # check if peak is larger than threshold * max(1,nbinned/1200)
        peak_score = v0max.max()
        if peak_score < peak_threshold:
            break
        ls = lxs[imap]

        # make square of initial pixels based on spatial scale of peak
        yi, xi = int(yi), int(xi)
        ypix0, xpix0, lam0 = add_square(yi, xi, ls, Lyc, Lxc)

        # project movie into square to get time series
        tproj = (mov[:, ypix0 * Lxc + xpix0] * lam0[0]).sum(axis=-1)
        if active_percentile > 0:
            threshold = min(Th2, np.percentile(tproj, active_percentile))
        else:
            threshold = Th2
        active_frames = np.nonzero(tproj > threshold)[0]  # frames with activity > Th2

        # get square around seed
        if extract_patches:
            mask = mov[active_frames].mean(axis=0).reshape(Lyc, Lxc)
            patches.append(utils.square_mask(mask, mask_window, yi, xi))
            seeds.append([yi, xi])

        # extend mask based on activity similarity
        for j in range(3):
            ypix0, xpix0, lam0 = iter_extend(ypix0, xpix0, mov, Lyc, Lxc, active_frames)
            tproj = mov[:, ypix0 * Lxc + xpix0] @ lam0
            active_frames = np.nonzero(tproj > threshold)[0]
            if len(active_frames) < 1:
                #if tj < max_ROIs/2: # TODO: nmasks is undefined
                #    continue
                #else:
                break
            
        if len(active_frames) < 1:
            #if tj < max_ROIs/2:
            #    continue
            #else:
            break

        # check if ROI should be split
        split_score, ipack = two_comps(mov[:, ypix0 * Lxc + xpix0], lam0, threshold)
        if split_score > 1.25:
            lam0, xp, active_frames = ipack
            tproj[active_frames] = xp
            ix = lam0 > lam0.max() / 5
            xpix0 = xpix0[ix]
            ypix0 = ypix0[ix]
            lam0 = lam0[ix]
            ymed = np.median(ypix0)
            xmed = np.median(xpix0)
            imin = np.argmin((xpix0 - xmed)**2 + (ypix0 - ymed)**2)
            med = [ypix0[imin], xpix0[imin]]

        rejection_reason = roi_size_rejection_reason(ypix0, xpix0)

        # update residual on raw movie
        mov[np.ix_(active_frames,
                   ypix0 * Lxc + xpix0)] -= tproj[active_frames][:, np.newaxis] * lam0
        # update filtered movie
        ys, xs, lms = multiscale_mask(ypix0, xpix0, lam0, Lyp, Lxp)
        for j in range(nscales):
            movu[j][np.ix_(active_frames, xs[j] + Lxp[j] * ys[j])] -= np.outer(
                tproj[active_frames], lms[j])
            Mx = movu[j][:, xs[j] + Lxp[j] * ys[j]]
            V1[j][ys[j], xs[j]] = (Mx**2 * np.float32(Mx > threshold)).sum(axis=0)**.5

        if rejection_reason is None:
            accepted = len(stats)
            v_max[accepted] = peak_score
            ihop[accepted] = imap
            v_split[accepted] = split_score
            stats.append({
                "ypix": ypix0.astype(int),
                "xpix": xpix0.astype(int),
                "lam": lam0 * sdmov[ypix0, xpix0],
                "med": med,
                "footprint": imap
            })
        else:
            n_rejected_size[rejection_reason] += 1

        if tj % 500 == 0:
            t1 = time.time() - t0
            logger.info(
                f"ROIs: {len(stats)}, candidates: {tj},\t "
                f"current peak score: {peak_score:0.4f}, "
                f"minimum peak score: {peak_threshold:0.4f}, "
                f"time: {t1:0.2f}sec, "
                f"size rejected: {rejection_summary()}"
            )
        tj += 1

    if len(stats) < max_ROIs and tj >= max_peaks_to_check:
        stopped_by_peak_limit = True
        logger.info(
            f"Stopped sparsery after checking max_peaks_to_check={max_peaks_to_check}; "
            f"accepted {len(stats)}/{max_ROIs} ROIs"
        )

    if any(n_rejected_size.values()):
        logger.info(
            "Rejected sparsery candidates outside ROI size limits: "
            + rejection_summary()
        )

    new_settings = {
        "Vmax": v_max,
        "ihop": ihop,
        "Vsplit": v_split,
        "Vcorr": v_corr,
        "Vcorr_scales": I.astype(np.float32, copy=False),
        "Vcorr_signal_scales": I_for_threshold.astype(np.float32, copy=False),
        "signal_peak_threshold": peak_threshold,
        "signal_peak_threshold_unscaled": peak_threshold_unscaled,
        "n_sparsery_candidates": tj,
        "max_peaks_to_check": max_peaks_to_check,
        "sparsery_stopped_by_peak_limit": stopped_by_peak_limit,
        "n_sparsery_rejected_size": n_rejected_size,
        "Vmap": np.asanyarray(
            v_map, dtype="object"
        ),  # needed so that scipy.io.savemat doesn"t fail in runpipeline with latest numpy (v1.24.3). dtype="object" is needed to have numpy array with elements having diff sizes
        "spatscale_pix": spatscale_pix,
    }
    if v_map_masked is not None:
        new_settings["Vmap_signal"] = np.asanyarray(v_map_masked, dtype="object")

    return new_settings, stats
