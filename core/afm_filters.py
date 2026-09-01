"""
core/afm_filters.py

Implementación ligera de filtros AFM orientados a HS-AFM: plane leveling (robusto con iteraciones y opcional RANSAC),
line-by-line leveling, median filter y despike/outlier removal con preview.

Diseñado para ser usado por gui/afm_loader.py
"""

import numpy as np
from scipy.ndimage import median_filter as _median_filter
from scipy.ndimage import gaussian_filter

# Optional imports
try:
    from sklearn.linear_model import RANSACRegressor
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def plane_fit_subtract(img, order=1, robust=True, iters=3, ransac=False, ransac_iters=200, ransac_thresh=2.0):
    """
    Fit a 2D polynomial surface of given order and subtract it from img.

    Parameters
    ----------
    img : 2D array
    order : int
        Polynomial order (0 = constant, 1 = plane)
    robust : bool
        If True use iterative sigma-clipping to reduce influence of outliers.
    iters : int
        Number of robust iterations when robust=True.
    ransac : bool
        If True and sklearn is available, run RANSAC for robust plane fitting (only for order<=1).
    ransac_iters : int
        Max trials for RANSAC.
    ransac_thresh : float
        Residual threshold for RANSAC.

    Returns
    -------
    out : 2D array
        img - fitted_background
    """
    arr = np.asarray(img, dtype=float)
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    X_base = _poly_design_matrix(xx, yy, order)

    def fit_least_squares(y_vec, X):
        c, *_ = np.linalg.lstsq(X, y_vec, rcond=None)
        return c

    def build_bg(coeffs, X):
        return (X @ coeffs).reshape(h, w)

    y = arr.ravel()
    X = X_base

    if ransac and HAS_SKLEARN and order <= 1:
        # Use RANSAC against the full set — memory intensive for large images, but robust.
        try:
            model = RANSACRegressor(min_samples=0.5, max_trials=ransac_iters, residual_threshold=ransac_thresh)
            model.fit(X, y)
            inlier_mask = model.inlier_mask_
            coeffs = np.concatenate((model.estimator_.coef_.ravel(), np.atleast_1d(model.estimator_.intercept_))) if hasattr(model, "estimator_") else fit_least_squares(y[inlier_mask], X[inlier_mask])
            bg = build_bg(coeffs, X)
            return arr - bg
        except Exception:
            # fallback to iterative robust fit
            ransac = False

    # Iterative robust fit (sigma clip)
    mask = np.ones_like(y, dtype=bool)
    coeffs = None
    for k in range(max(1, iters)):
        if not mask.any():
            break
        coeffs = fit_least_squares(y[mask], X[mask])
        bg = build_bg(coeffs, X).ravel()
        resid = y - bg
        if not robust or k == iters - 1:
            break
        med = np.nanmedian(resid[mask])
        std = np.nanstd(resid[mask])
        if std == 0 or np.isnan(std):
            break
        mask = np.abs(resid - med) < (2.5 * std)

    if coeffs is None:
        coeffs = fit_least_squares(y, X)
    bg = build_bg(coeffs, X)
    return arr - bg


def _poly_design_matrix(xx, yy, order):
    """Construct design matrix for 2D polynomial fitting (pairwise powers with i+j<=order).
    Returns shape (Npix, ncoeffs).
    """
    h, w = xx.shape
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append((xx ** i) * (yy ** j))
    X = np.column_stack([t.ravel() for t in terms])
    return X


def line_level(img, method='median', fit='offset', clip_sigma=None):
    """
    Line-by-line leveling. Works along rows (assumes fast-scan is horizontal).

    method: 'median' | 'mean' | 'linear'
    fit: 'offset' | 'slope' — whether to subtract only offset or also remove slope per-line
    clip_sigma: optional sigma for clipping outliers before computing per-line stats
    """
    arr = np.asarray(img, dtype=float)
    out = arr.copy()
    h, w = arr.shape
    for i in range(h):
        row = arr[i].copy()
        if clip_sigma is not None:
            med = np.nanmedian(row)
            std = np.nanstd(row)
            keep = np.abs(row - med) < (clip_sigma * std)
            if keep.sum() < 3:
                keep = np.ones_like(keep, dtype=bool)
            vals = row[keep]
        else:
            vals = row
        if method == 'median':
            center = np.nanmedian(vals)
            if fit == 'offset':
                out[i] = row - center
            else:
                # slope: fit a first-order polynomial to row
                x = np.arange(w)
                A = np.vstack([x, np.ones_like(x)]).T
                m, b = np.linalg.lstsq(A, row, rcond=None)[0]
                out[i] = row - (m * x + b)
        elif method == 'mean':
            center = np.nanmean(vals)
            if fit == 'offset':
                out[i] = row - center
            else:
                x = np.arange(w)
                A = np.vstack([x, np.ones_like(x)]).T
                m, b = np.linalg.lstsq(A, row, rcond=None)[0]
                out[i] = row - (m * x + b)
        elif method == 'linear':
            # Fit linear trend (slope + intercept) using least squares
            x = np.arange(w)
            A = np.vstack([x, np.ones_like(x)]).T
            m, b = np.linalg.lstsq(A, row, rcond=None)[0]
            if fit == 'offset':
                out[i] = row - b
            else:
                out[i] = row - (m * x + b)
        else:
            # default median offset
            c = np.nanmedian(vals)
            out[i] = row - c
    return out


def median_filter(img, size=3):
    """Wrapper around scipy.ndimage.median_filter that ensures odd sizes and handles NaNs by filling them with local median first."""
    if size % 2 == 0:
        size = max(1, size - 1)
    arr = np.asarray(img, dtype=float)
    # Replace NaNs with local median
    if np.isnan(arr).any():
        nanmask = np.isnan(arr)
        tmp = arr.copy()
        tmp[nanmask] = np.nanmedian(arr[~nanmask]) if (~nanmask).any() else 0
        arr = tmp
    return _median_filter(arr, size=size)


def despike_outliers(img, k_sigma=3.0, neigh=3, replace_with='median'):
    """
    Detect and replace outliers based on local median & std in a neighborhood.

    Parameters
    ----------
    img : 2D array
    k_sigma : float
        Threshold in sigmas for considering a pixel an outlier.
    neigh : int
        Neighborhood radius (odd window size = 2*neigh+1)
    replace_with : 'median'|'gaussian'|'local_mean'
        Strategy to replace detected outliers.

    Returns
    -------
    out : 2D array
    mask : 2D bool array where True indicates an outlier replaced.
    """
    arr = np.asarray(img, dtype=float)
    h, w = arr.shape
    # pad and compute local median/std via generic_filter-like approach but using median_filter for simplicity
    win = 2 * neigh + 1
    local_med = _median_filter(arr, size=win)
    # local std: approximate with gaussian filter on squared deviations
    sq = (arr - local_med) ** 2
    local_var = gaussian_filter(sq, sigma=max(1, neigh/2.0))
    local_std = np.sqrt(local_var)
    # Avoid zeros
    local_std[local_std == 0] = 1e-12
    diff = np.abs(arr - local_med)
    mask = diff > (k_sigma * local_std)
    out = arr.copy()
    if replace_with == 'median':
        out[mask] = local_med[mask]
    elif replace_with == 'gaussian':
        g = gaussian_filter(arr, sigma=max(0.5, neigh/2.0))
        out[mask] = g[mask]
    else:
        out[mask] = local_med[mask]
    return out, mask


def apply_to_stack(stack, func, progress_callback=None, **kwargs):
    """
    Apply a 2D->2D function to every frame of a stack and return a new stack.
    progress_callback(i, n) can be provided to update a progress bar.
    """
    arr = np.asarray(stack)
    if arr.ndim != 3:
        raise ValueError("stack must be T x H x W")
    out = np.empty_like(arr, dtype=float)
    n = arr.shape[0]
    for i in range(n):
        out[i] = func(arr[i], **kwargs)
        if progress_callback is not None:
            try:
                progress_callback(i + 1, n)
            except Exception:
                pass
    return out
