import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass
from scipy.ndimage import map_coordinates

@dataclass
class KymogramData:
    kymo: np.ndarray
    axis_x_nm: np.ndarray
    axis_t_s: np.ndarray
    metadata: dict


def extract_kymograph(stack, line_points, meta, radius_px=0, method='mean', subpixel=False):
    """Extract kymograph along a polyline with optional perpendicular averaging.

    Parameters
    ----------
    stack : array-like
        Frames array (T, H, W)
    line_points : sequence of (x, y)
        Polyline in image coordinates (x horizontal, y vertical)
    meta : dict
        Metadata containing pixel_size/frame_rate (optional)
    radius_px : int
        Radius in pixels for perpendicular averaging (default 0: single-pixel profile)
    method : {'mean', 'median', 'max'}
        Aggregation method across perpendicular samples.
    subpixel : bool
        If True, use bilinear interpolation (map_coordinates order=1) for sampling; otherwise round to nearest integer.

    Returns
    -------
    kymo : np.ndarray
    axis_x_nm : np.ndarray
    axis_t_s : np.ndarray
    """
    # Aceptar meta None o incompleto
    pixel_size = None
    if meta is not None:
        pixel_size = meta.get("pixel_size", None) or meta.get("pixel_size_nm", None) or meta.get("pixel_size_x", None)

    if pixel_size is None:
        # fallback visual; si es crítico para resultados científicos, lanzar ValueError en su lugar
        pixel_size = 1.0
        print("WARNING: extract_kymograph: meta.pixel_size missing; using fallback pixel_size=1.0")
    frame_rate = None
    if meta is not None:
        frame_rate = meta.get("frame_rate", None) or meta.get("real_fps", None) or meta.get("fps", None)

    if frame_rate is None:
        # fallback visual; si es crítico para resultados científicos, lanzar ValueError en su lugar
        frame_rate = 1.0
        print("WARNING: extract_kymograph: meta.frame_rate missing; using fallback frame_rate=1.0")

    xs = np.array([p[0] for p in line_points])
    ys = np.array([p[1] for p in line_points])

    # handle degenerate
    if len(xs) < 2:
        # return empty
        return np.zeros((len(stack), 0)), np.array([]), np.arange(len(stack)) / frame_rate

    # compute cumulative distances along the polyline
    seg_lengths = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    total_length_px = np.sum(seg_lengths)
    total_length_nm = total_length_px * pixel_size

    n_samples = max(2, int(np.ceil(total_length_px)) + 1)
    t = np.linspace(0, 1, n_samples)

    fx = interp1d(np.linspace(0, 1, len(xs)), xs)
    fy = interp1d(np.linspace(0, 1, len(ys)), ys)

    sample_x = fx(t)
    sample_y = fy(t)

    # Precompute perpendicular offsets if radius_px > 0
    if radius_px and radius_px > 0:
        # compute tangent vectors along the polyline samples
        dx = np.gradient(sample_x)
        dy = np.gradient(sample_y)
        norms = np.sqrt(dx*dx + dy*dy) + 1e-12
        ux = dx / norms
        uy = dy / norms
        # perpendicular vectors (normalized)
        px = -uy
        py = ux
        # sample offsets
        radii = np.arange(-radius_px, radius_px+1)
        ofs_x = np.outer(px, radii)
        ofs_y = np.outer(py, radii)
        # final sample grid per sample point: shape (n_samples, n_offsets)
        sample_grid_x = sample_x[:, None] + ofs_x
        sample_grid_y = sample_y[:, None] + ofs_y
    else:
        sample_grid_x = sample_x[:, None]
        sample_grid_y = sample_y[:, None]

    kymo = []
    H = None
    W = None
    for frame in stack:
        if H is None:
            try:
                H, W = frame.shape
            except Exception:
                frame = np.asarray(frame)
                H, W = frame.shape
        # clamp indices
        sx = np.clip(sample_grid_x, 0, W-1)
        sy = np.clip(sample_grid_y, 0, H-1)

        if subpixel:
            # map_coordinates expects coordinates as (dim, indices), where first axis is y then x
            coords = np.vstack((sy.ravel(), sx.ravel()))
            vals = map_coordinates(frame, coords, order=1, mode='nearest').reshape(sy.shape)
        else:
            ix = np.round(sx).astype(int)
            iy = np.round(sy).astype(int)
            vals = frame[iy, ix]

        # vals shape: (n_samples, n_offsets)
        if vals.ndim == 2:
            if method == 'mean':
                prof = np.nanmean(vals, axis=1)
            elif method == 'median':
                prof = np.nanmedian(vals, axis=1)
            elif method == 'max':
                prof = np.nanmax(vals, axis=1)
            else:
                prof = np.nanmean(vals, axis=1)
        else:
            prof = vals
        kymo.append(prof)

    kymo = np.asarray(kymo)

    axis_x_nm = np.linspace(0, total_length_nm, n_samples)
    axis_t_s = np.arange(len(stack)) / frame_rate

    return kymo, axis_x_nm, axis_t_s
