import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class KymogramData:
    """
    Lightweight container for a kymogram and its metadata/provenance.
    - data: 2D numpy array (frames, spatial_samples)
    - metadata: dict with keys like pixel_size_nm, frame_rate, time_per_frame, source_name, etc.
    - provenance: free-form dict (loader, timestamp, source_file...)
    """
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


def extract_kymograph(stack, line_points, meta):
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
        frame_rate = (
            meta.get("real_fps")
            or meta.get("fps")
            or meta.get("frame_rate")
        )

    if frame_rate is None:
        # fallback visual; si es crítico para resultados científicos, lanzar ValueError en su lugar
        frame_rate = 1.0
        print("WARNING: extract_kymograph: meta.frame_rate missing; using fallback frame_rate=1.0")

    xs = np.array([p[0] for p in line_points])
    ys = np.array([p[1] for p in line_points])

    distances = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    total_length_px = np.sum(distances)
    total_length_nm = total_length_px * pixel_size

    n_samples = max(2, int(np.ceil(total_length_px)) + 1)
    t = np.linspace(0, 1, n_samples)

    fx = interp1d(np.linspace(0, 1, len(xs)), xs)
    fy = interp1d(np.linspace(0, 1, len(ys)), ys)

    kymo = []
    for frame in stack:
        profile = frame[fy(t).astype(int), fx(t).astype(int)]
        kymo.append(profile)

    kymo = np.array(kymo)

    axis_x_nm = np.linspace(0, total_length_nm, n_samples)
    axis_t_s = np.arange(len(stack)) / frame_rate

    return kymo, axis_x_nm, axis_t_s
