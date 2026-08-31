import numpy as np
from PySide6.QtGui import QImage

def frame_to_qimage_safe(frame, percentile_clip=(0.5, 99.5)):
    """
    Convierte cualquier frame (float32/int/uint8) a QImage grayscale seguro.
    - Normaliza float/int a uint8 usando percentiles para evitar outliers.
    - Asegura contiguidad y usa arr.strides[0] como bytes_per_line.
    - Devuelve una copia de QImage para evitar dependencias de memoria.
    """
    if frame is None:
        return QImage()

    arr = np.asarray(frame)

    if arr.ndim != 2:
        raise ValueError("frame must be 2D grayscale")

    # Reemplazar NaN por mínimo
    if np.isnan(arr).any():
        arr = arr.copy()
        arr[np.isnan(arr)] = np.nanmin(arr)

    # Si no es uint8, normalizar con percentiles y convertir
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        lo, hi = percentile_clip
        lo_v = np.percentile(a, lo)
        hi_v = np.percentile(a, hi)
        if hi_v <= lo_v:
            lo_v = np.nanmin(a)
            hi_v = np.nanmax(a)
            if hi_v <= lo_v:
                hi_v = lo_v + 1.0
        a = np.clip(a, lo_v, hi_v)
        a = ((a - lo_v) / (hi_v - lo_v) * 255.0).astype(np.uint8)
        arr = a

    # Asegurar contiguidad
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)

    h, w = arr.shape
    bytes_per_line = arr.strides[0]

    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    return qimg.copy()

def extend_meta_with_stack_info(meta, stack, drift=None, ecc_transforms=None):
    """
    Añade metadatos derivados para trazabilidad completa del análisis.
    No altera los metadatos físicos (nm/pixel, frame_rate, etc.).
    """
    if meta is None:
        meta = {}

    extended = dict(meta)

    # Tamaño original (si existe en meta)
    x_orig = meta.get("x_pixels", None)
    y_orig = meta.get("y_pixels", None)

    # Tamaño actual del stack
    n_frames = len(stack)
    y_curr, x_curr = stack[0].shape

    # Padding aplicado (si lo hay)
    pad_x = x_curr - x_orig if x_orig is not None else None
    pad_y = y_curr - y_orig if y_orig is not None else None

    # Duración total del vídeo
    frame_rate = meta.get("frame_rate", None)
    total_time_s = n_frames / frame_rate if frame_rate else None

    extended.update({
        "x_pixels_original": x_orig,
        "y_pixels_original": y_orig,
        "x_pixels_current": x_curr,
        "y_pixels_current": y_curr,
        "padding_x_px": pad_x,
        "padding_y_px": pad_y,
        "n_frames": n_frames,
        "total_time_s": total_time_s,
    })
    # Drift acumulado (si se pasa)
    if drift is not None:
        extended["drift_dx_px"] = drift[:, 1].tolist()
        extended["drift_dy_px"] = drift[:, 0].tolist()

    # ECC transforms (si se pasan)
    if ecc_transforms is not None:
        extended["ecc_transforms"] = [m.tolist() for m in ecc_transforms]

    return extended