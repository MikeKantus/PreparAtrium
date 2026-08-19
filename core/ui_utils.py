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
