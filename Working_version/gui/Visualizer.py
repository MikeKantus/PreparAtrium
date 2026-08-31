import os
import numpy as np
import cv2
import h5py
import tifffile
import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QFileDialog
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


def numpy_to_qimage(arr):
    """Convierte un array 2D en QImage."""
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        a -= np.nanmin(a)
        rng = np.nanmax(a)
        if rng == 0:
            rng = 1
        arr = (a / rng * 255).astype(np.uint8)

    h, w = arr.shape
    return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()


class TestViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minimal AFM Viewer")
        self.resize(600, 600)

        self.label = QLabel("No image loaded")
        self.label.setAlignment(Qt.AlignCenter)

        self.btn_open = QPushButton("Open file")
        self.btn_open.clicked.connect(self.open_file)

        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.btn_open)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select file",
            "",
            "Images (*.asd, *.stp, *.jpk);;All files (*)"
        )

        if not path:
            return

        # Cargar imagen con QImage directamente
        img = QImage(path)
        if img.isNull():
            self.label.setText("Could not load image")
            return

        pix = QPixmap.fromImage(img)
        self.label.setPixmap(pix.scaled(
            self.label.width(),
            self.label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

try:
    from playnano import AFMImage
    HAS_PLAYNANO = True
except Exception:
    HAS_PLAYNANO = False


def _find_hdf5_dataset(f):
    """Busca el primer dataset 2D/3D en un HDF5."""
    def walk(group):
        for k, v in group.items():
            if isinstance(v, h5py.Dataset):
                if v.ndim in (2, 3):
                    return v
            elif isinstance(v, h5py.Group):
                res = walk(v)
                if res is not None:
                    return res
        return None
    return walk(f)


def read_afm_file(path):
    ext = os.path.splitext(path)[1].lower()

    # 1) JPK
    if ext == ".jpk":
        arr = read_jpk(path)
        if arr is not None:
            return arr, {"source_file": path}

    # 2) ASD (Asylum)
    if ext == ".asd":
        arr = read_asylum_asd(path)
        if arr is not None:
            return arr, {"source_file": path}

    # 3) STP (Nanoscope)
    if ext == ".stp":
        arr = read_nanoscope_stp(path)
        if arr is not None:
            return arr, {"source_file": path}

    # 4) HDF5 genérico
    try:
        with h5py.File(path, "r") as f:
            ds = _find_hdf5_dataset(f)
            if ds is not None:
                arr = ds[()]
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                return arr, {"source_file": path}
    except:
        pass

    # 5) TIFF genérico
    try:
        with tifffile.TiffFile(path) as tif:
            arr = tif.asarray()
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            return arr, {"source_file": path}
    except:
        pass

    raise ValueError(f"Unsupported AFM format: {path}")

def read_asylum_asd(path):
    try:
        with h5py.File(path, "r") as f:
            # Canales típicos
            candidates = [
                "Data/Image/Height",
                "Data/Image/Deflection",
                "Data/Image/Amplitude"
            ]

            for c in candidates:
                if c in f:
                    arr = f[c][()]
                    if arr.ndim == 2:
                        return arr[np.newaxis, ...]
                    elif arr.ndim == 3:
                        return arr
            raise ValueError("No AFM image channels found in ASD file")
    except Exception as e:
        print("ASD reader error:", e)
        return None
def read_nanoscope_stp(path):
    try:
        with open(path, "rb") as f:
            data = f.read()

        text = data[:5000].decode(errors="ignore")

        # Extraer parámetros
        import re
        off = int(re.search(r"Data offset:\s*(\d+)", text).group(1))
        length = int(re.search(r"Data length:\s*(\d+)", text).group(1))
        bpp = int(re.search(r"Bytes/pixel:\s*(\d+)", text).group(1))
        width = int(re.search(r"Width:\s*(\d+)", text).group(1))
        height = int(re.search(r"Height:\s*(\d+)", text).group(1))

        raw = data[off:off+length]

        if bpp == 2:
            arr = np.frombuffer(raw, dtype=np.uint16)
        elif bpp == 4:
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            raise ValueError("Unsupported bytes/pixel")

        arr = arr.reshape(height, width)
        return arr[np.newaxis, ...]

    except Exception as e:
        print("STP reader error:", e)
        return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TestViewer()
    viewer.show()
    sys.exit(app.exec())