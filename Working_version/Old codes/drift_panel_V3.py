#gui/drift_panel.py
import sys
import time
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QSlider, QProgressBar,
    QGridLayout, QDialog, QSizePolicy, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal

from scipy.ndimage import shift as nd_shift

from core.ui_utils import frame_to_qimage_safe


# ============================================================
#                   BASIC UTILITIES
# ============================================================

def read_avi_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    return np.array(frames)


def numpy_to_qimage(frame):
    """
    Conversión segura de 2D numpy array (grayscale) a QImage.
    Usa arr.strides[0] como bytes_per_line, asegura contiguidad y devuelve copia.
    """
    import numpy as np
    from PySide6.QtGui import QImage

    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError("numpy_to_qimage: frame must be 2D grayscale")

    # Normalizar si no es uint8 (no forzar si ya es uint8)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        a = a - np.nanmin(a)
        rng = np.nanmax(a)
        if rng == 0 or np.isnan(rng):
            rng = 1.0
        arr = (a / rng * 255.0).astype(np.uint8)

    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)

    h, w = arr.shape
    bytes_per_line = arr.strides[0]
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    return qimg.copy()

    

# ============================================================
#                   SEGMENTATION UTILITIES
# ============================================================

def sample_mask_otsu(frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask // 255


def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ============================================================
#                   DRIFT METHODS (Optical Flow)
# ============================================================

# def drift_optical_flow(ref, img):
#     flow = cv2.calcOpticalFlowFarneback(
#         ref, img, None,
#         pyr_scale=0.5, levels=3, winsize=21,
#         iterations=5, poly_n=7, poly_sigma=1.5, flags=0
#     )
#     dy = np.mean(flow[..., 1])
#     dx = np.mean(flow[..., 0])
#     return np.array([dy, dx])

def drift_template_matching(ref, img, template_size=64):
    """
    ref: frame de referencia (primer frame)
    img: frame actual
    template_size: tamaño del template cuadrado
    """

    H, W = ref.shape
    cy, cx = H // 2, W // 2
    half = template_size // 2

    # Template centrado en el frame de referencia
    template = ref[cy-half:cy+half, cx-half:cx+half].astype(np.float32)

    # Correlación normalizada (NCC)
    res = cv2.matchTemplate(img.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)

    # Máximo → posición del template
    _, _, _, max_loc = cv2.minMaxLoc(res)
    y, x = max_loc

    # Convertir a drift relativo al centro
    dy = y - (cy - half)
    dx = x - (cx - half)

    return np.array([dy, dx])

def compute_raw_drift(frames, template_size=64):
    ref = frames[0]
    drifts = []

    for i in range(len(frames)):
        if i == 0:
            drifts.append([0.0, 0.0])
        else:
            dy, dx = drift_template_matching(ref, frames[i], template_size)
            drifts.append([dy, dx])

    return np.array(drifts)



def compute_optimal_canvas(frames, drifts):
    H, W = frames[0].shape

    dy_min = drifts[:, 0].min()
    dy_max = drifts[:, 0].max()
    dx_min = drifts[:, 1].min()
    dx_max = drifts[:, 1].max()

    pad_top    = int(abs(dy_min))
    pad_bottom = int(abs(dy_max))
    pad_left   = int(abs(dx_min))
    pad_right  = int(abs(dx_max))

    H_pad = H + pad_top + pad_bottom
    W_pad = W + pad_left + pad_right

    return H_pad, W_pad, pad_top, pad_left


def align_with_auto_canvas(frames, drifts):
    H, W = frames[0].shape
    H_pad, W_pad, pad_top, pad_left = compute_optimal_canvas(frames, drifts)

    aligned = []
    masks = []

    for i, f in enumerate(frames):
        dy, dx = drifts[i]

        canvas = np.zeros((H_pad, W_pad), dtype=f.dtype)
        mask = np.zeros((H_pad, W_pad), dtype=np.uint8)

        y0 = pad_top
        x0 = pad_left

        canvas[y0:y0+H, x0:x0+W] = f
        mask[y0:y0+H, x0:x0+W] = 1

        aligned.append(nd_shift(canvas, shift=(dy, dx), mode="constant", cval=0))
        masks.append(nd_shift(mask, shift=(dy, dx), mode="constant", cval=0))

    return np.array(aligned), np.array(masks)



# ============================================================
#                   MASK PROPAGATION
# ============================================================

def propagate_mask(mask0, drifts, ecc_transforms=None, H_pad=None, W_pad=None):
    propagated = []

    H, W = mask0.shape

    if H_pad is not None and W_pad is not None:
        y0 = (H_pad - H) // 2
        x0 = (W_pad - W) // 2

    for i in range(len(drifts)):
        dy, dx = drifts[i]

        if H_pad is not None:
            mask_canvas = np.zeros((H_pad, W_pad), dtype=np.uint8)
            mask_canvas[y0:y0+H, x0:x0+W] = mask0
        else:
            mask_canvas = mask0.copy()

        mask_shifted = nd_shift(mask_canvas, shift=(dy, dx), mode="constant", cval=0)

        if ecc_transforms is not None:
            warp = ecc_transforms[i]
            mask_shifted = cv2.warpAffine(
                mask_shifted.astype(np.uint8),
                warp,
                (mask_shifted.shape[1], mask_shifted.shape[0]),
                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        propagated.append(mask_shifted)

    return np.array(propagated)


# ============================================================
#                   ECC FIRST (con padding)
# ============================================================

def ecc_align_first(frames, mask_frames, drifts):
    H, W = frames[0].shape

    # Usa el padding calculado por drift
    H_pad, W_pad, pad_top, pad_left = compute_optimal_canvas(frames, drifts)

    y0 = pad_top
    x0 = pad_left


    ref_canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
    ref_canvas[y0:y0+H, x0:x0+W] = frames[0].astype(np.float32)

    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

    aligned = []
    masks_out = []
    ecc_transforms = [warp_matrix.copy()]

    aligned.append(ref_canvas.copy())

    mask0_canvas = np.zeros((H_pad, W_pad), dtype=np.uint8)
    mask0_canvas[y0:y0+H, x0:x0+W] = mask_frames[0]
    masks_out.append(mask0_canvas)

    for i in range(1, len(frames)):
        canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
        canvas[y0:y0+H, x0:x0+W] = frames[i].astype(np.float32)

        cc, warp_matrix = cv2.findTransformECC(
            ref_canvas, canvas, warp_matrix, warp_mode, criteria
        )

        aligned_img = cv2.warpAffine(
            canvas, warp_matrix, (W_pad, H_pad),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        mask_canvas = np.zeros((H_pad, W_pad), dtype=np.uint8)
        mask_canvas[y0:y0+H, x0:x0+W] = mask_frames[i]

        aligned_mask = cv2.warpAffine(
            mask_canvas, warp_matrix, (W_pad, H_pad),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        aligned.append(aligned_img)
        masks_out.append(aligned_mask)
        ecc_transforms.append(warp_matrix.copy())
        
    return np.array(aligned), np.array(masks_out), ecc_transforms, H_pad, W_pad
    

# ============================================================
#                   ECC FINAL (sin padding)
# ============================================================

def ecc_align_final(frames, mask_frames):
    H_pad, W_pad = frames.shape[1], frames.shape[2]


    ref_canvas = frames[0].astype(np.float32)

    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

    aligned = [ref_canvas.copy()]
    masks_out = [mask_frames[0].astype(np.uint8)]
    ecc_transforms = [warp_matrix.copy()]

    for i in range(1, len(frames)):
        img = frames[i].astype(np.float32)

        cc, warp_matrix = cv2.findTransformECC(
            ref_canvas, img, warp_matrix, warp_mode, criteria
        )

        aligned_img = cv2.warpAffine(
            img, warp_matrix, (W_pad, H_pad),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        mask_i = mask_frames[i].astype(np.uint8)
        aligned_mask = cv2.warpAffine(
            mask_i, warp_matrix, (W_pad, H_pad),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        aligned.append(aligned_img)
        masks_out.append(aligned_mask)
        ecc_transforms.append(warp_matrix.copy())

    return np.array(aligned), np.array(masks_out), ecc_transforms


# ============================================================
#                   WORKERS
# ============================================================

class DriftWorker(QThread):
    progress_signal = Signal(int, float)
    finished_signal = Signal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, frames):
        super().__init__()
        self.original_frames = frames

    def run(self):
        n = len(self.original_frames)
        start_time = time.time()

        drifts = compute_raw_drift(self.original_frames)
        aligned, masks = align_with_auto_canvas(self.original_frames, drifts)

        for i in range(1, n):
            pct = int((i / (n - 1)) * 100)
            elapsed = time.time() - start_time
            remaining = (elapsed / i) * (n - i)
            self.progress_signal.emit(pct, remaining)

        self.finished_signal.emit(aligned, masks, drifts)


# ============================================================
#                   DRIFT PLOT WINDOW
# ============================================================

class PlotWindow(QDialog):
    def __init__(self, drifts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Drift plot")
        self.resize(600, 400)

        # Crear figura con matplotlib pero renderizar a buffer PNG
        fig, ax = plt.subplots()
        ax.plot(drifts[:, 1], label="dx")
        ax.plot(drifts[:, 0], label="dy")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Drift (pixels)")
        ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.canvas.print_png(buf)
        buf.seek(0)
        img = QImage.fromData(buf.getvalue())
        pix = QPixmap.fromImage(img)

        # Mostrar en un QLabel dentro de un QDialog (sin plt.show())
        label = QLabel(self)
        label.setPixmap(pix.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        layout = QVBoxLayout(self)
        layout.addWidget(label)
        self.setLayout(layout)

        # cerrar la figura de matplotlib para liberar memoria
        plt.close(fig)

# ============================================================
#                   DRIFT WINDOW (FULLY FIXED)
# ============================================================
class DriftWindow(QWidget):
    def __init__(self, stack=None, meta=None):
        super().__init__()

        # Store incoming data
        self.stack = stack
        self.meta = meta
        # --- NEW STACK ARCHITECTURE ---
        self.original_frames = stack.copy() if stack is not None else None
        self.current_frames  = stack.copy() if stack is not None else None
        self.processed_frames = None

        self.original_masks = np.ones_like(self.original_frames, dtype=np.uint8) if stack is not None else None
        self.current_masks  = self.original_masks.copy() if stack is not None else None
        self.processed_masks = None

        self.current_drifts = None
        self.processed_drifts = None

        self.current_ecc_transforms = None
        self.processed_ecc_transforms = None

        # Internal buffers (solo inicializar los que NO dependen del stack)
        self.drift_frames = None
        self.drift_masks = None
        self.drift_drifts = None
        self.initial_ecc_frame = None
        self.initial_ecc_masks = None
        self.ecc_frames = None
        self.ecc_masks = None

        self.setWindowTitle("PreparAtrium – Drift & ECC Alignment")
        self.setMinimumSize(900, 500)
        self.resize(1300, 700)

        # ============================================================
        #                   CREATE ALL WIDGETS
        # ============================================================

        # LOAD controls (kept as before)
        self.btn_load = QPushButton("Load video")
        self.slider_original = QSlider(Qt.Horizontal)
        self.trim_start = QSlider(Qt.Horizontal)
        self.trim_end = QSlider(Qt.Horizontal)
        self.btn_trim = QPushButton("Apply trim")
        self.btn_restore = QPushButton("Restore original")

        # ECC / drift controls
        self.btn_align_initial_ecc = QPushButton("Align ECC (first pass)")
        self.slider_initial_ecc = QSlider(Qt.Horizontal)
        self.btn_align_optical_flow = QPushButton("Template Matching")
        self.btn_drift_plot = QPushButton("Show drift (Template Matching drift plot)")
        self.btn_align_fine_ecc = QPushButton("Fine ECC alignment")
        self.btn_save_fine_ecc = QPushButton("Save aligned Fine ECC video")
        self.btn_open_kymo = QPushButton("Open Kymograph Panel")

        # STATUS
        self.progress = QProgressBar()
        self.status_label = QLabel("Status: waiting...")

        # ============================================================
        #                   PREVIEW LABELS (size policy + minimums)
        # ============================================================
        from PySide6.QtWidgets import QSizePolicy

        self.label_original_left = QLabel("Original video not loaded")
        self.label_original_left.setAlignment(Qt.AlignCenter)
        self.label_original_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_original_left.setMinimumSize(300, 220)
        self.label_original_left.setScaledContents(False)

        self.label_initial_ecc = QLabel("Initial ECC not available")
        self.label_initial_ecc.setAlignment(Qt.AlignCenter)
        self.label_initial_ecc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_initial_ecc.setMinimumSize(300, 220)
        self.label_initial_ecc.setScaledContents(False)
        self.slider_initial_ecc.setMaximum(0)

        self.label_drift = QLabel("Optical Flow not available")
        self.label_drift.setAlignment(Qt.AlignCenter)
        self.label_drift.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_drift.setMinimumSize(300, 220)
        self.label_drift.setScaledContents(False)
        self.slider_drift = QSlider(Qt.Horizontal)
        self.slider_drift.setMaximum(0)

        self.label_fine_ecc = QLabel("Fine ECC not available")
        self.label_fine_ecc.setAlignment(Qt.AlignCenter)
        self.label_fine_ecc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_fine_ecc.setMinimumSize(300, 220)
        self.label_fine_ecc.setScaledContents(False)
        self.slider_fine_ecc = QSlider(Qt.Horizontal)
        self.slider_fine_ecc.setMaximum(0)

        # ============================================================
        #                   BUILD 2x2 GRID WITH CONTAINERS
        # ============================================================
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        # Helper to create a panel: preview on top, controls row below
        def _make_panel(title_widget, preview_label, controls_widgets):
            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(6, 6, 6, 6)
            v.setSpacing(6)
            # title (optional)
            if title_widget is not None:
                v.addWidget(title_widget, 0)
            # preview (expanding)
            v.addWidget(preview_label, 1)
            # controls row
            controls_row = QHBoxLayout()
            controls_row.setSpacing(6)
            for w in controls_widgets:
                # ensure sliders/buttons don't expand vertically
                if isinstance(w, QSlider):
                    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                    w.setFixedHeight(18)
                else:
                    w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                controls_row.addWidget(w)
            controls_row.addStretch()
            v.addLayout(controls_row)
            return container

       # LEFT PANEL — ORIGINAL STACK
        title_left = QLabel("Original stack")
        title_left.setAlignment(Qt.AlignCenter)

        self.label_original_left = QLabel()
        self.label_original_left.setAlignment(Qt.AlignCenter)
        self.label_original_left.setMinimumSize(300, 300)
        self.label_original_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Buttons
        self.btn_trim_left = QPushButton("Trim")
        self.btn_restore_left = QPushButton("Restore original")
        self.btn_add_to_drift = QPushButton("Add to drift correction")

        left_controls = QHBoxLayout()
        left_controls.addWidget(self.btn_trim_left)
        left_controls.addWidget(self.btn_restore_left)
        left_controls.addWidget(self.btn_add_to_drift)

        left_panel = QVBoxLayout()
        left_panel.addWidget(title_left)
        left_panel.addWidget(self.label_original_left, 1)
        left_panel.addWidget(self.slider_original)
        left_panel.addLayout(left_controls)

        grid.addLayout(left_panel, 0, 0, 2, 1)


        # RIGHT PANEL — CURRENT (top) + PROCESSED (bottom)

        # Titles
        title_current = QLabel("Current stack")
        title_current.setAlignment(Qt.AlignCenter)

        title_processed = QLabel("Processed preview")
        title_processed.setAlignment(Qt.AlignCenter)

        # Video labels
        self.label_current = QLabel()
        self.label_current.setAlignment(Qt.AlignCenter)
        self.label_current.setMinimumSize(400, 350)
        self.label_current.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.label_processed = QLabel()
        self.label_processed.setAlignment(Qt.AlignCenter)
        self.label_processed.setMinimumSize(400, 350)
        self.label_processed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Accept button
        self.btn_accept_preview = QPushButton("Accept preview")

        right_panel = QVBoxLayout()
        right_panel.addWidget(title_current)
        right_panel.addWidget(self.label_current, 1)
        right_panel.addWidget(title_processed)
        right_panel.addWidget(self.label_processed, 1)
        right_panel.addWidget(self.btn_accept_preview)

        grid.addLayout(right_panel, 0, 1, 2, 1)


        # BOTTOM STATUS
        bottom_panel = QWidget()
        bottom_v = QVBoxLayout(bottom_panel)
        bottom_v.setContentsMargins(6, 6, 6, 6)
        bottom_v.setSpacing(6)
        bottom_v.addWidget(self.progress)
        bottom_v.addWidget(self.status_label)

        main_container = QVBoxLayout()
        main_container.setContentsMargins(6, 6, 6, 6)
        main_container.setSpacing(8)
        main_container.addLayout(grid, 1)
        main_container.addWidget(bottom_panel, 0)

        self.setLayout(main_container)

        # ============================================================
        #                   CONNECT SIGNALS
        # ============================================================
        self.btn_load.clicked.connect(self.load_video)
        self.slider_original.valueChanged.connect(self.update_original_frame)

        self.btn_trim.clicked.connect(self.apply_trim)
        self.btn_restore.clicked.connect(self.restore_original)

        self.btn_align_initial_ecc.clicked.connect(self.align_initial_ecc)
        self.slider_initial_ecc.valueChanged.connect(self.update_initial_ecc_frame)

        self.btn_align_optical_flow.clicked.connect(self.align_optical_flow)
        self.slider_drift.valueChanged.connect(self.update_optical_flow_frame)
        self.btn_drift_plot.clicked.connect(self.show_drift)
        self.btn_add_to_drift.clicked.connect(self.add_original_to_current)
        self.btn_accept_preview.clicked.connect(self.accept_preview)



        self.btn_align_fine_ecc.clicked.connect(self.align_fine_ecc)
        self.btn_save_fine_ecc.clicked.connect(self.save_fine_aligned_video)
        self.slider_fine_ecc.valueChanged.connect(self.update_fine_ecc_frame)
        self.btn_open_kymo.clicked.connect(self.open_kymo_panel)

        # ============================================================
        #                   LOAD STACK IF PROVIDED
        # ============================================================
        if self.stack is not None:
            self.original_frames = self.stack.copy()
            self.current_frames  = self.stack.copy()
            self.frames = self.stack.copy()  # <- recuperar compatibilidad con update_original_frame

            self.slider_original.setMaximum(len(self.frames) - 1)
            self.trim_start.setMaximum(len(self.frames) - 1)
            self.trim_end.setMaximum(len(self.frames) - 1)
            self.trim_end.setValue(len(self.frames) - 1)

            self.drift_masks = np.ones_like(self.frames, dtype=np.uint8)
            self.update_original_frame(0)
            # Disable manual loading
            self.btn_load.setEnabled(False)
            self.btn_load.setVisible(False)
            self.status_label.setText(f"Video loaded from AFMLoader: {len(self.original_frames)} frames")


    def add_original_to_current(self):
        self.current_frames = self.original_frames.copy()
        self.current_masks  = self.original_masks.copy()
        self.current_drifts = None
        self.current_ecc_transforms = None
        self.update_current_preview()
    def accept_preview(self):
        if self.processed_frames is None:
            self.status_label.setText("No processed preview to accept")
            return

        self.current_frames = self.processed_frames.copy()
        self.current_masks  = self.processed_masks.copy()
        self.current_drifts = self.processed_drifts
        self.current_ecc_transforms = self.processed_ecc_transforms

        self.processed_frames = None
        self.processed_masks = None
        self.processed_drifts = None
        self.processed_ecc_transforms = None

        self.update_current_preview()
        self.update_processed_preview()
        self.status_label.setText("Preview accepted")


    def build_drift_panel_ui(self):
        """
        Construye la UI del panel de drift con 4 áreas (cada una: visor + botones).
        Llamar desde __init__ de DriftWindow: self.build_drift_panel_ui()
        """

        # --- Widgets comunes de estado/control global ---
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.back_button = QPushButton("Back to video processing")

        # --- AREA 1: Load / Trim / Preview original video ---
        # Visor
        self.label_original_left = QLabel("Original video")
        self.label_original_left.setAlignment(Qt.AlignCenter)
        self.label_original_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_original_left.setMinimumSize(120, 90)

        # Botones
        self.btn_load_video = QPushButton("Load video")
        self.btn_trim_start = QPushButton("Trim Start")
        self.btn_trim_end = QPushButton("Trim End")
        self.btn_restore_original = QPushButton("Restore original")
        # slider de trim opcional
        self.slider_trim = QSlider(Qt.Horizontal)
        self.slider_trim.setEnabled(False)

        # Agrupar botones area1
        area1_tools = QHBoxLayout()
        area1_tools.addWidget(self.btn_load_video)
        area1_tools.addWidget(self.btn_trim_start)
        area1_tools.addWidget(self.btn_trim_end)
        area1_tools.addWidget(self.btn_restore_original)

        # --- AREA 2: ECC first pass ---
        self.label_ecc_first = QLabel("ECC aligned (first pass)")
        self.label_ecc_first.setAlignment(Qt.AlignCenter)
        self.label_ecc_first.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_ecc_first.setMinimumSize(120, 90)

        self.btn_align_ecc_first = QPushButton("Align ECC (first pass)")
        self.btn_show_ecc_first = QPushButton("Show ECC aligned video")

        area2_tools = QHBoxLayout()
        area2_tools.addWidget(self.btn_align_ecc_first)
        area2_tools.addWidget(self.btn_show_ecc_first)

        # --- AREA 3: Optical Flow ---
        self.label_optflow = QLabel("ECC + Optical Flow")
        self.label_optflow.setAlignment(Qt.AlignCenter)
        self.label_optflow.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_optflow.setMinimumSize(120, 90)

        self.btn_align_optical_flow = QPushButton("Align using Template Matching")
        self.btn_show_optflow = QPushButton("Show ECC-Optical Flow aligned video")
        # slider para navegar frames
        self.slider_drift = QSlider(Qt.Horizontal)
        self.slider_drift.setEnabled(False)

        area3_tools = QHBoxLayout()
        area3_tools.addWidget(self.btn_align_optical_flow)
        area3_tools.addWidget(self.btn_show_optflow)

        # --- AREA 4: Fine ECC, save, kymograph ---
        self.label_fine_ecc = QLabel("Fine ECC aligned")
        self.label_fine_ecc.setAlignment(Qt.AlignCenter)
        self.label_fine_ecc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_fine_ecc.setMinimumSize(120, 90)

        self.btn_align_fine_ecc = QPushButton("Align using fine ECC")
        self.btn_save_fine_ecc = QPushButton("Save Fine ECC aligned video")
        self.btn_go_kymolizer = QPushButton("Go to: Kymograph analysis")

        area4_tools = QHBoxLayout()
        area4_tools.addWidget(self.btn_align_fine_ecc)
        area4_tools.addWidget(self.btn_save_fine_ecc)
        area4_tools.addWidget(self.btn_go_kymolizer)

        # --- Layout principal: grid 2x2 para las 4 áreas ---
        grid = QGridLayout()
        grid.setSpacing(8)
        # fila 0: area1 | area2
        # fila 1: area3 | area4

        # Area 1 vertical box (visor + slider + botones)
        box1 = QVBoxLayout()
        box1.addWidget(self.label_original_left, stretch=1)
        box1.addWidget(self.slider_trim)
        box1.addLayout(area1_tools)

        # Area 2 vertical box
        box2 = QVBoxLayout()
        box2.addWidget(self.label_ecc_first, stretch=1)
        box2.addLayout(area2_tools)

        # Area 3 vertical box
        box3 = QVBoxLayout()
        box3.addWidget(self.label_optflow, stretch=1)
        box3.addWidget(self.slider_drift)
        box3.addLayout(area3_tools)

        # Area 4 vertical box
        box4 = QVBoxLayout()
        box4.addWidget(self.label_fine_ecc, stretch=1)
        box4.addLayout(area4_tools)

        # Insertar cajas en la grid
        grid.addLayout(box1, 0, 0)
        grid.addLayout(box2, 0, 1)
        grid.addLayout(box3, 1, 0)
        grid.addLayout(box4, 1, 1)

        # Hacer que las columnas y filas escalen proporcionalmente
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        # --- Pie de panel: controles globales (status, progress, back) ---
        footer = QHBoxLayout()
        footer.addWidget(self.back_button)
        footer.addItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        footer.addWidget(self.status_label)
        footer.addWidget(self.progress)

        # --- Layout final del widget (vertical) ---
        main_v = QVBoxLayout()
        main_v.addLayout(grid, stretch=1)
        main_v.addLayout(footer)

        # Aplicar layout al widget principal (asumiendo self es QWidget)
        self.setLayout(main_v)

        # --- Conexiones (placeholders: conecta a tus métodos existentes) ---
        self.btn_load_video.clicked.connect(self.load_video)                      # implementar load_video
        self.btn_trim_start.clicked.connect(self.trim_start)                     # implementar trim_start
        self.btn_trim_end.clicked.connect(self.trim_end)                         # implementar trim_end
        self.btn_restore_original.clicked.connect(self.restore_original)         # implementar restore_original

        self.btn_align_ecc_first.clicked.connect(self.align_ecc_first)          # implementar align_ecc_first
        self.btn_show_ecc_first.clicked.connect(lambda: self.update_ecc_first_frame(0))

        self.btn_align_optical_flow.clicked.connect(self.align_optical_flow)     # implementar align_optical_flow
        self.btn_show_optflow.clicked.connect(lambda: self.update_optical_flow_frame(0))
        self.slider_drift.valueChanged.connect(self.update_optical_flow_frame)

        self.btn_align_fine_ecc.clicked.connect(self.align_fine_ecc)             # implementar align_fine_ecc
        self.btn_save_fine_ecc.clicked.connect(self.save_fine_ecc_video)        # implementar save_fine_ecc_video
        self.btn_go_kymolizer.clicked.connect(self.open_kymo_panel)             # implementar open_kymo_panel

        # slider_trim connection (si tu update acepta idx)
        self.slider_trim.valueChanged.connect(self.update_trim_preview)

        # Asegurar que los labels aceptan mouse events si usas clicks
        self.label_original_left.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.label_ecc_first.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.label_optflow.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.label_fine_ecc.setAttribute(Qt.WA_TransparentForMouseEvents, False)

        # Inicializar estados
        self.slider_trim.setValue(0)
        self.slider_drift.setValue(0)
        self.progress.setValue(0)
        self.status_label.setText("Ready")
    # ============================================================
    #                   VIDEO LOADING & DISPLAY
    # ============================================================

    def open_kymo_panel(self):
        if self.ecc_frames is None:
            self.status_label.setText("Run ECC final pass first")
            return
        if self.ecc_frames is not None and len(self.ecc_frames) > 0:
            f = self.ecc_frames[0]
            
        from gui.kymo_panel import KymoPanel
        self.kymo_window = KymoPanel(stack=self.ecc_frames, meta=self.meta)
        self.kymo_window.show()


    def load_video(self):
        path = QFileDialog.getOpenFileName(
            self, "Select video", "", "AVI Files (*.avi)"
        )[0]
        if not path:
            return

        self.original_frames = read_avi_frames(path)
        self.original_frames = self.original_frames.copy()

        self.slider_original.setMaximum(len(self.original_frames) - 1)
        self.slider_drift.setMaximum(0)
        self.slider_fine_ecc.setMaximum(0)

        self.trim_start.setMaximum(len(self.original_frames) - 1)
        self.trim_end.setMaximum(len(self.original_frames) - 1)
        self.trim_end.setValue(len(self.original_frames) - 1)

        self.update_original_frame(0)
        self.status_label.setText(f"Video loaded: {len(self.original_frames)} frames")
    
    def update_original_frame(self, idx):
        if self.original_frames is None:
            return
        idx = max(0, min(idx, len(self.original_frames) - 1))
        frame = self.original_frames[idx]

        # Convertir a QImage de forma segura (si tienes frame_to_qimage_safe, úsala)
        try:
            # Si tienes una función numpy_to_qimage en este archivo, úsala; si usas core.ui_utils.frame_to_qimage_safe, importa y usa esa.
            from core.ui_utils import frame_to_qimage_safe
            qimg = frame_to_qimage_safe(frame)
        except Exception:
            # Fallback simple: asegurar uint8 y contiguo
            import numpy as np
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                a = arr.astype(np.float32)
                a = a - np.nanmin(a)
                rng = np.nanmax(a) or 1.0
                arr = (a / rng * 255.0).astype(np.uint8)
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            from PySide6.QtGui import QImage
            h, w = arr.shape
            bytes_per_line = arr.strides[0]
            qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()

        pix = QPixmap.fromImage(qimg)

        # Usar label_original_left (no label_preview)
        # Mostrar en el panel izquierdo (label_original_left)
        if hasattr(self, "label_original_left") and self.label_original_left is not None:
            target_w = max(1, self.label_original_left.width())
            target_h = max(1, self.label_original_left.height())
            self.label_original_left.setPixmap(
                pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        else:
            # Fallback: si por alguna razón label_original no existe, intenta asignar a label_drift
            if hasattr(self, "label_drift") and self.label_drift is not None:
                target_w = max(1, self.label_drift.width())
                target_h = max(1, self.label_drift.height())
                self.label_drift.setPixmap(pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_original_preview(self):
        frame = self.original_frames[0]
        qimg = numpy_to_qimage(frame)
        self.label_original_left.setPixmap(QPixmap.fromImage(qimg))

    def update_current_preview(self, idx=0):
        frame = self.current_frames[idx]
        qimg = numpy_to_qimage(frame)
        self.label_current.setPixmap(QPixmap.fromImage(qimg))

    def update_processed_preview(self, idx=0):
        if self.processed_frames is None:
            self.label_processed.setText("No processed preview")
            return
        frame = self.processed_frames[idx]
        qimg = numpy_to_qimage(frame)
        self.label_processed.setPixmap(QPixmap.fromImage(qimg))


    # ============================================================
    #                   TRIM
    # ============================================================

    def apply_trim(self):
        if self.original_frames is None:
            return
        s = self.trim_start.value()
        e = self.trim_end.value()
        self.original_frames = self.original_frames[s:e+1]
        self.slider_original.setMaximum(len(self.original_frames) - 1)
        self.update_original_frame(0)
        self.status_label.setText(f"Trim applied: {s} → {e}")

    def restore_original(self):
        if self.original_frames is None:
            return
        self.original_frames = self.original_frames.copy()
        self.slider_original.setMaximum(len(self.original_frames) - 1)
        self.trim_start.setMaximum(len(self.original_frames) - 1)
        self.trim_end.setMaximum(len(self.original_frames) - 1)
        self.trim_end.setValue(len(self.original_frames) - 1)
        self.update_original_frame(0)
        self.status_label.setText("Original video restored")

    # ============================================================
    #                   ECC FIRST PASS
    # ============================================================
    
    def align_initial_ecc(self):
        if self.original_frames is None:
            self.status_label.setText("Load a video first")
            return

        self.status_label.setText("Aligning ECC (first pass)...")
        self.progress.setValue(0)

        frame0 = self.original_frames[0].astype(np.uint8)
        mask0 = sample_mask_otsu(frame0)
        mask0 = clean_mask(mask0)

        zero_drift = np.zeros((len(self.original_frames), 2))
        mask_drift = propagate_mask(mask0, zero_drift)

        ecc_frames, ecc_masks_raw, ecc_transforms, H_pad, W_pad = ecc_align_first(
            self.original_frames,
            mask_drift,
            self.drift_drifts   # ← el drift calculado previamente
        )

        # --- Recorte automático del padding ---
        nonzero = np.where(ecc_frames > 0)
        if len(nonzero[0]) > 0:
            ymin, ymax = nonzero[0].min(), nonzero[0].max()
            xmin, xmax = nonzero[1].min(), nonzero[1].max()

            # Si el recorte es demasiado pequeño, ignorarlo
            if (ymax - ymin) < 20 or (xmax - xmin) < 20:
                print("ECC.align_initial_ecc: auto-crop too small → using full ECC frames")
            else:
                ecc_frames = ecc_frames[:, ymin:ymax+1, xmin:xmax+1]
                ecc_masks_raw = ecc_masks_raw[:, ymin:ymax+1, xmin:xmax+1]
                print("ECC.align_initial_ecc: auto-cropped padding → new shape =", ecc_frames.shape)
        else:
            print("ECC.align_initial_ecc: WARNING → padding crop failed (all zeros)")

        # --- Propagar máscara con ECC ---
        mask_ecc = propagate_mask(
            mask0,
            zero_drift,
            ecc_transforms,
            H_pad=H_pad,
            W_pad=W_pad
        )

        mask_union = np.max(mask_ecc, axis=0)
        ys, xs = np.where(mask_union > 0)

        if len(ys) > 0 and len(xs) > 0:
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            if (ymax - ymin) < 20 or (xmax - xmin) < 20:
                self.initial_ecc_frame = ecc_frames
                self.initial_ecc_masks = mask_ecc
            else:
                self.initial_ecc_frame = ecc_frames[:, ymin:ymax+1, xmin:xmax+1]
                self.initial_ecc_masks = mask_ecc[:, ymin:ymax+1, xmin:xmax+1]
        else:
            self.initial_ecc_frame = ecc_frames
            self.initial_ecc_masks = mask_ecc


        if self.initial_ecc_frame is None:
            self.status_label.setText("ECC failed: empty initial ECC frame")
            return

        self.slider_initial_ecc.setMaximum(len(self.initial_ecc_frame) - 1)
        self.update_initial_ecc_frame(0)

        self.status_label.setText("ECC first pass completed")
        self.progress.setValue(100)


    def update_initial_ecc_frame(self, idx):
        if self.initial_ecc_frame is None:
            return
        idx = max(0, min(idx, len(self.initial_ecc_frame) - 1))
        frame = self.initial_ecc_frame[idx]

        # usar la utilidad segura
        try:
            qimg = frame_to_qimage_safe(frame)
        except Exception:
            qimg = numpy_to_qimage(frame)

        pix = QPixmap.fromImage(qimg)
        if hasattr(self, "label_initial_ecc") and self.label_initial_ecc is not None:
            target_w = max(1, self.label_initial_ecc.width())
            target_h = max(1, self.label_initial_ecc.height())
            self.label_initial_ecc.setPixmap(pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

       
    # ============================================================
    #                   OPTICAL FLOW
    # ============================================================

    def run_template_matching(self):
        drifts = compute_raw_drift(self.current_frames)
        aligned, masks = align_with_auto_canvas(self.current_frames, drifts)

        self.processed_frames = aligned
        self.processed_masks  = masks
        self.processed_drifts = drifts
        self.processed_ecc_transforms = None

        self.update_processed_preview()
    def run_ecc_first(self):
        drifts = self.current_drifts or compute_raw_drift(self.current_frames)

        ecc_frames, ecc_masks, ecc_transforms, H_pad, W_pad = ecc_align_first(
            self.current_frames,
            self.current_masks,
            drifts
        )

        self.processed_frames = ecc_frames
        self.processed_masks  = ecc_masks
        self.processed_drifts = drifts
        self.processed_ecc_transforms = ecc_transforms

        self.update_processed_preview()
    def run_ecc_final(self):
        ecc_frames, ecc_masks, ecc_transforms = ecc_align_final(
            self.current_frames,
            self.current_masks
        )

        self.processed_frames = ecc_frames
        self.processed_masks  = ecc_masks
        self.processed_drifts = self.current_drifts
        self.processed_ecc_transforms = ecc_transforms

        self.update_processed_preview()

    def align_optical_flow(self):
        if self.initial_ecc_frames is None:
            self.status_label.setText("Run initial ECC first")
            return

        self.status_label.setText("Running Template Matching...")
        self.progress.setValue(0)

        # Drift basado en Template Matching
        drifts = compute_raw_drift(self.initial_ecc_frames)

        # Alineación con padding automático
        aligned, masks = align_with_auto_canvas(self.initial_ecc_frames, drifts)

        self.drift_frames = aligned
        self.drift_masks = masks
        self.drift_drifts = drifts

        self.slider_drift.setMaximum(len(self.drift_frames) - 1)
        self.update_optical_flow_frame(0)

        self.status_label.setText("Template Matching completed")
        self.progress.setValue(100)

    def update_progress(self, pct, remaining):
        self.progress.setValue(pct)
        self.status_label.setText(
            f"{pct}% completed — Estimated remaining time: {remaining:.1f} s"
        )

    def finish_optical_flow(self, aligned, masks, drifts):
        """
        Handler llamado cuando el worker de drift termina.
        Guarda resultados, recorta por la máscara y actualiza la UI de forma segura.
        """
                # Guardar resultados
        self.drift_drifts = drifts
        self.drift_frames = aligned
        self.drift_masks = masks

        # Si no hay frames, salir
        if self.drift_frames is None or len(self.drift_frames) == 0:
            self.status_label.setText("Optical Flow produced no frames")
            self.progress.setValue(100)
            return

        # Unión de máscaras y recorte (una sola vez)
        try:
            mask_union = np.max(self.drift_masks, axis=0)
            ys, xs = np.where(mask_union > 0)
            if len(ys) > 0 and len(xs) > 0:
                ymin, ymax = ys.min(), ys.max()
                xmin, xmax = xs.min(), xs.max()
                self.drift_frames = self.drift_frames[:, ymin:ymax+1, xmin:xmax+1]
                self.drift_masks = self.drift_masks[:, ymin:ymax+1, xmin:xmax+1]
        except Exception:
            # si algo falla con las máscaras, continuar sin recorte
            pass

        # Normalizar a uint8 y asegurar contigüidad (evita problemas con float32)
        try:
            arr = np.asarray(self.drift_frames)
            if arr.dtype != np.uint8:
                a = arr.astype(np.float32)
                a = a - np.nanmin(a)
                rng = np.nanmax(a)
                if rng == 0 or np.isnan(rng):
                    rng = 1.0
                arr = (a / rng * 255.0).astype(np.uint8)
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            self.drift_frames = arr
        except Exception:
            # si falla la normalización, dejar los frames tal cual y confiar en el fallback de update
            pass

        # Actualizar slider y mostrar primer frame
        n = len(self.drift_frames)
        self.slider_drift.setMaximum(max(0, n - 1))
        self.slider_drift.setEnabled(n > 1)

        # Forzar actualización del frame 0 (usa la función robusta)
        try:
            self.update_optical_flow_frame(0)
        except Exception:
            # fallback: mostrar primer frame manualmente
            try:
                frame = self.drift_frames[0]
                try:
                    qimg = frame_to_qimage_safe(frame)
                except Exception:
                    qimg = numpy_to_qimage(frame.astype(np.uint8))
                pix = QPixmap.fromImage(qimg)
                if hasattr(self, "label_drift") and self.label_drift is not None:
                    target_w = max(1, self.label_drift.width())
                    target_h = max(1, self.label_drift.height())
                    self.label_drift.setPixmap(pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except Exception:
                pass

        self.status_label.setText("alignment completed")
        self.progress.setValue(100)


    def update_optical_flow_frame(self, idx=None):
        """
        Mostrar el frame idx de drift_frames de forma segura.
        Si idx es None, lee el valor del slider_drift.
        """
        import numpy as np
        from PySide6.QtGui import QImage

        # Determinar índice
        if idx is None:
            if hasattr(self, "slider_drift"):
                try:
                    idx = int(self.slider_drift.value())
                except Exception:
                    idx = 0
            else:
                idx = 0

        # Selección segura de frames
        frames = None
        for name in ("drift_frames", "ecc_frames", "frames", "stack"):
            candidate = getattr(self, name, None)
            if candidate is not None:
                frames = candidate
                break

        if frames is None:
            return
        try:
            if len(frames) == 0:
                return
        except Exception:
            pass

        idx = max(0, min(int(idx), len(frames) - 1))
        frame = frames[idx]
        if frame is None:
            return

        # Obtener máscara si existe
        mask = None
        if hasattr(self, "drift_masks") and self.drift_masks is not None:
            try:
                if idx < len(self.drift_masks):
                    mask = self.drift_masks[idx]
            except Exception:
                mask = None

        # Preparar imagen de display (copia para no modificar original)
        display = np.asarray(frame).copy()
        if mask is not None:
            try:
                display[mask == 0] = 255
            except Exception:
                pass

        # Conversión segura a QImage
        try:
            qimg = frame_to_qimage_safe(display)
        except Exception:
            arr = np.asarray(display)
            # si color, convertir a gris
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                try:
                    import cv2
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                except Exception:
                    arr = arr[..., 0]

            # manejar NaNs
            if np.isnan(arr).any():
                arr = arr.copy()
                arr[np.isnan(arr)] = np.nanmin(arr)

            # normalizar a uint8
            if arr.dtype != np.uint8:
                a = arr.astype(np.float32)
                a = a - np.nanmin(a)
                rng = np.nanmax(a)
                if rng == 0 or np.isnan(rng):
                    rng = 1.0
                arr = (a / rng * 255.0).astype(np.uint8)

            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)

            h, w = arr.shape[:2]
            bytes_per_line = arr.strides[0]
            qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()

        # Crear pixmap y asignar al label correspondiente
        pix = QPixmap.fromImage(qimg)
        lbl = getattr(self, "label_drift", None) or getattr(self, "label_original_left", None) or getattr(self, "label_frame", None)
        if lbl is None:
            return

        target_w = max(1, lbl.width())
        target_h = max(1, lbl.height())
        lbl.setPixmap(pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def show_drift(self):
        if self.drift_drifts is None:
            self.status_label.setText("Run Optical Flow first")
            return
        win = PlotWindow(self.drift_drifts, parent=self)
        win.exec()

    # ============================================================
    #                   ECC FINAL (WITH MASK)
    # ============================================================

    def align_fine_ecc(self):
        if self.drift_frames is None:
            self.status_label.setText("Run Template Matching first")
            return

        self.status_label.setText("Aligning ECC (final pass)...")
        self.progress.setValue(0)

        # Máscara inicial del vídeo TM
        frame0 = self.drift_frames[0].astype(np.uint8)
        mask0 = sample_mask_otsu(frame0)
        mask0 = clean_mask(mask0)

        # Propagar máscara usando drift de Template Matching
        mask_tm = propagate_mask(
            mask0,
            self.drift_drifts,
            ecc_transforms=None,   # no usamos ECC aquí
            H_pad=self.drift_frames.shape[1],
            W_pad=self.drift_frames.shape[2]
        )

        ecc_frames, ecc_masks_raw, ecc_transforms = ecc_align_final(
            self.drift_frames,
            mask_tm
        )

        self.ecc_frames = ecc_frames
        self.ecc_masks = ecc_masks_raw
        self.ecc_transforms = ecc_transforms

        self.slider_fine_ecc.setMaximum(len(self.ecc_frames) - 1)
        self.update_fine_ecc_frame(0)

        self.status_label.setText("ECC final pass completed")
        self.progress.setValue(100)

        # Metadatos extendidos
        from core.ui_utils import extend_meta_with_stack_info
        self.meta = extend_meta_with_stack_info(
            self.meta,
            self.ecc_frames,
            drift=self.drift_drifts,
            ecc_transforms=self.ecc_transforms
        )

        # Abrir panel de kymos
        panel = KymoPanel(self.ecc_frames, self.meta)


    def update_fine_ecc_frame(self, idx=None):
        """
        Mostrar el frame idx de ecc_frames de forma segura.
        Acepta idx=None (leer del slider) o idx=int (desde slider).
        """
        # 1) Determinar idx si no se pasó
        if idx is None:
            if hasattr(self, "slider_fine_ecc"):
                try:
                    idx = int(self.slider_fine_ecc.value())
                except Exception:
                    idx = 0
            else:
                idx = 0

        # 2) Selección segura de frames (evitar evaluar numpy arrays en booleanos)
        frames = None
        for name in ("ecc_frames", "frames", "stack"):
            candidate = getattr(self, name, None)
            if candidate is not None:
                frames = candidate
                break

        # 3) Validaciones
        if frames is None:
            return
        try:
            if len(frames) == 0:
                return
        except Exception:
            # len no aplica: asumimos que frames es indexable
            pass

        # Debug seguro (frames ya definido)
        try:
            print("DEBUG update_fine_ecc_frame idx", idx, "frames type", type(frames), "len", len(frames))
        except Exception:
            print("DEBUG update_fine_ecc_frame: unable to print frames info")
          
        except Exception:
            print("DEBUG update_fine_ecc_frame: unable to print frames info")

        # 4) Normalizar índice y obtener frame
        idx = max(0, min(int(idx), len(frames) - 1))
        frame = frames[idx]
        if frame is None:
            return

        # 5) Conversión a QImage (usar util si existe, fallback robusto)
        try:
            qimg = frame_to_qimage_safe(frame)
        except Exception:
            import numpy as np
            from PySide6.QtGui import QImage
            arr = np.asarray(frame)

            # si color, convertir a gris
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                try:
                    import cv2
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                except Exception:
                    arr = arr[..., 0]

            # manejar NaNs
            if np.isnan(arr).any():
                arr = arr.copy()
                arr[np.isnan(arr)] = np.nanmin(arr)

            # normalizar a uint8
            if arr.dtype != np.uint8:
                a = arr.astype(np.float32)
                a = a - np.nanmin(a)
                rng = np.nanmax(a)
                if rng == 0 or np.isnan(rng):
                    rng = 1.0
                arr = (a / rng * 255.0).astype(np.uint8)

            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)

            h, w = arr.shape[:2]
            bytes_per_line = arr.strides[0]
            qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()

        # 6) Crear pixmap y asignar al label correspondiente
        pix = QPixmap.fromImage(qimg)
        lbl = getattr(self, "label_fine_ecc", None) or getattr(self, "label_original_left", None) or getattr(self, "label_frame", None)
        if lbl is None:
            return

        target_w = max(1, lbl.width())
        target_h = max(1, lbl.height())
        lbl.setPixmap(pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def save_fine_aligned_video(self):
        if self.ecc_frames is None:
            self.status_label.setText("No ECC frames to save")
            return

        path = QFileDialog.getSaveFileName(
            self, "Save aligned video", "", "AVI Files (*.avi)"
        )[0]
        if not path:
            return

        H, W = self.ecc_frames[0].shape
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), 20, (W, H), False)

        for f in self.ecc_frames:
            out.write(f.astype(np.uint8))

        out.release()
        self.status_label.setText("Aligned video saved")
    
# ============================================================
#                   MAIN ENTRY POINT
# ============================================================

#if __name__ == "__main__":
#    app = QApplication(sys.argv)
#    win = MainWindow()
#    win.show()
#    sys.exit(app.exec())
