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



def compute_optimal_canvas(frames, drifts, extra_pad=50):
    H, W = frames[0].shape

    dy_min = drifts[:, 0].min()
    dy_max = drifts[:, 0].max()
    dx_min = drifts[:, 1].min()
    dx_max = drifts[:, 1].max()

    pad_top    = int(abs(dy_min)) + extra_pad
    pad_bottom = int(abs(dy_max)) + extra_pad
    pad_left   = int(abs(dx_min)) + extra_pad
    pad_right  = int(abs(dx_max)) + extra_pad

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

def ecc_align_first(frames, mask_frames, drifts, H_pad, W_pad, pad_top, pad_left):
    H, W = frames[0].shape

    # Canvas de referencia
    y0 = pad_top
    x0 = pad_left

    ref_canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
    ref_canvas[y0:y0+H, x0:x0+W] = frames[0].astype(np.float32)

    warp_mode = cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

    aligned = []
    masks_out = []
    ecc_transforms = []

    # Primer frame sin ECC
    aligned.append(ref_canvas.copy())

    mask0_canvas = np.zeros((H_pad, W_pad), dtype=np.uint8)
    mask0_canvas[y0:y0+H, x0:x0+W] = mask_frames[0]
    masks_out.append(mask0_canvas)
    ecc_transforms.append(np.eye(2, 3, dtype=np.float32))

    # ECC para el resto
    for i in range(1, len(frames)):
        canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
        canvas[y0:y0+H, x0:x0+W] = frames[i].astype(np.float32)

        warp_matrix = np.eye(2, 3, dtype=np.float32)  # reiniciar SIEMPRE

        try:
            cc, warp_matrix = cv2.findTransformECC(
                ref_canvas, canvas, warp_matrix, warp_mode, criteria
            )
        except cv2.error:
            # fallback seguro
            warp_matrix = np.eye(2, 3, dtype=np.float32)

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

    return np.array(aligned), np.array(masks_out), ecc_transforms


# ============================================================
#                   ECC FINAL (sin padding)
# ============================================================

def ecc_align_final(frames, mask_frames, H_pad, W_pad, pad_top, pad_left):
    H, W = frames[0].shape

    y0 = pad_top
    x0 = pad_left

    ref_canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
    ref_canvas[y0:y0+H, x0:x0+W] = frames[0].astype(np.float32)

    warp_mode = cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

    aligned = []
    masks_out = []
    ecc_transforms = []

    aligned.append(ref_canvas.copy())

    mask0_canvas = np.zeros((H_pad, W_pad), dtype=np.uint8)
    mask0_canvas[y0:y0+H, x0:x0+W] = mask_frames[0]
    masks_out.append(mask0_canvas)
    ecc_transforms.append(np.eye(2, 3, dtype=np.float32))

    for i in range(1, len(frames)):
        canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
        canvas[y0:y0+H, x0:x0+W] = frames[i].astype(np.float32)

        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            cc, warp_matrix = cv2.findTransformECC(
                ref_canvas, canvas, warp_matrix, warp_mode, criteria
            )
        except cv2.error:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

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

    return np.array(aligned), np.array(masks_out), ecc_transforms

class PlotWindow(QDialog):
    def __init__(self, drifts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Drift plot")
        self.resize(600, 400)

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

        label = QLabel(self)
        label.setPixmap(pix.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        layout = QVBoxLayout(self)
        layout.addWidget(label)
        self.setLayout(layout)

        plt.close(fig)

class DriftWindow(QWidget):
    def __init__(self, stack=None, meta=None):
        super().__init__()

        # ============================
        #   STORE STACK + META
        # ============================
        self.stack = stack
        self.meta = meta

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

        # Para ECC final (stack alineado que se manda al kymo)
        self.ecc_frames = None
        self.ecc_masks = None

        # ============================
        #   WINDOW CONFIG
        # ============================
        self.setWindowTitle("PreparAtrium – Drift & ECC Alignment")
        self.setMinimumSize(900, 500)
        self.resize(1300, 700)

        # ============================
        #   LEFT PANEL — ORIGINAL STACK
        # ============================
        self.label_original_left = QLabel("Original stack")
        self.label_original_left.setAlignment(Qt.AlignCenter)
        self.label_original_left.setMinimumSize(300, 300)
        self.label_original_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Trim sliders ---
        self.slider_trim_start = QSlider(Qt.Horizontal)
        self.slider_trim_end   = QSlider(Qt.Horizontal)

        self.slider_trim_start.setEnabled(True)
        self.slider_trim_end.setEnabled(True)

        # --- Trim buttons ---
        self.btn_apply_trim = QPushButton("Apply trim")
        self.btn_restore_original = QPushButton("Restore original")
        self.btn_add_to_drift = QPushButton("Add to drift correction")

        # ============================
        #   RIGHT PANEL — CURRENT + PROCESSED
        # ============================
        self.label_current = QLabel("Current stack")
        self.label_current.setAlignment(Qt.AlignCenter)
        self.label_current.setMinimumSize(400, 350)
        self.label_current.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.label_processed = QLabel("Processed preview")
        self.label_processed.setAlignment(Qt.AlignCenter)
        self.label_processed.setMinimumSize(400, 350)
        self.label_processed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.btn_accept_preview = QPushButton("Accept preview")

        # ============================
        #   DRIFT / ECC CONTROLS
        # ============================
        # ECC first pass (sobre current_frames/current_masks)
        self.btn_align_initial_ecc = QPushButton("ECC first pass")
        self.slider_initial_ecc = QSlider(Qt.Horizontal)
        self.slider_initial_ecc.setMaximum(0)

        # Optical Flow (Farneback) – método clásico
        self.btn_align_optical_flow = QPushButton("Optical Flow (Farneback)")
        self.btn_drift_plot = QPushButton("Show drift (Optical Flow)")
        self.slider_drift = QSlider(Qt.Horizontal)
        self.slider_drift.setMaximum(0)

        # Template Matching – nuevo método
        self.btn_align_template = QPushButton("Template Matching")

        # Fine ECC (sobre lo que tengas en current_frames/current_masks)
        self.btn_align_fine_ecc = QPushButton("Fine ECC")
        self.slider_fine_ecc = QSlider(Qt.Horizontal)
        self.slider_fine_ecc.setMaximum(0)

        # Guardar y kymograph
        self.btn_save_video = QPushButton("Save aligned video + metadata")
        self.btn_open_kymo = QPushButton("Open Kymograph Panel")

        # ============================
        #   STATUS
        # ============================
        self.progress = QProgressBar()
        self.status_label = QLabel("Status: waiting...")

        # ============================
        #   BUILD LAYOUT (clean 2×2)
        # ============================
        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setContentsMargins(8, 8, 8, 8)

        # LEFT PANEL
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Original stack"))
        self.slider_original = QSlider(Qt.Horizontal)
        self.slider_original.setMaximum(0)
        left_panel.addWidget(self.label_original_left, 1)

        left_panel.addWidget(QLabel("Original video"))
        left_panel.addWidget(self.label_original_left, 1)
        left_panel.addWidget(self.slider_original)


        left_panel.addWidget(QLabel("Trim start"))
        left_panel.addWidget(self.slider_trim_start)
        left_panel.addWidget(QLabel("Trim end"))
        left_panel.addWidget(self.slider_trim_end)

        trim_buttons = QHBoxLayout()
        trim_buttons.addWidget(self.btn_apply_trim)
        trim_buttons.addWidget(self.btn_restore_original)
        trim_buttons.addWidget(self.btn_add_to_drift)
        left_panel.addLayout(trim_buttons)

        grid.addLayout(left_panel, 0, 0, 2, 1)

        # RIGHT PANEL
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Current stack"))
        right_panel.addWidget(self.label_current, 1)

        right_panel.addWidget(QLabel("Processed preview"))
        right_panel.addWidget(self.label_processed, 1)
        right_panel.addWidget(self.btn_accept_preview)

        grid.addLayout(right_panel, 0, 1, 2, 1)

        # BOTTOM PANEL — DRIFT/ECC controls
        bottom = QVBoxLayout()

        ecc_row = QHBoxLayout()
        ecc_row.addWidget(self.btn_align_initial_ecc)
        ecc_row.addWidget(self.slider_initial_ecc)
        bottom.addLayout(ecc_row)

        # Optical Flow row
        of_row = QHBoxLayout()
        of_row.addWidget(self.btn_align_optical_flow)
        of_row.addWidget(self.btn_drift_plot)
        of_row.addWidget(self.slider_drift)
        bottom.addLayout(of_row)

        # Template Matching row
        tm_row = QHBoxLayout()
        tm_row.addWidget(self.btn_align_template)
        bottom.addLayout(tm_row)

        fine_row = QHBoxLayout()
        fine_row.addWidget(self.btn_align_fine_ecc)
        fine_row.addWidget(self.slider_fine_ecc)
        bottom.addLayout(fine_row)

        save_row = QHBoxLayout()
        save_row.addWidget(self.btn_save_video)
        save_row.addWidget(self.btn_open_kymo)
        bottom.addLayout(save_row)

        bottom.addWidget(self.progress)
        bottom.addWidget(self.status_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(grid, 1)
        main_layout.addLayout(bottom, 0)

        self.setLayout(main_layout)

        # ============================
        #   CONNECT SIGNALS
        # ============================
        #Original video
        self.slider_original.valueChanged.connect(self.update_original_frame)

        # Trim
        self.slider_trim_start.valueChanged.connect(self.update_trim_preview)
        self.slider_trim_end.valueChanged.connect(self.update_trim_preview)

        self.btn_apply_trim.clicked.connect(self.apply_trim)
        self.btn_restore_original.clicked.connect(self.restore_original)
        self.btn_add_to_drift.clicked.connect(self.add_original_to_current)

        # ECC first pass
        self.btn_align_initial_ecc.clicked.connect(self.align_initial_ecc)
        self.slider_initial_ecc.valueChanged.connect(self.update_initial_ecc_frame)

        # Optical Flow (Farneback)
        self.btn_align_optical_flow.clicked.connect(self.align_optical_flow_farneback)
        self.slider_drift.valueChanged.connect(self.update_optical_flow_frame)
        self.btn_drift_plot.clicked.connect(self.show_drift_plot)

        # Template Matching
        self.btn_align_template.clicked.connect(self.align_template_matching)

        # Fine ECC
        self.btn_align_fine_ecc.clicked.connect(self.align_fine_ecc)
        self.slider_fine_ecc.valueChanged.connect(self.update_fine_ecc_frame)

        # Save + kymo
        self.btn_save_video.clicked.connect(self.save_fine_aligned_video)
        self.btn_open_kymo.clicked.connect(self.open_kymo_panel)

        # Accept preview
        self.btn_accept_preview.clicked.connect(self.accept_preview)

        # ============================
        #   LOAD STACK IF PROVIDED
        # ============================
        if self.original_frames is not None:
            self.slider_original.setMaximum(len(self.original_frames) - 1)
            self.slider_trim_start.setMaximum(len(self.original_frames) - 1)
            self.slider_trim_end.setMaximum(len(self.original_frames) - 1)
            self.slider_trim_end.setValue(len(self.original_frames) - 1)

            self.update_original_frame(0)
            self.update_current_preview()
            self.status_label.setText(f"Video loaded: {len(self.original_frames)} frames")

    # ============================================================
    #                   VIDEO LOADING & DISPLAY
    # ============================================================

    def open_kymo_panel(self):
        if self.ecc_frames is None:
            self.status_label.setText("Run Fine ECC first to generate ECC-aligned stack")
            return

        from gui.kymo_panel import KymoPanel
        self.kymo_window = KymoPanel(stack=self.ecc_frames, meta=self.meta)
        self.kymo_window.show()

    def load_video(self):
        path = QFileDialog.getOpenFileName(
            self, "Select video", "", "AVI Files (*.avi)"
        )[0]
        if not path:
            return

        frames = read_avi_frames(path)
        self.original_frames = frames.copy()
        self.current_frames  = frames.copy()
        self.processed_frames = None

        self.original_masks = np.ones_like(self.original_frames, dtype=np.uint8)
        self.current_masks  = self.original_masks.copy()
        self.processed_masks = None

        self.current_drifts = None
        self.processed_drifts = None
        self.current_ecc_transforms = None
        self.processed_ecc_transforms = None

        self.slider_trim_start.setMaximum(len(self.original_frames) - 1)
        self.slider_trim_end.setMaximum(len(self.original_frames) - 1)
        self.slider_trim_end.setValue(len(self.original_frames) - 1)

        self.slider_initial_ecc.setMaximum(len(self.current_frames) - 1)
        self.slider_drift.setMaximum(len(self.current_frames) - 1)
        self.slider_fine_ecc.setMaximum(len(self.current_frames) - 1)

        self.update_original_frame(0)
        self.update_current_preview()
        self.update_processed_preview()
        self.status_label.setText(f"Video loaded: {len(self.original_frames)} frames")

    def update_original_frame(self, idx):
        if self.original_frames is None:
            return

        idx = max(0, min(idx, len(self.original_frames) - 1))
        frame = self.original_frames[idx]

        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)

        target_w = max(1, self.label_original_left.width())
        target_h = max(1, self.label_original_left.height())
        self.label_original_left.setPixmap(
            pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def update_current_preview(self):
        if self.current_frames is None:
            return

        frame = self.current_frames[0]
        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)

        target_w = max(1, self.label_current.width())
        target_h = max(1, self.label_current.height())
        self.label_current.setPixmap(
            pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def update_processed_preview(self):
        if self.processed_frames is None:
            self.label_processed.clear()
            return

        frame = self.processed_frames[0]
        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)

        target_w = max(1, self.label_processed.width())
        target_h = max(1, self.label_processed.height())
        self.label_processed.setPixmap(
            pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ============================================================
    #                   TRIM / RESTORE / ACCEPT
    # ============================================================

    def update_trim_preview(self):
        if self.original_frames is None:
            return
        # opcional: mostrar frame en label_original_left según slider_trim_start/slider_trim_end

    def apply_trim(self):
        if self.original_frames is None:
            return

        start = min(self.slider_trim_start.value(), self.slider_trim_end.value())
        end   = max(self.slider_trim_start.value(), self.slider_trim_end.value())

        self.current_frames = self.original_frames[start:end+1].copy()
        self.current_masks  = self.original_masks[start:end+1].copy()

        self.current_drifts = None
        self.current_ecc_transforms = None

        self.slider_initial_ecc.setMaximum(len(self.current_frames) - 1)
        self.slider_drift.setMaximum(len(self.current_frames) - 1)
        self.slider_fine_ecc.setMaximum(len(self.current_frames) - 1)

        self.update_current_preview()
        self.status_label.setText(f"Trim applied: {len(self.current_frames)} frames")

    def restore_original(self):
        if self.original_frames is None:
            return

        self.current_frames = self.original_frames.copy()
        self.current_masks  = self.original_masks.copy()
        self.current_drifts = None
        self.current_ecc_transforms = None

        self.slider_initial_ecc.setMaximum(len(self.current_frames) - 1)
        self.slider_drift.setMaximum(len(self.current_frames) - 1)
        self.slider_fine_ecc.setMaximum(len(self.current_frames) - 1)

        self.update_current_preview()
        self.status_label.setText("Restored original stack")

    def add_original_to_current(self):
        if self.original_frames is None:
            return
        self.current_frames = self.original_frames.copy()
        self.current_masks  = self.original_masks.copy()
        self.current_drifts = None
        self.current_ecc_transforms = None
        self.update_current_preview()
        self.status_label.setText("Original stack copied to current")

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

    # ============================================================
    #                   ECC FIRST / FINE
    # ============================================================

    def align_initial_ecc(self):
        # 1. Drift si no existe
        if self.current_drifts is None:
            self.current_drifts = compute_raw_drift(self.current_frames, template_size=64)

        # 2. Padding grande
        H_pad, W_pad, pad_top, pad_left = compute_optimal_canvas(
            self.current_frames, self.current_drifts, extra_pad=50
        )

        # 3. ECC first con padding correcto
        aligned, masks_out, ecc_transforms = ecc_align_first(
            self.current_frames, self.current_masks,
            self.current_drifts,
            H_pad, W_pad, pad_top, pad_left
        )

        # 4. Guardar preview
        self.processed_frames = aligned
        self.processed_masks = masks_out
        self.processed_ecc_transforms = ecc_transforms
        self.processed_drifts = self.current_drifts

        # 5. Slider
        self.slider_initial_ecc.setMaximum(len(self.processed_frames) - 1)

        # 6. Preview
        self.update_processed_preview()
        self.status_label.setText("ECC first pass completed")


    def update_initial_ecc_frame(self, idx):
        if self.processed_frames is None:
            return
        idx = max(0, min(idx, len(self.processed_frames) - 1))
        frame = self.processed_frames[idx]
        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)
        target_w = max(1, self.label_processed.width())
        target_h = max(1, self.label_processed.height())
        self.label_processed.setPixmap(
            pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def align_fine_ecc(self):
        if self.current_frames is None:
            self.status_label.setText("No current stack for Fine ECC")
            return

        # drift obligatorio
        if self.current_drifts is None:
            self.current_drifts = compute_raw_drift(self.current_frames, template_size=64)

        # padding grande
        H_pad, W_pad, pad_top, pad_left = compute_optimal_canvas(
            self.current_frames, self.current_drifts, extra_pad=50
        )

        aligned, masks_out, ecc_transforms = ecc_align_final(
            self.current_frames, self.current_masks,
            H_pad, W_pad, pad_top, pad_left
        )

        self.processed_frames = aligned
        self.processed_masks = masks_out
        self.processed_ecc_transforms = ecc_transforms
        self.processed_drifts = self.current_drifts

        self.ecc_frames = aligned.copy()
        self.ecc_masks = masks_out.copy()

        self.slider_fine_ecc.setMaximum(len(self.processed_frames) - 1)
        self.update_processed_preview()
        self.status_label.setText("Fine ECC completed")


    def update_fine_ecc_frame(self, idx):
        if self.processed_frames is None:
            return
        idx = max(0, min(idx, len(self.processed_frames) - 1))
        frame = self.processed_frames[idx]
        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)
        target_w = max(1, self.label_processed.width())
        target_h = max(1, self.label_processed.height())
        self.label_processed.setPixmap(
            pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ============================================================
    #                   DRIFT: OPTICAL FLOW + TEMPLATE MATCHING
    # ============================================================

    def align_optical_flow_farneback(self):
        if self.current_frames is None:
            self.status_label.setText("No current stack for Optical Flow")
            return

        # Usa drift_optical_flow clásico (de tu versión vieja)
        drifts = compute_raw_drift(self.current_frames)  # si quieres, haz otra función específica
        aligned, masks = align_with_auto_canvas(self.current_frames, drifts)

        self.processed_frames = aligned
        self.processed_masks = masks
        self.processed_drifts = drifts
        self.processed_ecc_transforms = self.current_ecc_transforms

        self.slider_drift.setMaximum(len(self.processed_frames) - 1)
        self.update_processed_preview()
        self.status_label.setText("Optical Flow alignment completed")

    def align_template_matching(self):
        if self.current_frames is None:
            self.status_label.setText("No current stack for Template Matching")
            return

        # Usa compute_raw_drift basado en drift_template_matching (ya definido arriba)
        drifts = compute_raw_drift(self.current_frames, template_size=64)
        aligned, masks = align_with_auto_canvas(self.current_frames, drifts)

        self.processed_frames = aligned
        self.processed_masks = masks
        self.processed_drifts = drifts
        self.processed_ecc_transforms = self.current_ecc_transforms

        self.slider_drift.setMaximum(len(self.processed_frames) - 1)
        self.update_processed_preview()
        self.status_label.setText("Template Matching alignment completed")

    def update_optical_flow_frame(self, idx):
        if self.processed_frames is None:
            return
        idx = max(0, min(idx, len(self.processed_frames) - 1))
        frame = self.processed_frames[idx]
        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)
        target_w = max(1, self.label_processed.width())
        target_h = max(1, self.label_processed.height())
        self.label_processed.setPixmap(
            pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def show_drift_plot(self):
        if self.processed_drifts is None:
            self.status_label.setText("No drift data to plot")
            return

        dlg = PlotWindow(self.processed_drifts, parent=self)
        dlg.exec()

    # ============================================================
    #                   SAVE VIDEO + METADATA
    # ============================================================

    def save_fine_aligned_video(self):
        if self.ecc_frames is None:
            self.status_label.setText("No ECC-aligned stack to save")
            return

        path = QFileDialog.getSaveFileName(
            self, "Save aligned video", "", "AVI Files (*.avi)"
        )[0]
        if not path:
            return

        # Aquí escribes self.ecc_frames a AVI y guardas meta (self.meta, self.current_drifts, self.current_ecc_transforms, etc.)
        # TODO: implementar escritura de vídeo + metadata según tu formato

        self.status_label.setText(f"Aligned video saved to: {path}")


# ============================================================
#                   MAIN ENTRY POINT
# ============================================================

#if __name__ == "__main__":
#    app = QApplication(sys.argv)
#    win = MainWindow()
#    win.show()
#    sys.exit(app.exec())
