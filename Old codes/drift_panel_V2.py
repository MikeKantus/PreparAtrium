#gui/drift_panel.py
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QSlider, QProgressBar,
    QGridLayout, QDialog
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
    h, w = frame.shape
    bytes_per_line = w
    return QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
    

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

def drift_optical_flow(ref, img):
    flow = cv2.calcOpticalFlowFarneback(
        ref, img, None,
        pyr_scale=0.5, levels=3, winsize=21,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0
    )
    dy = np.mean(flow[..., 1])
    dx = np.mean(flow[..., 0])
    return np.array([dy, dx])


def compute_raw_drift(frames):
    ref = frames[0]
    drifts = []

    for i in range(len(frames)):
        if i == 0:
            drifts.append([0.0, 0.0])
        else:
            dy, dx = drift_optical_flow(ref, frames[i])
            drifts.append([dy, dx])

    return np.array(drifts)


def compute_optimal_canvas(frames, drifts):
    H, W = frames[0].shape

    dy_min, dy_max = drifts[:, 0].min(), drifts[:, 0].max()
    dx_min, dx_max = drifts[:, 1].min(), drifts[:, 1].max()

    H_pad = H + int(abs(dy_min) + abs(dy_max))
    W_pad = W + int(abs(dx_min) + abs(dx_max))

    return H_pad, W_pad


def align_with_auto_canvas(frames, drifts):
    H, W = frames[0].shape
    H_pad, W_pad = compute_optimal_canvas(frames, drifts)

    aligned = []
    masks = []

    for i, f in enumerate(frames):
        dy, dx = drifts[i]

        canvas = np.zeros((H_pad, W_pad), dtype=f.dtype)
        mask = np.zeros((H_pad, W_pad), dtype=np.uint8)

        y0 = (H_pad - H) // 2
        x0 = (W_pad - W) // 2

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

def ecc_align_first(frames, mask_frames):
    H, W = frames[0].shape

    pad = max(H, W)
    H_pad = H + pad
    W_pad = W + pad

    y0 = (H_pad - H) // 2
    x0 = (W_pad - W) // 2

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
    H_pad, W_pad = frames[0].shape

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
        self.frames = frames

    def run(self):
        n = len(self.frames)
        start_time = time.time()

        drifts = compute_raw_drift(self.frames)
        aligned, masks = align_with_auto_canvas(self.frames, drifts)

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

        # Internal buffers
        self.frames = None
        self.original_frames = None
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

        # LOAD
        self.btn_load = QPushButton("Load video")
        self.label_original = QLabel("Original video not loaded")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.slider_original = QSlider(Qt.Horizontal)

        # TRIM
        self.trim_start = QSlider(Qt.Horizontal)
        self.trim_end = QSlider(Qt.Horizontal)
        self.btn_trim = QPushButton("Apply trim")
        self.btn_restore = QPushButton("Restore original")

        # INITIAL ECC
        self.label_initial_ecc = QLabel("Initial ECC not available")
        self.label_initial_ecc.setAlignment(Qt.AlignCenter)
        self.btn_align_initial_ecc = QPushButton("Align ECC (first pass)")
        self.slider_initial_ecc = QSlider(Qt.Horizontal)
        self.slider_initial_ecc.setMaximum(0)

        # OPTICAL FLOW
        self.btn_align_optical_flow = QPushButton("Optical Flow Alignment")
        self.btn_drift_plot = QPushButton("Show drift (Optical Flow)")
        self.label_drift = QLabel("Optical Flow not available")
        self.label_drift.setAlignment(Qt.AlignCenter)
        self.slider_drift = QSlider(Qt.Horizontal)
        self.slider_drift.setMaximum(0)

        # FINE ECC
        self.label_fine_ecc = QLabel("Fine ECC not available")
        self.label_fine_ecc.setAlignment(Qt.AlignCenter)
        self.slider_fine_ecc = QSlider(Qt.Horizontal)
        self.slider_fine_ecc.setMaximum(0)
        self.btn_align_fine_ecc = QPushButton("Fine ECC alignment")
        self.btn_save_fine_ecc = QPushButton("Save aligned Fine ECC video")
        self.btn_open_kymo = QPushButton("Open Kymograph Panel")

        # STATUS
        self.progress = QProgressBar()
        self.status_label = QLabel("Status: waiting...")

        # ============================================================
        #                   BUILD GRID LAYOUT
        # ============================================================

        grid = QGridLayout()

        # A1 — ORIGINAL VIDEO PANEL
        panel_A1 = QVBoxLayout()
        panel_A1.addWidget(self.btn_load)
        panel_A1.addWidget(self.label_original)
        panel_A1.addWidget(self.slider_original)

        panel_A1.addWidget(QLabel("Trim start"))
        panel_A1.addWidget(self.trim_start)
        panel_A1.addWidget(QLabel("Trim end"))
        panel_A1.addWidget(self.trim_end)

        trim_buttons = QHBoxLayout()
        trim_buttons.addWidget(self.btn_trim)
        trim_buttons.addWidget(self.btn_restore)
        panel_A1.addLayout(trim_buttons)

        grid.addLayout(panel_A1, 0, 0)

        # B1 — INITIAL ECC
        panel_B1 = QVBoxLayout()
        panel_B1.addWidget(QLabel("Initial ECC"))
        panel_B1.addWidget(self.label_initial_ecc)
        panel_B1.addWidget(self.slider_initial_ecc)
        panel_B1.addWidget(self.btn_align_initial_ecc)

        grid.addLayout(panel_B1, 0, 1)

        # C1 — OPTICAL FLOW
        panel_C1 = QVBoxLayout()
        panel_C1.addWidget(QLabel("Optical Flow"))
        panel_C1.addWidget(self.label_drift)
        panel_C1.addWidget(self.slider_drift)
        panel_C1.addWidget(self.btn_align_optical_flow)
        panel_C1.addWidget(self.btn_drift_plot)

        grid.addLayout(panel_C1, 0, 2)

        # A2 — FINE ECC
        panel_A2 = QVBoxLayout()
        panel_A2.addWidget(QLabel("Fine ECC"))
        panel_A2.addWidget(self.label_fine_ecc)
        panel_A2.addWidget(self.slider_fine_ecc)
        panel_A2.addWidget(self.btn_align_fine_ecc)
        panel_A2.addWidget(self.btn_save_fine_ecc)
        panel_A2.addWidget(self.btn_open_kymo)

        grid.addLayout(panel_A2, 1, 0)

        # BOTTOM STATUS
        bottom_panel = QVBoxLayout()
        bottom_panel.addWidget(self.progress)
        bottom_panel.addWidget(self.status_label)

        main_container = QVBoxLayout()
        main_container.addLayout(grid)
        main_container.addLayout(bottom_panel)

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

        self.btn_align_fine_ecc.clicked.connect(self.align_fine_ecc)
        self.btn_save_fine_ecc.clicked.connect(self.save_fine_aligned_video)
        self.slider_fine_ecc.valueChanged.connect(self.update_fine_ecc_frame)
        self.btn_open_kymo.clicked.connect(self.open_kymo_panel)

        # ============================================================
        #                   LOAD STACK IF PROVIDED
        # ============================================================

        if self.stack is not None:
            self.frames = self.stack.copy()
            self.original_frames = self.frames.copy()

            self.slider_original.setMaximum(len(self.frames) - 1)
            self.trim_start.setMaximum(len(self.frames) - 1)
            self.trim_end.setMaximum(len(self.frames) - 1)
            self.trim_end.setValue(len(self.frames) - 1)

            self.update_original_frame(0)
            self.status_label.setText(f"Video loaded from AFMLoader: {len(self.frames)} frames")
    # ============================================================
    #                   VIDEO LOADING & DISPLAY
    # ============================================================

    def open_kymo_panel(self):
        if self.ecc_frames is None:
            self.status_label.setText("Run ECC2 first")
            return

        from gui.kymo_panel import KymoPanel
        self.kymo_window = KymoPanel(self.ecc_frames, self.meta)
        self.kymo_window.show()

    def load_video(self):
        path = QFileDialog.getOpenFileName(
            self, "Select video", "", "AVI Files (*.avi)"
        )[0]
        if not path:
            return

        self.frames = read_avi_frames(path)
        self.original_frames = self.frames.copy()

        self.slider_original.setMaximum(len(self.frames) - 1)
        self.slider_drift.setMaximum(0)
        self.slider_fine_ecc.setMaximum(0)

        self.trim_start.setMaximum(len(self.frames) - 1)
        self.trim_end.setMaximum(len(self.frames) - 1)
        self.trim_end.setValue(len(self.frames) - 1)

        self.update_original_frame(0)
        self.status_label.setText(f"Video loaded: {len(self.frames)} frames")

    def update_original_frame(self, idx):
        if self.frames is None:
            return
        idx = max(0, min(idx, len(self.frames) - 1))
        frame = self.frames[idx]
        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)
        # si escalas:
        pix = pix.scaled(self.label_preview.width(), self.label_preview.height(), Qt.KeepAspectRatio)
        self.label_preview.setPixmap(pix)


    # ============================================================
    #                   TRIM
    # ============================================================

    def apply_trim(self):
        if self.frames is None:
            return
        s = self.trim_start.value()
        e = self.trim_end.value()
        self.frames = self.frames[s:e+1]
        self.slider_original.setMaximum(len(self.frames) - 1)
        self.update_original_frame(0)
        self.status_label.setText(f"Trim applied: {s} → {e}")

    def restore_original(self):
        if self.original_frames is None:
            return
        self.frames = self.original_frames.copy()
        self.slider_original.setMaximum(len(self.frames) - 1)
        self.trim_start.setMaximum(len(self.frames) - 1)
        self.trim_end.setMaximum(len(self.frames) - 1)
        self.trim_end.setValue(len(self.frames) - 1)
        self.update_original_frame(0)
        self.status_label.setText("Original video restored")

    # ============================================================
    #                   ECC FIRST PASS
    # ============================================================

    def align_initial_ecc(self):
        if self.frames is None:
            self.status_label.setText("Load a video first")
            return

        self.status_label.setText("Aligning ECC (first pass)...")
        self.progress.setValue(0)

        frame0 = self.frames[0].astype(np.uint8)
        mask0 = sample_mask_otsu(frame0)
        mask0 = clean_mask(mask0)

        zero_drift = np.zeros((len(self.frames), 2))
        mask_drift = propagate_mask(mask0, zero_drift)

        ecc_frames, ecc_masks_raw, ecc_transforms, H_pad, W_pad = ecc_align_first(
            self.frames, mask_drift
        )

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

            cropped_frames = ecc_frames[:, ymin:ymax+1, xmin:xmax+1]
            cropped_masks = mask_ecc[:, ymin:ymax+1, xmin:xmax+1]

            self.initial_ecc_frame = cropped_frames
            self.initial_ecc_masks = cropped_masks

        self.slider_initial_ecc.setMaximum(len(self.initial_ecc_frame) - 1)
        self.update_initial_ecc_frame(0)

        self.status_label.setText("ECC first pass completed")
        self.progress.setValue(100)

    def update_initial_ecc_frame(self, idx):
        if self.initial_ecc_frame is None:
            return
        idx = max(0, min(idx, len(self.initial_ecc_frame) - 1))
        frame = self.initial_ecc_frame[idx].astype(np.uint8)
        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)
        # si escalas:
        pix = pix.scaled(self.label_preview.width(), self.label_preview.height(), Qt.KeepAspectRatio)
        self.label_preview.setPixmap(pix)


    # ============================================================
    #                   OPTICAL FLOW
    # ============================================================

    def align_optical_flow(self):
        if self.initial_ecc_frame is None:
            self.status_label.setText("Run ECC first")
            return

        self.status_label.setText("Aligning drift (Optical Flow)...")
        self.progress.setValue(0)

        self.worker_drift = DriftWorker(self.initial_ecc_frame)
        self.worker_drift.progress_signal.connect(self.update_progress)
        self.worker_drift.finished_signal.connect(self.finish_optical_flow)
        self.worker_drift.start()

    def update_progress(self, pct, remaining):
        self.progress.setValue(pct)
        self.status_label.setText(
            f"{pct}% completed — Estimated remaining time: {remaining:.1f} s"
        )

    def finish_optical_flow(self, aligned, masks, drifts):
        self.drift_drifts = drifts
        self.drift_frames = aligned
        self.drift_masks = masks

        mask_union = np.max(self.drift_masks, axis=0)
        ys, xs = np.where(mask_union > 0)

        if len(ys) > 0 and len(xs) > 0:
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            self.drift_frames = self.drift_frames[:, ymin:ymax+1, xmin:xmax+1]
            self.drift_masks = self.drift_masks[:, ymin:ymax+1, xmin:xmax+1]

        self.slider_drift.setMaximum(len(self.drift_frames) - 1)
        self.update_optical_flow_frame(0)

        frame = self.drift_frames[0].astype(np.uint8)
        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)
        # si escalas:
        pix = pix.scaled(self.label_preview.width(), self.label_preview.height(), Qt.KeepAspectRatio)
        self.label_preview.setPixmap(pix)


        self.status_label.setText("Optical Flow alignment completed")
        self.progress.setValue(100)

    def update_optical_flow_frame(self, idx):
        if self.drift_frames is None:
            return
        idx = max(0, min(idx, len(self.drift_frames) - 1))

        frame = self.drift_frames[idx]
        mask = self.drift_masks[idx]

        display = frame.copy()
        display[mask == 0] = 255

        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)
        # si escalas:
        pix = pix.scaled(self.label_preview.width(), self.label_preview.height(), Qt.KeepAspectRatio)
        self.label_preview.setPixmap(pix)


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
            self.status_label.setText("Run Optical Flow first")
            return

        self.status_label.setText("Aligning ECC (final pass)...")
        self.progress.setValue(0)

        frame0 = self.drift_frames[0].astype(np.uint8)
        mask0 = sample_mask_otsu(frame0)
        mask0 = clean_mask(mask0)

        mask_drift = propagate_mask(mask0, self.drift_drifts)

        ecc_frames, ecc_masks_raw, ecc_transforms = ecc_align_final(
            self.drift_frames, mask_drift
        )

        self.ecc_frames = ecc_frames
        self.ecc_masks = ecc_masks_raw

        self.slider_fine_ecc.setMaximum(len(self.ecc_frames) - 1)
        self.update_fine_ecc_frame(0)

        self.status_label.setText("ECC final pass completed")
        self.progress.setValue(100)

    def update_fine_ecc_frame(self, idx):
        if self.ecc_frames is None:
            return
        idx = max(0, min(idx, len(self.ecc_frames) - 1))

        frame = self.ecc_frames[idx]
        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)
        # si escalas:
        pix = pix.scaled(self.label_preview.width(), self.label_preview.height(), Qt.KeepAspectRatio)
        self.label_preview.setPixmap(pix)

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
