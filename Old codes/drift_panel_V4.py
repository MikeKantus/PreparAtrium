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
    Safely convert a 2D NumPy array (grayscale) to QImage.
    Use arr.strides[0] as bytes_per_line, ensure contiguity, and return a copy.
    """
    import numpy as np
    from PySide6.QtGui import QImage

    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError("numpy_to_qimage: frame must be 2D grayscale")

    # Normalize unless the array is already uint8.
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
    # Ensure uint8.
    if frame.dtype != np.uint8:
        a = frame.astype(np.float32)
        a = a - np.nanmin(a)
        rng = np.nanmax(a)
        if rng == 0 or np.isnan(rng):
            rng = 1.0
        frame_u8 = (a / rng * 255.0).astype(np.uint8)
    else:
        frame_u8 = frame

    # Blur.
    blur = cv2.GaussianBlur(frame_u8, (5, 5), 0)

    # Otsu threshold.
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
    ref: reference frame (first frame)
    img: current frame
    template_size: square template size
    """

    H, W = ref.shape
    cy, cx = H // 2, W // 2
    half = template_size // 2

    # Center the template in the reference frame.
    template = ref[cy-half:cy+half, cx-half:cx+half].astype(np.float32)

    # Normalized cross-correlation (NCC).
    res = cv2.matchTemplate(img.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)

    # Maximum correlation gives the template position.
    _, _, _, max_loc = cv2.minMaxLoc(res)
    y, x = max_loc

    # Convert the match position into drift relative to the center.
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
    """
    Calculate the smallest canvas containing all translated frames.
    This removes unused padding and keeps only what is required.
    """
    H, W = frames[0].shape

    dy = drifts[:, 0]
    dx = drifts[:, 1]

    # Minimum and maximum coordinates occupied by any frame.
    y_min = dy.min()
    y_max = dy.max()
    x_min = dx.min()
    x_max = dx.max()

    # Bounding box global
    top    = int(max(0, -y_min))
    bottom = int(max(0,  y_max))
    left   = int(max(0, -x_min))
    right  = int(max(0,  x_max))

    H_pad = H + top + bottom
    W_pad = W + left + right

    return H_pad, W_pad, top, left

def align_with_auto_canvas(frames, drifts):
    H, W = frames[0].shape
    H_pad, W_pad, top, left = compute_optimal_canvas(frames, drifts)

    aligned = []
    masks = []

    for i, f in enumerate(frames):
        dy, dx = drifts[i]

        canvas = np.zeros((H_pad, W_pad), dtype=f.dtype)
        mask = np.zeros((H_pad, W_pad), dtype=np.uint8)

        # Base position of the frame without drift.
        y0 = top
        x0 = left

        # Insert the frame.
        canvas[y0:y0+H, x0:x0+W] = f
        mask[y0:y0+H, x0:x0+W] = 1

        # The measured drift is the motion of the current frame relative to
        # the reference, so alignment applies its inverse.
        inverse_shift = (-dy, -dx)
        aligned.append(nd_shift(canvas, shift=inverse_shift, mode="constant", cval=0,
                    order=1, prefilter=False))
        masks.append(nd_shift(mask, shift=inverse_shift, mode="constant", cval=0,
                      order=0, prefilter=False).astype(np.uint8))

    return np.array(aligned), np.array(masks)


def crop_to_used_area(aligned, masks):
    """
    Crop unused padding while preserving all pixels required by any frame.
    """
    combined = np.sum(masks, axis=0)

    ys, xs = np.where(combined > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    cropped_frames = aligned[:, y_min:y_max+1, x_min:x_max+1]
    cropped_masks  = masks[:,  y_min:y_max+1, x_min:x_max+1]

    return cropped_frames, cropped_masks

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
#                   ECC FIRST (WITH PADDING)
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

        # ECC can fail for an individual frame with little overlap or contrast.
        # Keep that frame unchanged and continue processing the stack.
        frame_warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, frame_warp = cv2.findTransformECC(
                ref_canvas, canvas, frame_warp, warp_mode, criteria
            )
        except cv2.error:
            frame_warp = np.eye(2, 3, dtype=np.float32)

        aligned_img = cv2.warpAffine(
            canvas, frame_warp, (W_pad, H_pad),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        mask_canvas = np.zeros((H_pad, W_pad), dtype=np.uint8)
        mask_canvas[y0:y0+H, x0:x0+W] = mask_frames[i]

        aligned_mask = cv2.warpAffine(
            mask_canvas, frame_warp, (W_pad, H_pad),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        aligned.append(aligned_img)
        masks_out.append(aligned_mask)
        ecc_transforms.append(frame_warp.copy())

    return np.array(aligned), np.array(masks_out), ecc_transforms, H_pad, W_pad


# ============================================================
#                   ECC FINAL (WITHOUT PADDING)
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

def ecc_transforms_to_drifts(ecc_transforms):
    """
    Convert ECC matrices (2x3) into translations (dy, dx).
    """
    drifts = []
    for M in ecc_transforms:
        dy = float(M[1, 2])
        dx = float(M[0, 2])
        drifts.append([dy, dx])
    return np.array(drifts, dtype=float)

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
        aligned, masks = crop_to_used_area(aligned, masks)

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

        # Create a Matplotlib figure and render it into a PNG buffer.
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

        # Show the plot in a QLabel inside a QDialog (without plt.show()).
        label = QLabel(self)
        label.setPixmap(pix.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        layout = QVBoxLayout(self)
        layout.addWidget(label)
        self.setLayout(layout)

        # Close the Matplotlib figure to release memory.
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
        self.tm_frames = None
        self.tm_masks = None
        self.tm_drifts = None
        self.drift_frames = None
        self.drift_masks = None
        self.drift_drifts = None
        self.initial_ecc_frame = None
        self.initial_ecc_masks = None
        self.ecc_frames = None
        self.ecc_masks = None

        self.setWindowTitle("PreparAtrium – Drift and ECC Alignment")
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

        self.label_original = QLabel("Original video not loaded")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_original.setMinimumSize(300, 220)
        self.label_original.setScaledContents(False)

        self.label_initial_ecc = QLabel("Initial ECC not available")
        self.label_initial_ecc.setAlignment(Qt.AlignCenter)
        self.label_initial_ecc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_initial_ecc.setMinimumSize(300, 220)
        self.label_initial_ecc.setScaledContents(False)
        self.slider_initial_ecc.setMaximum(0)

        self.label_drift = QLabel("Template Matching not available")
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

        # A1 — ORIGINAL VIDEO PANEL
        title_A1 = QLabel("Original")
        title_A1.setAlignment(Qt.AlignCenter)
        panel_A1 = _make_panel(title_A1, self.label_original,
                            [self.btn_load, self.slider_original])
        # add trim controls below preview inside same panel
        # create a small row for trim sliders/buttons
        trim_row = QHBoxLayout()
        trim_row.setSpacing(6)
        trim_row.addWidget(QLabel("Start"))
        trim_row.addWidget(self.trim_start)
        trim_row.addWidget(QLabel("End"))
        trim_row.addWidget(self.trim_end)
        trim_row.addWidget(self.btn_trim)
        trim_row.addWidget(self.btn_restore)
        # append trim_row to panel_A1 layout
        panel_A1.layout().addLayout(trim_row)

        grid.addWidget(panel_A1, 0, 0)

        # B1 — INITIAL ECC
        title_B1 = QLabel("Initial ECC")
        title_B1.setAlignment(Qt.AlignCenter)
        panel_B1 = _make_panel(title_B1, self.label_initial_ecc,
                            [self.btn_align_initial_ecc, self.slider_initial_ecc])
        grid.addWidget(panel_B1, 0, 1)

        # C1 — TEMPLATE MATCHING
        title_C1 = QLabel("Template Matching")
        title_C1.setAlignment(Qt.AlignCenter)
        panel_C1 = _make_panel(title_C1, self.label_drift,
                            [self.btn_align_optical_flow, self.btn_drift_plot, self.slider_drift])
        grid.addWidget(panel_C1, 1, 0)

        # A2 — FINE ECC
        title_A2 = QLabel("Fine ECC")
        title_A2.setAlignment(Qt.AlignCenter)
        panel_A2 = _make_panel(title_A2, self.label_fine_ecc,
                            [self.btn_align_fine_ecc, self.btn_save_fine_ecc, self.btn_open_kymo, self.slider_fine_ecc])
        grid.addWidget(panel_A2, 1, 1)

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

        # Connect Template Matching once during initialization.
        self.btn_align_optical_flow.clicked.connect(self.align_template_matching)

        # Slider -> Template Matching viewer.
        self.slider_drift.valueChanged.connect(self.update_template_matching_frame)

        # Drift plot (uses tm_drifts).
        self.btn_drift_plot.clicked.connect(lambda: PlotWindow(self.tm_drifts, parent=self).exec())

        # Connect the remaining pipeline controls.
        self.btn_load.clicked.connect(self.load_video)
        self.slider_original.valueChanged.connect(self.update_original_frame)

        self.btn_trim.clicked.connect(self.apply_trim)
        self.btn_restore.clicked.connect(self.restore_original)

        self.btn_align_initial_ecc.clicked.connect(self.align_initial_ecc)
        self.slider_initial_ecc.valueChanged.connect(self.update_initial_ecc_frame)

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
    #                   VIDEO LOADING and DISPLAY
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

        # Convert safely to QImage using frame_to_qimage_safe when available.
        try:
            # Use the local helper only as a fallback.
            from core.ui_utils import frame_to_qimage_safe
            qimg = frame_to_qimage_safe(frame)
        except Exception:
            # Simple fallback: ensure uint8 and contiguous storage.
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

        # Usar label_original (no label_preview)
        if hasattr(self, "label_original") and self.label_original is not None:
            target_w = max(1, self.label_original.width())
            target_h = max(1, self.label_original.height())
            self.label_original.setPixmap(pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # Fallback if label_original does not exist.
            if hasattr(self, "label_drift") and self.label_drift is not None:
                target_w = max(1, self.label_drift.width())
                target_h = max(1, self.label_drift.height())
                self.label_drift.setPixmap(pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))



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
    #                  UPDATE PROGRESS
    # ============================================================
    def update_progress(self, pct, remaining):
            self.progress.setValue(pct)
            self.status_label.setText(
                f"{pct}% completed — Estimated remaining time: {remaining:.1f} s"
            )

    # ============================================================
    #                   ECC FIRST PASS
    # ============================================================

    def align_initial_ecc(self):
        if self.frames is None:
            self.status_label.setText("Load a video first")
            return

        self.status_label.setText("Aligning ECC (first pass)...")
        self.progress.setValue(0)

        # 1. Mask from first ORIGINAL frame
        mask0 = sample_mask_otsu(self.frames[0])
        mask0 = clean_mask(mask0)

        # 2. No drift yet → drift is zero
        zero_drifts = np.zeros((len(self.frames), 2), dtype=float)

        # 3. ECC alignment with mask (MoviTrack pattern)
        ecc_frames, ecc_masks_raw, ecc_transforms, H_pad, W_pad = ecc_align_first(
            self.frames, np.stack([mask0]*len(self.frames), axis=0)
        )

        # 4. Propagate mask through ECC transforms
        mask_ecc = propagate_mask(
            mask0,
            zero_drifts,
            ecc_transforms,
            H_pad=H_pad,
            W_pad=W_pad
        )

        # 5. Auto‑crop (MoviTrack pattern)
        mask_union = np.max(mask_ecc, axis=0)
        ys, xs = np.where(mask_union > 0)

        if len(ys) > 0 and len(xs) > 0:
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            ecc_frames = ecc_frames[:, ymin:ymax+1, xmin:xmax+1]
            mask_ecc   = mask_ecc[:, ymin:ymax+1, xmin:xmax+1]

        # 6. Save results
        self.initial_ecc_frame = ecc_frames
        self.initial_ecc_masks = mask_ecc

        self.slider_initial_ecc.setMaximum(len(self.initial_ecc_frame) - 1)
        self.update_initial_ecc_frame(0)

        self.status_label.setText("Initial ECC completed")
        self.progress.setValue(100)


    # This function must remain at class indentation.
    def update_initial_ecc_frame(self, idx):
        if self.initial_ecc_frame is None:
            return
        idx = max(0, min(idx, len(self.initial_ecc_frame) - 1))
        frame = self.initial_ecc_frame[idx]

        try:
            qimg = frame_to_qimage_safe(frame)
        except Exception:
            qimg = numpy_to_qimage(frame)

        pix = QPixmap.fromImage(qimg)
        if hasattr(self, "label_initial_ecc") and self.label_initial_ecc is not None:
            target_w = max(1, self.label_initial_ecc.width())
            target_h = max(1, self.label_initial_ecc.height())
            self.label_initial_ecc.setPixmap(
                pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    # ============================================================
    #                  TEMPLATE MATCHING
    # ============================================================

    def align_template_matching(self):
        """
        Correct drift by Template Matching on self.initial_ecc_frame.
        Store results in self.tm_frames, self.tm_masks, and self.tm_drifts,
        then update the panel (which now uses Template Matching).
        """
        if self.initial_ecc_frame is None:
            self.status_label.setText("Run Initial ECC first")
            return

        self.status_label.setText("Aligning drift (Template Matching)...")
        self.progress.setValue(0)

        frames = self.initial_ecc_frame
        ref = frames[0].astype(np.float32)
        H, W = ref.shape

        # Center the template and limit it to the frame size.
        template_size = min(64, H // 2, W // 2)
        if template_size < 8:
            self.status_label.setText("Frame too small for Template Matching")
            return

        half = template_size // 2
        cy, cx = H // 2, W // 2
        y0 = cy - half
        y1 = cy + half
        x0 = cx - half
        x1 = cx + half

        template = ref[y0:y1, x0:x1]

        drifts = []
        max_dy = max(1, H // 4)
        max_dx = max(1, W // 4)
        for f in frames:
            img = f.astype(np.float32)
            if float(np.std(template)) < 1e-6:
                drifts.append([0.0, 0.0])
                continue

            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            # Ignore matches outside a bounded neighborhood of the reference
            # position; black padding otherwise produces false matches.
            valid = np.zeros_like(res, dtype=bool)
            y_min = max(0, y0 - max_dy)
            y_max = min(res.shape[0] - 1, y0 + max_dy)
            x_min = max(0, x0 - max_dx)
            x_max = min(res.shape[1] - 1, x0 + max_dx)
            valid[y_min:y_max + 1, x_min:x_max + 1] = True
            res[~valid] = -np.inf
            _, _, _, max_loc = cv2.minMaxLoc(res)
            y, x = max_loc
            dy = float(y - y0)
            dx = float(x - x0)
            drifts.append([dy, dx])

        drifts = np.array(drifts, dtype=float)

        # Align on an automatic canvas and crop unused padding.
        aligned, masks = align_with_auto_canvas(frames, drifts)
        aligned, masks = crop_to_used_area(aligned, masks)

        self.tm_frames = aligned
        self.tm_masks = masks
        self.tm_drifts = drifts

        # Update the slider and show the first frame.
        self.slider_drift.setMaximum(len(self.tm_frames) - 1)
        self.update_template_matching_frame(0)

        self.status_label.setText("Template Matching drift completed")
        self.progress.setValue(100)

    def update_template_matching_frame(self, idx=None):
        """
        Show frame idx from self.tm_frames in self.label_drift,
        using self.tm_masks to mark areas outside the field of view.
        """
        if self.tm_frames is None:
            return

        if idx is None:
            try:
                idx = int(self.slider_drift.value())
            except Exception:
                idx = 0

        idx = max(0, min(idx, len(self.tm_frames) - 1))
        frame = self.tm_frames[idx]

        mask = None
        if self.tm_masks is not None and idx < len(self.tm_masks):
            mask = self.tm_masks[idx]

        display = np.asarray(frame).copy()
        if mask is not None:
            try:
                display[mask == 0] = 255
            except Exception:
                pass

        try:
            qimg = frame_to_qimage_safe(display)
        except Exception:
            qimg = numpy_to_qimage(display)

        pix = QPixmap.fromImage(qimg)

        target_w = max(1, self.label_drift.width())
        target_h = max(1, self.label_drift.height())
        self.label_drift.setPixmap(
            pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


    # ============================================================
    #                   OPTICAL FLOW
    # ============================================================


    def align_optical_flow(self):
        if self.initial_ecc_frame is None:
            self.status_label.setText("Run ECC first")
            return

        self.status_label.setText("Aligning drift (Template Matching)...")
        self.progress.setValue(0)

        drifts = compute_raw_drift(self.initial_ecc_frame)
        aligned, masks = align_with_auto_canvas(self.initial_ecc_frame, drifts)
        aligned, masks = crop_to_used_area(aligned, masks)

        self.drift_frames = aligned
        self.drift_masks  = masks
        self.drift_drifts = drifts

        self.slider_drift.setMaximum(len(self.drift_frames) - 1)
        self.update_optical_flow_frame(0)

        self.status_label.setText("Template Matching drift completed")
        self.progress.setValue(100)

    def finish_optical_flow(self, aligned, masks, drifts):
        """
        Handle completion of the drift worker.
        Store results, crop using the mask, and update the UI safely.
        """
            # Store results.
        self.drift_drifts = drifts
        self.drift_frames = aligned
        self.drift_masks = masks

        # Exit if no frames were produced.
        if self.drift_frames is None or len(self.drift_frames) == 0:
            self.status_label.setText("Optical Flow produced no frames")
            self.progress.setValue(100)
            return

        # Union masks and crop once.
        try:
            mask_union = np.max(self.drift_masks, axis=0)
            ys, xs = np.where(mask_union > 0)
            if len(ys) > 0 and len(xs) > 0:
                ymin, ymax = ys.min(), ys.max()
                xmin, xmax = xs.min(), xs.max()
                self.drift_frames = self.drift_frames[:, ymin:ymax+1, xmin:xmax+1]
                self.drift_masks = self.drift_masks[:, ymin:ymax+1, xmin:xmax+1]
        except Exception:
            # Continue without cropping if mask processing fails.
            pass

        # Normalize to uint8 and ensure contiguous storage.
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
            # Leave frames unchanged and rely on the display fallback.
            pass

        # Update the slider and show the first frame.
        n = len(self.drift_frames)
        self.slider_drift.setMaximum(max(0, n - 1))
        self.slider_drift.setEnabled(n > 1)

        # Force an update of frame 0 using the robust display function.
        try:
            self.update_optical_flow_frame(0)
        except Exception:
            # Fallback: display the first frame manually.
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

        self.status_label.setText("Optical Flow alignment completed")
        self.progress.setValue(100)


    def update_optical_flow_frame(self, idx=None):
        """
        Safely display frame idx from drift_frames.
        If idx is None, read the value from slider_drift.
        """
        import numpy as np
        from PySide6.QtGui import QImage

        # Determine the frame index.
        if idx is None:
            if hasattr(self, "slider_drift"):
                try:
                    idx = int(self.slider_drift.value())
                except Exception:
                    idx = 0
            else:
                idx = 0

        # Select the available frame array safely.
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

        # Get a mask if one exists.
        mask = None
        if hasattr(self, "drift_masks") and self.drift_masks is not None:
            try:
                if idx < len(self.drift_masks):
                    mask = self.drift_masks[idx]
            except Exception:
                mask = None

        # Prepare a display copy without modifying the source frame.
        display = np.asarray(frame).copy()
        if mask is not None:
            try:
                display[mask == 0] = 255
            except Exception:
                pass

        # Convert safely to QImage.
        try:
            qimg = frame_to_qimage_safe(display)
        except Exception:
            arr = np.asarray(display)
            # Convert color images to grayscale.
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                try:
                    import cv2
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                except Exception:
                    arr = arr[..., 0]

            # Handle NaNs.
            if np.isnan(arr).any():
                arr = arr.copy()
                arr[np.isnan(arr)] = np.nanmin(arr)

            # Normalize to uint8.
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

        # Create a pixmap and assign it to the corresponding label.
        pix = QPixmap.fromImage(qimg)
        lbl = getattr(self, "label_drift", None) or getattr(self, "label_original", None) or getattr(self, "label_frame", None)
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
        if self.tm_frames is None or self.tm_drifts is None:
            self.status_label.setText("Run Template Matching first")
            return

        self.status_label.setText("Aligning fine ECC...")
        self.progress.setValue(0)

        # Fine ECC must consume the stack produced by Template Matching.
        # Its masks are already in the same canvas and coordinate system.
        frames = self.tm_frames
        masks = self.tm_masks
        if masks is None or len(masks) != len(frames):
            masks = np.ones_like(frames, dtype=np.uint8)

        ecc_frames, ecc_masks_raw, ecc_transforms = ecc_align_final(
            frames, masks
        )

        mask_ecc = ecc_masks_raw

        mask_union = np.max(mask_ecc, axis=0)
        ys, xs = np.where(mask_union > 0)

        if len(ys) > 0 and len(xs) > 0:
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            ecc_frames = ecc_frames[:, ymin:ymax+1, xmin:xmax+1]
            mask_ecc   = mask_ecc[:, ymin:ymax+1, xmin:xmax+1]

        self.ecc_frames = ecc_frames
        self.ecc_masks  = mask_ecc

        self.slider_fine_ecc.setMaximum(len(self.ecc_frames) - 1)
        self.update_fine_ecc_frame(0)

        self.status_label.setText("Fine ECC completed")
        self.progress.setValue(100)


    def update_fine_ecc_frame(self, idx=None):
        """
        Safely display frame idx from ecc_frames.
        Accept idx=None (read from the slider) or idx=int.
        """
        # 1) Determine idx when it was not supplied.
        if idx is None:
            if hasattr(self, "slider_fine_ecc"):
                try:
                    idx = int(self.slider_fine_ecc.value())
                except Exception:
                    idx = 0
            else:
                idx = 0

        # 2) Select frames safely (do not evaluate NumPy arrays as booleans).
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
        # 4) Normalize the index and get the frame.
        idx = max(0, min(int(idx), len(frames) - 1))
        frame = frames[idx]
        if frame is None:
            return

        # 5) Convert to QImage (use the helper when available).
        try:
            qimg = frame_to_qimage_safe(frame)
        except Exception:
            import numpy as np
            from PySide6.QtGui import QImage
            arr = np.asarray(frame)

            # Convert color images to grayscale.
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                try:
                    import cv2
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                except Exception:
                    arr = arr[..., 0]

            # Handle NaNs.
            if np.isnan(arr).any():
                arr = arr.copy()
                arr[np.isnan(arr)] = np.nanmin(arr)

            # Normalize to uint8.
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

        # 6) Create a pixmap and assign it to the corresponding label.
        pix = QPixmap.fromImage(qimg)
        lbl = getattr(self, "label_fine_ecc", None) or getattr(self, "label_original", None) or getattr(self, "label_frame", None)

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