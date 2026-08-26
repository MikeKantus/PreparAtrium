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
from PySide6.QtCore import Qt

from scipy.ndimage import shift as nd_shift

from core.ui_utils import frame_to_qimage_safe
from core.drift_tools import (
    sample_mask_otsu,
    clean_mask,
    propagate_mask,
    ecc_align_first_global,
    ecc_align_final,
    align_with_auto_canvas,
    crop_to_used_area,
)
from core.drift_pipeline import DriftPipeline


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
        self.current_stack = None
        self.processed_stack = None
        self.processed_masks = None
        self.processed_drifts = None
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
        self.btn_align_initial_ecc_seq = QPushButton("ECC (sequential)")
        self.btn_align_tm_seq = QPushButton("TM (sequential)")
        self.btn_align_tm = QPushButton("Template Matching")
        self.btn_drift_plot = QPushButton("Show drift (Template Matching drift plot)")
        self.btn_align_fine_ecc = QPushButton("Fine ECC alignment")
        self.btn_accept_preview = QPushButton("Accept preview")
        self.btn_discard_preview = QPushButton("Discard preview")
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

        self.label_current = QLabel("Current stack not available")
        self.label_current.setAlignment(Qt.AlignCenter)
        self.label_current.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_current.setMinimumSize(300, 220)
        self.label_current.setScaledContents(False)
        self.slider_current = QSlider(Qt.Horizontal)
        self.slider_current.setMaximum(0)

        self.label_processed = QLabel("Processed stack not available")
        self.label_processed.setAlignment(Qt.AlignCenter)
        self.label_processed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_processed.setMinimumSize(300, 220)
        self.label_processed.setScaledContents(False)
        self.slider_processed = QSlider(Qt.Horizontal)
        self.slider_processed.setMaximum(0)

        # Compatibility aliases for code that still refers to stage-specific viewers.
        self.label_initial_ecc = self.label_processed
        self.label_drift = self.label_processed
        self.label_fine_ecc = self.label_processed
        self.slider_initial_ecc = self.slider_processed
        self.slider_drift = self.slider_processed
        self.slider_fine_ecc = self.slider_processed

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

        # B1 — CURRENT STACK
        title_B1 = QLabel("Current stack")
        title_B1.setAlignment(Qt.AlignCenter)
        panel_B1 = _make_panel(title_B1, self.label_current, [self.slider_current])
        grid.addWidget(panel_B1, 0, 1)

        # C1 — PROCESSING CONTROLS
        title_C1 = QLabel("Processing controls")
        title_C1.setAlignment(Qt.AlignCenter)
        processing_buttons = [
            self.btn_align_initial_ecc,
            self.btn_align_initial_ecc_seq,
            self.btn_align_tm,
            self.btn_align_tm_seq,
            self.btn_drift_plot,
            self.btn_align_fine_ecc,
            self.btn_accept_preview,
            self.btn_discard_preview,
            self.btn_save_fine_ecc,
            self.btn_open_kymo,
        ]
        panel_C1 = QWidget()
        controls_layout = QVBoxLayout(panel_C1)
        controls_layout.setContentsMargins(6, 6, 6, 6)
        controls_layout.setSpacing(6)
        controls_layout.addWidget(title_C1)
        controls_grid = QGridLayout()
        controls_grid.setSpacing(6)
        for index, button in enumerate(processing_buttons):
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            controls_grid.addWidget(button, index // 2, index % 2)
        controls_layout.addLayout(controls_grid)
        controls_layout.addStretch()
        grid.addWidget(panel_C1, 1, 0)

        # A2 — PROCESSED STACK
        title_A2 = QLabel("Processed stack preview")
        title_A2.setAlignment(Qt.AlignCenter)
        panel_A2 = _make_panel(title_A2, self.label_processed, [self.slider_processed])
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
        self.btn_align_initial_ecc_seq.clicked.connect(self.align_initial_ecc_sequential)
        self.btn_align_tm_seq.clicked.connect(self.align_template_matching_sequential)

        # Connect Template Matching once during initialization.
        self.btn_align_tm.clicked.connect(self.align_template_matching)

        # Slider -> Template Matching viewer.
        self.slider_current.valueChanged.connect(self.update_current_frame)
        self.slider_processed.valueChanged.connect(self.update_processed_frame)

        # Drift plot (uses the latest Template Matching drift).
        self.btn_drift_plot.clicked.connect(self.show_drift)

        # Connect the remaining pipeline controls.
        self.btn_load.clicked.connect(self.load_video)
        self.slider_original.valueChanged.connect(self.update_original_frame)

        self.btn_trim.clicked.connect(self.apply_trim)
        self.btn_restore.clicked.connect(self.restore_original)

        self.btn_align_initial_ecc.clicked.connect(self.align_initial_ecc)
        self.btn_align_fine_ecc.clicked.connect(self.align_fine_ecc)
        self.btn_accept_preview.clicked.connect(self.accept_preview)
        self.btn_discard_preview.clicked.connect(self.discard_preview)
        self.btn_save_fine_ecc.clicked.connect(self.save_fine_aligned_video)
        self.btn_open_kymo.clicked.connect(self.open_kymo_panel)


        # ============================================================
        #                   LOAD STACK IF PROVIDED
        # ============================================================
        if self.stack is not None:
            self.frames = self.stack.copy()
            self.original_frames = self.frames.copy()
            self.current_stack = self.frames.copy()

            self.slider_original.setMaximum(len(self.frames) - 1)
            self.slider_current.setMaximum(len(self.current_stack) - 1)
            self.trim_start.setMaximum(len(self.frames) - 1)
            self.trim_end.setMaximum(len(self.frames) - 1)
            self.trim_end.setValue(len(self.frames) - 1)

            self.update_original_frame(0)
            self.update_current_frame(0)
            self.status_label.setText(f"Video loaded from AFMLoader: {len(self.frames)} frames")

    # ============================================================
    #                   VIDEO LOADING and DISPLAY
    # ============================================================

    def open_kymo_panel(self):
        if self.current_stack is None:
            self.status_label.setText("No current stack available")
            return

        from gui.kymo_panel import KymoPanel
        self.kymo_window = KymoPanel(stack=self.current_stack, meta=self.meta)
        self.kymo_window.show()


    def load_video(self):
        path = QFileDialog.getOpenFileName(
            self, "Select video", "", "AVI Files (*.avi)"
        )[0]
        if not path:
            return

        self.frames = read_avi_frames(path)
        self.original_frames = self.frames.copy()
        self.current_stack = self.frames.copy()
        self.processed_stack = None

        self.slider_original.setMaximum(len(self.frames) - 1)
        self.slider_current.setMaximum(len(self.current_stack) - 1)
        self.slider_processed.setMaximum(0)

        self.trim_start.setMaximum(len(self.frames) - 1)
        self.trim_end.setMaximum(len(self.frames) - 1)
        self.trim_end.setValue(len(self.frames) - 1)

        self.update_original_frame(0)
        self.update_current_frame(0)
        self.label_processed.setText("Processed stack not available")
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

    def _display_stack_frame(self, stack, idx, label, mask_stack=None):
        if stack is None or len(stack) == 0:
            return
        idx = max(0, min(int(idx), len(stack) - 1))
        display = np.asarray(stack[idx]).copy()
        if mask_stack is not None and idx < len(mask_stack):
            display[np.asarray(mask_stack[idx]) == 0] = 255
        try:
            qimg = frame_to_qimage_safe(display)
        except Exception:
            qimg = numpy_to_qimage(display)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(max(1, label.width()), max(1, label.height()),
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

    def update_current_frame(self, idx=None):
        if idx is None:
            idx = self.slider_current.value()
        self._display_stack_frame(self.current_stack, idx, self.label_current)

    def update_processed_frame(self, idx=None):
        if idx is None:
            idx = self.slider_processed.value()
        self._display_stack_frame(
            self.processed_stack, idx, self.label_processed, self.processed_masks
        )

    def _set_processed_stack(self, stack, masks=None, drifts=None):
        self.processed_stack = np.asarray(stack).copy()
        self.processed_masks = masks
        self.processed_drifts = drifts
        self.slider_processed.setMaximum(max(0, len(self.processed_stack) - 1))
        self.slider_processed.setValue(0)
        self.update_processed_frame(0)

    def accept_preview(self):
        if self.processed_stack is None:
            self.status_label.setText("No processed preview to accept")
            return
        self.current_stack = self.processed_stack.copy()
        self.slider_current.setMaximum(len(self.current_stack) - 1)
        self.slider_current.setValue(0)
        self.update_current_frame(0)
        self.processed_stack = None
        self.processed_masks = None
        self.processed_drifts = None
        self.slider_processed.setMaximum(0)
        self.label_processed.clear()
        self.label_processed.setText("Processed stack not available")
        self.status_label.setText("Preview accepted as current stack")

    def discard_preview(self):
        if self.processed_drifts is not None:
            self.tm_drifts = None
        self.processed_stack = None
        self.processed_masks = None
        self.processed_drifts = None
        self.slider_processed.setMaximum(0)
        self.label_processed.clear()
        self.label_processed.setText("Processed stack not available")
        self.status_label.setText("Preview discarded")



    # ============================================================
    #                   TRIM
    # ============================================================

    def apply_trim(self):
        if self.frames is None:
            return
        s = self.trim_start.value()
        e = self.trim_end.value()
        self.frames = self.frames[s:e+1]
        self.current_stack = self.frames.copy()
        self.processed_stack = None
        self.processed_masks = None
        self.processed_drifts = None
        self.tm_drifts = None
        self.slider_original.setMaximum(len(self.frames) - 1)
        self.slider_current.setMaximum(len(self.current_stack) - 1)
        self.slider_processed.setMaximum(0)
        self.update_original_frame(0)
        self.update_current_frame(0)
        self.label_processed.setText("Processed stack not available")
        self.status_label.setText(f"Trim applied: {s} → {e}")

    def restore_original(self):
        if self.original_frames is None:
            return
        self.frames = self.original_frames.copy()
        self.current_stack = self.frames.copy()
        self.processed_stack = None
        self.processed_masks = None
        self.processed_drifts = None
        self.tm_drifts = None
        self.slider_original.setMaximum(len(self.frames) - 1)
        self.slider_current.setMaximum(len(self.current_stack) - 1)
        self.slider_processed.setMaximum(0)
        self.trim_start.setMaximum(len(self.frames) - 1)
        self.trim_end.setMaximum(len(self.frames) - 1)
        self.trim_end.setValue(len(self.frames) - 1)
        self.update_original_frame(0)
        self.update_current_frame(0)
        self.label_processed.setText("Processed stack not available")
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
        if self.current_stack is None:
            self.status_label.setText("Load a video first")
            return

        self.status_label.setText("Aligning ECC (first pass)...")
        self.progress.setValue(0)
        self.tm_drifts = None

        # 1. Mask from first ORIGINAL frame
        mask0 = sample_mask_otsu(self.current_stack[0])
        mask0 = clean_mask(mask0)

        # 2. No drift yet → drift is zero
        zero_drifts = np.zeros((len(self.current_stack), 2), dtype=float)

        # 3. ECC alignment with mask (MoviTrack pattern)
        ecc_frames, ecc_masks_raw, ecc_transforms, H_pad, W_pad = ecc_align_first_global(
            self.current_stack, np.stack([mask0]*len(self.current_stack), axis=0)
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
        self._set_processed_stack(ecc_frames, mask_ecc)

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
    def align_initial_ecc_sequential(self):
        if self.current_stack is None:
            self.status_label.setText("Load a video first")
            return

        from core.drift_pipeline import DriftPipeline
        self.tm_drifts = None
        pipeline = DriftPipeline(self.current_stack)

        ecc_frames, ecc_masks, transforms, H_pad, W_pad = pipeline.run_ecc1_sequential()

        self.initial_ecc_frame = ecc_frames
        self.initial_ecc_masks = ecc_masks
        self._set_processed_stack(ecc_frames, ecc_masks)

        self.status_label.setText("Sequential ECC completed")

    # ============================================================
    #                  TEMPLATE MATCHING
    # ============================================================

    def align_template_matching(self):
        if self.current_stack is None:
            self.status_label.setText("Load a video first")
            return

        self.status_label.setText("Aligning drift (Template Matching)...")
        self.progress.setValue(0)

        from core.drift_pipeline import DriftPipeline
        pipeline = DriftPipeline(self.current_stack)

        aligned, masks, drifts, conf = pipeline.run_tm_global(self.current_stack)

        # Save results
        self.tm_frames = aligned
        self.tm_masks = masks
        self.tm_drifts = drifts
        self.tm_confidence = conf

        # Smooth drift
        drifts_smooth, segments = DriftPipeline.process_tm_drifts(drifts, conf)
        self.tm_drifts = drifts_smooth
        self.tm_segments = segments

        self._set_processed_stack(aligned, masks, drifts)

        self.status_label.setText("Template Matching completed")
        self.progress.setValue(100)

    def align_template_matching_sequential(self):
        if self.current_stack is None:
            self.status_label.setText("Load a video first")
            return

        from core.drift_pipeline import DriftPipeline
        pipeline = DriftPipeline(self.current_stack)

        aligned, masks, drifts, conf = pipeline.run_tm_sequential(self.current_stack)

        self.tm_frames = aligned
        self.tm_masks = masks
        self.tm_drifts = drifts
        self.tm_confidence = conf

        self._set_processed_stack(aligned, masks, drifts)

        self.status_label.setText("Sequential TM completed")

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


    # def align_optical_flow(self):
    #     if self.initial_ecc_frame is None:
    #         self.status_label.setText("Run ECC first")
    #         return

    #     self.status_label.setText("Aligning drift (Optical Flow)...")
    #     self.progress.setValue(0)

    #     from core.drift_pipeline import DriftPipeline
    #     pipeline = DriftPipeline(self.initial_ecc_frame)

    #     aligned, masks, drifts, conf = pipeline.run_tm_global(self.initial_ecc_frame)

    #     self.drift_frames = aligned
    #     self.drift_masks = masks
    #     self.drift_drifts = drifts

    #     self.slider_drift.setMaximum(len(self.drift_frames) - 1)
    #     self.update_optical_flow_frame(0)

    #     self.status_label.setText("Optical Flow drift completed")
    #     self.progress.setValue(100)


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
        if self.current_stack is None:
            self.status_label.setText("Load a video first")
            return
        if self.tm_drifts is None:
            self.status_label.setText("Run Template Matching first")
            return

        self.status_label.setText("Aligning fine ECC...")
        self.progress.setValue(0)

        # Fine ECC must consume the stack produced by Template Matching.
        # Its masks are already in the same canvas and coordinate system.
        frames = self.current_stack
        masks = self.tm_masks if self.tm_masks is not None else None
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
        self._set_processed_stack(ecc_frames, mask_ecc)

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
        if self.current_stack is None:
            self.status_label.setText("No current stack to save")
            return

        path = QFileDialog.getSaveFileName(
            self, "Save aligned video", "", "AVI Files (*.avi)"
        )[0]
        if not path:
            return

        H, W = self.current_stack[0].shape
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), 20, (W, H), False)

        for f in self.current_stack:
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