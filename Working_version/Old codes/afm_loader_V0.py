# gui/afm_loader.py
# (Este archivo es la versión corregida: histograma en panel aparte y sin "Open Drift Panel")

import os
import io
import json
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QListWidget,
    QFileDialog, QComboBox, QSlider, QProgressBar, QApplication,
    QSpinBox, QSizePolicy, QCheckBox, QListWidgetItem, QFrame, QGridLayout
)
from PySide6.QtGui import QPixmap, QImage, QIcon, QFont
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from gui.metadata_reader import read_metadata

# Optional libs
try:
    from playnano import AFMImage
    HAS_PLAYNANO = True
except Exception:
    HAS_PLAYNANO = False

try:
    import pnanolocz
    from pnanolocz import leveling, flattening
    HAS_PNANOLOCZ = True
except Exception:
    HAS_PNANOLOCZ = False


def numpy_to_qimage(frame):
    if frame is None:
        return QImage()
    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError("Frame must be 2D")
    if arr.dtype != np.uint8:
        f = arr.astype(np.float32)
        f = f - np.nanmin(f)
        rng = np.nanmax(f)
        if rng == 0:
            rng = 1.0
        f = (f / rng * 255.0).astype(np.uint8)
        arr = f
    h, w = arr.shape
    bytes_per_line = w
    return QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8)


class ProcessingThread(QThread):
    finished = Signal(np.ndarray)

    def __init__(self, stack, level_method, flat_method, lo_pct, hi_pct):
        super().__init__()
        self.stack = stack.astype(np.float32)
        self.level_method = level_method
        self.flat_method = flat_method
        self.lo_pct = lo_pct
        self.hi_pct = hi_pct

    def run(self):
        stack = self.stack.copy()
        try:
            lo = np.percentile(stack, self.lo_pct)
            hi = np.percentile(stack, self.hi_pct)
            if hi <= lo:
                hi = lo + 1e-6
            stack = np.clip(stack, lo, hi)
        except Exception:
            pass

        if self.level_method != "None" and HAS_PNANOLOCZ:
            try:
                if self.level_method == "Plane":
                    leveled = [leveling.plane_level(f) for f in stack]
                    stack = np.stack(leveled, axis=0)
                elif self.level_method == "Line":
                    leveled = [leveling.line_level(f) for f in stack]
                    stack = np.stack(leveled, axis=0)
            except Exception:
                pass

        if self.flat_method != "None" and HAS_PNANOLOCZ:
            try:
                if self.flat_method == "Histogram":
                    stack = flattening.flatten_histogram(stack)
                elif self.flat_method == "Polynomial":
                    stack = flattening.flatten_polynomial(stack)
            except Exception:
                pass

        self.finished.emit(stack)


class AFMLoaderWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(480)
        self.raw_stack = None
        self.processed_stack = None
        self.meta = {}
        self.current_frame = 0

        # fonts & sizes
        self.btn_font = QFont()
        self.btn_font.setPointSize(10)

        # Top controls
        self.btn_open = QPushButton("Open folder / files")
        self.btn_open.setIcon(QIcon.fromTheme("folder-open"))
        self.btn_open.setMinimumHeight(36)
        self.btn_open.setFont(self.btn_font)
        self.btn_open.clicked.connect(self.open_folder_or_files)

        # File list with thumbnails
        self.list_files = QListWidget()
        self.list_files.setIconSize(QSize(96, 64))
        self.list_files.setSelectionMode(QListWidget.ExtendedSelection)
        # connect to selection handler (method implemented below)
        self.list_files.itemSelectionChanged.connect(self.on_list_selection_changed)

        # Preview
        self.label_preview = QLabel("Preview")
        self.label_preview.setAlignment(Qt.AlignCenter)
        self.label_preview.setScaledContents(False)            # no escalar el contenido para forzar cambio de tamaño del label
        self.label_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_preview.setMinimumSize(640, 480)
        self.label_preview.setMaximumSize(1280, 960)
        self.label_preview.setFrameShape(QFrame.StyledPanel)

        # Metadata frame (panel)
        self.meta_frame = QFrame()
        self.meta_frame.setFrameShape(QFrame.StyledPanel)

        # Leveling / flatten
        self.combo_level = QComboBox()
        self.combo_level.addItems(["None", "Plane", "Line"])
        self.combo_flatten = QComboBox()
        self.combo_flatten.addItems(["None", "Histogram", "Polynomial"])

        # Histogram sliders (moved under preview for preview control)
        self.slider_lower = QSlider(Qt.Horizontal)
        self.slider_lower.setRange(0, 100)
        self.slider_lower.setValue(0)
        self.slider_upper = QSlider(Qt.Horizontal)
        self.slider_upper.setRange(0, 100)
        self.slider_upper.setValue(100)
        self.slider_lower.valueChanged.connect(self.on_histogram_slider_changed)
        self.slider_upper.valueChanged.connect(self.on_histogram_slider_changed)

        # Overlay options
        self.checkbox_overlay = QCheckBox("Overlay timestamp (s)")
        self.checkbox_overlay.setChecked(False)
        self.checkbox_overlay_frame = QCheckBox("Overlay frame #")
        self.checkbox_overlay_frame.setChecked(False)

        # Metadata display (summary labels; detailed grid created below)
        self.meta_fps = QLabel("FPS: -")
        self.meta_size = QLabel("Size (nm): -")
        self.meta_pixels = QLabel("Pixels: -")

        # Playback controls
        self.btn_play = QPushButton("Play")
        self.btn_play.setMinimumHeight(36)
        self.btn_play.setFont(self.btn_font)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setMinimumHeight(36)
        self.btn_pause.setFont(self.btn_font)
        self.btn_prev = QPushButton("Prev")
        self.btn_prev.setMinimumHeight(36)
        self.btn_prev.setFont(self.btn_font)
        self.btn_next = QPushButton("Next")
        self.btn_next.setMinimumHeight(36)
        self.btn_next.setFont(self.btn_font)

        self.spin_frame = QSpinBox()
        self.spin_frame.setMinimum(0)
        self.spin_frame.valueChanged.connect(self.on_spin_frame_changed)

        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setMinimum(0)
        self.slider_time.valueChanged.connect(self.on_slider_time_changed)

        self.btn_play.clicked.connect(self.start_play)
        self.btn_pause.clicked.connect(self.stop_play)
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next.clicked.connect(self.next_frame)

        # Apply / send / save
        self.btn_apply = QPushButton("Apply filters")
        self.btn_apply.setMinimumHeight(36)
        self.btn_apply.setFont(self.btn_font)
        self.btn_apply.clicked.connect(self.apply_filters)

        self.btn_send = QPushButton("Send to Drift")
        self.btn_send.setMinimumHeight(36)
        self.btn_send.setFont(self.btn_font)
        self.btn_send.clicked.connect(self.send_to_drift)

        self.btn_save = QPushButton("Save metadata + video")
        self.btn_save.setMinimumHeight(36)
        self.btn_save.setFont(self.btn_font)
        self.btn_save.clicked.connect(self.save_metadata_and_video)

        # Status / progress
        self.status_label = QLabel("Status: ready")
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        # Layout assembly
        left_col = QVBoxLayout()
        left_col.addWidget(self.btn_open)
        left_col.addWidget(QLabel("Files in folder / selection"))
        left_col.addWidget(self.list_files)
        left_col.addStretch()
        left_col.addWidget(QLabel("Leveling"))
        left_col.addWidget(self.combo_level)
        left_col.addWidget(QLabel("Flatten"))
        left_col.addWidget(self.combo_flatten)
        left_col.addWidget(self.btn_apply)

        center_col = QVBoxLayout()
        center_col.addWidget(self.label_preview)

        # histogram sliders under preview (controls preview clipping only)
        hist_controls = QVBoxLayout()
        hist_controls.addWidget(QLabel("Histogram lower %"))
        hist_controls.addWidget(self.slider_lower)
        hist_controls.addWidget(QLabel("Histogram upper %"))
        hist_controls.addWidget(self.slider_upper)
        hist_controls.addWidget(self.checkbox_overlay)
        hist_controls.addWidget(self.checkbox_overlay_frame)
        center_col.addLayout(hist_controls)

        play_row = QHBoxLayout()
        play_row.addWidget(self.btn_prev)
        play_row.addWidget(self.btn_play)
        play_row.addWidget(self.btn_pause)
        play_row.addWidget(self.btn_next)
        play_row.addWidget(QLabel("Frame:"))
        play_row.addWidget(self.spin_frame)
        center_col.addLayout(play_row)
        center_col.addWidget(self.slider_time)
        center_col.addWidget(self.progress)
        center_col.addWidget(self.status_label)

        right_col = QVBoxLayout()
        right_col.addStretch()
        right_col.addWidget(self.meta_fps)
        right_col.addWidget(self.meta_size)
        right_col.addWidget(self.meta_pixels)
        right_col.addStretch()
        right_col.addWidget(self.btn_send)
        right_col.addWidget(self.btn_save)

        # --- Metadata panel (grid) ---
        # Import QGridLayout at top of file if not already imported:
        # from PySide6.QtWidgets import QGridLayout
        meta_layout = QGridLayout(self.meta_frame)

        # Header
        meta_layout.addWidget(QLabel("<b>Parameter</b>"), 0, 0)
        meta_layout.addWidget(QLabel("<b>Value</b>"), 0, 1)

        # Rows to show
        self.meta_labels = {}
        rows = [
            ("Num Imgs", "num_imgs"),
            ("X-Range (nm)", "x_range_nm"),
            ("Speed (fps)", "frame_rate"),
            ("Line/s (Hz)", "line_rate"),
            ("y pixels", "y_pixels"),
            ("x pixels", "x_pixels"),
            ("Pixel/nm", "pixel_size_nm"),
            ("Channel", "channel")
        ]
        for i, (label_text, key) in enumerate(rows, start=1):
            meta_layout.addWidget(QLabel(label_text), i, 0)
            val_label = QLabel("-")
            meta_layout.addWidget(val_label, i, 1)
            self.meta_labels[key] = val_label

        # Add metadata frame to right column
        right_col.addWidget(QLabel("Metadata"))
        right_col.addWidget(self.meta_frame)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_col, 0)
        main_layout.addLayout(center_col, 1)
        main_layout.addLayout(right_col, 0)
        self.setLayout(main_layout)

        # playback timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)


    # -------------------------
    # Loading folder / files
    # -------------------------
    def open_folder_or_files(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with AFM files")
        if folder:
            self.status_label.setText(f"Scanning folder: {folder}")
            QApplication.processEvents()
            self._populate_from_folder(folder)
            return

        filters = "JPK/AFM files (*.jpk *.h5 *.hdf5 *.asd *.spm *.aris *.h5-jpk);;NumPy files (*.npy *.npz);;Video files (*.avi *.mp4 *.mov);;All files (*)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Open AFM files", "", filters)
        if not paths:
            return
        self.status_label.setText("Loading selected files...")
        QApplication.processEvents()
        self._populate_from_file_list(paths)
    def update_metadata_panel(self):
        """Actualizar los labels del panel de metadatos desde self.meta y self.raw_stack."""
        # Num Imgs
        n = len(self.raw_stack) if self.raw_stack is not None else 0
        self.meta_labels["num_imgs"].setText(str(n))

        # X-Range (nm)
        xr = self.meta.get("x_range_nm", None)
        yr = self.meta.get("y_range_nm", None)
        if xr is not None and yr is not None:
            try:
                self.meta_labels["x_range_nm"].setText(f"{float(xr):.3g} × {float(yr):.3g}")
            except Exception:
                self.meta_labels["x_range_nm"].setText(str(xr))
        else:
            self.meta_labels["x_range_nm"].setText(str(self.meta.get("x_range_nm", "-")))

        # Speed (fps)
        fps = self.meta.get("frame_rate", None)
        self.meta_labels["frame_rate"].setText(str(fps) if fps is not None else "-")

        # Line/s (Hz)
        line_rate = self.meta.get("line_rate", self.meta.get("lines_per_second", None))
        self.meta_labels["line_rate"].setText(str(line_rate) if line_rate is not None else "-")

        # Pixels (y,x)
        if self.raw_stack is not None:
            try:
                h = int(self.raw_stack.shape[1])
                w = int(self.raw_stack.shape[2])
                self.meta_labels["y_pixels"].setText(str(h))
                self.meta_labels["x_pixels"].setText(str(w))
            except Exception:
                self.meta_labels["y_pixels"].setText("-")
                self.meta_labels["x_pixels"].setText("-")
        else:
            self.meta_labels["y_pixels"].setText("-")
            self.meta_labels["x_pixels"].setText("-")

        # Pixel/nm
        px = self.meta.get("pixel_size_nm", None)
        self.meta_labels["pixel_size_nm"].setText(str(px) if px is not None else "-")

        # Channel
        channel = self.meta.get("channel", self.meta.get("channel_name", "unknown"))
        self.meta_labels["channel"].setText(str(channel) if channel is not None else "unknown")

        # Update summary labels too (if los tienes)
        try:
            self.meta_fps.setText(f"FPS: {fps}" if fps is not None else "FPS: -")
            if px is not None:
                self.meta_size.setText(f"Pixel size (nm): {px}")
            else:
                self.meta_size.setText("Size (nm): -")
            if self.raw_stack is not None:
                h = self.raw_stack.shape[1]; w = self.raw_stack.shape[2]
                self.meta_pixels.setText(f"Pixels: {w} × {h}")
            else:
                self.meta_pixels.setText("Pixels: -")
        except Exception:
            pass


    def _populate_from_folder(self, folder):
        valid_exts = (".jpk", ".h5", ".hdf5", ".asd", ".spm", ".aris", ".h5-jpk", ".npy", ".npz", ".avi", ".mp4", ".mov")
        files = []
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(valid_exts):
                files.append(os.path.join(folder, fname))
        if not files:
            self.status_label.setText("No AFM files found in folder.")
            return
        self._populate_from_file_list(files)

    def _populate_from_file_list(self, paths):
        """
        Ahora: solo lista los archivos en self.list_files (con miniatura y nframes)
        y guarda la lista en self._file_index. No concatena automáticamente.
        La carga real del stack se hace cuando el usuario selecciona items (on_list_selection_changed).
        """
        self.list_files.clear()
        self._file_index = []   # lista de rutas en el mismo orden que los items
        meta_accum = None

        for p in paths:
            try:
                frames, file_meta = self._read_file_to_frames(p)
                nframes = len(frames)
                thumb = self._make_thumbnail(frames[0])
                item = QListWidgetItem(QIcon(thumb), f"{os.path.basename(p)}  —  {nframes} frames")
                # guardamos la ruta en el item para mapear fácilmente
                item.setData(Qt.UserRole, p)
                self.list_files.addItem(item)
                self._file_index.append(p)
                if meta_accum is None:
                    meta_accum = file_meta
            except Exception as e:
                item = QListWidgetItem(f"{os.path.basename(p)}  —  ERROR: {e}")
                item.setData(Qt.UserRole, p)
                self.list_files.addItem(item)

        # Guardamos source_files para referencia, pero NO concatenamos aquí
        self.meta["source_files"] = paths
        # Actualizamos etiquetas de metadatos con la info del primer archivo (si existe)
        if meta_accum is not None:
            # no sobreescribimos meta completa, solo añadimos campos detectados
            self.meta.update(meta_accum)
        self._update_meta_labels()
        self.update_metadata_panel()
        self.status_label.setText(f"Found {len(self._file_index)} files. Select one or more to build the video.")


    def _read_file_to_frames(self, p):
        # 1. Leer frames (playnano, h5py, numpy, video)
        frames = self._read_frames_only(p)

        # 2. Leer metadatos usando Photosynthesis
        file_meta = read_metadata(p)
        file_meta["source_file"] = p

        return frames, file_meta
    def _read_frames_only(self, p):
        # playnano si existe
        if HAS_PLAYNANO:
            afm = AFMImage(p)
            data = afm.data
            return data[np.newaxis, ...] if data.ndim == 2 else data

        # hdf5
        try:
            import h5py
            with h5py.File(p, "r") as f:
                ds = next(iter(f.values()))
                arr = np.asarray(ds)
                return arr if arr.ndim == 3 else arr[np.newaxis, ...]
        except:
            pass

        # numpy
        ext = os.path.splitext(p)[1].lower()
        if ext in (".npy", ".npz"):
            arr = np.load(p)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr = arr[list(arr.keys())[0]]
            return arr if arr.ndim == 3 else arr[np.newaxis, ...]

        # video
        cap = cv2.VideoCapture(p)
        if cap.isOpened():
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            return np.stack(frames, axis=0)

        raise ValueError("Unsupported file format")


    def _make_thumbnail(self, frame, thumb_w=160, thumb_h=96):
        try:
            img = frame.astype(np.float32)
            img = img - np.nanmin(img)
            rng = np.nanmax(img)
            if rng == 0:
                rng = 1.0
            img = (img / rng * 255.0).astype(np.uint8)
            qimg = numpy_to_qimage(img)
            pix = QPixmap.fromImage(qimg)
            pix = pix.scaled(thumb_w, thumb_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return pix
        except Exception:
            return QPixmap(thumb_w, thumb_h)

    def populate_list(self):
        self.list_files.clearSelection()
        if self.raw_stack is None:
            return
        n = len(self.raw_stack)
        self.spin_frame.setMaximum(max(0, n - 1))
        self.slider_time.setMaximum(max(0, n - 1))
        self.current_frame = 0

    # -------------------------
    # Selection handler (now implemented)
    # -------------------------
    def on_list_selection_changed(self):
        """
        Cuando el usuario selecciona uno o varios archivos en la lista,
        cargamos solo esos archivos y concatenamos sus frames en raw_stack.
        """
        selected_items = self.list_files.selectedItems()
        if not selected_items:
            return

        sel_paths = []
        for it in selected_items:
            p = it.data(Qt.UserRole)
            if p:
                sel_paths.append(p)

        if not sel_paths:
            self.status_label.setText("No valid files selected.")
            return

        all_frames = []
        meta_accum = None
        total_frames = 0
        for p in sel_paths:
            try:
                frames, file_meta = self._read_file_to_frames(p)
                all_frames.append(np.asarray(frames))
                total_frames += len(frames)
                if meta_accum is None:
                    meta_accum = file_meta
            except Exception as e:
                self.status_label.setText(f"Error loading {os.path.basename(p)}: {e}")

        if len(all_frames) == 0:
            self.status_label.setText("No valid frames loaded from selection.")
            return

        try:
            new_stack = np.concatenate(all_frames, axis=0)
        except Exception as e:
            self.status_label.setText(f"Error concatenating selected frames: {e}")
            return

        # Actualizar stacks y meta
        self.raw_stack = new_stack
        self.processed_stack = self.raw_stack.copy()
        # meta_accum contiene metadatos del primer archivo seleccionado
        if meta_accum:
            self.meta.update(meta_accum)
        self.meta["total_frames"] = total_frames
        self.meta["source_files"] = sel_paths

        # Actualizar UI
        self._update_meta_labels()
        self.update_metadata_panel()
        self.populate_list()      # ajusta spin/slider
        self.update_preview()
        # Histograma está desactivado si lo comentaste; si no, actualizaría:
        # self.update_histogram()
        self.status_label.setText(f"Loaded {total_frames} frames from {len(sel_paths)} selected files")


    # -------------------------
    # Histogram preview sliders
    # -------------------------
    def on_histogram_slider_changed(self, _val=None):
        base = self.processed_stack if self.processed_stack is not None else self.raw_stack
        if base is None:
            return
        lo_pct = self.slider_lower.value()
        hi_pct = self.slider_upper.value()
        if hi_pct <= lo_pct:
            self.status_label.setText("Histogram upper must be > lower")
            return
        try:
            lo = np.percentile(base, lo_pct)
            hi = np.percentile(base, hi_pct)
            preview_stack = np.clip(base, lo, hi)
            idx = max(0, min(self.current_frame, len(preview_stack) - 1))
            frame = preview_stack[idx]
            frame_disp = self._overlay_frame(frame, idx)
            qimg = numpy_to_qimage(frame_disp.astype(np.float32))
            pix = QPixmap.fromImage(qimg)
            pix = pix.scaled(self.label_preview.width(), self.label_preview.height(), Qt.KeepAspectRatio)
            self.label_preview.setPixmap(pix)
            # IMPORTANT: histogram rendering only in right panel (label_hist)
            #self._update_histogram_from_array(preview_stack)
            self.status_label.setText(f"Preview clipping: {lo_pct}% - {hi_pct}%")
        except Exception as e:
            self.status_label.setText(f"Preview error: {e}")

    def _update_histogram_from_array(self, arr):
        data = arr.flatten()
        data = data[~np.isnan(data)]
        fig = plt.figure(figsize=(3, 2), dpi=100)
        ax = fig.add_subplot(111)
        ax.hist(data, bins=128, color="#2c7fb8")
        ax.set_xlabel("Height (a.u.)")
        ax.set_ylabel("Counts")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.canvas.print_png(buf)
        buf.seek(0)
        img = QImage.fromData(buf.getvalue())
        pix = QPixmap.fromImage(img)
        pix = pix.scaled(self.label_hist.width(), self.label_hist.height(), Qt.KeepAspectRatio)
        self.label_hist.setPixmap(pix)
        plt.close(fig)

    # -------------------------
    # Full processing (background)
    # -------------------------
    def apply_filters(self):
        if self.raw_stack is None:
            self.status_label.setText("Load a stack first")
            return
        level_method = self.combo_level.currentText()
        flat_method = self.combo_flatten.currentText()
        lo_pct = self.slider_lower.value()
        hi_pct = self.slider_upper.value()
        self.progress.setVisible(True)
        QApplication.processEvents()
        self._proc_thread = ProcessingThread(self.raw_stack, level_method, flat_method, lo_pct, hi_pct)
        self._proc_thread.finished.connect(self._on_processing_finished)
        self._proc_thread.start()
        self.status_label.setText("Processing...")

    def _on_processing_finished(self, stack):
        self.processed_stack = stack
        self.progress.setVisible(False)
        self.populate_list()
        self.update_preview()
        #self.update_histogram()
        self.update_metadata_panel()
        self.status_label.setText("Filters applied")

    # -------------------------
    # Overlay & preview helpers
    # -------------------------
    def _overlay_frame(self, frame, idx):
        """Return frame with overlay text scaled to image size."""
        arr = frame.copy().astype(np.float32)
        overlay_texts = []
        if self.checkbox_overlay.isChecked():
            fps = self.meta.get("frame_rate", None) or 10
            seconds = idx / float(fps)
            overlay_texts.append(f"{seconds:.2f} s")
        if self.checkbox_overlay_frame.isChecked():
            overlay_texts.append(f"Frame {idx}")
        if not overlay_texts:
            return arr

        # Normalize to uint8 0-255
        img = arr - np.nanmin(arr)
        rng = np.nanmax(img)
        if rng == 0:
            rng = 1.0
        img8 = (img / rng * 255.0).astype(np.uint8)

        # Convert to BGR for drawing
        bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

        text = " | ".join(overlay_texts)

        # Compute scale and thickness based on image size (min dimension)
        h, w = img8.shape[:2]
        base = min(h, w)
        # scale: small images -> 0.4, medium -> 0.6, large -> 1.0 (tunable)
        scale = max(0.25, min(0.7, base / 600.0))
        thickness = max(1, int(round(scale * 2)))

        # Text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = int(round(6 * scale))

        # Position: top-left with margin
        x0, y0 = 8, 8
        # Draw semi-transparent rectangle
        rect_w = tw + pad * 2
        rect_h = th + pad * 2
        overlay = bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + rect_w, y0 + rect_h), (0, 0, 0), -1)
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0, bgr)

        # Put white text
        text_x = x0 + pad
        text_y = y0 + pad + th
        cv2.putText(bgr, text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Convert back to grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return gray


    def update_preview(self):
        if self.processed_stack is None:
            self.label_preview.setText("Preview")
            return
        idx = max(0, min(self.current_frame, len(self.processed_stack) - 1))
        frame = self.processed_stack[idx]
        frame_disp = self._overlay_frame(frame, idx)
        qimg = numpy_to_qimage(frame_disp.astype(np.float32))
        pix = QPixmap.fromImage(qimg)

        target_size = self.label_preview.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
           target_size = QSize(480, 360)

        pix = pix.scaled(self.label_preview.width(), self.label_preview.height(), Qt.KeepAspectRatio)
        self.label_preview.setPixmap(pix)

        self.spin_frame.blockSignals(True)
        self.spin_frame.setValue(idx)
        self.spin_frame.blockSignals(False)
        self.slider_time.blockSignals(True)
        self.slider_time.setValue(idx)
        self.slider_time.blockSignals(False)

    #def update_histogram(self):
    #    if self.processed_stack is None:
    #        self.label_hist.setText("Histogram")
    #        return
    #    self._update_histogram_from_array(self.processed_stack)

    # -------------------------
    # Playback
    # -------------------------
    def start_play(self):
        if self.processed_stack is None:
            self.status_label.setText("No stack to play")
            return
        fps = self.meta.get("frame_rate", 10) or 10
        interval = int(max(1, 1000 / fps))
        self._timer.start(interval)
        self.status_label.setText("Playing")

    def stop_play(self):
        self._timer.stop()
        self.status_label.setText("Paused")

    def _advance_frame(self):
        if self.processed_stack is None:
            return
        self.current_frame = (self.current_frame + 1) % len(self.processed_stack)
        self.update_preview()

    def prev_frame(self):
        if self.processed_stack is None:
            return
        self.current_frame = max(0, self.current_frame - 1)
        self.update_preview()

    def next_frame(self):
        if self.processed_stack is None:
            return
        self.current_frame = min(len(self.processed_stack) - 1, self.current_frame + 1)
        self.update_preview()

    def on_spin_frame_changed(self, val):
        self.current_frame = val
        self.update_preview()

    def on_slider_time_changed(self, val):
        self.current_frame = val
        self.update_preview()

    # -------------------------
    # Send / Save
    # -------------------------
    def send_to_drift(self):
        if self.processed_stack is None:
            self.status_label.setText("No processed stack to send")
            return
        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, "load_afm"):
                parent.load_afm(self.processed_stack, self.meta)
                self.status_label.setText("Sent stack to Drift Correction (parent.load_afm called)")
            elif parent is not None:
                setattr(parent, "afm_stack", self.processed_stack)
                setattr(parent, "afm_meta", self.meta)
                self.status_label.setText("Sent stack to parent attributes (afm_stack / afm_meta).")
            else:
                self.status_label.setText("No parent to send to")
        except Exception as e:
            self.status_label.setText(f"Error sending to parent: {e}")

    def save_metadata_and_video(self):
        if self.processed_stack is None:
            self.status_label.setText("No processed stack to save")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder to save")
        if not folder:
            return
        meta_path = os.path.join(folder, "afm_metadata.json")
        try:
            with open(meta_path, "w") as f:
                json.dump(self.meta, f, indent=2, default=str)
        except Exception as e:
            self.status_label.setText(f"Error saving metadata: {e}")
            return
        try:
            H, W = self.processed_stack[0].shape
            out_path = os.path.join(folder, "afm_processed.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = self.meta.get("frame_rate", 10) or 10
            writer = cv2.VideoWriter(out_path, fourcc, float(fps), (W, H), False)
            for f in self.processed_stack:
                arr = f.astype(np.float32)
                arr = arr - np.nanmin(arr)
                rng = np.nanmax(arr)
                if rng == 0:
                    rng = 1.0
                arr = (arr / rng * 255.0).astype(np.uint8)
                writer.write(arr)
            writer.release()
        except Exception as e:
            self.status_label.setText(f"Error saving video: {e}")
            return
        self.status_label.setText(f"Saved metadata and video to {folder}")

    # -------------------------
    # Helpers
    # -------------------------
    def _update_meta_labels(self):
        fps = self.meta.get("frame_rate", None)
        if fps is None:
            self.meta_fps.setText("FPS: -")
        else:
            self.meta_fps.setText(f"FPS: {fps}")

        px = self.meta.get("pixel_size_nm", None)
        if px is None:
            self.meta_size.setText("Size (nm): -")
        else:
            xr = self.meta.get("x_range_nm", None)
            yr = self.meta.get("y_range_nm", None)
            if xr is not None and yr is not None:
                self.meta_size.setText(f"Size (nm): {xr} × {yr}")
            else:
                self.meta_size.setText(f"Pixel size (nm): {px}")

        pixels = self.meta.get("pixels", None)
        if pixels is None and self.raw_stack is not None:
            h, w = self.raw_stack.shape[1], self.raw_stack.shape[2]
            self.meta_pixels.setText(f"Pixels: {w} × {h}")
        elif pixels is not None:
            self.meta_pixels.setText(f"Pixels: {pixels}")
        else:
            self.meta_pixels.setText("Pixels: -")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self.update_preview()
            self.update_histogram()
        except Exception:
            pass


# Backwards compatibility name
AFMLoaderWindow = AFMLoaderWidget
