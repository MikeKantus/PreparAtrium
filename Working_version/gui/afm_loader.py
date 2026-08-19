# gui/afm_loader.py
# (Este archivo es la versión corregida: histograma en panel aparte y sin "Open Drift Panel")

import os
import io
import json
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py
import tifffile

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QListWidget,
    QFileDialog, QComboBox, QSlider, QProgressBar, QApplication,
    QSpinBox, QSizePolicy, QCheckBox, QListWidgetItem, QFrame, QGridLayout,
    QToolButton, QSplitter
)
from PySide6.QtGui import QPixmap, QImage, QIcon, QFont
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from core.ui_utils import frame_to_qimage_safe

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
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.setMinimumWidth(480)
        self.raw_stack = None
        self.processed_stack = None
        self.meta = {}
        self.current_file_or_folder = os.getcwd()
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

        # --- LEFT COLUMN: Explorer (combo + preview) arriba, file list abajo (splitter) ---
        # Top explorer widgets
        self.combo_parent_files = QComboBox()
        self.combo_parent_files.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo_parent_files.setToolTip("Selecciona carpeta/fichero en el nivel superior")

        self.btn_refresh_files = QToolButton()
        self.btn_refresh_files.setText("Refresh")
        self.btn_refresh_files.setToolTip("Refrescar lista de ficheros/carpeta")

        self.btn_open_in_explorer = QToolButton()
        self.btn_open_in_explorer.setText("Open folder")
        self.btn_open_in_explorer.setToolTip("Abrir carpeta en el explorador del sistema")

        # Preview list (miniaturas horizontales)
        self.list_file_preview = QListWidget()
        self.list_file_preview.setViewMode(QListWidget.IconMode)
        self.list_file_preview.setIconSize(QSize(120, 80))
        self.list_file_preview.setResizeMode(QListWidget.Adjust)
        self.list_file_preview.setMovement(QListWidget.Static)
        self.list_file_preview.setMaximumHeight(140)
        self.list_file_preview.setSpacing(6)
        self.list_file_preview.itemActivated.connect(self.on_preview_activated)
        self.list_file_preview.itemClicked.connect(lambda it: self.list_file_preview.setCurrentItem(it))

        # Top bar layout (combo + buttons)
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Explore:"))
        top_bar.addWidget(self.combo_parent_files)
        top_bar.addWidget(self.btn_refresh_files)
        top_bar.addWidget(self.btn_open_in_explorer)

        # Build a widget for the top explorer area
        explorer_widget = QFrame()
        explorer_layout = QVBoxLayout(explorer_widget)
        explorer_layout.setContentsMargins(4, 4, 4, 4)
        explorer_layout.addLayout(top_bar)
        explorer_layout.addWidget(self.list_file_preview)

        # Now create a splitter so the top explorer and the file list share the left column
        self.left_splitter = QSplitter(Qt.Vertical)
        # Put explorer widget on top
        self.left_splitter.addWidget(explorer_widget)
        # Put the file list below
        self.left_splitter.addWidget(self.list_files)
        # Set initial sizes (top smaller than bottom)
        self.left_splitter.setSizes([200, 1200])
        self.left_splitter.setMaximumWidth(350)

        # Now create the left_col layout and add the splitter and the leveling controls below
        left_col = QVBoxLayout()
        left_col.addWidget(self.left_splitter)
        left_col.addStretch()
        left_col.addWidget(QLabel("Leveling"))
        left_col.addWidget(self.combo_level)
        left_col.addWidget(QLabel("Flatten"))
        left_col.addWidget(self.combo_flatten)
        left_col.addWidget(self.btn_apply)

        # Explorer connections
        self.btn_refresh_files.clicked.connect(lambda: self.populate_parent_combo(getattr(self, "current_file_or_folder", os.getcwd())))
        self.btn_open_in_explorer.clicked.connect(self.open_selected_folder_in_explorer)
        self.combo_parent_files.currentIndexChanged.connect(lambda idx: self.refresh_file_preview())
        # Si quieres que la preview se llene al inicio, llama populate_parent_combo tras definir current_file_or_folder

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
            ("Pixel/nm", "pixel_size_nm"),
            ("X-Range (nm)", "x_range_nm"),
            ("Line/s", "frame_rate"),
            ("FPS", "real_FPS"),
            ("y pixels", "y_pixels"),
            ("x pixels", "x_pixels"),
           
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

        self.populate_parent_combo(self.current_file_or_folder)
        # Inicializar explorer con la carpeta actual o con el directorio de trabajo
        base = getattr(self, "current_file_or_folder", None) or os.getcwd()
        try:
            self.populate_parent_combo(base)
        except Exception:
            # no bloquear si algo falla en startup
            pass


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
        self.list_files.clear()
        self._file_index = []
        self.meta = {}   # reiniciar metadatos para nueva selección

        for p in paths:
            try:
                frames, file_meta = self._read_file_to_frames(p)

                # Miniatura
                thumb = self._make_thumbnail(frames[0])
                item = QListWidgetItem(QIcon(thumb), f"{os.path.basename(p)}  —  {len(frames)} frames")
                item.setData(Qt.UserRole, p)
                self.list_files.addItem(item)
                self._file_index.append(p)

                # 1) Extraer metadatos extendidos primero
                extra_meta = self._read_metadata_jpk(p)
                if isinstance(extra_meta, dict):
                    for k, v in extra_meta.items():
                        if k not in self.meta or self.meta[k] in (None, "-", "unknown"):
                            self.meta[k] = v


                # 2) Rellenar huecos con metadatos básicos del loader
                for k, v in file_meta.items():
                    if k not in self.meta or self.meta[k] in (None, "-", "unknown"):
                        self.meta[k] = v

            except Exception as e:
                item = QListWidgetItem(f"{os.path.basename(p)}  —  ERROR: {e}")
                item.setData(Qt.UserRole, p)
                self.list_files.addItem(item)

        # Actualizar panel de metadatos
        self.update_metadata_panel()
        self.status_label.setText(f"Found {len(self._file_index)} files. Select one or more to build the video.")

    def _read_metadata_jpk(self, path):
        """
        Lee metadatos de archivos JPK usando TIFF tags.
        Basado en el script funcional proporcionado por Miguel.
        """

        import tifffile

        meta = {}

        # --- SCAN TAGS ---
        scan_fields = {
            "x_origin_nm": 32832,
            "y_origin_nm": 32833,
            "x_range_nm": 32834,
            "y_range_nm": 32835,
            "x_pixels": 32838,
            "y_pixels": 32839,
            "frame_rate": 32841,   # normalizamos nombre
        }

        # --- CANTILEVER / FEEDBACK ---
        cantilever_keys = {
            "amplitude",
            "calibration-environment",
            "cantilever-id",
            "cantilever-name",
            "defined",
            "frequency",
            "geometry",
            "qFactor",
            "sensitivity",
            "spring-constant",
        }

        feedback_keys = {
            "setpoint-feedback-settings.relative-setpoint"
        }

        def extract_scan(tags):
            scan = {}
            for key, code in scan_fields.items():
                value = tags.get(code)
                if value is None:
                    continue

                try:
                    if "nm" in key:
                        val = float(value)
                        # si está en metros, convertir a nm
                        scan[key] = val * 1e9 if val < 1 else val
                    elif "pixels" in key:
                        scan[key] = int(value)
                    else:
                        scan[key] = float(value)
                except Exception:
                    pass

            return scan

        def extract_cantilever_and_feedback(text):
            cantilever = {}
            feedback = {}

            for line in text.splitlines():
                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # CANTILEVER
                if key.startswith("cantilever-calibration-info."):
                    short = key.replace("cantilever-calibration-info.", "")
                    if short in cantilever_keys:
                        cantilever[short] = value

                # FEEDBACK
                if key.startswith("feedback-mode.setpoint-feedback-settings."):
                    short = key.replace("feedback-mode.setpoint-feedback-settings.", "")
                    full_key = f"setpoint-feedback-settings.{short}"
                    if full_key in feedback_keys:
                        feedback[full_key] = value

            return cantilever, feedback

        # --- LECTURA TIFF ---
        try:
            with tifffile.TiffFile(path) as tif:
                scan = {}
                cantilever = {}
                feedback = {}

                for page in tif.pages:
                    tags = {tag.code: tag.value for tag in page.tags.values()}

                    # SCAN
                    if not scan:
                        scan = extract_scan(tags)

                    # CANTILEVER + FEEDBACK
                    for value in tags.values():
                        if isinstance(value, str) and "cantilever-calibration-info" in value:
                            c, f = extract_cantilever_and_feedback(value)
                            cantilever.update(c)
                            feedback.update(f)

                # fusionar todo
                meta.update(scan)
                meta.update(cantilever)
                meta.update(feedback)

                # canal (si existe)
                if "channel" not in meta:
                    meta["channel"] = None

                return meta

        except Exception as e:
            print(f"[_read_metadata_jpk] ERROR leyendo TIFF: {e}")

        # Si falla, devolver dict vacío
        return {}

    def _read_file_to_frames(self, p):
        """
        Return (frames_array, file_meta). Robust extraction of common metadata keys.
        If playnano is available, try many attribute names. If HDF5, inspect attrs.
        """
        # Helper to try many keys
        def pick_first(d, keys):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return None

        if HAS_PLAYNANO:
            afm = AFMImage(p)
            data = afm.data
            frames = data[np.newaxis, ...] if data.ndim == 2 else data

            # Collect attributes from object (try __dict__, dir, and getattr)
            obj_attrs = {}
            try:
                obj_attrs.update(getattr(afm, "__dict__", {}))
            except Exception:
                pass
            # also try dir() to list properties
            for name in dir(afm):
                try:
                    val = getattr(afm, name)
                    # avoid callables
                    if not callable(val):
                        obj_attrs[name] = val
                except Exception:
                    pass

            # Candidate keys for each metadata field
            pixel_keys = ["pixel_size", "pixel_size_nm", "pixel_size_x", "pixel_size_y", "pixel"]
            fps_keys = ["frame_rate", "fps", "frameRate", "scan_rate"]
            x_range_keys = ["x_range", "x_range_nm", "x_range_nm_total", "x_range_um"]
            y_range_keys = ["y_range", "y_range_nm", "y_range_nm_total", "y_range_um"]
            pixels_keys = ["shape", "pixels", "image_shape", "size"]
            channel_keys = ["channel", "channel_name", "chan", "channelId"]
            line_keys = ["line_rate", "lines_per_second", "line_frequency", "scan_rate_hz"]

            file_meta = {}
            # Píxeles siempre desde frames, nunca desde metadatos
            y_pixels = frames.shape[1]
            x_pixels = frames.shape[2]
            file_meta["pixels"] = (x_pixels, y_pixels)
            file_meta["x_pixels"] = x_pixels
            file_meta["y_pixels"] = y_pixels

            file_meta["frame_rate"] = pick_first(obj_attrs, fps_keys)
            file_meta["x_range_nm"] = pick_first(obj_attrs, x_range_keys)
            file_meta["y_range_nm"] = pick_first(obj_attrs, y_range_keys)
            file_meta["pixels"] = pick_first(obj_attrs, pixels_keys)
            file_meta["channel"] = pick_first(obj_attrs, channel_keys) or "unknown"
            file_meta["line_rate"] = pick_first(obj_attrs, line_keys)
            file_meta["source_file"] = p

            # If some fields are still None, try to infer from available data
            if file_meta["pixels"] is None:
                try:
                    file_meta["pixels"] = (frames.shape[2], frames.shape[1])
                except Exception:
                    file_meta["pixels"] = None

            # If nothing found for x_range but pixel_size and pixels exist, compute x_range
            try:
                if file_meta["x_range_nm"] is None and file_meta["pixel_size_nm"] is not None and file_meta["pixels"] is not None:
                    px = float(file_meta["pixel_size_nm"])
                    w = int(file_meta["pixels"][0]) if isinstance(file_meta["pixels"], (list, tuple)) else int(file_meta["pixels"])
                    file_meta["x_range_nm"] = px * w
            except Exception:
                pass

            # Debug: if critical fields missing, show available keys in status_label (short)
            missing = []
            for k in ("x_range_nm", "frame_rate", "line_rate", "channel"):
                if file_meta.get(k) in (None, "unknown"):
                    missing.append(k)
            if missing:
                # show a short debug hint (not too verbose)
                keys_found = ", ".join(list(obj_attrs.keys())[:20])
                self.status_label.setText(f"Metadata missing: {missing}. Example attrs: {keys_found} ...")
                print("\n=== DIAGNÓSTICO JPK ===")
                print(f"Archivo: {p}")
                print(f"frames.shape: {frames.shape}")
            return np.asarray(frames), file_meta      

        try:
            print(f"PlayNano data.shape: {afm.data.shape}")
        except Exception:
            print("PlayNano data.shape: ERROR")

        # Diagnóstico de atributos PlayNano
        print("Atributos PlayNano relevantes:")
        for key in ["shape", "pixels", "image_shape", "size"]:
            try:
                print(f"  {key}: {getattr(afm, key, 'NO EXISTE')}")
            except Exception:
                print(f"  {key}: ERROR")

        print("=== FIN DIAGNÓSTICO ===\n")
        # HDF5 fallback (same as previous but with more attribute name checks)
        try:
            import h5py
            with h5py.File(p, "r") as f:
                # find dataset
                def find_dataset(group):
                    for k, v in group.items():
                        if isinstance(v, h5py.Dataset):
                            if v.ndim in (2, 3):
                                return v
                        elif isinstance(v, h5py.Group):
                            res = find_dataset(v)
                            if res is not None:
                                return res
                    return None
                ds = find_dataset(f)
                if ds is None:
                    raise ValueError("No image dataset found in HDF5")
                arr = np.asarray(ds)
                frames = arr if arr.ndim == 3 else arr[np.newaxis, ...]

                # collect attrs
                attrs = {}
                try:
                    attrs.update({k: v for k, v in f.attrs.items()})
                except Exception:
                    pass
                # flatten group attrs too (first level)
                for name, grp in f.items():
                    try:
                        if hasattr(grp, "attrs"):
                            for k, v in grp.attrs.items():
                                if k not in attrs:
                                    attrs[k] = v
                    except Exception:
                        pass

                # pick keys
                file_meta = {}
                file_meta["pixel_size_nm"] = pick_first(attrs, ["pixel_size_nm", "pixel_size", "pixel_size_x"])
                file_meta["frame_rate"] = pick_first(attrs, ["frame_rate", "fps"])
                file_meta["x_range_nm"] = pick_first(attrs, ["x_range_nm", "x_range"])
                file_meta["y_range_nm"] = pick_first(attrs, ["y_range_nm", "y_range"])
                file_meta["pixels"] = (frames.shape[2], frames.shape[1])
                file_meta["channel"] = pick_first(attrs, ["channel", "channel_name"]) or "unknown"
                file_meta["line_rate"] = pick_first(attrs, ["line_rate", "lines_per_second"])
                file_meta["source_file"] = p

                return np.asarray(frames), file_meta
        except Exception:
            pass

        # numpy / video fallback (as before)
        ext = os.path.splitext(p)[1].lower()
        if ext in (".npy", ".npz"):
            data = np.load(p)
            frames = data[list(data.keys())[0]] if isinstance(data, np.lib.npyio.NpzFile) else data
            if frames.ndim == 2:
                frames = frames[np.newaxis, ...]
            file_meta = {"pixel_size_nm": None, "frame_rate": None, "source_file": p, "channel": "unknown"}
            return np.asarray(frames), file_meta

        cap = cv2.VideoCapture(p)
        if cap.isOpened():
            frames_list = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames_list.append(gray)
            cap.release()
            if len(frames_list) == 0:
                raise ValueError("No frames in video")
            frames = np.stack(frames_list, axis=0)
            file_meta = {"pixel_size_nm": None, "frame_rate": None, "source_file": p, "channel": "unknown"}
            print("=== META DESPUÉS DE _read_file_to_frames ===")
            print(f"x_pixels: {file_meta.get('x_pixels')}")
            print(f"y_pixels: {file_meta.get('y_pixels')}")
            print(f"pixels (tuple): {file_meta.get('pixels')}")
            print("===========================================\n")

            return frames, file_meta

        raise ValueError("Unsupported file format or missing playnano/h5py")


    def _make_thumbnail(self, frame, thumb_w=160, thumb_h=96):
        """
        Crear miniatura segura a partir de cualquier tipo de frame.
        Usa frame_to_qimage_safe para normalizar y copiar datos.
        """
        try:
            # Asegurar que trabajamos con una copia y tipo manejable
            img = np.asarray(frame)
            # Si es float, dejar que frame_to_qimage_safe haga la normalización
            qimg = frame_to_qimage_safe(img)
            pix = QPixmap.fromImage(qimg)
            pix = pix.scaled(thumb_w, thumb_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return pix
        except Exception:
            return QPixmap(thumb_w, thumb_h)

    def update_metadata_panel(self):
        # Diagnóstico opcional
        print("=== DIAGNÓSTICO PANEL METADATOS ===")
        print("self.meta:", self.meta)
        if self.raw_stack is not None:
            print("raw_stack.shape:", self.raw_stack.shape)
        print("=== FIN DIAGNÓSTICO PANEL ===")

        # Num Imgs
        num_imgs = self.meta.get("total_frames")
        if num_imgs is None and self.raw_stack is not None:
            num_imgs = self.raw_stack.shape[0]
        self.meta_labels["num_imgs"].setText(str(num_imgs) if num_imgs is not None else "-")

        # --- X/Y pixels: usar SIEMPRE los metadatos JPK si existen ---
        x_pixels = self.meta.get("x_pixels")
        y_pixels = self.meta.get("y_pixels")

        # fallback solo si no hay metadatos
        if (x_pixels is None or y_pixels is None) and self.raw_stack is not None:
            y_pixels = self.raw_stack.shape[-2]
            x_pixels = self.raw_stack.shape[-1]

        self.meta_labels["x_pixels"].setText(str(x_pixels) if x_pixels is not None else "-")
        self.meta_labels["y_pixels"].setText(str(y_pixels) if y_pixels is not None else "-")

        # --- FPS reales ---
        frame_rate = self.meta.get("frame_rate")
        real_fps = None
        try:
            if frame_rate is not None and y_pixels not in (None, 0):
                real_fps = float(frame_rate) / float(y_pixels)
        except Exception:
            real_fps = None

        self.meta_labels["real_FPS"].setText(f"{real_fps:.3f}" if real_fps is not None else "-")


        # X-Range (nm)
        x_range = self.meta.get("x_range_nm")
        self.meta_labels["x_range_nm"].setText(str(x_range) if x_range is not None else "-")

        # Frame rate
        frame_rate = self.meta.get("frame_rate")
        self.meta_labels["frame_rate"].setText(str(frame_rate) if frame_rate is not None else "-")

        # Channel
        channel = self.meta.get("channel")
        self.meta_labels["channel"].setText(str(channel) if channel not in (None, "unknown") else "-")

        # Pixel size (si quieres derivarlo)
        pixel_size = self.meta.get("pixel_size_nm")
        if pixel_size is None and x_range is not None and x_pixels not in (None, 0):
            pixel_size = x_range / x_pixels
        self.meta_labels["pixel_size_nm"].setText(str(pixel_size) if pixel_size is not None else "-")


            
    def open_selected_folder_in_explorer(self):
        """
        Abre un cuadro de diálogo para elegir una carpeta
        y actualiza el panel explorer con esa carpeta.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select folder to explore")

        if folder:
            self.current_file_or_folder = folder
            self.populate_parent_combo(folder)

    def populate_list(self):
        self.list_files.clearSelection()
        if self.raw_stack is None:
            return
        n = len(self.raw_stack)
        self.spin_frame.setMaximum(max(0, n - 1))
        self.slider_time.setMaximum(max(0, n - 1))
        self.current_frame = 0

    def populate_parent_combo(self, folder):
        """
        Muestra en el panel superior la carpeta que contiene los archivos del panel inferior.
        """
        if not folder or not os.path.isdir(folder):
            return

        self.current_file_or_folder = folder

        # Limpiar combo
        self.combo_parent_files.clear()

        # Añadir carpeta actual
        self.combo_parent_files.addItem(folder)

        # Añadir contenido del folder
        for entry in sorted(os.listdir(folder)):
            full_path = os.path.join(folder, entry)
            self.combo_parent_files.addItem(full_path)

        # Actualizar panel inferior si se selecciona algo
        self.refresh_file_preview()

    def refresh_file_preview(self):
        """
        Muestra el contenido del folder seleccionado en el panel superior.
        """
        folder = self.combo_parent_files.currentText()

        if not os.path.isdir(folder):
            self.list_file_preview.clear()
            return

        self.list_file_preview.clear()

        for entry in sorted(os.listdir(folder)):
            full_path = os.path.join(folder, entry)

            icon = QIcon.fromTheme("folder") if os.path.isdir(full_path) else QIcon.fromTheme("text-x-generic")

            item = QListWidgetItem(icon, entry)
            item.setData(Qt.UserRole, full_path)
            self.list_file_preview.addItem(item)

    # -------------------------
    # Selection handler (now implemented)
    # -------------------------
    def on_preview_activated(self, item):
        path = item.data(Qt.UserRole)

        if os.path.isdir(path):
            # Mostrar archivos AFM dentro de esa carpeta
            self._populate_from_folder(path)
            self.current_file_or_folder = path
            self.populate_parent_combo(path)

    def on_list_selection_changed(self):
        """
        Cuando el usuario selecciona uno o varios archivos en la lista,
        cargamos solo esos archivos y concatenamos sus frames en raw_stack.
        También enriquecemos los metadatos usando read_metadata(path).
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

        # ---------------------------------------------------------
        # 1) Cargar frames de todos los archivos seleccionados
        # ---------------------------------------------------------
        for p in sel_paths:
            try:
                frames, file_meta = self._read_file_to_frames(p)
                all_frames.append(np.asarray(frames))
                total_frames += len(frames)

                # metadatos base del primer archivo
                if meta_accum is None:
                    meta_accum = file_meta

            except Exception as e:
                self.status_label.setText(f"Error loading {os.path.basename(p)}: {e}")

        if len(all_frames) == 0:
            self.status_label.setText("No valid frames loaded from selection.")
            return

        # ---------------------------------------------------------
        # 2) Concatenar frames
        # ---------------------------------------------------------
        try:
            new_stack = np.concatenate(all_frames, axis=0)
        except Exception as e:
            self.status_label.setText(f"Error concatenating selected frames: {e}")
            return

        # ---------------------------------------------------------
        # 3) Actualizar stacks y metadatos base
        # ---------------------------------------------------------
        self.raw_stack = new_stack
        self.processed_stack = self.raw_stack.copy()

        # metadatos base del loader
        if meta_accum:
            self.meta.update(meta_accum)

        self.meta["total_frames"] = total_frames
        self.meta["source_files"] = sel_paths

        # ---------------------------------------------------------
        # 4) Metadatos extendidos usando read_metadata(path)
        #    (JPK, HDF5, TIFF, SPM, ASD, BMP/JPG, AVI/MP4)
        # ---------------------------------------------------------
        try:
            extra_meta = self._read_metadata_jpk(sel_paths[0])  # solo del primer archivo
            if isinstance(extra_meta, dict):
                for k, v in extra_meta.items():
                    # no sobrescribir valores válidos del loader
                    if k not in self.meta or self.meta[k] in (None, "-", "unknown"):
                        self.meta[k] = v
        except Exception as e:
            print(f"[Metadata] Error extracting extended metadata: {e}")

        # ---------------------------------------------------------
        # 5) Actualizar UI
        # ---------------------------------------------------------
        self.update_metadata_panel()
        self.populate_list()      # ajusta spin/slider
        self.update_preview()

        self.status_label.setText(
            f"Loaded {total_frames} frames from {len(sel_paths)} selected files"
        )


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

        ps = np.asarray(self.processed_stack).astype(np.float32)
        # clip by percentiles to avoid outliers (ajusta si quieres)
        lo, hi = np.percentile(ps, [0.5, 99.5])
        ps = np.clip(ps, lo, hi)

        # normalize to uint8
        ps = ((ps - ps.min()) / (ps.max() - ps.min() + 1e-12) * 255.0).astype(np.uint8)

        # ensure contiguous memory
        self.processed_stack = np.ascontiguousarray(ps)
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
        """
        Return frame with overlay text scaled to image size.
        Always returns a uint8, C-contiguous 2D array (grayscale).
        """
        import cv2
        import numpy as np

        # Work on a copy and ensure numpy array
        arr = np.asarray(frame)

        # Replace NaNs if any
        if np.isnan(arr).any():
            arr = arr.copy()
            arr[np.isnan(arr)] = np.nanmin(arr)

        # Convert to float32 for safe processing
        if arr.dtype != np.float32:
            arr_f = arr.astype(np.float32)
        else:
            arr_f = arr.copy()

        # If no overlays requested, return a uint8 contiguous copy
        overlay_texts = []
        if self.checkbox_overlay.isChecked():
            fps = self.meta.get("frame_rate", None) or 10
            seconds = idx / float(fps)
            overlay_texts.append(f"{seconds:.2f} s")
        if self.checkbox_overlay_frame.isChecked():
            overlay_texts.append(f"Frame {idx}")
        if not overlay_texts:
            # normalize to uint8 and return contiguous
            img8 = arr_f - np.nanmin(arr_f)
            rng = np.nanmax(img8)
            if rng == 0 or np.isnan(rng):
                rng = 1.0
            img8 = (img8 / rng * 255.0).astype(np.uint8)
            return np.ascontiguousarray(img8)

        # --- Normalize using percentiles to avoid outliers (robust) ---
        lo_pct, hi_pct = 0.5, 99.5
        lo_v = np.percentile(arr_f, lo_pct)
        hi_v = np.percentile(arr_f, hi_pct)
        if hi_v <= lo_v:
            lo_v = np.nanmin(arr_f)
            hi_v = np.nanmax(arr_f)
            if hi_v <= lo_v:
                hi_v = lo_v + 1.0
        img_clip = np.clip(arr_f, lo_v, hi_v)

        # Scale to 0-255 uint8
        img8 = ((img_clip - lo_v) / (hi_v - lo_v) * 255.0).astype(np.uint8)

        # Convert to BGR for drawing overlay
        bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

        text = " | ".join(overlay_texts)

        # Compute scale and thickness based on image size (min dimension)
        h, w = img8.shape[:2]
        base = min(h, w)
        scale = max(0.35, min(1.0, base / 600.0))
        thickness = max(1, int(round(scale * 2)))

        # Text size and padding
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = int(round(6 * scale))

        # Position: top-left with margin
        x0, y0 = 8, 8
        rect_w = tw + pad * 2
        rect_h = th + pad * 2

        # Draw semi-transparent rectangle
        overlay = bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + rect_w, y0 + rect_h), (0, 0, 0), -1)
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0, bgr)

        # Put white text
        text_x = x0 + pad
        text_y = y0 + pad + th
        cv2.putText(bgr, text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Convert back to grayscale and ensure contiguous uint8
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(gray.astype(np.uint8))



    def update_preview(self):
        """
        Mostrar el frame actual usando frame_to_qimage_safe.
        Llamar a esta función después de actualizar self.current_frame,
        self.processed_stack o self.raw_stack.
        """
        base = self.processed_stack if self.processed_stack is not None else self.raw_stack
        if base is None:
            self.label_preview.clear()
            return

        idx = max(0, min(self.current_frame, len(base) - 1))
        frame = base[idx]

        # Si aplicas overlays, trabaja sobre copia y no modifiques 'frame' original
        frame_disp = self._overlay_frame(frame, idx)

        # Conversión segura a QImage y QPixmap
        qimg = frame_to_qimage_safe(frame_disp)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.label_preview.width(), self.label_preview.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_preview.setPixmap(pix)

        
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
        stack = self.processed_stack if self.processed_stack is not None else self.raw_stack
        meta = self.meta

        if stack is None:
            self.status_label.setText("No stack to send")
            return

        if self.main_window is not None:
            self.main_window.load_afm(stack, meta)
            self.main_window.open_drift_panel()   # ← ESTA ES LA CLAVE
            self.status_label.setText("Sent stack to drift panel")
        else:
            self.status_label.setText("ERROR: main_window not assigned")


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
    
    def resizeEvent(self, event):
        """
        Forzar re-render del frame actual cuando el widget cambia de tamaño,
        para que disp_w/disp_h se recalculen y el mapeo clic<->imagen siga siendo exacto.
        """
        # Llamar al resizeEvent de la superclase para mantener comportamiento por defecto
        try:
            super().resizeEvent(event)
        except Exception:
            # En caso raro de que la superclase no tenga resizeEvent, ignorar
            pass

        # Forzar re-render del frame actual (si existe update_frame)
        try:
            current = getattr(self, "current_frame", 0)
            # Si update_frame acepta idx, lo llamamos; si no, llamamos sin parámetros
            try:
                self.update_frame(current)
            except TypeError:
                self.update_frame()
        except Exception:
            # No queremos que un error de redibujo rompa el resize
            pass



# Backwards compatibility name
AFMLoaderWindow = AFMLoaderWidget
