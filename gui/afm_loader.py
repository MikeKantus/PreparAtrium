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
    QToolButton, QSplitter, QGroupBox, QLineEdit
)
from PySide6.QtGui import QPixmap, QImage, QIcon, QFont
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from core.ui_utils import frame_to_qimage_safe
from core.preloader_hsafm import preload_hsafm_folder

try:
    from playnano.io.loader import load_afm_stack
    HAS_PLAYNANO = True
    print("DEBUG: PlayNano importado correctamente")
except Exception as e:
    HAS_PLAYNANO = False
    print("DEBUG: PlayNano NO se pudo importar:", e)



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
        print("DEBUG ProcessingThread: stack type =", type(self.stack))
        print("DEBUG ProcessingThread: stack ndim =", getattr(self.stack, "ndim", None))
        print("DEBUG ProcessingThread: stack shape =", getattr(self.stack, "shape", None))
        print("DEBUG ProcessingThread: stack dtype =", getattr(self.stack, "dtype", None))
        stack = self.stack.copy()
        try:
            lo = np.percentile(stack, self.lo_pct)
            hi = np.percentile(stack, self.hi_pct)
            if hi <= lo:
                hi = lo + 1e-6
            stack = np.clip(stack, lo, hi)
        except Exception:
            pass

        self.finished.emit(stack)

class AFMLoaderWidget(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.speed_multiplier = 1.0
        self.setMinimumWidth(200)
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
        self.list_file_preview.setIconSize(QSize(64, 48))
        self.list_file_preview.setResizeMode(QListWidget.Adjust)
        self.list_file_preview.setMovement(QListWidget.Static)
        self.list_file_preview.setMaximumHeight(140)
        self.list_file_preview.setSpacing(6)
        self.list_file_preview.setSelectionMode(QListWidget.SingleSelection)
        self.list_file_preview.setSelectionBehavior(QListWidget.SelectItems)
        self.list_file_preview.setEditTriggers(QListWidget.NoEditTriggers)
        self.list_file_preview.setMouseTracking(True)

        self.list_file_preview.itemClicked.connect(self.preview_folder_contents)
        self.list_file_preview.itemDoubleClicked.connect(self.enter_folder)

       

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
        self.left_splitter.setSizes([800, 1200])
        self.left_splitter.setMaximumWidth(500)

        self.group_advanced = QGroupBox("Advanced Leveling / Flattening")
       


        # Now create the left_col layout and add the splitter and the leveling controls below
        left_col = QVBoxLayout()
        left_col.addWidget(self.left_splitter)
        left_col.addWidget(QLabel("Leveling"))
        left_col.addWidget(self.combo_level)
        left_col.addWidget(QLabel("Flatten"))
        left_col.addWidget(self.combo_flatten)
        left_col.addWidget(self.btn_apply)
        #left_col.addStretch()
     
        
        
        # --- Advanced Leveling / Flattening Controls ---
       
        layout_adv = QVBoxLayout()
        # Window size (nm)
        self.slider_window_nm = QSlider(Qt.Horizontal)
        self.slider_window_nm.setMinimum(1)
        self.slider_window_nm.setMaximum(50)
        self.slider_window_nm.setValue(5)
        layout_adv.addWidget(QLabel("Window size (nm)"))
        layout_adv.addWidget(self.slider_window_nm)

        # Step size (nm)
        self.slider_step_nm = QSlider(Qt.Horizontal)
        self.slider_step_nm.setMinimum(1)
        self.slider_step_nm.setMaximum(20)
        self.slider_step_nm.setValue(2)
        layout_adv.addWidget(QLabel("Step size (nm)"))
        layout_adv.addWidget(self.slider_step_nm)

        # Block size (px)
        self.slider_block_px = QSlider(Qt.Horizontal)
        self.slider_block_px.setMinimum(8)
        self.slider_block_px.setMaximum(256)
        self.slider_block_px.setValue(64)
        layout_adv.addWidget(QLabel("Block size (px)"))
        layout_adv.addWidget(self.slider_block_px)

        # Polynomial order
        self.slider_poly_order = QSlider(Qt.Horizontal)
        self.slider_poly_order.setMinimum(1)
        self.slider_poly_order.setMaximum(5)
        self.slider_poly_order.setValue(2)
        layout_adv.addWidget(QLabel("Polynomial order"))
        layout_adv.addWidget(self.slider_poly_order)

        # Smoothing sigma
        self.slider_smooth_sigma = QSlider(Qt.Horizontal)
        self.slider_smooth_sigma.setMinimum(0)
        self.slider_smooth_sigma.setMaximum(10)
        self.slider_smooth_sigma.setValue(0)
        layout_adv.addWidget(QLabel("Smoothing sigma"))
        layout_adv.addWidget(self.slider_smooth_sigma)

        # Iterations
        self.slider_iterations = QSlider(Qt.Horizontal)
        self.slider_iterations.setMinimum(1)
        self.slider_iterations.setMaximum(5)
        self.slider_iterations.setValue(1)
        layout_adv.addWidget(QLabel("Iterations"))
        layout_adv.addWidget(self.slider_iterations)

        # Apply advanced pipeline button
        self.btn_apply_advanced = QPushButton("Apply Advanced Leveling")
        layout_adv.addWidget(self.btn_apply_advanced)

        self.group_advanced.setLayout(layout_adv)
        left_col.addWidget(self.group_advanced)

        # Explorer connections
        self.btn_refresh_files.clicked.connect(lambda: self.populate_parent_combo(getattr(self, "current_file_or_folder", os.getcwd())))
        self.btn_apply_advanced.clicked.connect(lambda: self.apply_advanced_pipeline(self.processed_stack))
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
        speed_row = QHBoxLayout()
        self.speed_input = QLineEdit()
        self.speed_input.setPlaceholderText("Speed multiplier (e.g. 0.5, 1, 2)")
        self.speed_input.setText("1.0")

        self.btn_set_speed = QPushButton("Set speed")
        self.btn_set_speed.clicked.connect(self.update_speed)

        speed_row.addWidget(QLabel("Speed:"))
        speed_row.addWidget(self.speed_input)
        speed_row.addWidget(self.btn_set_speed)

        center_col.addLayout(speed_row)

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
            ("FPS", "real_fps"),
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
    def preview_folder_contents(self, item):
        path = item.data(Qt.UserRole)

        if not os.path.isdir(path):
            return

        # ⭐ PANEL INFERIOR: solo .jpk ⭐
        self.list_files.clear()
        jpk_files = []

        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isfile(full) and name.lower().endswith((".jpk", ".avi")):
                it = QListWidgetItem(name)
                it.setData(Qt.UserRole, full)
                self.list_files.addItem(it)
                jpk_files.append(full)

        # ⭐ thumbnails SOLO de archivos .jpk
        if jpk_files:
            self._populate_from_file_list(jpk_files)

        # Opcional: actualizar etiqueta de estado
        self.status_label.setText(f"Preview folder: {os.path.basename(path)}")


        # ⭐ thumbnails SOLO de archivos .jpk
        self._populate_from_file_list(jpk_files)

    def enter_folder(self, item):
        path = item.data(Qt.UserRole)

        if not os.path.isdir(path):
            return

        self.current_folder = path

        # ⭐ PANEL SUPERIOR: SOLO carpetas ⭐
        self.list_file_preview.clear()
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                it = QListWidgetItem(name)
                it.setData(Qt.UserRole, full)
                self.list_file_preview.addItem(it)

        # ⭐ PANEL INFERIOR: SOLO .jpk ⭐
        self.list_files.clear()
        jpk_files = []   # ← ← ← AÑADIR ESTO

        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isfile(full) and name.lower().endswith((".jpk",".avi")):
                it = QListWidgetItem(name)
                it.setData(Qt.UserRole, full)
                self.list_files.addItem(it)
                jpk_files.append(full)

        # thumbnails
        self._populate_from_file_list(jpk_files)


    def open_folder_or_files(self):
        # 1) Usuario selecciona carpeta HS-AFM
        folder = QFileDialog.getExistingDirectory(self, "Select folder with AFM files")

        if folder:
            self.status_label.setText(f"Preloading HS-AFM folder: {folder}")
            QApplication.processEvents()

            # Carpeta donde guardaremos TIFF + JSON
            output_folder = os.path.join(folder, "_preloaded")
            os.makedirs(output_folder, exist_ok=True)

            # Ejecutar preloader (TIFF + JSON)
            try:
                from core.preloader_hsafm import preload_hsafm_folder
                generated_tiffs = preload_hsafm_folder(folder, output_folder, self._read_metadata_jpk)
            except Exception as e:
                self.status_label.setText(f"Preloader error: {e}")
                return

            # Mostrar TIFFs en la interfaz
            self._populate_from_file_list(generated_tiffs)
            return

        # 2) Usuario selecciona archivos individuales (TIFF, AVI, etc.)
        filters = (
            "Preloaded AFM TIFF (*.tif);;"
            "Video files (*.avi *.mp4 *.mov);;"
            "HDF5 files (*.h5 *.hdf5);;"
            "NumPy files (*.npy *.npz);;"
            "All files (*)"
        )

        paths, _ = QFileDialog.getOpenFileNames(self, "Open AFM files", "", filters)
        if not paths:
            return

        # Filtrar fuera cualquier .jpk (ya no deben procesarse aquí)
        paths = [p for p in paths if not p.lower().endswith(".jpk")]

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
        paths = [p for p in paths if not p.lower().endswith(".jpk")]

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

        except Exception:
            pass
        return {}
    def _is_metadata_frame(self, frame):
        # Si los primeros bytes son ASCII → es metadatos
        flat = frame.ravel()
        return all(32 <= v <= 126 for v in flat[:20])

    def _parse_metadata_frame(self, frame):
        flat = frame.ravel()

        # Leer hasta encontrar un byte 0 (fin del JSON)
        end = np.where(flat == 0)[0]
        if len(end) > 0:
            end = end[0]
        else:
            end = len(flat)

        meta_bytes = bytes(flat[:end])
        try:
            meta_json = meta_bytes.decode("utf-8")
            return json.loads(meta_json)
        except:
            return {}
    def build_stack_from_tiffs(self, selected_tiffs):
        frames = []
        metas = []

        for tiff_path in selected_tiffs:
            frame, meta = self.load_tiff_with_metadata(tiff_path)
            frames.append(frame)
            metas.append(meta)

        stack = np.stack(frames)
        return stack, metas
    def load_single_jpk(path):
        from playnano.io.loader import load_afm_stack

        afm = load_afm_stack(path)   # carga un solo frame
        frames = afm.data            # shape (1, H, W)
        meta = {
            "pixel_size_nm": afm.pixel_size_nm,
            "channel": afm.channel,
            "frame_metadata": afm.frame_metadata,
        }
        print("DEBUG: resolución del stack =", frames.shape)

        return frames, meta
    def load_tiff_with_metadata(self, tiff_path):
        import tifffile
        frame = tifffile.imread(tiff_path)

        # Link TIFF → JPK
        jpk_path = tiff_path.replace(".tif", ".jpk")
        json_path = tiff_path.replace(".tif", ".json")

        # Prefer JSON metadata (faster)
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                meta = json.load(f)
        else:
            # Fallback: read metadata from JPK
            meta = self._read_metadata_jpk(jpk_path)

        return frame, meta

    def _read_file_to_frames(self, p):
        """
        Return (frames_array, file_meta). Robust extraction of common metadata keys.
        If playnano is available, try many attribute names. If HDF5, inspect attrs.
        """
        # --- LECTURA DE VIDEOS AVI CON METADATOS INCRUSTADOS ---
        ext = os.path.splitext(p)[1].lower()

        # Si es TIFF → cargar imagen + metadatos JSON/JPK
        if ext == ".jpk":
            raise ValueError("Direct JPK loading is disabled. Use TIFF+JSON preloader.")

        if ext == ".tif":
            frame, meta = self.load_tiff_with_metadata(p)
            return np.array([frame]), meta

        if p.lower().endswith(".avi"):
            cap = cv2.VideoCapture(p)
            ok, first_frame = cap.read()

            if not ok:
                raise ValueError("Cannot read AVI file")

            # Detectar si el primer frame es metadatos
            if self._is_metadata_frame(first_frame):
                meta = self._parse_metadata_frame(first_frame)

                # Leer frames reales
                frames = []
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    frames.append(f)

                cap.release()

                # Convertir a numpy
                frames = np.array(frames)

                # Devolver frames + metadatos reconstruidos
                return frames, meta

            else:
                # AVI normal sin metadatos incrustados
                frames = []
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    frames.append(f)

                cap.release()

                return np.array(frames), {"source_file": p}

            

        # Si hay muchos .jpk → es HS-AFM
        if ext == ".jpk" and HAS_PLAYNANO:
            print("DEBUG LOADER: detectado HS-AFM → cargando carpeta con PlayNano")
            afm = load_afm_stack(folder)
            frames = afm.data

            # Seleccionar solo el archivo que el usuario eligió
            sorted_files = sorted(jpk_files)
            idx = sorted_files.index(os.path.basename(p))
            frames = frames[idx:idx+1]

            meta = {
                "pixel_size_nm": afm.pixel_size_nm,
                "channel": afm.channel,
                "frame_metadata": [afm.frame_metadata[idx]],
            }

            print("DEBUG LOADER: resolución real:", frames.shape)
            return frames, meta   
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
            
            return frames, file_meta
            print("DEBUG channels:", afm.channels)
        raise ValueError("Unsupported file format or missing playnano/h5py")
        
    def _make_thumbnail(self, frame, thumb_w=160, thumb_h=160):
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
            if frame_rate is not None and x_pixels not in (None, 0):
                real_fps = float(frame_rate) / float(x_pixels)
        except Exception:
            real_fps = None

        self.meta_labels["real_fps"].setText(f"{real_fps:.3f}" if real_fps is not None else "-")


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

        # ⭐ 0) Resetear metadatos para evitar mezclas entre archivos
        self.meta = {}

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

        # ⭐ metadatos base del loader (solo del primer archivo)
        if meta_accum:
            for k, v in meta_accum.items():
                if v not in (None, "-", "unknown"):
                    self.meta[k] = v

        self.meta["total_frames"] = total_frames
        self.meta["source_files"] = sel_paths

        # ---------------------------------------------------------
        # 4) Metadatos extendidos usando read_metadata(path)
        # ---------------------------------------------------------
        try:
            extra_meta = self._read_metadata_jpk(sel_paths[0])  # solo del primer archivo
            if isinstance(extra_meta, dict):
                for k, v in extra_meta.items():
                    # no sobrescribir valores válidos del loader
                    if k not in self.meta or self.meta[k] in (None, "-", "unknown"):
                        self.meta[k] = v
        except Exception:
            pass  # silencioso

        # ---------------------------------------------------------
        # 5) Actualizar UI
        # ---------------------------------------------------------
        self.update_metadata_panel()
        self.populate_list()      # ajusta spin/slider
        self.update_preview()

        self.status_label.setText(
            f"Loaded {total_frames} frames from {len(sel_paths)} selected files"
        )
    def advanced_level_flatten(
        self,
        stack,
        meta,
        window_nm,
        step_nm,
        block_px,
        poly_order,
        smooth_sigma,
        iterations
    ):
        import numpy as np
        import cv2

        # ⭐ IMPORTAR AQUÍ (garantizado)
        from pnanolocz import leveling, flattening

        # Pixel size
        px_nm = meta.get("pixel_size_nm", None)
        if px_nm is None:
            px_nm = (meta.get("x_range_nm", 1000) / meta.get("x_pixels", 512))

        window_px = max(4, int(window_nm / px_nm))
        step_px = max(2, int(step_nm / px_nm))

        def local_plane_level(frame):
            h, w = frame.shape
            out = frame.copy()
            for y in range(0, h - window_px, step_px):
                for x in range(0, w - window_px, step_px):
                    block = frame[y:y+window_px, x:x+window_px]
                    leveled = leveling.plane_level(block)
                    out[y:y+window_px, x:x+window_px] = leveled
            return out

        def block_flatten(frame):
            h, w = frame.shape
            out = frame.copy()
            for y in range(0, h, block_px):
                for x in range(0, w, block_px):
                    block = frame[y:y+block_px, x:x+block_px]
                    flat = flattening.flatten_histogram(block)
                    out[y:y+block_px, x:x+block_px] = flat
            return out

        def polynomial_detrend(frame):
            yy, xx = np.indices(frame.shape)
            X = np.column_stack([
                np.ones_like(xx).ravel(),
                xx.ravel(), yy.ravel(),
                (xx*yy).ravel(),
                (xx**2).ravel(),
                (yy**2).ravel()
            ])[:, :poly_order+3]
            y = frame.ravel()
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            trend = (X @ coef).reshape(frame.shape)
            return frame - trend

        def smooth(frame):
            if smooth_sigma <= 0:
                return frame
            return cv2.GaussianBlur(frame, (0, 0), smooth_sigma)

        new_stack = stack.astype(np.float32).copy()

        for _ in range(iterations):
            for i in range(len(new_stack)):
                f = new_stack[i]
                f = local_plane_level(f)
                f = block_flatten(f)
                f = polynomial_detrend(f)
                f = smooth(f)
                new_stack[i] = f

        return new_stack


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
    def closeEvent(self, event):
        try:
            if hasattr(self, "_thread") and self._thread is not None:
                self._thread.quit()
                self._thread.wait()
        except Exception:
            pass
        event.accept()

    # -------------------------
    # Full processing (background)
    # -------------------------
    def apply_filters(self):
        # 1) Elegir stack de entrada
        base_stack = self.processed_stack if self.processed_stack is not None else self.raw_stack

        print("DEBUG apply_filters: processed_stack =", type(self.processed_stack), 
            getattr(self.processed_stack, "shape", None))

        print("DEBUG apply_filters: raw_stack =", type(self.raw_stack), 
            getattr(self.raw_stack, "shape", None))

        print("DEBUG apply_filters: base_stack =", type(base_stack), 
            getattr(base_stack, "shape", None))


        if base_stack is None or not isinstance(base_stack, np.ndarray):
            self.status_label.setText("No valid stack to filter.")
            return


        level_method = self.combo_level.currentText()
        flat_method = self.combo_flatten.currentText()
        lo_pct = self.slider_lower.value()
        hi_pct = self.slider_upper.value()

        self.status_label.setText("Applying basic filters...")
        QApplication.processEvents()

        self._thread = ProcessingThread(
            base_stack,
            level_method,
            flat_method,
            lo_pct,
            hi_pct
        )
        self._thread.finished.connect(self._after_basic_filters)
        self._thread.start()
        

    def _after_basic_filters(self, basic_stack):
        if basic_stack is None or not isinstance(basic_stack, np.ndarray):
            self.status_label.setText("Basic filter error: invalid stack")
            return

        self.processed_stack = basic_stack
        self.current_frame = 0
        self.update_preview()
        self.populate_list()

    def advanced_level_flatten(
        self,
        stack,
        meta,
        window_nm=5.0,
        step_nm=2.0,
        block_px=64,
        poly_order=2,
        smooth_sigma=0.0,
        iterations=1
    ):
        """
        Pipeline avanzado de leveling + flattening inspirado en NanoLocz.
        Todos los parámetros son configurables desde la interfaz.
        """
        import numpy as np
        import cv2

        # --- Pixel size ---
        px_nm = meta.get("pixel_size_nm", None)
        if px_nm is None:
            px_nm = (meta.get("x_range_nm", 1000) / meta.get("x_pixels", 512))

        # --- Convertir nm → px ---
        window_px = max(4, int(window_nm / px_nm))
        step_px = max(2, int(step_nm / px_nm))

        # --- Funciones internas ---
        def local_plane_level(frame):
            h, w = frame.shape
            out = frame.copy()

            for y in range(0, h - window_px, step_px):
                for x in range(0, w - window_px, step_px):
                    block = frame[y:y+window_px, x:x+window_px]
                    leveled = leveling.plane_level(block)
                    out[y:y+window_px, x:x+window_px] = leveled

            return out

        def block_flatten(frame):
            h, w = frame.shape
            out = frame.copy()

            for y in range(0, h, block_px):
                for x in range(0, w, block_px):
                    block = frame[y:y+block_px, x:x+block_px]
                    flat = flattening.flatten_histogram(block)
                    out[y:y+block_px, x:x+block_px] = flat

            return out

        def polynomial_detrend(frame):
            yy, xx = np.indices(frame.shape)
            X = np.column_stack([
                np.ones_like(xx).ravel(),
                xx.ravel(), yy.ravel(),
                (xx*yy).ravel(),
                (xx**2).ravel(),
                (yy**2).ravel()
            ])[:, :poly_order+3]

            y = frame.ravel()
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            trend = (X @ coef).reshape(frame.shape)
            return frame - trend

        def smooth(frame):
            if smooth_sigma <= 0:
                return frame
            return cv2.GaussianBlur(frame, (0, 0), smooth_sigma)

        # --- Pipeline ---
        new_stack = stack.astype(np.float32).copy()

        for _ in range(iterations):
            for i in range(len(new_stack)):
                f = new_stack[i]
                f = local_plane_level(f)
                f = block_flatten(f)
                f = polynomial_detrend(f)
                f = smooth(f)
                new_stack[i] = f

        return new_stack
        

    def _on_processing_finished(self, stack):
        self.processed_stack = stack
        self.progress.setVisible(False)
        self.populate_list()
        self.update_preview()
        #self.update_histogram()
        self.update_metadata_panel()
        self.status_label.setText("Filters applied")
    def run_advanced_pipeline(self, basic_stack):
        """
        Wrapper que recoge parámetros de la interfaz (cuando existan)
        y llama al pipeline avanzado.
        """

        # Cuando añadamos sliders, leeremos aquí:
        # window_nm = self.slider_window_nm.value()
        # step_nm = self.slider_step_nm.value()
        # block_px = self.slider_block_px.value()
        # poly_order = self.slider_poly_order.value()
        # smooth_sigma = self.slider_smooth.value()
        # iterations = self.slider_iterations.value()

        # Por ahora, valores por defecto:
        window_nm = 5.0
        step_nm = 2.0
        block_px = 64
        poly_order = 2
        smooth_sigma = 0.0
        iterations = 1

        return self.advanced_level_flatten(
            basic_stack,
            self.meta,
            window_nm,
            step_nm,
            block_px,
            poly_order,
            smooth_sigma,
            iterations
        )
    def apply_advanced_pipeline(self, stack=None):
        if stack is None:
            stack = self.processed_stack if self.processed_stack is not None else self.raw_stack

        if stack is None or not isinstance(stack, np.ndarray):
            self.status_label.setText("Advanced pipeline error: no valid stack loaded.")
            return None

        window_nm = self.slider_window_nm.value()
        step_nm = self.slider_step_nm.value()
        block_px = self.slider_block_px.value()
        poly_order = self.slider_poly_order.value()
        smooth_sigma = self.slider_smooth_sigma.value()
        iterations = self.slider_iterations.value()

        try:

            processed = self.advanced_level_flatten(
                stack,
                self.meta,
                window_nm,
                step_nm,
                block_px,
                poly_order,
                smooth_sigma,
                iterations
            )
        except Exception as e:
            self.status_label.setText(f"Advanced pipeline error: {e}")
            return None

        # Encadenar: actualizar processed_stack
        self.processed_stack = processed
        self.current_frame = 0
        self.update_preview()

        return processed

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

        arr = np.asarray(frame)

        if np.isnan(arr).any():
            arr = arr.copy()
            arr[np.isnan(arr)] = np.nanmin(arr)

        if arr.dtype != np.float32:
            arr_f = arr.astype(np.float32)
        else:
            arr_f = arr.copy()

        overlay_texts = []
        if self.checkbox_overlay.isChecked():
            fps = self.meta.get("frame_rate", None) or 10
            seconds = idx / float(fps)
            overlay_texts.append(f"{seconds:.2f} s")
        if self.checkbox_overlay_frame.isChecked():
            overlay_texts.append(f"Frame {idx}")
        if not overlay_texts:
            img8 = arr_f - np.nanmin(arr_f)
            rng = np.nanmax(img8)
            if rng == 0 or np.isnan(rng):
                rng = 1.0
            img8 = (img8 / rng * 255.0).astype(np.uint8)
            return np.ascontiguousarray(img8)

        lo_pct, hi_pct = 0.5, 99.5
        lo_v = np.percentile(arr_f, lo_pct)
        hi_v = np.percentile(arr_f, hi_pct)
        if hi_v <= lo_v:
            lo_v = np.nanmin(arr_f)
            hi_v = np.nanmax(arr_f)
            if hi_v <= lo_v:
                hi_v = lo_v + 1.0
        img_clip = np.clip(arr_f, lo_v, hi_v)

        img8 = ((img_clip - lo_v) / (hi_v - lo_v) * 255.0).astype(np.uint8)

        bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        text = " | ".join(overlay_texts)

        h, w = img8.shape[:2]

        # ⭐ Escala relativa al ancho, para tamaño visual consistente
        reference_width = 800  # ajusta este valor si quieres texto más grande/pequeño
        scale = w / reference_width
        scale = max(0.5, min(scale, 1.5))  # límites razonables

        thickness = max(1, int(round(scale * 2)))

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = int(round(6 * scale))

        x0, y0 = 8, 8
        rect_w = tw + pad * 2
        rect_h = th + pad * 2

        overlay = bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + rect_w, y0 + rect_h), (0, 0, 0), -1)
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0, bgr)

        text_x = x0 + pad
        text_y = y0 + pad + th
        cv2.putText(bgr, text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

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
       # Convertir a 2D si viene en RGB
        if frame_disp.ndim == 3:
            frame_disp = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2GRAY)

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
        fps = self.meta.get("real_fps") or self.meta.get("frame_rate", 10)
        interval = int(max(1, fps / self.speed_multiplier))
        self._timer.start(interval)
        self.status_label.setText("Playing")
    def update_speed(self):
        try:
            m = float(self.speed_input.text())
            if m <= 0:
                raise ValueError
            self.speed_multiplier = m
            self.status_label.setText(f"Speed multiplier set to {m}")
        except:
            self.status_label.setText("Invalid speed multiplier")


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
        print("DEBUG send_to_drift: stack type =", type(stack))
        print("DEBUG send_to_drift: stack shape =", getattr(stack, "shape", None))

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
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save video",
            "",
            "AVI Files (*.avi);;MP4 Files (*.mp4)"
        )

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
            meta_frame = self._make_metadata_frame()
            writer.write(meta_frame)
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
    def _make_metadata_frame(self):
        meta_json = json.dumps(self.meta)
        meta_bytes = meta_json.encode("utf-8")

        # Tamaño fijo del frame
        H, W = 200, 200
        frame = np.zeros((H, W), dtype=np.uint8)

        # Escribir bytes en los primeros píxeles
        flat = frame.ravel()
        for i, b in enumerate(meta_bytes):
            if i < flat.size:
                flat[i] = b

        return frame

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
