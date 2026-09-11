# gui/afm_loader.py
# (Este archivo es la versión corregida: histograma en panel aparte y sin "Open Drift Panel")

import os
import io
import json
import time
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py
import tifffile
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QListWidget,
    QFileDialog, QComboBox, QSlider, QProgressBar, QApplication,
    QSpinBox, QSizePolicy, QCheckBox, QListWidgetItem, QFrame, QGridLayout,
    QToolButton, QSplitter, QGroupBox, QLineEdit,
)
from PySide6.QtGui import QPixmap, QImage, QIcon, QFont
from AFMReader.asd import load_asd
from AFMReader.spm import load_spm
from PySide6.QtCore import Qt, QTimer, QSize
from core.ui_utils import frame_to_qimage_safe
from core.preloader_hsafm import preload_hsafm_folder
from core.afm_filters import despike_outliers, line_level, median_filter, plane_fit_subtract
from core.video_annotations import annotate_frame
from playnano.processing.filters import (
    remove_plane,
    row_median_align,
    zero_mean,
    polynomial_flatten,
    gaussian_filter,
    vertical_flip,
)

try:
    from playnano.io.loader import load_afm_stack
    HAS_PLAYNANO = True
except Exception as e:
    HAS_PLAYNANO = False
    print("DEBUG: PlayNano CANNOT be imported", e)



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


def _natural_path_key(path):
    """Sort frame files consistently, including numeric portions of their names."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", os.path.basename(path))]


def _z_scale_limit_nm(metadata):
    """Return a positive physical Z-scale limit from common AFM metadata fields."""
    metadata = metadata or {}
    for key in ("z_scale_max_nm", "z_scale_nm", "z_max_nm", "z_range_nm", "z_scale", "z_max"):
        try:
            value = float(metadata[key])
            if np.isfinite(value) and value > 0:
                return value
        except (KeyError, TypeError, ValueError):
            continue
    try:
        extension = float(metadata["z_piezo_extension"])
        gain = float(metadata.get("z_piezo_gain", 1.0))
        value = extension * gain
        return value if np.isfinite(value) and value > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def _persist_z_scale_metadata(metadata, frames, json_path):
    """Add physical Z limits from converted frames and persist them in the sidecar JSON."""
    values = np.asarray(frames, dtype=float)
    values = values[np.isfinite(values)]
    if values.size:
        metadata["z_scale_min_nm"] = float(np.min(values))
        metadata["z_scale_max_nm"] = float(np.max(values))
    with open(json_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return metadata


def _zero_baseline_per_frame(frames):
    """Shift each finite frame independently so its minimum height is zero."""
    normalized = np.asarray(frames, dtype=np.float32).copy()
    for frame in normalized:
        finite = np.isfinite(frame)
        if finite.any():
            frame[finite] -= np.min(frame[finite])
    return normalized
def normalize_meta_with_aliases(meta, aliases):
    normalized = {}
    for key, alias_list in aliases.items():
        for alias in alias_list:
            if alias in meta:
                normalized[key] = meta[alias]
                break
    return normalized

class AFMLoaderWidget(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.speed_multiplier = 1.0
        self.setMinimumWidth(200)
        self.original_stack = None
        self.processed_stack = None
        self.meta = {}
        self.current_file_or_folder = os.getcwd()
        self.current_frame = 0
        self.thumbnail_cache = {}
        import logging
        # Silenciar AFMReader completamente
        logging.getLogger("AFMReader").setLevel(logging.WARNING)
        logging.getLogger("jpk").setLevel(logging.WARNING)
        logging.getLogger("AFMReader.jpk").setLevel(logging.WARNING)


        # Equivalencias de metadatos entre formatos
        self.meta_aliases = {
            "num_imgs": ["num_imgs", "n_frames", "frames", "frame_count"],
            "pixel_size_nm": ["pixel_size_nm", "px_size_nm", "nm_per_pixel", "pixel_nm"],
            "x_range_nm": ["x_range_nm", "scan_size_x_nm", "range_x_nm", "x_nm"],
            "y_range_nm": ["y_range_nm", "scan_size_y_nm", "range_y_nm","y_nm"],
            "frame_rate": ["frame_rate", "line_rate_hz", "scan_rate_hz"],
            "real_fps": ["FPS", "real_fps", "fps","real_fps_asd"],
            "x_pixels": ["x_pixels", "x_num_pix", "width_px"],
            "y_pixels": ["y_pixels", "y_num_pix", "height_px"],
            "channel": ["channel", "mode", "signal"]
        }


        # fonts and sizes
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
        self.list_files.setIconSize(QSize(48, 48))
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
        self.combo_overlay_text_size = QComboBox()
        self.combo_overlay_text_size.addItems(["Small", "Medium", "Large"])
        self.combo_overlay_text_size.setCurrentText("Medium")
        self.combo_overlay_text_color = QComboBox()
        self.combo_overlay_text_color.addItems(["White", "Black", "Yellow", "Cyan", "Red"])
        self.combo_overlay_text_color.setCurrentText("White")
        self.checkbox_scale_bar = QCheckBox("Scale bar (1/5 width)")
        self.checkbox_scale_bar.setChecked(False)
        self.combo_scale_bar_color = QComboBox()
        self.combo_scale_bar_color.addItems(["White", "Black", "Yellow", "Cyan", "Red"])
        self.combo_scale_bar_color.setCurrentText("White")
        self.combo_color_palette = QComboBox()
        self.combo_color_palette.addItems(["Grayscale", "Viridis", "Plasma", "Turbo", "Hot"])
        self.z_grayscale_max_input = QLineEdit()
        self.z_grayscale_max_input.setPlaceholderText("Automatic")
        self.z_grayscale_max_input.setToolTip("Maximum height in nm shown by the grayscale")
        self.z_grayscale_max_input.editingFinished.connect(self._update_grayscale_z_max)

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
        
        #video stacks
        self.original_stack = None
        self.current_stack = None
        self.processed_stack = None

        # --- Advanced Leveling / Flattening Controls ---
       
        layout_adv = QVBoxLayout()
        self.chk_plane_level = QCheckBox("Robust plane leveling")
        self.chk_plane_level.setChecked(True)
        self.spin_plane_order = QSpinBox()
        self.spin_plane_order.setRange(1, 3)
        self.spin_plane_order.setValue(1)
        self.spin_plane_iterations = QSpinBox()
        self.spin_plane_iterations.setRange(1, 10)
        self.spin_plane_iterations.setValue(3)
        plane_layout = QHBoxLayout()
        plane_layout.addWidget(QLabel("Plane order"))
        plane_layout.addWidget(self.spin_plane_order)
        plane_layout.addWidget(QLabel("Robust passes"))
        plane_layout.addWidget(self.spin_plane_iterations)
        layout_adv.addWidget(self.chk_plane_level)
        layout_adv.addLayout(plane_layout)

        self.combo_line_level = QComboBox()
        self.combo_line_level.addItems(["No line leveling", "Median offset", "Median slope", "Mean offset", "Mean slope"])
        layout_adv.addWidget(QLabel("Line leveling"))
        layout_adv.addWidget(self.combo_line_level)

        self.chk_median_filter = QCheckBox("Median filter")
        self.spin_median_size = QSpinBox()
        self.spin_median_size.setRange(1, 15)
        self.spin_median_size.setSingleStep(2)
        self.spin_median_size.setValue(3)
        median_layout = QHBoxLayout()
        median_layout.addWidget(self.chk_median_filter)
        median_layout.addWidget(QLabel("Kernel"))
        median_layout.addWidget(self.spin_median_size)
        layout_adv.addLayout(median_layout)

        self.chk_despike = QCheckBox("Remove local spikes")
        self.spin_despike_sigma = QSpinBox()
        self.spin_despike_sigma.setRange(1, 10)
        self.spin_despike_sigma.setValue(3)
        self.spin_despike_neighborhood = QSpinBox()
        self.spin_despike_neighborhood.setRange(1, 7)
        self.spin_despike_neighborhood.setValue(3)
        despike_layout = QHBoxLayout()
        despike_layout.addWidget(self.chk_despike)
        despike_layout.addWidget(QLabel("Sigma"))
        despike_layout.addWidget(self.spin_despike_sigma)
        despike_layout.addWidget(QLabel("Radius"))
        despike_layout.addWidget(self.spin_despike_neighborhood)
        layout_adv.addLayout(despike_layout)

        # Apply advanced pipeline button, accept and restart
        self.btn_apply_advanced = QPushButton("Apply Advanced Leveling")
        self.btn_accept = QPushButton("Accept preview")
        self.btn_accept.setMinimumHeight(36)
        self.btn_accept.setFont(self.btn_font)
        self.btn_restart = QPushButton("Restart editing")
        self.btn_restart.setMinimumHeight(36)
        self.btn_restart.setFont(self.btn_font)
        row = QHBoxLayout()
        row.addWidget(self.btn_apply_advanced)
        row.addWidget(self.btn_accept)
        row.addWidget(self.btn_restart)
        layout_adv.addLayout(row)

        self.group_advanced.setLayout(layout_adv)
        left_col.addWidget(self.group_advanced)
        #Video accept and restart
        
        

        # Explorer connections
        self.btn_refresh_files.clicked.connect(lambda: self.populate_parent_combo(getattr(self, "current_file_or_folder", os.getcwd())))
        self.btn_apply_advanced.clicked.connect(lambda: self.apply_advanced_pipeline(self.current_stack))
        self.btn_open_in_explorer.clicked.connect(self.open_selected_folder_in_explorer)
        self.combo_parent_files.currentIndexChanged.connect(lambda idx: self.refresh_file_preview())
        self.btn_accept.clicked.connect(self.accept_preview)
        self.btn_restart.clicked.connect(self.restart_editing)
        for control in (self.checkbox_overlay, self.checkbox_overlay_frame, self.checkbox_scale_bar):
            control.toggled.connect(self.update_preview)
        self.combo_overlay_text_size.currentTextChanged.connect(self.update_preview)
        self.combo_overlay_text_color.currentTextChanged.connect(self.update_preview)
        self.combo_scale_bar_color.currentTextChanged.connect(self.update_preview)
        self.combo_color_palette.currentTextChanged.connect(self.update_preview)
        
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
        hist_controls.addWidget(QLabel("Overlay text size"))
        hist_controls.addWidget(self.combo_overlay_text_size)
        hist_controls.addWidget(QLabel("Overlay text color"))
        hist_controls.addWidget(self.combo_overlay_text_color)
        hist_controls.addWidget(self.checkbox_scale_bar)
        hist_controls.addWidget(QLabel("Scale bar color"))
        hist_controls.addWidget(self.combo_scale_bar_color)
        hist_controls.addWidget(QLabel("Video palette"))
        hist_controls.addWidget(self.combo_color_palette)
        hist_controls.addWidget(QLabel("Grayscale Z max (nm)"))
        hist_controls.addWidget(self.z_grayscale_max_input)
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
        self.speed_input.setPlaceholderText("Speed multiplier")
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
            ("Z min (nm)", "z_display_min_nm"),
            ("Z max (nm)", "z_display_max_nm"),
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

        # 1) JPK
        jpk_files = [f for f in os.listdir(path) if f.lower().endswith(".jpk")]
        if jpk_files:
            generated_tiffs = preload_hsafm_folder(path, path, self._read_metadata_jpk)
            self._populate_from_file_list(generated_tiffs)
            self.status_label.setText(f"Preview folder: {os.path.basename(path)}")
            return

                # --- ASD ---
       # --- ASD ---
        asd_files = [f for f in os.listdir(path) if f.lower().endswith(".asd")]
        if asd_files:
            generated_tiffs = []

            for fname in asd_files:
                full = os.path.join(path, fname)
                base = os.path.splitext(full)[0]
                out_json = base + ".json"

                # ------------------------------------------------------------
                # 1) Leer SOLO el header del ASD para obtener num_frames
                #    (sin cargar los frames → carga instantánea)
                # ------------------------------------------------------------
                try:
                    header_only = load_asd(full, channel="TP", read_frames=False)
                    header_meta = header_only[2] if isinstance(header_only, (list, tuple)) else {}
                    num_frames = header_meta.get("num_frames")
                except Exception:
                    num_frames = None

                # ------------------------------------------------------------
                # 2) Comprobar si ya existen TIFFs y JSON
                # ------------------------------------------------------------
                tiffs = sorted([
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.startswith(os.path.basename(base) + "_frame") and f.lower().endswith(".tif")
                ])

                if os.path.exists(out_json) and num_frames is not None and len(tiffs) == num_frames:
                    # Ya está todo generado → usar TIFFs existentes
                    generated_tiffs.extend(tiffs)
                    continue

                # ------------------------------------------------------------
                # 3) Cargar ASD COMPLETO solo si faltan TIFFs
                # ------------------------------------------------------------
                result = load_asd(full, channel="TP")
                frames = result[0]
                meta   = result[2]

                # 🔵 Normalizar metadatos usando alias del pipeline
                meta = normalize_meta_with_aliases(meta, self.meta_aliases)

                # 🔵 Completar metadatos faltantes
                # FPS
                if "real_fps" not in meta:
                    if "frame_time" in meta:
                        meta["real_fps"] = 1000.0 / float(meta["frame_time"])
                    elif "frame_time_ms" in meta:
                        meta["real_fps"] = 1000.0 / float(meta["frame_time_ms"])
                    elif "frame_time_s" in meta:
                        meta["real_fps"] = 1.0 / float(meta["frame_time_s"])
                    elif "fps" in meta:
                        meta["real_fps"] = float(meta["fps"])

                # Rango X/Y
                if "x_range_nm" not in meta and "x_nm" in meta:
                    meta["x_range_nm"] = float(meta["x_nm"])
                if "y_range_nm" not in meta and "y_nm" in meta:
                    meta["y_range_nm"] = float(meta["y_nm"])

                # Pixel size
                if "pixel_size_nm" not in meta:
                    if "x_range_nm" in meta and "x_pixels" in meta:
                        meta["pixel_size_nm"] = meta["x_range_nm"] / meta["x_pixels"]

                # Canal
                if "channel" not in meta:
                    meta["channel"] = "TP"

                # Asegurar frames 3D
                if frames.ndim == 2:
                    frames = frames[np.newaxis, ...]

                # ------------------------------------------------------------
                # 4) Guardar TIFFs
                # ------------------------------------------------------------
                for i, frame in enumerate(frames):
                    tif_path = f"{base}_frame{i:04d}.tif"
                    tifffile.imwrite(tif_path, frame.astype(np.float32))
                    generated_tiffs.append(tif_path)

                # ------------------------------------------------------------
                # 5) Guardar JSON global
                # ------------------------------------------------------------
                with open(out_json, "w") as f:
                    json.dump(meta, f, indent=2)

            # ------------------------------------------------------------
            # 6) Mostrar TIFFs en el panel
            # ------------------------------------------------------------
            self._populate_from_file_list(generated_tiffs)
            self.status_label.setText(f"Preview folder: {os.path.basename(path)}")
            return



        # 3) STP
        stp_files = [f for f in os.listdir(path) if f.lower().endswith((".stp", ".spm", ".stm"))]
        if stp_files:
            generated_tiffs = []
            for fname in stp_files:
                full = os.path.join(path, fname)
                base = os.path.splitext(full)[0]
                out_json = base + ".json"

                if os.path.exists(out_json):
                    with open(out_json, "r", encoding="utf-8") as metadata_file:
                        meta = json.load(metadata_file)
                    tiffs = sorted([
                        os.path.join(path, f)
                        for f in os.listdir(path)
                        if f.startswith(os.path.basename(base) + "_frame") and f.lower().endswith(".tif")
                    ])
                    if len(tiffs) == 1:
                        if "z_scale_max_nm" not in meta:
                            meta = _persist_z_scale_metadata(meta, tifffile.imread(tiffs[0]), out_json)
                        generated_tiffs.extend(tiffs)
                        continue

                image, px_nm = load_spm(full, channel="Height")
                frames = image[np.newaxis, :]

                meta = {
                    "pixel_size_nm": px_nm,
                    "x_pixels": frames.shape[2],
                    "y_pixels": frames.shape[1],
                    "x_range_nm": frames.shape[2] * px_nm,
                    "y_range_nm": frames.shape[1] * px_nm,
                    "frame_rate": None,
                    "channel": "Height",
                    "num_imgs": frames.shape[0]
                }
                meta = _persist_z_scale_metadata(meta, frames, out_json)

                for i, frame in enumerate(frames):
                    tif_path = f"{base}_frame{i}.tif"
                    if not os.path.exists(tif_path):
                        tifffile.imwrite(tif_path, frame.astype(np.float32))
                    generated_tiffs.append(tif_path)

            self._populate_from_file_list(generated_tiffs)
            self.status_label.setText(f"Preview folder: {os.path.basename(path)}")
            return

        # 4) Normal TIFF/AVI
        self.list_files.clear()
        tif_files = []
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if name.lower().endswith((".tif", ".avi")):
                it = QListWidgetItem(name)
                it.setData(Qt.UserRole, full)
                self.list_files.addItem(it)
                tif_files.append(full)

        if tif_files:
            self._populate_from_file_list(tif_files)

        self.status_label.setText(f"Preview folder: {os.path.basename(path)}")


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

        # ⭐ PANEL INFERIOR: SOLO TIFF y AVI ⭐
        self.list_files.clear()
        tif_files = []

        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isfile(full) and name.lower().endswith((".tif", ".avi")):
                it = QListWidgetItem(name)
                it.setData(Qt.UserRole, full)
                self.list_files.addItem(it)
                tif_files.append(full)

        # thumbnails
        self._populate_from_file_list(tif_files)

    def open_folder_or_files(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with AFM files")

        if folder:
            self.status_label.setText(f"Preloading HS-AFM folder: {folder}")
            QApplication.processEvents()

            # Carpeta donde guardaremos TIFF + JSON
            output_folder = os.path.join(folder, "_preloaded")
            os.makedirs(output_folder, exist_ok=True)

            # Ejecutar preloader
            try:
                generated_tiffs = preload_hsafm_folder(folder, output_folder, self._read_metadata_jpk)
            except Exception as e:
                self.status_label.setText(f"Preloader error: {e}")
                return

            # ⭐ CAMBIAR CARPETA ACTUAL A _preloaded ⭐
            self.current_folder = output_folder

            # ⭐ REFRESCAR EXPLORADOR PARA MOSTRAR TIFF ⭐
            self.populate_parent_combo(output_folder)

            # ⭐ MOSTRAR TIFF EN PANEL INFERIOR ⭐
            self._populate_from_file_list(generated_tiffs)

            return

        # Si el usuario selecciona archivos individuales
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

        paths = [p for p in paths if not p.lower().endswith(".jpk")]

        self.status_label.setText("Loading selected files...")
        QApplication.processEvents()

        self._populate_from_file_list(paths)


   
    def _populate_from_folder(self, folder):
        valid_exts = (".tif", ".avi", ".mp4", ".mov", ".h5", ".hdf5", ".npy", ".npz", ".asd", ".stp", ".stm")
        files = []
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(valid_exts):
                files.append(os.path.join(folder, fname))
        if not files:
            self.status_label.setText("No AFM files found in folder.")
            return
        self._populate_from_file_list(files)

    def _populate_from_file_list(self, paths):
        paths = [p for p in paths if not p.lower().endswith(".jpk")]
        previous_signal_state = self.list_files.blockSignals(True)
        self.list_files.clear()
        self._file_index = []
        self.meta = {}   # reiniciar metadatos para nueva selección
    
        # Nombre base del vídeo original (primer archivo de la lista)
        if paths:
            first = paths[0]
            self.meta["source_name"] = os.path.splitext(os.path.basename(first))[0]
    
        for p in paths:
            try:
                # -----------------------------
                # 1) Thumbnail rápido con caché
                # -----------------------------
                if p not in self.thumbnail_cache:
                    try:
                        thumb = tifffile.imread(p, key=0)
                        thumb_small = cv2.resize(thumb, (48, 48))
                        self.thumbnail_cache[p] = thumb_small
                    except Exception:
                        self.thumbnail_cache[p] = np.zeros((48, 48), dtype=np.uint8)
    
                thumb_small = self.thumbnail_cache[p]
                qimg = numpy_to_qimage(thumb_small)
    
                # Crear item de la lista
                item = QListWidgetItem(QIcon(QPixmap.fromImage(qimg)), os.path.basename(p))
                item.setData(Qt.UserRole, p)
                self.list_files.addItem(item)
                self._file_index.append(p)
    
                # -----------------------------
                # 2) Cargar metadatos desde JSON global
                # -----------------------------
                base = os.path.splitext(p)[0]
                json_guess = base.split("_frame")[0] + ".json"
    
                if os.path.exists(json_guess):
                    try:
                        with open(json_guess, "r") as f:
                            asd_meta = json.load(f)
    
                        for panel_key in self.meta_aliases.keys():
                            val = self.resolve_meta_value(asd_meta, panel_key)
                            if val is not None:
                                self.meta[panel_key] = val
    
                    except Exception as e:
                        print("DEBUG: error reading JSON ASD:", e)
    
                # -----------------------------
                # 3) Metadatos extendidos (JPK/STP/etc)
                # -----------------------------
                extra_meta = self._read_metadata_jpk(p)
                if isinstance(extra_meta, dict):
                    for panel_key in self.meta_aliases.keys():
                        val = self.resolve_meta_value(extra_meta, panel_key)
                        if val is not None:
                            self.meta[panel_key] = val
    
            except Exception as e:
                item = QListWidgetItem(f"{os.path.basename(p)}  —  ERROR: {e}")
                item.setData(Qt.UserRole, p)
                self.list_files.addItem(item)
    
        # -----------------------------
        # 4) Actualizar panel de metadatos
        # -----------------------------
        self.list_files.blockSignals(previous_signal_state)
        self.update_metadata_panel()
        self.status_label.setText(
            f"Found {len(self._file_index)} files. Select one or more to build the video."
        )

  
    
    
    def resolve_meta_value(self, meta_dict, key):
        """
        Devuelve el valor del metadato 'key' buscando en todas sus equivalencias.
        """
        aliases = self.meta_aliases.get(key, [key])
        for name in aliases:
            if name in meta_dict:
                return meta_dict[name]
        return None
    def _read_metadata_jpk(self, path):
        """
        Lee metadatos de archivos JPK, ASD, STP/SPM y TIFF generados.
        Devuelve SIEMPRE un diccionario.
        """
    
        meta = {}
    
        path_lower = path.lower()
        base = os.path.splitext(path)[0]
        json_path = base + ".json"
    
        # 1) Si existe JSON asociado → usarlo directamente
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print("DEBUG error reading JSON:", e)
    
        # 2) STP/SPM → JSON generado en preview_folder_contents
        if path_lower.endswith((".stp", ".spm", ".stm")):
            return {}
    
        # 3) ASD → JSON generado en preview_folder_contents
        if path_lower.endswith(".asd"):
            return {}
    
        # 4) TIFF normal → intentar leer JSON global
        if path_lower.endswith(".tif"):
            out_json = base + ".json"
            if os.path.exists(out_json):
                try:
                    with open(out_json, "r") as f:
                        return json.load(f)
                except Exception as e:
                    print("DEBUG error leyendo JSON:", e)
                    return {}
            return {}
    
        # 5) JPK → lógica completa original
        if path_lower.endswith(".jpk"):
            scan_fields = {
                "x_origin_nm": 32832,
                "y_origin_nm": 32833,
                "x_range_nm": 32834,
                "y_range_nm": 32835,
                "x_pixels": 32838,
                "y_pixels": 32839,
                "frame_rate": 32841,
            }
    
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
                    if key.startswith("cantilever-calibration-info."):
                        short = key.replace("cantilever-calibration-info.", "")
                        if short in cantilever_keys:
                            cantilever[short] = value
                    if key.startswith("feedback-mode.setpoint-feedback-settings."):
                        short = key.replace("feedback-mode.setpoint-feedback-settings.", "")
                        full_key = f"setpoint-feedback-settings.{short}"
                        if full_key in feedback_keys:
                            feedback[full_key] = value
                return cantilever, feedback
    
            try:
                with tifffile.TiffFile(path) as tif:
                    scan = {}
                    cantilever = {}
                    feedback = {}
    
                    for page in tif.pages:
                        tags = {tag.code: tag.value for tag in page.tags.values()}
    
                        if not scan:
                            scan = extract_scan(tags)
    
                        for value in tags.values():
                            if isinstance(value, str) and "cantilever-calibration-info" in value:
                                c, f = extract_cantilever_and_feedback(value)
                                cantilever.update(c)
                                feedback.update(f)
    
                    meta.update(scan)
                    meta.update(cantilever)
                    meta.update(feedback)
    
                    if "channel" not in meta:
                        meta["channel"] = None
    
                    return meta
    
            except Exception:
                return {}
    
        # 6) Otros formatos → sin metadatos
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
    def load_tiff_with_metadata(self, tiff_path):
        frame = tifffile.imread(tiff_path)
    
        jpk_path = tiff_path.replace(".tif", ".jpk")
        json_path = tiff_path.replace(".tif", ".json")
    
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                meta = json.load(f)
        else:
            meta = self._read_metadata_jpk(jpk_path)
    
        meta = dict(meta or {})
    
        if os.path.isfile(jpk_path):
            meta["_source_format"] = "jpk"
    
        frame_base = os.path.splitext(tiff_path)[0]
        if "_frame" in frame_base and os.path.isfile(frame_base.rsplit("_frame", 1)[0] + ".asd"):
            meta["_source_format"] = "asd"
    
        return frame, meta

    def _read_file_to_frames(self, p):
        """
        Return (frames_array, file_meta). Robust extraction of common metadata keys.
        If playnano is available, try many attribute names. If HDF5, inspect attrs.
        """
        # --- LECTURA DE VIDEOS AVI CON METADATOS INCRUSTADOS ---
        ext = os.path.splitext(p)[1].lower()
        # ------------------------------------------------------------
        # STP / SPM files (Bruker) — MODE A: TIFF PER FRAME
        # ------------------------------------------------------------
        if p.lower().endswith((".stp", ".spm", ".stm")):
            from AFMReader.stp import load_spm

            base = os.path.splitext(p)[0]
            out_json = base + ".json"

            # Si ya existe JSON y TIFF → no reprocesar
            if os.path.exists(out_json):
                with open(out_json, "r") as f:
                    meta = json.load(f)

                tiffs = sorted([f for f in os.listdir(os.path.dirname(p))
                                if f.startswith(os.path.basename(base) + "_frame") and f.lower().endswith(".tif")])
                if len(tiffs) == 1:
                    frames = [tifffile.imread(os.path.join(os.path.dirname(p), t)) for t in tiffs]
                    frames = np.stack(frames, axis=0)
                    if "z_scale_max_nm" not in meta:
                        meta = _persist_z_scale_metadata(meta, frames, out_json)
                    return frames, meta

            # Decodificar STP
            image, px_nm = load_spm(p, channel="Height")
            frames = image[np.newaxis, ...]

            meta = {
                "pixel_size_nm": px_nm,
                "x_pixels": frames.shape[2],
                "y_pixels": frames.shape[1],
                "x_range_nm": frames.shape[2] * px_nm,
                "y_range_nm": frames.shape[1] * px_nm,
                "frame_rate": None,
                "channel": "Height",
                "num_imgs": frames.shape[0]
            }
            meta = _persist_z_scale_metadata(meta, frames, out_json)

            # Guardar TIFF por frame
            for i, frame in enumerate(frames):
                tif_path = f"{base}_frame{i}.tif"
                if not os.path.exists(tif_path):
                    tifffile.imwrite(tif_path, frame.astype(np.float32))

            return frames, meta


        # ------------------------------------------------------------
        # ASD files (Asylum Research) — MODE A: TIFF PER FRAME
        # ------------------------------------------------------------
        if p.lower().endswith(".asd"):
            from AFMReader.asd import load_asd


            base = os.path.splitext(p)[0]
            out_json = base + ".json"

            # Si ya existe JSON y TIFFs → no reprocesar
            if os.path.exists(out_json):
                # cargar metadatos
                with open(out_json, "r") as f:
                    meta = json.load(f)

                # cargar todos los TIFF generados
                tiffs = sorted([f for f in os.listdir(os.path.dirname(p))
                                if f.startswith(os.path.basename(base) + "_frame") and f.lower().endswith(".tif")])
                expected_frames = meta.get("num_imgs")
                if tiffs and (expected_frames is None or len(tiffs) == expected_frames):
                    frames = [tifffile.imread(os.path.join(os.path.dirname(p), t)) for t in tiffs]
                    meta = dict(meta)
                    meta["_source_format"] = "asd"
                    return np.stack(frames, axis=0), meta

            # Decodificar ASD
            obj = load_asd(p)
            frames = obj.data
            meta = obj.metadata or {}

            # Normalizar
            if frames.ndim == 2:
                frames = frames[np.newaxis, ...]

            # Guardar TIFF por frame
            for i, frame in enumerate(frames):
                tif_path = f"{base}_frame{i:04d}.tif"
                if not os.path.exists(tif_path):
                    tifffile.imwrite(tif_path, frame.astype(np.float32))

            # Guardar metadatos globales
            with open(out_json, "w") as f:
                json.dump(meta, f, indent=2)

            meta = dict(meta)
            meta["_source_format"] = "asd"
            return frames, meta

        # TIFF: load the image and its JSON/JPK metadata.
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
        try:

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

                def pick_first(values, keys):
                    for key in keys:
                        if key in values:
                            value = values[key]
                            if isinstance(value, bytes):
                                value = value.decode("utf-8", errors="replace")
                            return value
                    return None

                # Pick known metadata keys.
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

    def update_metadata_json(path, extra_meta, read_metadata_func):
        """
        Fuse new external metadata with the extracted metadata. 
        Merge new external metadata with the existing metadata (JSON + hardware).
        - path: Path to the original data file (TIFF/JPK/etc.)
        - extra_meta: Dictionary containing new metadata to merge (e.g., uv_state, uv_on_frame)
        - read_metadata_func: the function _read_metadata_jpk
        """
            json_path = path.with_suffix(".json")
        
            # 1) Leer metadatos existentes (JSON o hardware)
            base_meta = read_metadata_func(str(path))

            # 2) Fusionar con los nuevos
            merged = {**base_meta, **extra_meta}

            # 3) Guardar JSON actualizado
            try:
                with open(json_path, "w") as f:
                    json.dump(merged, f, indent=4)
            except Exception as e:
                print("DEBUG: error writing merged metadata:", e)

            return merged

    def update_metadata_panel(self):
        # --- Num Imgs ---
        num_imgs = self.resolve_meta_value(self.meta, "num_imgs")
        if num_imgs is None and self.original_stack is not None:
            num_imgs = self.original_stack.shape[0]
        self.meta_labels["num_imgs"].setText(str(num_imgs) if num_imgs is not None else "-")

        # --- X/Y pixels ---
        x_pixels = self.resolve_meta_value(self.meta, "x_pixels")
        y_pixels = self.resolve_meta_value(self.meta, "y_pixels")

        if (x_pixels is None or y_pixels is None) and self.original_stack is not None:
            y_pixels = self.original_stack.shape[-2]
            x_pixels = self.original_stack.shape[-1]

        self.meta_labels["x_pixels"].setText(str(x_pixels) if x_pixels is not None else "-")
        self.meta_labels["y_pixels"].setText(str(y_pixels) if y_pixels is not None else "-")

        # --- FPS reales ---
        real_fps = self.resolve_meta_value(self.meta, "real_fps")
        frame_rate = self.resolve_meta_value(self.meta, "frame_rate")

        # fallback: derive FPS if possible
        if real_fps is None and frame_rate is not None:
            try:
                real_fps = float(frame_rate)
            except Exception:
                real_fps = None

        self.meta_labels["real_fps"].setText(f"{real_fps:.3f}" if real_fps is not None else "-")

        # --- X-Range (nm) ---
        x_range = self.resolve_meta_value(self.meta, "x_range_nm")
        self.meta_labels["x_range_nm"].setText(str(x_range) if x_range is not None else "-")

        # --- Z min / Z max ---
        z_min = self.resolve_meta_value(self.meta, "z_display_min_nm") or \
                self.resolve_meta_value(self.meta, "z_data_min_nm")
        z_max = self.resolve_meta_value(self.meta, "z_display_max_nm") or \
                self.resolve_meta_value(self.meta, "z_data_max_nm")

        self.meta_labels["z_display_min_nm"].setText(
            f"{float(z_min):.6g}" if z_min is not None else "-"
        )
        self.meta_labels["z_display_max_nm"].setText(
            f"{float(z_max):.6g}" if z_max is not None else "-"
        )

        # --- Frame rate (raw) ---
        self.meta_labels["frame_rate"].setText(str(frame_rate) if frame_rate is not None else "-")

        # --- Channel ---
        channel = self.resolve_meta_value(self.meta, "channel")
        self.meta_labels["channel"].setText(str(channel) if channel not in (None, "unknown") else "-")

        # --- Pixel size (nm/pixel) ---
        pixel_size = self.resolve_meta_value(self.meta, "pixel_size_nm")
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
        if self.original_stack is None:
            return
        n = len(self.original_stack)
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
        selected_items = self.list_files.selectedItems()
        if not selected_items:
            return
    
        sel_paths = sorted(
            (it.data(Qt.UserRole) for it in selected_items if it.data(Qt.UserRole)),
            key=_natural_path_key,
        )
        if not sel_paths:
            self.status_label.setText("No valid files selected.")
            return
    
        all_frames = []
        total_frames = 0
        file_metas = []
    
        for p in sel_paths:
            try:
                img, file_meta = self._read_file_to_frames(p)
    
                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                elif img.ndim != 3:
                    raise ValueError(f"Invalid TIFF shape: {img.shape}")
    
                if (file_meta or {}).get("_source_format") in ("asd", "jpk"):
                    img = _zero_baseline_per_frame(img)
    
                all_frames.append(img)
                file_metas.append(file_meta or {})
                total_frames += img.shape[0]
    
            except Exception as e:
                self.status_label.setText(f"Error loading {os.path.basename(p)}: {e}")
                return
    
        try:
            new_stack = np.concatenate(all_frames, axis=0)
        except Exception as e:
            self.status_label.setText(f"Error concatenating selected frames: {e}")
            return
    
        self.original_stack = new_stack.astype(np.float32)
        self.current_stack = self.original_stack.copy()
        self.processed_stack = None
        self.current_frame = 0
    
        meta_json = file_metas[0] if file_metas else {}
        for panel_key in self.meta_aliases.keys():
            val = self.resolve_meta_value(meta_json, panel_key)
            if val is not None:
                self.meta[panel_key] = val
    
        self.meta["total_frames"] = total_frames
        self.meta["source_files"] = sel_paths
    
        # Derive real_fps generically
        real_fps = self.resolve_meta_value(self.meta, "real_fps")
        frame_rate = self.resolve_meta_value(self.meta, "frame_rate")
        x_pixels = self.resolve_meta_value(self.meta, "x_pixels")
        y_pixels = self.resolve_meta_value(self.meta, "y_pixels")
    
        if real_fps is None:
            try:
                if frame_rate is not None and y_pixels not in (None, 0):
                    self.meta["real_fps"] = float(frame_rate) / float(y_pixels)
                elif frame_rate is not None and x_pixels not in (None, 0):
                    self.meta["real_fps"] = float(frame_rate) / float(x_pixels)
                else:
                    fps_alias = self.resolve_meta_value(self.meta, "real_fps")
                    if fps_alias is not None:
                        self.meta["real_fps"] = float(fps_alias)
            except Exception:
                pass
    
        finite_values = new_stack[np.isfinite(new_stack)]
        if finite_values.size:
            stack_min = float(np.min(finite_values))
            observed_max = float(np.max(finite_values))
            self.meta["z_data_min_nm"] = stack_min
            self.meta["z_data_max_nm"] = observed_max
    
        self.spin_frame.setMaximum(len(self.current_stack) - 1)
        self.slider_time.setMaximum(len(self.current_stack) - 1)
        self.spin_frame.setValue(0)
        self.slider_time.setValue(0)
    
        self.update_preview()
        self.update_metadata_panel()
    
        self.status_label.setText(
            f"Loaded {total_frames} frames from {len(sel_paths)} selected files"
        )

    # -------------------------
    # Histogram preview sliders
    # -------------------------
    def _set_grayscale_z_max_text(self):
        value = self.meta.get("z_display_max_nm")
        self.z_grayscale_max_input.blockSignals(True)
        self.z_grayscale_max_input.setText(f"{float(value):.6g}" if value is not None else "")
        self.z_grayscale_max_input.blockSignals(False)

    def _update_grayscale_z_max(self):
        text = self.z_grayscale_max_input.text().strip()
        if not text:
            automatic_max = self.meta.get("z_auto_display_max_nm")
            if automatic_max is not None:
                self.meta["z_display_max_nm"] = automatic_max
                self._set_grayscale_z_max_text()
            self.update_preview()
            return
        try:
            value = float(text)
            if not np.isfinite(value) or value <= 0:
                raise ValueError
        except ValueError:
            self.status_label.setText("Grayscale Z max must be a positive number in nm")
            self._set_grayscale_z_max_text()
            return
        self.meta["z_display_max_nm"] = value
        self.update_preview()
        self.update_metadata_panel()

    def on_histogram_slider_changed(self, _val=None):
        base = self.current_stack if self.current_stack is not None else self.original_stack
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
            preview_stack = np.clip(base, lo, hi).astype(np.float32)
            self.processed_stack = preview_stack
            idx = max(0, min(self.current_frame, len(preview_stack) - 1))
            frame = preview_stack[idx]
            frame_disp = self._overlay_frame(frame, idx)
            rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
            qimg = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0],
                          rgb_frame.strides[0], QImage.Format_RGB888).copy()
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
        if self.original_stack is None:
            self.status_label.setText("No stack loaded.")
            return

        stack = (self.current_stack if self.current_stack is not None else self.original_stack).copy().astype(np.float64)

        level_method = self.combo_level.currentText()
        flat_method = self.combo_flatten.currentText()

        # Aplicar filtros frame‑por‑frame
        for i in range(stack.shape[0]):
            frame = stack[i]

            # LEVELING
            if level_method == "Plane":
                frame = remove_plane(frame)
            elif level_method == "Line":
                frame = row_median_align(frame)

            # FLATTEN
            flat_method = self.combo_flatten.currentText()
            if flat_method == "Histogram":
                low = self.slider_lower.value()
                high = self.slider_upper.value()
                if high <= low:
                    self.status_label.setText("Histogram upper must be > lower")
                    return
                low_value, high_value = np.percentile(frame, [low, high])
                frame = np.clip(frame, low_value, high_value)
            elif flat_method == "Polynomial":
                frame = polynomial_flatten(frame, order=2)


            stack[i] = frame

        self.processed_stack = stack.astype(np.float32)
        self.update_preview()
        self.status_label.setText("Basic filters applied using PlayNano filters.py.")

       

    def _after_basic_filters(self, basic_stack):
        if basic_stack is None or not isinstance(basic_stack, np.ndarray):
            self.status_label.setText("Basic filter error: invalid stack")
            return

        self.processed_stack = basic_stack
        self.current_frame = 0
        self.update_preview()
        self.populate_list()

    def accept_preview(self):
        if self.processed_stack is None:
            self.status_label.setText("No processed stack to accept.")
            return

        self.current_stack = self.processed_stack.copy()
        self.status_label.setText("Preview accepted. Current stack updated.")
    def restart_editing(self):
        if self.original_stack is None:
            self.status_label.setText("No original stack loaded.")
            return

        self.current_stack = self.original_stack.copy()
        self.processed_stack = self.current_stack.copy()
        self.update_preview()
        self.status_label.setText("Editing restarted. Current stack reset.")

    def _on_processing_finished(self, stack):
        self.processed_stack = stack
        self.progress.setVisible(False)
        self.populate_list()
        self.update_preview()
        #self.update_histogram()
        self.update_metadata_panel()
        self.status_label.setText("Filters applied")
    def apply_advanced_pipeline(self, base_stack):
        if base_stack is None:
            self.status_label.setText("No stack loaded.")
            return

        stack = base_stack.copy().astype(np.float64)

        line_options = {
            "Median offset": ("median", "offset"),
            "Median slope": ("median", "slope"),
            "Mean offset": ("mean", "offset"),
            "Mean slope": ("mean", "slope"),
        }
        line_setting = line_options.get(self.combo_line_level.currentText())

        for index, frame in enumerate(stack):
            if self.chk_plane_level.isChecked():
                frame = plane_fit_subtract(
                    frame,
                    order=self.spin_plane_order.value(),
                    robust=True,
                    iters=self.spin_plane_iterations.value(),
                )
            if line_setting is not None:
                frame = line_level(frame, method=line_setting[0], fit=line_setting[1])
            if self.chk_median_filter.isChecked():
                frame = median_filter(frame, size=self.spin_median_size.value())
            if self.chk_despike.isChecked():
                frame, _ = despike_outliers(
                    frame,
                    k_sigma=float(self.spin_despike_sigma.value()),
                    neigh=self.spin_despike_neighborhood.value(),
                )
            stack[index] = frame

        self.processed_stack = stack.astype(np.float32)
        self.update_preview()
        self.status_label.setText("Advanced AFM filters applied.")

    # -------------------------
    # Overlay and preview helpers
    # -------------------------
    def _overlay_frame(self, frame, idx):
        return annotate_frame(
            frame, idx, self.meta,
            show_timestamp=self.checkbox_overlay.isChecked(),
            show_frame_number=self.checkbox_overlay_frame.isChecked(),
            text_size=self.combo_overlay_text_size.currentText(),
            text_color=self.combo_overlay_text_color.currentText(),
            show_scale_bar=self.checkbox_scale_bar.isChecked(),
            scale_bar_color=self.combo_scale_bar_color.currentText(),
            color_palette=self.combo_color_palette.currentText(),
        )




    def update_preview(self):
        """
        Mostrar el frame actual usando frame_to_qimage_safe.
        Llamar a esta función después de actualizar self.current_frame,
        self.processed_stack o self.original_stack.
        """
        base = self.processed_stack if self.processed_stack is not None else self.current_stack
        if base is None:
            self.label_preview.clear()
            return

        idx = max(0, min(self.current_frame, len(base) - 1))
        frame = base[idx]

        # Si aplicas overlays, trabaja sobre copia y no modifiques 'frame' original
        frame_disp = self._overlay_frame(frame, idx)

        rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0],
                      rgb_frame.strides[0], QImage.Format_RGB888).copy()
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
        if self.current_stack is None:
            self.status_label.setText("No stack to play")
            return
        try:
            fps = float(self.meta.get("real_fps") or self.meta.get("frame_rate") or 10)
        except (TypeError, ValueError):
            fps = 10.0
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
        if self.current_stack is None:
            return
        self.current_frame = (self.current_frame + 1) % len(self.current_stack)
        self.update_preview()

    def prev_frame(self):
        if self.current_stack is None:
            return
        self.current_frame = max(0, self.current_frame - 1)
        self.update_preview()

    def next_frame(self):
        if self.current_stack is None:
            return
        self.current_frame = min(len(self.current_stack) - 1, self.current_frame + 1)
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
        stack = self.current_stack if self.current_stack is not None else self.original_stack
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
        if self.current_stack is None:
            self.status_label.setText("No processed stack to save")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save video",
            "",
            "AVI Files (*.avi);;MP4 Files (*.mp4)"
        )

        if not path:
            return
        folder = os.path.dirname(path)
        meta_path = os.path.splitext(path)[0] + "_metadata.json"
        try:
            with open(meta_path, "w") as f:
                json.dump(self.meta, f, indent=2, default=str)
        except Exception as e:
            self.status_label.setText(f"Error saving metadata: {e}")
            return
        try:
            stack = np.asarray(self.current_stack)
            H, W = stack[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = self.meta.get("frame_rate", 10) or 10
            writer = cv2.VideoWriter(path, fourcc, float(fps), (W, H), True)
            if not writer.isOpened():
                raise OSError(f"Could not open video writer for {path}")
            for index, frame in enumerate(stack):
                writer.write(self._overlay_frame(frame, index))
            writer.release()
        except Exception as e:
            self.status_label.setText(f"Error saving video: {e}")
            return
        self.status_label.setText(f"Saved metadata and video to {path}")

    def _make_metadata_frame(self, height, width):
        meta_json = json.dumps(self.meta)
        meta_bytes = meta_json.encode("utf-8")

        frame = np.zeros((height, width), dtype=np.uint8)

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
