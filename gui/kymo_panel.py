import threading
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QSpinBox,
    QCheckBox,
)

from .kymo_analyzer import KymoAnalyzer
from .kymo_canvas import KymoCanvas
from .kymo_controller import KymoController
from .kymo_model import KymoModel
from .kymo_manager import KymoManager
from core.kymo_tools import KymogramData


class KymoPanel(QWidget):
    """Fixed-size Qt kymograph workspace backed by the MVC components."""

    def __init__(self, stack, meta=None):
        super().__init__()
        self.setWindowTitle("PreparAtrium - Kymograph analysis")
        self.setFixedSize(1536, 1024)

        self.model = KymoModel(np.asarray(stack), meta or {})
        self.model.current_frame = 0
        self.canvas = KymoCanvas(self.model)
        self.canvas.panel = self

        # The analyzer/controller API is shared with the standalone manager.
        self.analyzer = KymoAnalyzer(None, None)
        self.controller = KymoController(self.analyzer)
        self._build_ui()
        self._refresh_metadata()
        self._refresh_frame()

    def _build_ui(self):
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, max(0, self.model.n_frames - 1))
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.frame_label = QLabel("Frame 0")

        self.btn_new_line = QPushButton("New line")
        self.btn_delete_line = QPushButton("Delete last line")
        self.btn_end_profile = QPushButton("End profile")
        self.btn_export = QPushButton("Export kymograms")
        self.btn_analyzer = QPushButton("Open analyzer")
        self.btn_new_line.clicked.connect(self.start_new_line)
        self.btn_delete_line.clicked.connect(self.delete_last_line)
        self.btn_end_profile.clicked.connect(self.end_profile)
        self.btn_export.clicked.connect(self.export_all_kymos)
        self.btn_analyzer.clicked.connect(self.open_analyzer)

        # New controls: panoramic overlay, radius, subpixel, detect polymers
        self.chk_panorama = QCheckBox("Panoramic overlay")
        self.chk_panorama.setChecked(False)
        self.chk_panorama.stateChanged.connect(self._on_panorama_toggled)

        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(0, 10)
        self.spin_radius.setValue(1)
        self.spin_radius.setToolTip("Radius in pixels for perpendicular averaging")

        self.chk_subpixel = QCheckBox("Sub-pixel sampling (bilinear)")
        self.chk_subpixel.setChecked(False)

        self.btn_detect_polymers = QPushButton("Detect polymers")
        self.btn_detect_polymers.clicked.connect(self._on_detect_polymers)

        metadata_box = QGroupBox("Metadata")
        self.metadata_layout = QFormLayout(metadata_box)
        self.metadata_layout.setLabelAlignment(Qt.AlignLeft)

        self.kymo_list = QListWidget()
        self.kymo_list.itemSelectionChanged.connect(self._show_selected_kymo)
        self.kymo_preview = QLabel("Select a kymogram")
        self.kymo_preview.setAlignment(Qt.AlignCenter)
        self.kymo_preview.setMinimumSize(360, 360)
        self.kymo_preview.setStyleSheet("background-color: #111;")

        left_layout = QVBoxLayout()
        left_layout.addWidget(metadata_box)
        left_layout.addWidget(QLabel("Extracted kymograms"))
        left_layout.addWidget(self.kymo_list, 1)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        frame_controls = QHBoxLayout()
        frame_controls.addWidget(self.frame_label)
        frame_controls.addWidget(self.frame_slider, 1)

        button_layout = QHBoxLayout()
        for button in (self.btn_new_line, self.btn_delete_line,
                   self.btn_end_profile,
                        self.btn_export, self.btn_analyzer):
            button_layout.addWidget(button)

        # Add new control layout
        extra_controls = QHBoxLayout()
        extra_controls.addWidget(self.chk_panorama)
        extra_controls.addWidget(QLabel("Radius:"))
        extra_controls.addWidget(self.spin_radius)
        extra_controls.addWidget(self.chk_subpixel)
        extra_controls.addWidget(self.btn_detect_polymers)

        center_layout = QVBoxLayout()
        center_layout.addWidget(QLabel("Video"))
        center_layout.addWidget(self.canvas, 1)
        center_layout.addLayout(frame_controls)
        center_layout.addLayout(button_layout)
        center_layout.addLayout(extra_controls)
        center_widget = QWidget()
        center_widget.setLayout(center_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Selected kymogram"))
        right_layout.addWidget(self.kymo_preview, 1)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([280, 820, 360])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(splitter)

    def _on_panorama_toggled(self, state):
        self.canvas.panorama_overlay = bool(state)
        self.canvas.update()

    def _on_detect_polymers(self):
        # run detection and refresh canvas
        try:
            self.model.detect_polymers()
            self.canvas.update()
        except Exception as e:
            print("Polymer detection failed:", e)

    def _refresh_metadata(self):
        while self.metadata_layout.rowCount():
            self.metadata_layout.removeRow(0)
        values = {
            "Source": self.model.source_name,
            "Frames": self.model.n_frames,
            "Height": self.model.shape[1],
            "Width": self.model.shape[2],
            "Pixel size (nm/px)": self.model.pixel_size_nm,
            "Frame rate (fps)": self.model.frame_rate or "unknown",
        }
        for key, value in values.items():
            self.metadata_layout.addRow(QLabel(key), QLabel(str(value)))

    def _on_frame_changed(self, index):
        self.model.current_frame = int(index)
        self.frame_label.setText(f"Frame {index}")
        self._refresh_frame()

    def _refresh_frame(self):
        self.canvas.current_frame = self.model.current_frame
        self.canvas.update()

    def update_preview(self):
        self._refresh_frame()

    def start_new_line(self):
        self.canvas.active_line = []
        self.canvas.update()

    def delete_last_line(self):
        if self.model.manual_lines:
            self.model.delete_manual_line(len(self.model.manual_lines) - 1)
            self.canvas.update()

    def end_profile(self):
        if not self.canvas.active_line or len(self.canvas.active_line) < 2:
            return
        line = list(self.canvas.active_line)
        self.model.add_manual_line(line)
        self.canvas.active_line = []
        # use radius and subpixel options
        radius = int(self.spin_radius.value())
        subpixel = bool(self.chk_subpixel.isChecked())
        entry = self.model.add_kymograph_from_line(line, radius_px=radius, subpixel=subpixel)
        self.kymo_list.addItem(QListWidgetItem(entry["label"]))
        self.canvas.update()

    def finish_profile_from_canvas(self):
        """Finish the active line after a right-click on the video."""
        self.end_profile()

    def _show_selected_kymo(self):
        index = self.kymo_list.currentRow()
        if index < 0 or index >= len(self.model.kymos):
            return
        entry = self.model.kymos[index]
        self.controller.load_kymo_array(
            entry["kymo"], self.model.pixel_size_nm,
            self.model.time_per_frame or 1.0,
        )
        self._set_kymo_pixmap(entry["kymo"])

    def _set_kymo_pixmap(self, kymo):
        array = np.asarray(kymo, dtype=float)
        if array.size == 0:
            self.kymo_preview.setText("Empty kymogram")
            return
        low, high = np.nanmin(array), np.nanmax(array)
        if high <= low:
            image = np.zeros(array.shape, dtype=np.uint8)
        else:
            image = (np.clip((array - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        image = np.ascontiguousarray(image)
        qimage = QImage(image.data, image.shape[1], image.shape[0],
                        image.strides[0], QImage.Format_Grayscale8).copy()
        self.kymo_preview.setPixmap(QPixmap.fromImage(qimage).scaled(
            self.kymo_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def export_all_kymos(self):
        if not self.model.kymos:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            self.model.export_all_kymographs(folder)

    def open_analyzer(self):
        index = self.kymo_list.currentRow()
        if index >= 0:
            entry = self.model.kymos[index]
            # Build a normalized metadata dict for the analyzer
            meta = dict(self.model.meta) if hasattr(self.model, "meta") and self.model.meta else {}
            meta["pixel_size"] = self.model.pixel_size_nm
            if self.model.time_per_frame is not None:
                meta["time_per_frame"] = self.model.time_per_frame
            elif getattr(self.model, "frame_rate", None):
                meta["frame_rate"] = self.model.frame_rate

            kymo_data = KymogramData(data=np.asarray(entry["kymo"], dtype=float),
                                     metadata=meta,
                                     provenance={"source": self.model.source_name})

            # Schedule opening the KymoManager on the Qt main loop (safe for GUI backends)
            def _open():
                try:
                    KymoManager(file_paths=None, pixel_size=self.model.pixel_size_nm,
                                kymo_array=kymo_data.data, metadata=kymo_data.metadata)
                except Exception as e:
                    print("Failed to open KymoManager:", e)

            QTimer.singleShot(0, _open)
