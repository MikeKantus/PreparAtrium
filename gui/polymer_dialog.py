import copy
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QDialogButtonBox, QMessageBox
)


class PolymerDetectionDialog(QDialog):
    """
    Diálogo interactivo para ajustar los parámetros de filtro y detección de polímeros
    con vista previa en tiempo real sobre el canvas del quimograma.
    """

    def __init__(self, panel, parent=None):
        super().__init__(parent or panel)
        self.panel = panel
        self.model = panel.model
        self.canvas = panel.canvas

        # Copiar estado inicial de polímeros para restaurarlo si el usuario cancela
        self._initial_polymers = copy.deepcopy(self.model.polymers)

        self.setWindowTitle("Detección de Polímeros y Filtros")
        self.setMinimumWidth(400)
        self._build_ui()

        # Generar vista previa inicial
        self.run_preview()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Grupo de controles de filtro y binarización
        group = QGroupBox("Parámetros de Filtro y Segmentación")
        form = QFormLayout(group)

        # 1. Filtro Gaussiano (Sigma)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.0, 10.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(1.0)
        self.spin_sigma.setToolTip("Desviación estándar (σ) del filtro Gaussiano para suavizar ruido antes del umbralizado.")
        form.addRow(QLabel("Filtro Gaussiano (σ):"), self.spin_sigma)

        # 2. Método de Umbralizado
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Otsu", "Percentil"])
        self.combo_method.setToolTip("Método para calcular el umbral de binarización.")
        form.addRow(QLabel("Método de Umbral:"), self.combo_method)

        # 3. Sensibilidad del Umbral
        self.spin_sensitivity = QDoubleSpinBox()
        self.spin_sensitivity.setRange(0.1, 5.0)
        self.spin_sensitivity.setSingleStep(0.1)
        self.spin_sensitivity.setValue(1.0)
        self.spin_sensitivity.setToolTip("Factor de sensibilidad. Valores más altos detectan más estructuras.")
        form.addRow(QLabel("Sensibilidad:"), self.spin_sensitivity)

        # 4. Tamaño Mínimo (px)
        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(5, 5000)
        self.spin_min_size.setSingleStep(10)
        self.spin_min_size.setValue(50)
        self.spin_min_size.setToolTip("Área mínima en píxeles para considerar una región como polímero.")
        form.addRow(QLabel("Tamaño Mínimo (px):"), self.spin_min_size)

        # 5. Elongación Mínima (Ratio de aspecto)
        self.spin_elongation = QDoubleSpinBox()
        self.spin_elongation.setRange(1.0, 20.0)
        self.spin_elongation.setSingleStep(0.5)
        self.spin_elongation.setValue(2.0)
        self.spin_elongation.setToolTip("Relación mínima de elongación (Eje Mayor / Eje Menor).")
        form.addRow(QLabel("Elongación Mínima (L/W):"), self.spin_elongation)

        layout.addWidget(group)

        # Opción de Vista Previa Automática
        self.chk_live_preview = QCheckBox("Previsualizar en tiempo real en el video")
        self.chk_live_preview.setChecked(True)
        layout.addWidget(self.chk_live_preview)

        # Etiqueta de Estado / Recuento
        self.lbl_status = QLabel("Polímeros detectados: 0")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #00bcd4; font-size: 13px;")
        layout.addWidget(self.lbl_status)

        # Botón extra: Convertir polímeros a quimogramas
        self.btn_extract_kymos = QPushButton("Extraer quimogramas de polímeros detectados")
        self.btn_extract_kymos.setToolTip("Añade los polímeros detectados como líneas de quimograma a la lista.")
        self.btn_extract_kymos.clicked.connect(self._on_extract_kymos)
        layout.addWidget(self.btn_extract_kymos)

        # Conectar cambios en controles para actualización en tiempo real
        self.spin_sigma.valueChanged.connect(self._on_param_changed)
        self.combo_method.currentIndexChanged.connect(self._on_param_changed)
        self.spin_sensitivity.valueChanged.connect(self._on_param_changed)
        self.spin_min_size.valueChanged.connect(self._on_param_changed)
        self.spin_elongation.valueChanged.connect(self._on_param_changed)

        # Botones OK / Cancel
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.button(QDialogButtonBox.Ok).setText("Aplicar")
        btn_box.button(QDialogButtonBox.Cancel).setText("Cancelar")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_param_changed(self):
        if self.chk_live_preview.isChecked():
            self.run_preview()

    def run_preview(self):
        try:
            polys = self.model.detect_polymers(
                min_size_px=self.spin_min_size.value(),
                elongation_thresh=self.spin_elongation.value(),
                sigma=self.spin_sigma.value(),
                threshold_method=self.combo_method.currentText(),
                sensitivity=self.spin_sensitivity.value(),
                frame_idx=self.panel.model.current_frame,
            )
            count = len(polys) if polys is not None else 0
            self.lbl_status.setText(f"Polímeros detectados: {count}")
            self.canvas.update()
        except Exception as e:
            self.lbl_status.setText(f"Error en detección: {e}")

    def _on_extract_kymos(self):
        if not self.model.polymers:
            QMessageBox.information(self, "Información", "No hay polímeros detectados para extraer.")
            return

        radius = int(self.panel.spin_radius.value())
        subpixel = bool(self.panel.chk_subpixel.isChecked())
        added_count = 0
        for p in self.model.polymers:
            line = p.get("centerline")
            if line and len(line) >= 2:
                self.model.add_manual_line(line)
                entry = self.model.add_kymograph_from_line(line, radius_px=radius, subpixel=subpixel)
                self.panel.kymo_list.addItem(entry["label"])
                added_count += 1
        self.canvas.update()
        QMessageBox.information(self, "Éxito", f"Se extrajeron {added_count} quimogramas a la lista.")

    def reject(self):
        self.model.polymers = self._initial_polymers
        self.canvas.update()
        super().reject()
