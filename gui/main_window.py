#gui/main_window.py
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QFrame,
    QSplitter, QLabel, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve

from gui.afm_loader import AFMLoaderWidget
from gui.drift_panel import DriftWindow
from gui.kymo_panel import KymoPanel   # asegúrate de tener este archivo


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PreparAtrium – Temple Edition")
        self.setMinimumSize(1400, 900)
        # ============================================================
        #                   OVERLAY PARA DRIFT
        # ============================================================

        self.overlay = QWidget(self)
        self.overlay.setObjectName("overlay")
        self.overlay.hide()  # oculto al inicio

        overlay_layout = QVBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(50, 50, 50, 50)  # margen opcional
        self.overlay_layout = overlay_layout

        # ============================================================
        #                   SPLITTER PRINCIPAL (3 cámaras)
        # ============================================================

        self.splitter_main = QSplitter(Qt.Horizontal)
        self.splitter_main.setHandleWidth(4)

        # ============================================================
        #                   PANEL IZQUIERDO — Cámara de Archivos
        # ============================================================

        self.left_panel = QFrame()
        self.left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(self.left_panel)

        self.loader = AFMLoaderWidget(main_window=self)
        self.loader.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.loader)

        # ============================================================
        #                   PANEL CENTRAL — Cámara del Vídeo
        # ============================================================

        self.center_panel = QFrame()
        self.center_panel.setFrameShape(QFrame.StyledPanel)
        center_layout = QVBoxLayout(self.center_panel)

        self.video_label = QLabel("Video area")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        controls = QHBoxLayout()
        btn_prev = QPushButton("⟵")
        btn_play = QPushButton("▶")
        btn_pause = QPushButton("⏸")
        btn_next = QPushButton("⟶")

        self.btn_prev = btn_prev
        self.btn_play = btn_play
        self.btn_pause = btn_pause
        self.btn_next = btn_next


        for b in (btn_prev, btn_play, btn_pause, btn_next):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            controls.addWidget(b)

        center_layout.addWidget(self.video_label, stretch=5)
        center_layout.addLayout(controls, stretch=1)

        # ============================================================
        #                   PANEL DERECHO — Cámara de Herramientas
        # ============================================================

        self.right_panel = QFrame()
        self.right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(self.right_panel)

        btn_drift = QPushButton("Process drift")
        btn_drift.clicked.connect(self.open_drift_panel)

        btn_kymo = QPushButton("Kymograph analysis")
        btn_kymo.clicked.connect(self.open_kymo_panel)

        btn_export = QPushButton("Export metadata")

        right_layout.addWidget(btn_drift)
        right_layout.addWidget(btn_kymo)
        right_layout.addWidget(btn_export)
        right_layout.addStretch()

        # ============================================================
        #                   AÑADIR PANELES AL SPLITTER
        # ============================================================

        self.splitter_main.addWidget(self.left_panel)
        self.splitter_main.addWidget(self.center_panel)
        self.splitter_main.addWidget(self.right_panel)

        self.splitter_main.setSizes([400, 800, 0])  # panel derecho oculto

        # ============================================================
        #                   PANEL INFERIOR — Cámara de Histogramas
        # ============================================================

        self.bottom_panel = QFrame()
        self.bottom_panel.setFrameShape(QFrame.StyledPanel)
        bottom_layout = QHBoxLayout(self.bottom_panel)

        hist_left = QLabel("Histogram A")
        hist_right = QLabel("Histogram B")

        hist_left.setAlignment(Qt.AlignCenter)
        hist_right.setAlignment(Qt.AlignCenter)

        hist_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hist_right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        bottom_layout.addWidget(hist_left)
        bottom_layout.addWidget(hist_right)

        # ============================================================
        #                   CONTENEDOR PRINCIPAL
        # ============================================================

        container = QWidget()
        container_layout = QVBoxLayout(container)

        container_layout.addWidget(self.splitter_main, stretch=8)
        container_layout.addWidget(self.bottom_panel, stretch=2)

        self.setCentralWidget(container)
        # ============================================================
        #                   FONDO DEL TEMPLO
        # ============================================================
        import os
        bg_path = os.path.join(os.path.dirname(__file__), "assets", "Preparatrium_bg.png")
        bg_path = bg_path.replace("\\", "/")  # Qt necesita barras normales

        self.setStyleSheet(f"""
        QMainWindow {{
            background-image: url("{bg_path}");
            background-repeat: no-repeat;
            background-position: center;
            background-origin: content;
            background-attachment: fixed;
        }}
        """)


        # ============================================================
        #                   CAPA DE OPACIDAD DEL FONDO
        # ============================================================

        opacity_layer = QWidget(self)
        opacity_layer.setObjectName("opacity_layer")
        opacity_layer.lower()

        # ============================================================
        #                   OCULTAR PANELES AL INICIO
        # ============================================================

        self.center_panel.hide()
        self.right_panel.hide()
        self.bottom_panel.hide()

        # ============================================================
        #                   VARIABLES PARA DRIFT Y KYMO
        # ============================================================

        self.afm_stack = None
        self.afm_meta = None

        self.drift_widget = None
        self.kymo_panel = None
        # ============================================================
        #                   White color
        # ============================================================


        self.setStyleSheet(self.styleSheet() + """
            QLabel, QPushButton {
                color: white;
                font-weight: bold;
                font-size: 14px;
            }

            QPushButton {
                background-color: rgba(40,40,40,180);
                border: 1px solid rgba(255,255,255,120);
                padding: 6px;
                border-radius: 6px;
            }
        """)


    # ============================================================
    #                   MÉTODOS DEL MAINWINDOW
    # ============================================================

    def load_afm(self, stack, meta):
        """Recibe el vídeo desde AFMLoaderWidget."""
        self.afm_stack = stack
        self.afm_meta = meta

        self.video_label.setText("AFM stack loaded — ready for analysis")

        # Mostrar paneles con animación
        self.center_panel.show()
        self.bottom_panel.show()
        self.animate_right_panel()

    # ============================================================
    #                   ANIMACIÓN PANEL DERECHO
    # ============================================================

    def animate_right_panel(self):
        self.right_panel.show()

        # Animación del panel derecho
        anim_right = QPropertyAnimation(self.right_panel, b"maximumWidth")
        anim_right.setDuration(5000)
        anim_right.setStartValue(0)
        anim_right.setEndValue(350)
        anim_right.setEasingCurve(QEasingCurve.OutCubic)
        anim_right.start()

        # Animación del panel central (ligero ajuste)
        anim_center = QPropertyAnimation(self.center_panel, b"maximumWidth")
        anim_center.setDuration(5000)
        anim_center.setStartValue(900)
        anim_center.setEndValue(800)
        anim_center.setEasingCurve(QEasingCurve.OutCubic)
        anim_center.start()

        # Ajuste final del splitter (sin animación)
        self.splitter_main.setSizes([450, 800, 350])
        self.setMinimumSize(768, 512)
        self.resize(1536, 1024)
    # ============================================================
    #                   PANEL DE DRIFT (DESPLEGABLE)
    # ============================================================
    def resizeEvent(self, event):
        # Mantener ratio 1536x1024
        target_ratio = 1536 / 1024
        w = self.width()
        h = self.height()

        current_ratio = w / h

        if current_ratio > target_ratio:
            # ventana demasiado ancha → ajustamos ancho
            new_w = int(h * target_ratio)
            self.resize(new_w, h)
        else:
            # ventana demasiado alta → ajustamos alto
            new_h = int(w / target_ratio)
            self.resize(w, new_h)

        super().resizeEvent(event)

    def open_drift_panel(self):
        if self.afm_stack is None:
            self.video_label.setText("Load and send an AFM stack first")
            return

        # Parar el vídeo si existe
        if hasattr(self, "video_timer"):
            self.video_timer.stop()

        # Crear el widget de drift si no existe
        if self.drift_widget is None:
            self.drift_widget = DriftWindow(self.afm_stack, self.afm_meta)
            # Lo añadimos al right_panel layout
            self.right_panel.layout().addWidget(self.drift_widget)

        # --- 1) Ocultar/colapsar el panel izquierdo (loader) y poner botón de retorno ---
        # Guardar referencia al loader original para restaurar después
        if hasattr(self, "_loader_hidden") and self._loader_hidden:
            pass
        else:
            # ocultar loader widget y reemplazar por un botón simple
            self.loader.hide()
            self._return_btn_left = QPushButton("Return to video processing")
            self._return_btn_left.setFixedWidth(50)
            self._return_btn_left.clicked.connect(self.close_drift_panel)
            # crear un layout sencillo en left_panel si no existe
            left_layout = self.left_panel.layout()
            # limpiar widgets existentes (si quieres mantenerlos, omite)
            while left_layout.count():
                item = left_layout.takeAt(0)
                w = item.widget()
                if w:
                    w.hide()
            left_layout.addWidget(self._return_btn_left)
            self._loader_hidden = True

        # --- 2) Reemplazar el panel central por un botón (ocultar preview) ---
        # Ocultar controles de vídeo y mostrar botón de retorno grande en centro
        self.center_panel.hide()  # ocultamos el panel central completo
        # Crear botón grande en su lugar (si no existe)
        if not hasattr(self, "_return_btn_center"):
            self._return_btn_center = QPushButton("Return to video processing")
            self._return_btn_center.setFixedWidth(50)
            self._return_btn_center.setMinimumHeight(48)
            self._return_btn_center.clicked.connect(self.close_drift_panel)
            # Añadir al right place: lo colocamos en left_panel para que sea visible
            self.left_panel.layout().addWidget(self._return_btn_center)

           # --- 3) Mostrar drift widget y ajustar tamaños para que ocupe mucho espacio ---
        self.drift_widget.show()

        # Calcular tamaños en píxeles según el ancho actual de la ventana
        total_w = max(1, self.width())
        left_w = max(1, int(total_w * 0.05))   # 5%
        center_w = 0                           # colapsado
        right_w = max(1, total_w - left_w - center_w)  # 95%

        # Animación del panel derecho: ajustar maximumWidth al ancho calculado
        anim_right = QPropertyAnimation(self.right_panel, b"maximumWidth")
        anim_right.setDuration(700)
        anim_right.setStartValue(self.right_panel.maximumWidth() or 1500)
        anim_right.setEndValue(right_w)
        anim_right.setEasingCurve(QEasingCurve.OutCubic)
        anim_right.start()

        # Animación del panel central (lo colapsamos)
        anim_center = QPropertyAnimation(self.center_panel, b"maximumWidth")
        anim_center.setDuration(700)
        anim_center.setStartValue(self.center_panel.maximumWidth() or 10)
        anim_center.setEndValue(0)
        anim_center.setEasingCurve(QEasingCurve.OutCubic)
        anim_center.start()

        # Ajuste final del splitter: left 5%, center 0, right 95%
        # Qt espera valores en píxeles; pasamos la lista calculada.
        total_w = max(1, self.width())
        left_w = max(int(total_w * 0.05), 140)   # 5% pero al menos 140 px para contener botones
        center_w = 0
        right_w = max(1, total_w - left_w - center_w)
        self.splitter_main.setSizes([left_w, center_w, right_w])



        # Ocultar botones del panel derecho originales (si los hay)
        for i in range(self.right_panel.layout().count()):
            widget = self.right_panel.layout().itemAt(i).widget()
            if isinstance(widget, QPushButton) and widget is not self.drift_widget:
                widget.hide()

        # Asegurarse de ocultar controles de vídeo
        for b in (self.btn_prev, self.btn_play, self.btn_pause, self.btn_next):
            b.hide()


    def close_drift_panel(self):
        # Ocultar widget de drift
        if self.drift_widget is not None:
            self.drift_widget.hide()

        # R    # Restaurar animaciones y tamaños usando el ancho actual de la ventana
        total_w = max(1, self.width())
        left_w = max(1, int(total_w * 0.25))   # tamaño por defecto al cerrar (ajusta si quieres)
        center_w = max(1, int(total_w * 0.55))
        right_w = max(1, total_w - left_w - center_w)

        anim_right = QPropertyAnimation(self.right_panel, b"maximumWidth")
        anim_right.setDuration(700)
        anim_right.setStartValue(self.right_panel.maximumWidth())
        anim_right.setEndValue(right_w if right_w > 0 else 350)
        anim_right.setEasingCurve(QEasingCurve.OutCubic)
        anim_right.start()

        anim_center = QPropertyAnimation(self.center_panel, b"maximumWidth")
        anim_center.setDuration(700)
        anim_center.setStartValue(self.center_panel.maximumWidth())
        anim_center.setEndValue(center_w if center_w > 0 else 800)
        anim_center.setEasingCurve(QEasingCurve.OutCubic)
        anim_center.start()

        # Restaurar splitter a proporciones razonables (puedes ajustar)
        self.splitter_main.setSizes([left_w, center_w, right_w])


        # Restaurar loader (panel izquierdo)
        if hasattr(self, "_loader_hidden") and self._loader_hidden:
            # quitar botones de retorno añadidos
            try:
                self.left_panel.layout().removeWidget(self._return_btn_left)
                self._return_btn_left.deleteLater()
            except Exception:
                pass
            try:
                self.left_panel.layout().removeWidget(self._return_btn_center)
                self._return_btn_center.deleteLater()
            except Exception:
                pass

            # volver a mostrar loader
            self.loader.show()
            self._loader_hidden = False

        # Restaurar controles del vídeo
        for b in (self.btn_prev, self.btn_play, self.btn_pause, self.btn_next):
            b.show()

        # Restaurar botones del panel derecho
        for i in range(self.right_panel.layout().count()):
            widget = self.right_panel.layout().itemAt(i).widget()
            if isinstance(widget, QPushButton):
                widget.show()

        # Reanudar vídeo si existía timer
        if hasattr(self, "video_timer"):
            self.video_timer.start()



    # ============================================================
    #                   PANEL DE KYMOGRAMAS (DESPLEGABLE)
    # ============================================================

    def open_kymo_panel(self):
        if self.afm_stack is None:
            self.video_label.setText("Load and send an AFM stack first")
            return

        if self.kymo_panel is None:
            self.kymo_panel = KymoPanel(self.afm_stack, self.afm_meta)
            self.kymo_panel.setParent(self.right_panel)
            self.right_panel.layout().addWidget(self.kymo_panel)

            back_btn = QPushButton("Back to video")
            back_btn.clicked.connect(self.close_kymo_panel)
            self.right_panel.layout().addWidget(back_btn)

        self.kymo_panel.show()

        animation = QPropertyAnimation(self.splitter_main, b"sizes")
        animation.setDuration(700)
        animation.setStartValue(self.splitter_main.sizes())
        animation.setEndValue([200, 400, 800])
        animation.setEasingCurve(Qt.EasingCurve.OutCubic)
        animation.start()

    def close_kymo_panel(self):
        self.kymo_panel.hide()

        animation = QPropertyAnimation(self.splitter_main, b"sizes")
        animation.setDuration(700)
        animation.setStartValue(self.splitter_main.sizes())
        animation.setEndValue([350, 800, 350])
        animation.setEasingCurve(Qt.EasingCurve.OutCubic)
        animation.start()
