TEMPLE_STYLE = """
/* ============================================================
   ESTILO GLOBAL — Templo oscuro con acentos dorados
   ============================================================ */

QWidget {
    background-color: #1a1a1a;
    color: #e0d6b8;
    font-family: 'Cinzel', 'Trajan Pro', serif;
    font-size: 14px;
}

/* ============================================================
   LABELS — Inscripciones del templo
   ============================================================ */

QLabel {
    border: 2px solid #bfa76f;
    padding: 6px;
    background-color: #262018;
    font-weight: 500;
}

QLabel#title {
    font-size: 20px;
    font-weight: bold;
    color: #f0e6c8;
    border: none;
    padding: 10px;
}

/* ============================================================
   BOTONES — Botones rituales
   ============================================================ */

QPushButton {
    background-color: #3a2f1f;
    border: 2px solid #bfa76f;
    padding: 8px 12px;
    border-radius: 4px;
    font-weight: bold;
    color: #e0d6b8;
}

QPushButton:hover {
    background-color: #4a3f2f;
    border-color: #d8c48a;
}

QPushButton:pressed {
    background-color: #2a2418;
    border-color: #a8925f;
}

/* Botones de acción principal */
QPushButton#primary {
    background-color: #5a4a2f;
    border: 2px solid #e6d19a;
    color: #f8f2d0;
}

QPushButton#primary:hover {
    background-color: #6a5a3f;
}

/* ============================================================
   SLIDERS — Controles ceremoniales
   ============================================================ */

QSlider::groove:horizontal {
    height: 6px;
    background: #4a3f2f;
    border: 1px solid #bfa76f;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #d8c48a;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #f0e6c8;
}

/* ============================================================
   SCROLL AREAS — Pergaminos del templo
   ============================================================ */

QScrollArea {
    border: 2px solid #bfa76f;
    background-color: #1f1a14;
}

QScrollBar:vertical {
    background: #2a2418;
    width: 12px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #bfa76f;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #d8c48a;
}

/* ============================================================
   QFrame — Cámaras del templo
   ============================================================ */

QFrame {
    border: 2px solid #bfa76f;
    background-color: #262018;
    border-radius: 4px;
}

/* ============================================================
   QProgressBar — Barra de ritual
   ============================================================ */

QProgressBar {
    border: 2px solid #bfa76f;
    border-radius: 4px;
    text-align: center;
    background-color: #2a2418;
    color: #e0d6b8;
}

QProgressBar::chunk {
    background-color: #d8c48a;
    width: 20px;
}

/* ============================================================
   QComboBox — Selección de artefactos
   ============================================================ */

QComboBox {
    background-color: #3a2f1f;
    border: 2px solid #bfa76f;
    padding: 4px;
    color: #e0d6b8;
}

QComboBox QAbstractItemView {
    background-color: #2a2418;
    selection-background-color: #bfa76f;
    color: #e0d6b8;
}

/* ============================================================
   QLineEdit — Inscripciones editables
   ============================================================ */

QLineEdit {
    background-color: #2a2418;
    border: 2px solid #bfa76f;
    padding: 6px;
    color: #e0d6b8;
}

/* ============================================================
   QTabWidget — Pórticos del templo
   ============================================================ */

QTabWidget::pane {
    border: 2px solid #bfa76f;
    background: #262018;
}

QTabBar::tab {
    background: #3a2f1f;
    border: 2px solid #bfa76f;
    padding: 6px;
    margin-right: 2px;
    color: #e0d6b8;
}

QTabBar::tab:selected {
    background: #5a4a2f;
    border-color: #e6d19a;
    color: #f8f2d0;
}

/* ============================================================
   QToolTip — Susurros del templo
   ============================================================ */

QToolTip {
    background-color: #3a2f1f;
    color: #f8f2d0;
    border: 1px solid #bfa76f;
    padding: 4px;
    font-size: 12px;
}
"""
QMainWindow {
    background-image: url("assets/Preparatrium_bg.png");
    background-repeat: no-repeat;
    background-position: center;
    background-origin: content;
    background-attachment: fixed;
}

/* Capa de opacidad */
QWidget#opacity_layer {
    background-color: rgba(0, 0, 0, 0.35); /* 35% opacidad */
}
    self.setStyleSheet(f"""
    QMainWindow {{
        background-image: url("{bg_path}");
        background-repeat: no-repeat;
        background-position: center;
        background-origin: content;
        background-attachment: fixed;
    }}

    QWidget#overlay {{
        background-color: rgba(0, 0, 0, 0.75);  /* oscurece el fondo */
    }}
""")

