# gui/main_window.py
import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSplitter, QFrame, QMessageBox
)
from PySide6.QtCore import Qt

# Import the AFM loader widget (the file you already placed at gui/afm_loader.py)
# The module exposes AFMLoaderWidget and also AFMLoaderWindow for backwards compatibility.
try:
    from gui.afm_loader import AFMLoaderWidget
except Exception:
    # Fallback to the compatibility name if used elsewhere
    from gui.afm_loader import AFMLoaderWindow as AFMLoaderWidget

# Optional: import your DriftWindow if available. If not present, we'll open a placeholder.
try:
    from gui.drift_panel import DriftWindow
    HAS_DRIFT_WINDOW = True
except Exception:
    DriftWindow = None
    HAS_DRIFT_WINDOW = False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PreparAtrium")
        self.resize(1400, 900)

        # State to hold AFM data when loader sends it
        self.afm_stack = None
        self.afm_meta = None

        # Create loader widget (left) and work area (right)
        self.loader_widget = AFMLoaderWidget(parent=self)
        self.work_widget = QWidget()
        self._build_work_widget()

        # Splitter between loader and work area
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.loader_widget)
        self.splitter.addWidget(self.work_widget)
        # sensible initial sizes: loader narrow, work area wide
        self.splitter.setSizes([480, 900])

        # Wizard-style navigation (Next / Back)
        self.btn_next = QPushButton("Next")
        self.btn_back = QPushButton("Back")
        self.btn_back.setEnabled(False)
        self.btn_next.clicked.connect(self.on_next)
        self.btn_back.clicked.connect(self.on_back)

        nav_row = QHBoxLayout()
        nav_row.addStretch()
        nav_row.addWidget(self.btn_back)
        nav_row.addWidget(self.btn_next)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.splitter)
        main_layout.addLayout(nav_row)
        self.setLayout(main_layout)

        # Keep previous sizes to restore when pressing Back
        self._prev_sizes = self.splitter.sizes()

    def _build_work_widget(self):
        """Build the right-side work area (Drift / Kymo launcher)."""
        layout = QVBoxLayout(self.work_widget)
        title = QLabel("<b>Work area</b>")
        title.setFrameStyle(QFrame.NoFrame)
        layout.addWidget(title)

        # Info label that updates when AFM is loaded
        self.info_label = QLabel("No AFM loaded")
        layout.addWidget(self.info_label)

        # Button to open DriftWindow (kept for convenience but not required)
        # NOTE: The explicit 'Open Drift Panel' button requested to be removed from the loader.
        # We keep an optional button here only if DriftWindow exists.
        if HAS_DRIFT_WINDOW:
            self.btn_open_drift_internal = QPushButton("Open Drift Panel (work area)")
            self.btn_open_drift_internal.clicked.connect(self.open_drift_panel)
            layout.addWidget(self.btn_open_drift_internal)

        layout.addStretch()

    # -------------------------
    # Wizard navigation
    # -------------------------
    def on_next(self):
        """Collapse the loader panel to focus on the work area."""
        # Validate that there is a processed stack before moving on
        if getattr(self.loader_widget, "processed_stack", None) is None:
            # allow moving forward but warn user
            reply = QMessageBox.question(
                self, "No processed stack",
                "No processed stack found. Do you want to continue to the work area anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # Save current sizes and collapse left pane
        self._prev_sizes = self.splitter.sizes()
        total = sum(self._prev_sizes) or 1
        # set left to 0, right to total
        self.splitter.setSizes([0, total])
        self.btn_next.setEnabled(False)
        self.btn_back.setEnabled(True)

    def on_back(self):
        """Restore the loader panel sizes."""
        # restore previous sizes
        if hasattr(self, "_prev_sizes") and self._prev_sizes:
            self.splitter.setSizes(self._prev_sizes)
        else:
            self.splitter.setSizes([480, 900])
        self.btn_next.setEnabled(True)
        self.btn_back.setEnabled(False)

    # -------------------------
    # Receiving AFM from loader
    # -------------------------
    def load_afm(self, stack, meta):
        """
        Called by AFMLoaderWidget.send_to_drift().
        Save the stack/meta and optionally open the drift panel.
        """
        self.afm_stack = stack
        self.afm_meta = meta or {}
        nframes = len(stack) if stack is not None else 0
        fps = self.afm_meta.get("frame_rate", "unknown")
        self.info_label.setText(f"AFM loaded: {nframes} frames, FPS: {fps}")
        # Automatically switch to work area (collapse loader)
        self.on_next()
        # Optionally open DriftWindow automatically if available
        if HAS_DRIFT_WINDOW:
            try:
                # Pass stack and meta to the drift window
                self.drift_window = DriftWindow(self.afm_stack, self.afm_meta)
                self.drift_window.show()
            except Exception as e:
                QMessageBox.warning(self, "Drift Window", f"Could not open DriftWindow: {e}")

    # -------------------------
    # Open drift panel (work area button)
    # -------------------------
    def open_drift_panel(self):
        """Open the drift panel using the currently loaded AFM data."""
        if not HAS_DRIFT_WINDOW:
            QMessageBox.information(self, "Drift Panel", "DriftWindow not available in this installation.")
            return
        if self.afm_stack is None:
            QMessageBox.information(self, "Drift Panel", "Load and send an AFM stack first.")
            return
        try:
            self.drift_window = DriftWindow(self.afm_stack, self.afm_meta)
            self.drift_window.show()
        except Exception as e:
            QMessageBox.warning(self, "Drift Window", f"Could not open DriftWindow: {e}")

    # -------------------------
    # Convenience: close app
    # -------------------------
    def closeEvent(self, event):
        # perform any cleanup if needed
        event.accept()


# If run as script, start the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
