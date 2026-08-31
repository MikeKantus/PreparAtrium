import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem

from gui.afm_loader import AFMLoaderWidget


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_show_frame_updates_preview_and_current_frame():
    _app()
    widget = AFMLoaderWidget()

    widget.processed_stack = np.zeros((2, 4, 4), dtype=np.float32)
    widget.raw_stack = widget.processed_stack
    widget.current_frame = 1

    widget.update_preview = lambda: None

    widget._show_frame(0)

    assert widget.current_frame == 0


def test_selection_change_calls_show_frame_after_loading():
    _app()
    widget = AFMLoaderWidget()
    stack = np.zeros((2, 4, 4), dtype=np.float32)
    meta = {"frame_rate_hz": 10, "x_range_nm": 100, "y_range_nm": 200, "channel": "test"}

    item = QListWidgetItem("demo")
    item.setData(Qt.UserRole, "dummy_path")
    widget.list_files.addItem(item)
    widget.list_files.setCurrentItem(item)

    widget._read_file_to_frames = lambda _path: (stack, meta)
    widget.update_metadata_panel = lambda: None
    widget._show_frame = lambda idx: None

    widget.on_list_selection_changed()

    assert widget.raw_stack is stack
    assert widget.current_frame == 0
