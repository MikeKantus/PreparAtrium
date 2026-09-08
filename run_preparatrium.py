# run_preparatrium.py
# Entry point for the PreparAtrium application, ensuring dependencies are installed and launching the main window.
from pathlib import Path
import subprocess
import sys


def ensure_dependencies() -> None:
	requirements_file = Path(__file__).with_name("Requirements.txt")
	subprocess.check_call([
		sys.executable,
		"-m",
		"pip",
		"install",
		"-r",
		str(requirements_file),
	])


ensure_dependencies()

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

app = QApplication(sys.argv)
win = MainWindow()
win.show()
sys.exit(app.exec())
