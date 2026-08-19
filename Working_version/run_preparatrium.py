
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
import sys

app = QApplication(sys.argv)
win = MainWindow()
win.show()
sys.exit(app.exec())
