# gui/main.py

import tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path

from kymo_manager import KymoManager
from utils import get_pixel_size_from_tiff


def main():
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select kymographs",
        filetypes=[
            ("Kymographs", "*.tif *.tiff *.csv *.txt *.tsv"),
            ("All files", "*.*"),
        ]
    )
    if not file_paths:
        print("No files selected.")
        return

    is_tiff = Path(file_paths[0]).suffix.lower() in {".tif", ".tiff"}
    pixel_size = get_pixel_size_from_tiff(file_paths[0]) if is_tiff else None

    if pixel_size is None:
        ans = simpledialog.askstring(
            "Calibration",
            "Enter pixel size (nm/pixel):"
        )
        pixel_size = float(ans)

    KymoManager(file_paths, pixel_size)


if __name__ == "__main__":
    main()
