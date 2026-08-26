import tkinter as tk
from tkinter import filedialog, simpledialog

from kymo_manager import KymoManager
from utils import get_pixel_size_from_tiff


def main():
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select kymographs",
        filetypes=[("TIFF", "*.tif *.tiff"), ("All files", "*.*")]
    )
    if not file_paths:
        print("No files selected.")
        return

    pixel_size = get_pixel_size_from_tiff(file_paths[0])  # nm/pixel

    if pixel_size is None:
        ans = simpledialog.askstring(
            "Calibration",
            "Enter pixel size (nm/pixel):"
        )
        pixel_size = float(ans)

    KymoManager(file_paths, pixel_size)


if __name__ == "__main__":
    main()
