import tifffile
import tkinter as tk
from tkinter import filedialog

# ============================================================
# Selección de archivo
# ============================================================

root = tk.Tk()
root.withdraw()

path = filedialog.askopenfilename(
    title="Selecciona archivo JPK",
    filetypes=[("JPK files", "*.jpk"), ("All files", "*.*")]
)

if not path:
    print("No se seleccionó archivo.")
    exit()

print(f"\nArchivo seleccionado: {path}\n")


# ============================================================
# SCAN
# ============================================================

def extract_scan(tags):
    scan = {}

    fields = {
        "x_origin_nm": 32832,
        "y_origin_nm": 32833,
        "x_range_nm": 32834,
        "y_range_nm": 32835,
        "x_pixels": 32838,
        "y_pixels": 32839,
        "frame_rate_hz": 32841,
    }

    for key, code in fields.items():
        value = tags.get(code)
        if value is None:
            continue

        if "nm" in key:
            scan[key] = float(value) * 1e9
        elif "pixels" in key:
            scan[key] = int(value)
        else:
            scan[key] = float(value)

    return scan


# ============================================================
# CANTILEVER + FEEDBACK
# ============================================================

def extract_cantilever_and_feedback(text):
    cantilever = {}
    feedback = {}

    # Campos que queremos conservar
    cantilever_keys = {
        "amplitude",
        "calibration-environment",
        "cantilever-id",
        "cantilever-name",
        "defined",
        "frequency",
        "geometry",
        "qFactor",
        "sensitivity",
        "spring-constant",
    }

    feedback_keys = {
        "setpoint-feedback-settings.relative-setpoint"
    }

    for line in text.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        # CANTILEVER
        if key.startswith("cantilever-calibration-info."):
            short = key.replace("cantilever-calibration-info.", "")
            if short in cantilever_keys:
                cantilever[short] = value

        # FEEDBACK
        if key.startswith("feedback-mode.setpoint-feedback-settings."):
            short = key.replace("feedback-mode.setpoint-feedback-settings.", "")
            full_key = f"setpoint-feedback-settings.{short}"
            if full_key in feedback_keys:
                feedback[full_key] = value

    return cantilever, feedback


# ============================================================
# LECTURA DEL ARCHIVO
# ============================================================

scan = {}
cantilever = {}
feedback = {}

with tifffile.TiffFile(path) as tif:
    for page in tif.pages:
        tags = {tag.code: tag.value for tag in page.tags.values()}

        # SCAN
        if not scan:
            scan = extract_scan(tags)

        # CANTILEVER + FEEDBACK
        for value in tags.values():
            if isinstance(value, str) and "cantilever-calibration-info" in value:
                c, f = extract_cantilever_and_feedback(value)
                cantilever.update(c)
                feedback.update(f)


# ============================================================
# IMPRESIÓN FINAL
# ============================================================

print("=== METADATOS SCAN ===")
for k, v in scan.items():
    print(f"{k}: {v}")

print("\n=== METADATOS CANTILEVER ===")
for k, v in cantilever.items():
    print(f"{k}: {v}")

print("\n=== METADATOS FEEDBACK ===")
for k, v in feedback.items():
    print(f"{k}: {v}")