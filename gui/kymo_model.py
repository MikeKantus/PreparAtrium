# kymo_model.py
import numpy as np
import json
from core.kymo_tools import extract_kymograph


class KymoModel:
    """
    Modelo científico para el panel de kymogramas.
    No contiene GUI. Solo datos, metadatos y operaciones científicas.
    """

    def __init__(self, stack, meta):
        """
        Parameters
        ----------
        stack : np.ndarray
            Vídeo final ECC (frames, height, width)
        meta : dict
            Metadatos extendidos (pixel_size_nm, frame_rate, drift, ecc_transforms…)
        """
        self.stack = np.asarray(stack)
        self.meta = dict(meta) if meta is not None else {}
        # Nombre base del vídeo (si existe en metadatos)
        self.source_name = self.meta.get("source_name", "Exp")

        # --- líneas manuales dibujadas por el usuario ---
        self.manual_lines = []

        # --- polímeros detectados automáticamente (Size-o-Matic) ---
        # Cada polímero es un dict con:
        #   { "label", "mask", "centerline", "length", "mean_width" }
        self.polymers = []

        # --- centerlines (manuales o automáticos) ---
        self.centerlines = []
        # --- lista de kymogramas generados ---
        self.kymos = []

        # --- caché de kymogramas ---
        self._kymo_cache = {}

    # ============================================================
    #                   METADATOS FÍSICOS
    # ============================================================

    @property
    def pixel_size_nm(self):
        """
        Devuelve el tamaño de pixel en nm/pixel.
        """
        return (
            self.meta.get("pixel_size_nm")
            or self.meta.get("pixel_size")
            or self.meta.get("pixel_size_x")
            or 1.0
        )

    @property
    def frame_rate(self):
        """
        Devuelve la frecuencia de adquisición (frames/s).
        """
        return (
            self.meta.get("real_fps")
            or self.meta.get("fps")
            or self.meta.get("frame_rate")
        )

    @property
    def time_per_frame(self):
        """
        Devuelve el tiempo entre frames en segundos.
        """
        fr = self.frame_rate
        return 1.0 / fr if fr else None

    @property
    def total_time_s(self):
        """
        Devuelve la duración total del vídeo en segundos.
        """
        return self.meta.get("total_time_s")

    @property
    def n_frames(self):
        return len(self.stack)

    @property
    def shape(self):
        return self.stack.shape

    # ============================================================
    #                   LÍNEAS MANUALES
    # ============================================================

    def add_manual_line(self, pts):
        """
        Añade una línea manual dibujada por el usuario.
        pts : lista de (x, y) en coordenadas de imagen.
        """
        self.manual_lines.append(list(pts))
        self.centerlines.append(list(pts))

    def delete_manual_line(self, idx):
        if 0 <= idx < len(self.manual_lines):
            del self.manual_lines[idx]
            if idx < len(self.centerlines):
                del self.centerlines[idx]

    # ============================================================
    #                   POLÍMEROS DETECTADOS (Size-o-Matic)
    # ============================================================

    def add_polymer(self, polymer_dict):
        """
        Añade un polímero detectado automáticamente.
        polymer_dict debe contener:
            { "label", "mask", "centerline", "length", "mean_width" }
        """
        self.polymers.append(polymer_dict)
        self.centerlines.append(polymer_dict["centerline"])

    def clear_polymers(self):
        self.polymers = []
        self.centerlines = [list(line) for line in self.manual_lines]

    # ============================================================
    #                   KYMOGRAMAS
    # ============================================================

    def generate_kymograph(self, line):
        """
        Genera un kymograma para una línea dada.
        Usa extract_kymograph(stack, line, meta).
        Cachea resultados para acelerar la UI.
        """
        key = tuple((int(x), int(y)) for x, y in line)

        if key in self._kymo_cache:
            return self._kymo_cache[key]

        # Ensure normalized metadata (pixel size and frame rate / time per frame)
        meta_for_call = dict(self.meta) if self.meta else {}
        # Ensure pixel size (nm / pixel) is present
        meta_for_call.setdefault("pixel_size", self.pixel_size_nm)
        meta_for_call.setdefault("pixel_size_nm", self.pixel_size_nm)
        # Ensure frame_rate presence if available
        if self.frame_rate is not None:
            meta_for_call.setdefault("frame_rate", self.frame_rate)
            meta_for_call.setdefault("real_fps", self.frame_rate)

        # Call extractor with guaranteed metadata
        kymo, axis_x_nm, axis_t_s = extract_kymograph(self.stack, line, meta_for_call)
        result = {
            "kymo": kymo,
            "axis_x_nm": axis_x_nm,
            "axis_t_s": axis_t_s
        }

        self._kymo_cache[key] = result
        return result

    # ============================================================
    #                   UTILIDADES
    # ============================================================

    def get_frame(self, idx):
        """
        Devuelve el frame idx de forma segura.
        """
        idx = max(0, min(idx, len(self.stack) - 1))
        return self.stack[idx]

    def get_centerlines(self):
        """
        Devuelve todas las centerlines (manuales + automáticas).
        """
        return self.centerlines

    def clear_cache(self):
        """
        Limpia el caché de kymogramas.
        """
        self._kymo_cache = {}
# ============================================================
#                   KYMOGRAMAS GUARDADOS
# ============================================================

    def add_kymograph_from_line(self, line, label=None):
        result = self.generate_kymograph(line)

        # Frame asociado al kymo (primer punto de la línea)
        # Si la línea se dibuja en un frame concreto, el canvas ya sabe cuál es:
        frame_idx = getattr(self, "current_frame", None)
        if frame_idx is None:
            frame_idx = 0

        # Nombre base del experimento
        base = self.source_name

        # Nombre del kymo
        kymo_id = len(self.kymos) + 1
        label = label or f"{base}_frame{frame_idx:03d}_kymo_{kymo_id:03d}"

        entry = {
            "label": label,
            "line": list(line),
            "frame_idx": frame_idx,
            "kymo": result["kymo"],
            "axis_x_nm": result["axis_x_nm"],
            "axis_t_s": result["axis_t_s"]
        }

        entry["pixel_size_nm"] = self.pixel_size_nm
        entry["fps"] = self.frame_rate
        entry["time_per_frame_s"] = self.time_per_frame

        self.kymos.append(entry)
        return entry


    def delete_kymograph(self, idx):
        if 0 <= idx < len(self.kymos):
            del self.kymos[idx]

    def list_kymographs(self):
        return self.kymos

    def export_kymograph(self, idx, folder):
        import os, json, tifffile as tiff
        entry = self.kymos[idx]
        base = os.path.join(folder, entry["label"])

        # TIFF
        tiff.imwrite(base + ".tif", entry["kymo"].astype(np.float32))

        # JSON
        meta_out = {
            "pixel_size_nm": self.pixel_size_nm,
            "frame_rate_fps": self.frame_rate,
            "axis_x_nm": entry["axis_x_nm"].tolist(),
            "axis_t_s": entry["axis_t_s"].tolist(),
            "line_points": entry["line"]
        }
        with open(base + ".json", "w") as f:
            json.dump(meta_out, f, indent=2)

    def export_all_kymographs(self, folder):
        for i in range(len(self.kymos)):
            self.export_kymograph(i, folder)
