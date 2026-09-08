# gui/kymo_model.py
# Scientific model for handling kymograph data, including centerline extraction, polymer detection, and kymograph generation.
import numpy as np
import json
from scipy import ndimage

# skimage imports are optional — use them if available for nicer morphology/otsu
try:
    from skimage import filters, morphology, measure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
    # provide minimal fallbacks
    def _median_threshold(img):
        # fallback: simple percentile
        return np.percentile(img[~np.isnan(img)], 90)

    class morphology:
        @staticmethod
        def remove_small_objects(mask, min_size=10):
            # naive implementation using scipy.ndimage
            lbl, n = ndimage.label(mask)
            sizes = ndimage.sum(mask, lbl, range(1, n+1))
            good = np.zeros(n+1, dtype=bool)
            for i, s in enumerate(sizes, start=1):
                if s >= min_size:
                    good[i] = True
            return good[lbl]

    class measure:
        @staticmethod
        def label(mask):
            lbl, _ = ndimage.label(mask)
            return lbl

        @staticmethod
        def regionprops(lbl):
            # very small fallback: return objects with area and bbox approximations
            props = []
            for val in np.unique(lbl):
                if val == 0:
                    continue
                m = lbl == val
                coords = np.column_stack(np.where(m))
                miny, minx = coords.min(axis=0)
                maxy, maxx = coords.max(axis=0)
                area = coords.shape[0]
                class _Prop:
                    pass
                p = _Prop()
                p.bbox = (miny, minx, maxy, maxx)
                p.area = int(area)
                p.centroid = coords.mean(axis=0)
                p.major_axis_length = None
                p.minor_axis_length = None
                p.label = int(val)
                props.append(p)
            return props

    class filters:
        @staticmethod
        def threshold_otsu(img):
            return _median_threshold(img)

from core.kymo_tools import extract_kymograph


def _order_centerline_points(coords_yx):
    """Sort centerline coordinates (y, x) sequentially along the skeleton curve."""
    if len(coords_yx) == 0:
        return []
    if len(coords_yx) <= 2:
        return [(int(x), int(y)) for y, x in coords_yx]

    pts = np.column_stack((coords_yx[:, 1], coords_yx[:, 0])).astype(float)
    n = len(pts)

    centroid = pts.mean(axis=0)
    dists_from_center = np.linalg.norm(pts - centroid, axis=1)
    start_idx = np.argmax(dists_from_center)

    ordered = [pts[start_idx]]
    mask = np.ones(n, dtype=bool)
    mask[start_idx] = False

    curr_pt = pts[start_idx]
    while np.any(mask):
        rem_indices = np.where(mask)[0]
        rem_pts = pts[rem_indices]
        dists = np.linalg.norm(rem_pts - curr_pt, axis=1)
        nearest_rel = np.argmin(dists)
        nearest_idx = rem_indices[nearest_rel]

        if dists[nearest_rel] > 15.0:
            break

        curr_pt = pts[nearest_idx]
        ordered.append(curr_pt)
        mask[nearest_idx] = False

    return [(int(p[0]), int(p[1])) for p in ordered]


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
            Metadatos extendidos (pixel_size_nm, real_fps, drift, ecc_transforms…)
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

        # --- cachés ---
        self._kymo_cache = {}
        self._panorama_cache = None

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
    def real_fps(self):
        """
        Devuelve la frecuencia de adquisición (frames/s).
        """
        return (
            self.meta.get("real_fps")
            or self.meta.get("fps")
            or self.meta.get("frame_rate_fps")
        )

    @property
    def time_per_frame(self):
        """
        Devuelve el tiempo entre frames en segundos.
        """
        fr = self.real_fps
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

    def detect_polymers(self, min_size_px=50, elongation_thresh=2.0, sigma=1.0,
                        threshold_method="Otsu", sensitivity=1.0, frame_idx=None):
        """Detect candidate tubular polymers using filtering, thresholding + morphology.

        Results are stored in self.polymers as dicts with keys: mask, label, bbox, centroid, centerline.
        """
        if frame_idx is None:
            frame_idx = getattr(self, "current_frame", 0)
        frame_idx = max(0, min(self.n_frames - 1, frame_idx))
        raw_img = self.stack[frame_idx].astype(np.float32)

        # Normalizar
        img = raw_img - np.nanmin(raw_img)
        if np.nanmax(img) > 0:
            img = img / np.nanmax(img)

        # 1. Filtro Gaussiano para suavizar ruido
        if sigma > 0:
            if _HAS_SKIMAGE:
                filtered_img = filters.gaussian(img, sigma=sigma)
            else:
                filtered_img = ndimage.gaussian_filter(img, sigma=sigma)
        else:
            filtered_img = img

        # 2. Umbralización (Thresholding)
        if threshold_method == "Percentil":
            perc = max(10.0, min(99.0, 90.0 / max(0.1, sensitivity)))
            th = np.percentile(filtered_img[~np.isnan(filtered_img)], perc)
        else:  # "Otsu"
            try:
                if _HAS_SKIMAGE:
                    base_th = filters.threshold_otsu(filtered_img)
                else:
                    base_th = np.percentile(filtered_img[~np.isnan(filtered_img)], 90)
            except Exception:
                base_th = np.percentile(filtered_img[~np.isnan(filtered_img)], 90)
            th = base_th / max(0.1, sensitivity)

        mask = filtered_img >= th

        # 3. Limpieza morfológica
        min_obj_size = max(1, int(min_size_px // 4))
        mask = morphology.remove_small_objects(mask, min_size=min_obj_size)
        mask = morphology.binary_closing(mask, morphology.disk(3))
        mask = ndimage.binary_fill_holes(mask)

        labels = measure.label(mask)
        props = measure.regionprops(labels)

        self.polymers = []
        for i, prop in enumerate(props, start=1):
            if prop.area < min_size_px:
                continue
            minor_len = getattr(prop, 'axis_minor_length', getattr(prop, 'minor_axis_length', None))
            major_len = getattr(prop, 'axis_major_length', getattr(prop, 'major_axis_length', None))
            if minor_len is not None and minor_len > 0 and major_len is not None:
                elong = max(major_len / (minor_len + 1e-6), 1.0)
            else:
                elong = 1.0
            if elong < elongation_thresh:
                continue

            p = {
                'label': f'poly_{i}',
                'mask': (labels == prop.label),
                'bbox': prop.bbox,
                'centroid': prop.centroid,
                'area': prop.area,
                'major_axis_length': major_len,
                'minor_axis_length': minor_len,
            }
            # Skeletonize mask and order coordinates
            try:
                if _HAS_SKIMAGE:
                    skel = morphology.skeletonize(p['mask'])
                else:
                    skel = ndimage.distance_transform_edt(p['mask']) > 0
                coords = np.column_stack(np.where(skel))
                centerline = _order_centerline_points(coords)
            except Exception:
                centerline = []

            p['centerline'] = centerline
            self.polymers.append(p)

        self._panorama_cache = None
        return self.polymers

    # ============================================================
    #                   KYMOGRAMAS
    # ============================================================

    def generate_kymograph(self, line, radius_px=0, method='mean', subpixel=False):
        """
        Genera un kymograma para una línea dada.
        Usa extract_kymograph(stack, line, meta).
        Cachea resultados para acelerar la UI.
        """
        key = (tuple((int(x), int(y)) for x, y in line), int(radius_px), method, bool(subpixel))

        if key in self._kymo_cache:
            return self._kymo_cache[key]

        # Ensure normalized metadata (pixel size and frame rate / time per frame)
        meta_for_call = dict(self.meta) if self.meta else {}
        # Ensure pixel size (nm / pixel) is present
        meta_for_call.setdefault("pixel_size", self.pixel_size_nm)
        meta_for_call.setdefault("pixel_size_nm", self.pixel_size_nm)
        # Ensure real_fps presence if available
        if self.real_fps is not None:
            meta_for_call.setdefault("real_fps", self.real_fps)
            meta_for_call.setdefault("real_fps", self.real_fps)

        # Call extractor with guaranteed metadata
        kymo, axis_x_nm, axis_t_s = extract_kymograph(
            self.stack, line, meta_for_call, radius_px=radius_px, method=method)
        result = {
            "kymo": kymo,
            "axis_x_nm": axis_x_nm,
            "axis_t_s": axis_t_s
        }

        self._kymo_cache[key] = result
        return result

    # ============================================================
    #                   PANORAMA
    # ============================================================

    def build_panorama(self, mode='max'):
        """Build a panorama (projection) from the stack.

        mode: 'max' or 'mean'
        If drift offsets present in self.meta (drift_dx_px, drift_dy_px per frame) they will be used to place frames.
        """
        if self._panorama_cache is not None:
            return self._panorama_cache

        # Determine per-frame offsets from meta if available
        n = len(self.stack)
        offsets = None
        if 'drift_offsets_px' in self.meta:
            offsets = self.meta['drift_offsets_px']
            if len(offsets) != n:
                offsets = None
        elif 'drift_dx_px' in self.meta and 'drift_dy_px' in self.meta:
            dx = self.meta.get('drift_dx_px')
            dy = self.meta.get('drift_dy_px')
            if hasattr(dx, '__len__') and hasattr(dy, '__len__') and len(dx) == n and len(dy) == n:
                offsets = list(zip(dx, dy))

        H = self.stack.shape[1]
        W = self.stack.shape[2]

        if offsets is None:
            # simple projection matching original frame size
            if mode == 'max':
                pano = np.nanmax(self.stack, axis=0)
            else:
                pano = np.nanmean(self.stack, axis=0)
            self._pano_min_x = 0
            self._pano_min_y = 0
            self._panorama_cache = pano
            return pano

        # Compute required canvas size
        xs = [ox for ox, oy in offsets]
        ys = [oy for ox, oy in offsets]
        min_x = int(min(0, min(xs)))
        min_y = int(min(0, min(ys)))
        max_x = int(max(0, max(xs)))
        max_y = int(max(0, max(ys)))

        self._pano_min_x = min_x
        self._pano_min_y = min_y

        canvas_w = W + (max_x - min_x)
        canvas_h = H + (max_y - min_y)

        canvas = np.full((canvas_h, canvas_w), np.nan, dtype=np.float64)

        for i, frame in enumerate(self.stack):
            ox, oy = offsets[i]
            # top-left position in canvas
            x0 = int(ox - min_x)
            y0 = int(oy - min_y)
            x1 = x0 + W
            y1 = y0 + H
            # place frame
            sub = canvas[y0:y1, x0:x1]
            if mode == 'max':
                # combine with nanmax
                combined = np.fmax(np.nan_to_num(sub, nan=-np.inf), np.nan_to_num(frame, nan=-np.inf))
                combined[combined == -np.inf] = np.nan
                canvas[y0:y1, x0:x1] = combined
            else:
                # mean combination with counting
                if not hasattr(self, '_pano_acc'):
                    self._pano_acc = np.zeros_like(canvas, dtype=np.float64)
                    self._pano_cnt = np.zeros_like(canvas, dtype=np.int32)
                mask = ~np.isnan(frame)
                self._pano_acc[y0:y1, x0:x1][mask] += frame[mask]
                self._pano_cnt[y0:y1, x0:x1][mask] += 1

        if mode != 'max':
            with np.errstate(invalid='ignore', divide='ignore'):
                canvas = self._pano_acc / (self._pano_cnt + 1e-12)
            del self._pano_acc
            del self._pano_cnt

        self._panorama_cache = canvas
        return canvas

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
        Limpia el caché de kymogramas y panoramas.
        """
        self._kymo_cache = {}
        self._panorama_cache = None
        if hasattr(self, '_pano_acc'):
            del self._pano_acc
        if hasattr(self, '_pano_cnt'):
            del self._pano_cnt

    # ============================================================
    #                   KYMOGRAMAS GUARDADOS
    # ============================================================

    def add_kymograph_from_line(self, line, label=None, radius_px=0, method='mean', subpixel=False):
        result = self.generate_kymograph(line, radius_px=radius_px, method=method, subpixel=subpixel)

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
        entry["fps"] = self.real_fps
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
            "fps": self.real_fps,
            "axis_x_nm": entry["axis_x_nm"].tolist(),
            "axis_t_s": entry["axis_t_s"].tolist(),
            "line_points": entry["line"]
        }
        with open(base + ".json", "w") as f:
            json.dump(meta_out, f, indent=2)

    def export_all_kymographs(self, folder):
        for i in range(len(self.kymos)):
            self.export_kymograph(i, folder)
