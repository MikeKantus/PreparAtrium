import numpy as np
import h5py
import os

class AFMImage:
    """
    Clase compatible con PlayNano.AFMImage.
    Carga archivos JPK/HDF5 y expone:
        - data (stack 3D)
        - pixel_size_nm
        - x_range_nm
        - y_range_nm
        - frame_rate
        - line_rate
        - channel
        - metadata (dict completo)
    """

    def __init__(self, path):
        self.path = path
        self.data, self.metadata = self._load_hdf5(path)

        # Exponer atributos estilo PlayNano
        self.pixel_size_nm = self.metadata.get("pixel_size_nm")
        self.x_range_nm = self.metadata.get("x_range_nm")
        self.y_range_nm = self.metadata.get("y_range_nm")
        self.frame_rate = self.metadata.get("frame_rate")
        self.line_rate = self.metadata.get("line_rate")
        self.channel = self.metadata.get("channel", "unknown")

    # ---------------------------------------------------------
    # Loader HDF5 (JPK, H5-JPK, algunos ARIS/ASD)
    # ---------------------------------------------------------
    def _load_hdf5(self, path):
        with h5py.File(path, "r") as f:

            # Buscar dataset de imagen
            def find_dataset(group):
                for k, v in group.items():
                    if isinstance(v, h5py.Dataset):
                        if v.ndim in (2, 3):
                            return v
                    elif isinstance(v, h5py.Group):
                        res = find_dataset(v)
                        if res is not None:
                            return res
                return None

            ds = find_dataset(f)
            if ds is None:
                raise ValueError("No se encontró dataset de imagen en el archivo.")

            data = np.asarray(ds)
            if data.ndim == 2:
                data = data[np.newaxis, ...]

            # Extraer atributos del archivo y grupos
            attrs = dict(f.attrs)
            for name, grp in f.items():
                if hasattr(grp, "attrs"):
                    for k, v in grp.attrs.items():
                        attrs[k] = v

            # Construir metadatos limpios
            meta = {
                "pixel_size_nm": attrs.get("pixel_size_nm") or attrs.get("pixel_size"),
                "frame_rate": attrs.get("frame_rate") or attrs.get("fps"),
                "x_range_nm": attrs.get("x_range_nm") or attrs.get("x_range"),
                "y_range_nm": attrs.get("y_range_nm") or attrs.get("y_range"),
                "line_rate": attrs.get("line_rate") or attrs.get("lines_per_second"),
                "channel": attrs.get("channel") or attrs.get("channel_name"),
            }

            # Inferir rangos si faltan
            if meta["pixel_size_nm"] and data.ndim == 3:
                px = float(meta["pixel_size_nm"])
                h, w = data.shape[1], data.shape[2]
                if meta["x_range_nm"] is None:
                    meta["x_range_nm"] = px * w
                if meta["y_range_nm"] is None:
                    meta["y_range_nm"] = px * h

            return data, meta
