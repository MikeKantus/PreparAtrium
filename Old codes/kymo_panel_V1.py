import numpy as np
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
)
from PySide6.QtWidgets import QFileDialog
import json
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PySide6.QtCore import Qt, QPoint
# al resto de imports ya existentes
from core.ui_utils import frame_to_qimage_safe
from core.kymo_tools import extract_kymograph

def numpy_to_qimage(frame):
    h, w = frame.shape
    bytes_per_line = w
    return QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8)


class KymoPanel(QWidget):
    def __init__(self, stack, meta):
        super().__init__()

        self.stack = stack
# compatibilidad: exponer las frames con los nombres que usan otras funciones
        self.frames = stack
        self.ecc_frames = stack

        self.meta = meta
        self.current_frame = 0

        self.lines = []
        self.current_line = []

        # --- Widgets principales ---
        self.label_frame = QLabel()
        self.label_frame.setAlignment(Qt.AlignCenter)

        self.label_kymo_preview = QLabel("Kymo preview")
        self.label_kymo_preview.setAlignment(Qt.AlignCenter)

        # --- Botones ---
        self.btn_prev = QPushButton("Prev frame")
        self.btn_next = QPushButton("Next frame")
        self.btn_new_line = QPushButton("New line")
        self.btn_finish_line = QPushButton("Finish line")
        self.btn_delete_line = QPushButton("Delete last line")
        self.btn_export_kymos = QPushButton("Export all kymos")
        self.btn_go_kymolizer = QPushButton("Go to Kymolizer")

        # --- Conexiones ---
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_new_line.clicked.connect(self.start_new_line)
        self.btn_finish_line.clicked.connect(self.finish_line)
        self.btn_delete_line.clicked.connect(self.delete_last_line)
        self.btn_export_kymos.clicked.connect(self.export_all_kymos)
        self.btn_go_kymolizer.clicked.connect(self.go_to_kymolizer)

        # --- Layout vertical principal ---
        layout = QVBoxLayout()
        layout.addWidget(self.label_frame)

        # Navegación
        nav = QHBoxLayout()
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        layout.addLayout(nav)

        # Herramientas
        tools = QHBoxLayout()
        tools.addWidget(self.btn_new_line)
        tools.addWidget(self.btn_finish_line)
        tools.addWidget(self.btn_delete_line)
        tools.addWidget(self.btn_export_kymos)
        tools.addWidget(self.btn_go_kymolizer)
        layout.addLayout(tools)

        # --- Layout horizontal final ---
        main_layout = QHBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(self.label_kymo_preview)

        self.setLayout(main_layout)

        self.update_frame()


    # ============================================================
    #                   FRAME NAVIGATION
    # ============================================================

    def prev_frame(self):
        self.current_frame = max(0, self.current_frame - 1)
        self.update_frame()

    def next_frame(self):
        self.current_frame = min(len(self.stack) - 1, self.current_frame + 1)
        self.update_frame()

    # ============================================================
    #                   DRAWING LINES
    # ============================================================

    def start_new_line(self):
        self.current_line = []

    def finish_line(self):
        if len(self.current_line) > 1:
            self.lines.append(self.current_line.copy())
            self.current_line = []
            self.update_frame()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        lbl = getattr(self, "label_frame", None)
        if lbl is None:
            return

        pix = lbl.pixmap()
        if pix is None:
            return

        disp_w = pix.width()
        disp_h = pix.height()
        if disp_w == 0 or disp_h == 0:
            return

        lbl_w = lbl.width()
        lbl_h = lbl.height()
        x_offset = max(0, (lbl_w - disp_w) / 2.0)
        y_offset = max(0, (lbl_h - disp_h) / 2.0)

        pos = event.position() if hasattr(event, "position") else event.pos()
        click_x = pos.x()
        click_y = pos.y()

        rel_x = click_x - x_offset
        rel_y = click_y - y_offset
        if rel_x < 0 or rel_y < 0 or rel_x > disp_w or rel_y > disp_h:
            return

        frame = self.stack[self.current_frame]
        orig_h, orig_w = frame.shape[:2]
        if orig_w == 0 or orig_h == 0:
            return

        mapped_x = int(rel_x * (orig_w / float(disp_w)))
        mapped_y = int(rel_y * (orig_h / float(disp_h)))

        if not hasattr(self, "current_line") or self.current_line is None:
            self.current_line = []
        self.current_line.append((mapped_x, mapped_y))
        self.update_frame(self.current_frame)


    # ============================================================
    #                   FRAME DISPLAY WITH LINES
    # ============================================================

    def update_frame(self, idx=0):
        """
        Mostrar el frame idx de forma segura en el KymoPanel y dibujar las líneas
        guardadas (self.lines) y la línea en curso (self.current_line).
        """
        # Selección segura de frames (evita evaluar numpy arrays en contexto booleano)
        frames = None
        for name in ("frames", "ecc_frames", "stack"):
            candidate = getattr(self, name, None)
            if candidate is not None:
                frames = candidate
                break

        # Si no hay frames, limpiar label y salir
        if frames is None:
            for candidate in ("label_frame", "label_kymo_preview", "label_preview", "label_original", "label_kymo", "label"):
                if hasattr(self, candidate):
                    getattr(self, candidate).clear()
                    break
            return

        # Si frames existe pero está vacío (lista o array con len==0), salir también
        try:
            if len(frames) == 0:
                for candidate in ("label_frame", "label_kymo_preview", "label_preview", "label_original", "label_kymo", "label"):
                    if hasattr(self, candidate):
                        getattr(self, candidate).clear()
                        break
                return
        except Exception:
            # len no aplica: asumimos que frames es indexable
            pass

        # Normalizar índice
        idx = int(idx) if idx is not None else 0
        idx = max(0, min(idx, len(frames) - 1))
        frame = frames[idx]
        if frame is None:
            return

        # Conversión segura a QImage (preferir frame_to_qimage_safe)
        try:
            qimg = frame_to_qimage_safe(frame)
        except Exception:
            import numpy as np
            from PySide6.QtGui import QImage
            import cv2

            arr = np.asarray(frame)

            # Si es color, convertir a gris
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                try:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                except Exception:
                    arr = arr[..., 0]

            # Manejar NaNs
            if np.isnan(arr).any():
                arr = arr.copy()
                arr[np.isnan(arr)] = np.nanmin(arr)

            # Normalizar a uint8 si es necesario
            if arr.dtype != np.uint8:
                a = arr.astype(np.float32)
                a = a - np.nanmin(a)
                rng = np.nanmax(a)
                if rng == 0 or np.isnan(rng):
                    rng = 1.0
                arr = (a / rng * 255.0).astype(np.uint8)

            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)

            h, w = arr.shape[:2]
            bytes_per_line = arr.strides[0]
            qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()

        # Crear pixmap base sobre el que dibujar
        pix = QPixmap.fromImage(qimg)

        # Elegir el label disponible (prioridad)
        lbl = None
        for candidate in ("label_frame", "label_kymo_preview", "label_preview", "label_original", "label_kymo", "label"):
            if hasattr(self, candidate) and getattr(self, candidate) is not None:
                lbl = getattr(self, candidate)
                break

        if lbl is None:
            # No hay label conocido: nada que mostrar
            return

        # Escalar pixmap al tamaño del label manteniendo aspecto
        target_w = max(1, lbl.width())
        target_h = max(1, lbl.height())
        pix = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Preparar listas de líneas
        lines = getattr(self, "lines", []) or []
        current_line = getattr(self, "current_line", []) or []

        # dimensiones del pixmap mostrado (ya escalado)
        disp_w = float(pix.width())
        disp_h = float(pix.height())

        # dimensiones originales del frame
        orig_h, orig_w = frame.shape[:2]

        # Dibujar líneas sobre el pixmap con QPainter
        from PySide6.QtGui import QPainter, QPen, QColor
        painter = QPainter(pix)
        try:
            # Configurar pen (puedes ajustar color/anchura)
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)

            # Dibujar líneas ya guardadas
            for line in lines:
                if not line or len(line) < 2:
                    continue
                for i in range(len(line) - 1):
                    px1, py1 = line[i]
                    px2, py2 = line[i+1]
                    x1 = int(round(px1 * (disp_w / float(orig_w))))
                    y1 = int(round(py1 * (disp_h / float(orig_h))))
                    x2 = int(round(px2 * (disp_w / float(orig_w))))
                    y2 = int(round(py2 * (disp_h / float(orig_h))))
                    x1 = max(0, min(x1, int(disp_w) - 1))
                    y1 = max(0, min(y1, int(disp_h) - 1))
                    x2 = max(0, min(x2, int(disp_w) - 1))
                    y2 = max(0, min(y2, int(disp_h) - 1))
                    painter.drawLine(x1, y1, x2, y2)

            # Dibujar línea actual (si existe)
            if current_line and len(current_line) >= 2:
                pen_current = QPen(QColor(0, 255, 0))
                pen_current.setWidth(2)
                painter.setPen(pen_current)
                for i in range(len(current_line) - 1):
                    px1, py1 = current_line[i]
                    px2, py2 = current_line[i+1]
                    x1 = int(round(px1 * (disp_w / float(orig_w))))
                    y1 = int(round(py1 * (disp_h / float(orig_h))))
                    x2 = int(round(px2 * (disp_w / float(orig_w))))
                    y2 = int(round(py2 * (disp_h / float(orig_h))))
                    x1 = max(0, min(x1, int(disp_w) - 1))
                    y1 = max(0, min(y1, int(disp_h) - 1))
                    x2 = max(0, min(x2, int(disp_w) - 1))
                    y2 = max(0, min(y2, int(disp_h) - 1))
                    painter.drawLine(x1, y1, x2, y2)
        finally:
            painter.end()

        # Asignar pixmap final al label
        lbl.setPixmap(pix)

    # ============================================================
    #                   EXPORT KYMOGRAMS
    # ============================================================

    def export_all_kymos(self):
        if len(self.lines) == 0:
            print("No lines to export")
            return

        save_path = QFileDialog.getSaveFileName(
            self,
            "Save kymograms",
            "",
            "JSON Files (*.json)"
        )[0]

        if not save_path:
            return

        kymo_list = []

        for line in self.lines:
            kymo, axis_x_nm, axis_t_s = extract_kymograph(
                self.stack, line, self.meta
            )
            kymo_list.append({
                "line": line,
                "kymo": kymo.tolist(),
                "x_nm": axis_x_nm.tolist(),
                "t_s": axis_t_s.tolist()
            })

        with open(save_path, "w") as f:
            json.dump(kymo_list, f)

        print(f"Saved {len(kymo_list)} kymograms to {save_path}")

    def delete_last_line(self):
        if len(self.lines) > 0:
            self.lines.pop()
            self.update_frame()
    def go_to_kymolizer(self):
        print("Kymolizer will be implemented soon.")
    def update_kymo_preview(self):
        if len(self.lines) == 0:
            self.label_kymo_preview.clear()
            return

        last_line = self.lines[-1]

        # --- asegurar meta y pixel_size ---
        pixel_size = None
        if self.meta is not None:
            pixel_size = self.meta.get("pixel_size", None) or self.meta.get("pixel_size_nm", None) or self.meta.get("pixel_size_x", None)

        if pixel_size is None:
            # fallback seguro: 1.0 (documentar que es solo para visualización)
            pixel_size = 1.0
            print("WARNING: meta.pixel_size missing; using fallback pixel_size=1.0")

        # pasar una copia de meta con pixel_size garantizado
        meta_for_call = dict(self.meta) if self.meta else {}
        meta_for_call["pixel_size"] = pixel_size

        try:
            kymo, axis_x_nm, axis_t_s = extract_kymograph(self.stack, last_line, meta_for_call)
        except Exception as e:
            print("ERROR in extract_kymograph:", e)
            self.label_kymo_preview.setText("Kymo generation failed")
            return

        # Normalizar y mostrar
        import numpy as np
        from PySide6.QtGui import QImage, QPixmap
        kymo = np.asarray(kymo)
        if kymo.size == 0:
            self.label_kymo_preview.setText("Empty kymo")
            return

        kymo_img = (kymo - np.min(kymo)) / (np.max(kymo) - np.min(kymo) + 1e-12) * 255.0
        kymo_img = kymo_img.astype(np.uint8)

        qimg = QImage(kymo_img.data, kymo_img.shape[1], kymo_img.shape[0],
                    kymo_img.strides[0], QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self.label_kymo_preview.setPixmap(pix.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def finish_line(self):
        if len(self.current_line) > 1:
            self.lines.append(self.current_line.copy())
            self.current_line = []
            self.update_frame()
            self.update_kymo_preview()
    def numpy_to_qimage(frame):

    
        from PySide6.QtGui import QImage
        arr = _np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            # convertir a gris si es color
            try:
                import cv2
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            except Exception:
                arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError("numpy_to_qimage: frame must be 2D grayscale after conversion")
        if arr.dtype != _np.uint8:
            a = arr.astype(_np.float32)
            a = a - _np.nanmin(a)
            rng = _np.nanmax(a)
            if rng == 0 or _np.isnan(rng):
                rng = 1.0
            arr = (a / rng * 255.0).astype(_np.uint8)
        if not arr.flags['C_CONTIGUOUS']:
            arr = _np.ascontiguousarray(arr)
        h, w = arr.shape
        bytes_per_line = arr.strides[0]
        return QImage(arr.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
