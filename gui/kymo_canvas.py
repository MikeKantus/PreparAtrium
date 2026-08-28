# gui/kymo_canvas.py
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import (
    QPainter, QPen, QColor, QPixmap, QImage
)
from PySide6.QtCore import Qt, QPointF

from core.ui_utils import frame_to_qimage_safe


class KymoCanvas(QWidget):
    """
    Canvas visual para el panel de kymogramas.
    - Dibuja frames del vídeo ECC final
    - Dibuja líneas manuales
    - Dibuja polímeros detectados (centerlines)
    - Permite edición avanzada (añadir/mover puntos)
    - Permite zoom/pan fluido
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.panel = None


        # Estado de visualización
        self.current_frame = 0
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Edición de líneas
        self.active_line = []          # línea que el usuario está editando
        self.selected_point = None     # índice del punto seleccionado
        self.dragging = False

        # Panning con botón derecho
        self._right_drag = False
        self._last_mouse_pos = None

        # Parámetros visuales
        self.line_width = 2
        self.manual_color = QColor(255, 0, 0)
        self.polymer_color = QColor(255, 255, 0)  # polymers: yellow
        self.active_color = QColor(0, 200, 255)

        self.setMouseTracking(True)

    # ============================================================
    #                   DIBUJO PRINCIPAL
    # ============================================================

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # --- dibujar frame ---
        frame = self.model.get_frame(self.current_frame)
        qimg = frame_to_qimage_safe(frame)
        pix = QPixmap.fromImage(qimg)

        # aplicar zoom
        pix = pix.scaled(pix.width() * self.zoom, pix.height() * self.zoom,
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_x = self.pan_x + max(0, (self.width() - pix.width()) / 2)
        image_y = self.pan_y + max(0, (self.height() - pix.height()) / 2)
        painter.drawPixmap(int(image_x), int(image_y), pix)

        # --- dibujar panorama overlay si existe y está activado ---
        if getattr(self.panel, 'panorama_overlay', False):
            pano = self.model.build_panorama()
            try:
                qimg_p = frame_to_qimage_safe(pano)
                pix_p = QPixmap.fromImage(qimg_p)
                # escalar y centrar como el frame
                pix_p = pix_p.scaled(pix.width(), pix.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.setOpacity(0.35)
                painter.drawPixmap(int(image_x), int(image_y), pix_p)
                painter.setOpacity(1.0)
            except Exception:
                pass

        # --- dibujar polímeros detectados ---
        for poly in self.model.polymers:
            # draw mask as semi-transparent overlay
            mask = poly.get('mask')
            if mask is None:
                continue
            # convert mask to QImage
            try:
                h, w = mask.shape
                img = (mask.astype('uint8') * 255).astype('uint8')
                qimg_m = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
                pix_m = QPixmap.fromImage(qimg_m)
                pix_m = pix_m.scaled(pix.width(), pix.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.setOpacity(0.25)
                painter.drawPixmap(int(image_x), int(image_y), pix_m)
                painter.setOpacity(1.0)
            except Exception:
                pass
            # draw centerline
            for i in range(len(poly.get('centerline', [])) - 1):
                x0, y0 = poly['centerline'][i]
                x1, y1 = poly['centerline'][i+1]
                cx0 = x0 * self.zoom + image_x
                cy0 = y0 * self.zoom + image_y
                cx1 = x1 * self.zoom + image_x
                cy1 = y1 * self.zoom + image_y
                pen = QPen(self.polymer_color)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(int(cx0), int(cy0), int(cx1), int(cy1))

        # --- dibujar líneas manuales ---
        for line in self.model.manual_lines:
            self._draw_line(painter, line, self.manual_color)

        # --- dibujar línea activa ---
        if self.active_line:
            self._draw_line(painter, self.active_line, self.active_color)

    # ============================================================
    #                   DIBUJO DE LÍNEAS
    # ============================================================

    def _draw_line(self, painter, pts, color):
        if len(pts) < 2:
            return

        pen = QPen(color)
        pen.setWidth(self.line_width)
        painter.setPen(pen)

        for i in range(len(pts) - 1):
            x0, y0 = self._to_canvas(pts[i])
            x1, y1 = self._to_canvas(pts[i + 1])
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))

        # dibujar puntos
        for p in pts:
            x, y = self._to_canvas(p)
            painter.drawEllipse(QPointF(x, y), 3, 3)

    # ============================================================
    #                   COORDENADAS
    # ============================================================

    def _to_canvas(self, pt):
        """
        Convierte coordenadas de imagen → canvas (zoom + pan).
        """
        frame = self.model.get_frame(self.current_frame)
        x = pt[0] * self.zoom + self.pan_x + max(0, (self.width() - frame.shape[1] * self.zoom) / 2)
        y = pt[1] * self.zoom + self.pan_y + max(0, (self.height() - frame.shape[0] * self.zoom) / 2)
        return x, y

    def _from_canvas(self, x, y):
        """
        Convierte coordenadas canvas → imagen.
        """
        frame = self.model.get_frame(self.current_frame)
        offset_x = self.pan_x + max(0, (self.width() - frame.shape[1] * self.zoom) / 2)
        offset_y = self.pan_y + max(0, (self.height() - frame.shape[0] * self.zoom) / 2)
        ix = (x - offset_x) / self.zoom
        iy = (y - offset_y) / self.zoom
        return ix, iy

    # ============================================================
    #                   INTERACCIÓN DEL RATÓN
    # ============================================================

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            img_x, img_y = self._from_canvas(event.x(), event.y())
            frame = self.model.get_frame(self.current_frame)
            if not (0 <= img_x < frame.shape[1] and 0 <= img_y < frame.shape[0]):
                return

            # ¿estamos editando una línea activa?
            if self.active_line:
                idx = self._find_nearest_point(self.active_line, img_x, img_y)
                if idx is not None:
                    self.selected_point = idx
                    self.dragging = True
                else:
                    # añadir punto nuevo
                    self.active_line.append([img_x, img_y])
                    self.update()
            else:
                # empezar una nueva línea
                self.active_line = [[img_x, img_y]]
                self.update()

        elif event.button() == Qt.RightButton:
            # start panning
            self._right_drag = True
            self._last_mouse_pos = event.position() if hasattr(event, 'position') else event.pos()

    def mouseMoveEvent(self, event):
        if self._right_drag and self._last_mouse_pos is not None:
            pos = event.position() if hasattr(event, 'position') else event.pos()
            dx = pos.x() - self._last_mouse_pos.x()
            dy = pos.y() - self._last_mouse_pos.y()
            self.pan_x += dx
            self.pan_y += dy
            self._last_mouse_pos = pos
            self.update()
            return

        if self.dragging and self.selected_point is not None:
            img_x, img_y = self._from_canvas(event.x(), event.y())
            frame = self.model.get_frame(self.current_frame)
            img_x = max(0, min(frame.shape[1] - 1, img_x))
            img_y = max(0, min(frame.shape[0] - 1, img_y))
            self.active_line[self.selected_point] = [img_x, img_y]
            self.update()
            if self.panel:
                self.panel.update_preview()


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._right_drag = False
            self._last_mouse_pos = None
            return
        self.dragging = False
        self.selected_point = None

    # ============================================================
    #                   ZOOM / PAN
    # ============================================================

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        # position may be in QPointF or QPoint depending on Qt version
        pos = event.position() if hasattr(event, 'position') else event.pos()
        before_x, before_y = self._from_canvas(pos.x(), pos.y())
        if delta > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1

        self.zoom = max(0.1, min(self.zoom, 20))
        after_x, after_y = self._from_canvas(pos.x(), pos.y())
        # adjust pan so that the point under cursor stays fixed
        self.pan_x += (before_x - after_x) * self.zoom
        self.pan_y += (before_y - after_y) * self.zoom
        self.update()

    def keyPressEvent(self, event):
        step = 20
        if event.key() == Qt.Key_Left:
            self.pan_x += step
        elif event.key() == Qt.Key_Right:
            self.pan_x -= step
        elif event.key() == Qt.Key_Up:
            self.pan_y += step
        elif event.key() == Qt.Key_Down:
            self.pan_y -= step
        self.update()

    # ============================================================
    #                   UTILIDADES
    # ============================================================

    def _find_nearest_point(self, pts, x, y, tol=10):
        """
        Devuelve el índice del punto más cercano a (x,y) en coords imagen.
        """
        best = None
        best_dist = tol

        for i, (px, py) in enumerate(pts):
            d = np.hypot(px - x, py - y)
            if d < best_dist:
                best_dist = d
                best = i

        return best
