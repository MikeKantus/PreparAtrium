# gui/kymo_manager.py
# Manager for coordinating kymograph analysis, including file handling, in-memory kymos, and user interactions.
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox, RectangleSelector, RadioButtons

from .kymo_controller import KymoController
from .kymo_analyzer import KymoAnalyzer
from datetime import datetime


class KymoManager:
    def __init__(self, file_paths=None, pixel_size=None, kymo_array=None, kymo_arrays=None, metadata=None):
        """
        KymoManager supports two modes:
        - file-based: pass file_paths (list) and pixel_size (float) to load kymos from disk
                - memory-based: pass kymo_array or kymo_arrays and optional metadata (dict) to load
                    one or more in-memory kymographs without disk I/O.
        """
        # Normalize inputs
        self.file_paths = list(file_paths) if file_paths else []
        self.pixel_size = pixel_size
        self.current_index = 0

        # In-memory kymo support
        self._in_memory_kymos = []
        self._kymo_metadata = dict(metadata) if metadata is not None else {}
        if kymo_arrays is not None:
            self._in_memory_kymos = [np.asarray(kymo, dtype=float) for kymo in kymo_arrays]
        elif kymo_array is not None:
            self._in_memory_kymos = [np.asarray(kymo_array, dtype=float)]
        if self._in_memory_kymos:
            # Prefer metadata pixel_size if provided
            if self.pixel_size is None:
                self.pixel_size = self._kymo_metadata.get("pixel_size") or self._kymo_metadata.get("pixel_size_nm")

        # State per file
        self.kymo_states = {}

        # Current state (analysis)
        self.detected_ys = None
        self.detected_values = None
        self.detected_segments = []
        self.accepted_segments = set()

        self.manual_fit_mode = False
        self.manual_pick_mode = False
        self.profile_pick_mode = False
        self.pick_start = None
        self.pick_end = None

        # List of lines (slopes)
        self.lines = []
        self.next_line_id = 0

        # Saved slopes
        self.pinned_values = []
        self.slope_list_offset = 0
        self.saved_list_offset = 0

        # Template profiles
        self.profiles = []
        self.active_profile = None
        self.profile_colors = ['yellow', 'cyan', 'magenta', 'orange', 'white']

        # Create main figure and axes
        self.fig = plt.figure(figsize=(15, 10))

        # Layout: controls, image, profile, list
        self.ax_controls = plt.axes([0.02, 0.20, 0.22, 0.72])
        self.ax_controls.set_xticks([])
        self.ax_controls.set_yticks([])
        self.ax_controls.set_title("Controls")
        self.ax_metadata = plt.axes([0.02, 0.02, 0.22, 0.14])
        self.ax_metadata.set_axis_off()
        self.ax_metadata.set_title("Metadata", loc="left")

        self.ax_img = plt.axes([0.28, 0.38, 0.57, 0.48])
        self.ax_len = plt.axes([0.28, 0.08, 0.57, 0.20])

        # Panel for boxplot of saved slopes
        self.ax_boxplot = plt.axes([0.70, 0.10, 0.15, 0.15])
        self.ax_boxplot.set_xticks([])
        self.ax_boxplot.set_yticks([])
        self.ax_boxplot.set_title("Slope distribution", fontsize=5)

        self.ax_kymo_index = self.fig.add_axes([0.73, 0.91, 0.20, 0.04])
        self.ax_kymo_index.set_axis_off()

        # Slopes panel (top)
        self.ax_list = plt.axes([0.88, 0.52, 0.10, 0.34])
        self.ax_list.set_xticks([])
        self.ax_list.set_yticks([])
        self.ax_list.set_title("Slopes")

        # Saved slopes panel (bottom)
        self.ax_pinned = plt.axes([0.88, 0.08, 0.10, 0.34])
        self.ax_pinned.set_xticks([])
        self.ax_pinned.set_yticks([])
        self.ax_pinned.set_title("Saved")
        self.ax_list_scroll = plt.axes([0.865, 0.52, 0.010, 0.34])
        self.slider_list_scroll = Slider(self.ax_list_scroll, '', 0, 1, valinit=0, valstep=1, orientation='vertical')
        self.slider_list_scroll.on_changed(self._set_slope_list_offset)
        self.ax_pinned_scroll = plt.axes([0.865, 0.08, 0.010, 0.34])
        self.slider_pinned_scroll = Slider(self.ax_pinned_scroll, '', 0, 1, valinit=0, valstep=1, orientation='vertical')
        self.slider_pinned_scroll.on_changed(self._set_saved_list_offset)

        self.analyzer = KymoAnalyzer(self.ax_img, self.ax_len)
        self.controller = KymoController(self.analyzer)

        self.roi_selector = None
        self.roi_active = False

        self.build_controls()
        self._draw_metadata()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

        # If in-memory kymographs were provided, load the first one and enable navigation.
        if self._in_memory_kymos:
            try:
                self.load_current_kymo()
                plt.show(block=False)
                return
            except Exception as e:
                print("KymoManager: failed to load in-memory kymo:", e)

        # Otherwise, continue file-based initialization
        self.load_current_kymo()
        plt.show()

    def update_kymo_index_label(self):
        total = len(self._in_memory_kymos) or len(self.file_paths)
        current = self.current_index + 1
        self.ax_kymo_index.clear()
        self.ax_kymo_index.set_axis_off()
        self.ax_kymo_index.text(
            0.5, 0.5,
            f"Kymo {current} of {total}",
            ha='center', va='center', fontsize=12
        )

    def load_current_kymo(self):
        if self._in_memory_kymos:
            if self.current_index < 0 or self.current_index >= len(self._in_memory_kymos):
                return
            time_per_frame = self._kymo_metadata.get('time_per_frame')
            if time_per_frame is None:
                frame_rate = self._kymo_metadata.get('real_fps') or self._kymo_metadata.get('real_FPS') or self._kymo_metadata.get('fps') or self._kymo_metadata.get('frame_rate')
                time_per_frame = 1.0 / frame_rate if frame_rate else 1.0
            pixel_size = self.pixel_size or self._kymo_metadata.get('pixel_size') or self._kymo_metadata.get('pixel_size_nm') or 1.0
            self.controller.load_kymo_array(
                self._in_memory_kymos[self.current_index], pixel_size, time_per_frame
            )
            self.update_kymo_index_label()
            self._draw_metadata()
            return
        if not self.file_paths or self.current_index < 0 or self.current_index >= len(self.file_paths):
            return
        path = self.file_paths[self.current_index]
        self.controller.load_kymo(path, self.pixel_size)
        self.update_kymo_index_label()
        self._draw_metadata()

    def prev_kymo(self, event=None):
        total = len(self._in_memory_kymos) or len(self.file_paths)
        if total and self.current_index > 0:
            self.current_index -= 1
            self.load_current_kymo()

    def next_kymo(self, event=None):
        total = len(self._in_memory_kymos) or len(self.file_paths)
        if total and self.current_index < total - 1:
            self.current_index += 1
            self.load_current_kymo()

    # -------------------------
    # CONTROL PANEL
    # -------------------------
    def build_controls(self):
        def control_axes(bounds):
            return self.ax_controls.inset_axes(bounds)

        xL = 0.05
        w = 0.90
        xR = 0.53
        y = 0.87
        dy = 0.06

        # Navigation buttons between kymos
        ax_prev_kymo = plt.axes([0.35, 0.95, 0.10, 0.04])
        self.button_prev_kymo = Button(ax_prev_kymo, '← Previous')
        self.button_prev_kymo.on_clicked(self.prev_kymo)

        ax_next_kymo = plt.axes([0.65, 0.95, 0.10, 0.04])
        self.button_next_kymo = Button(ax_next_kymo, 'Next →')
        self.button_next_kymo.on_clicked(self.next_kymo)

        # Smoothing (left column)
        ax_smooth = control_axes([xL, y, w, 0.07])
        self.radio_smooth = RadioButtons(ax_smooth, ('Gaussian', 'Median', 'None'))
        self.radio_smooth.on_clicked(self.controller.set_smoothing_mode)

        y = 0.78
        ax_sigma = control_axes([xL, y, w, 0.045])
        self.slider_sigma = Slider(ax_sigma, 'Filter', 1, 9, valinit=3, valstep=1)
        self.slider_sigma.on_changed(self.controller.set_smoothing_param)

        ax_prev = control_axes([0.05, 0.18, 0.42, 0.05])
        self.button_preview = Button(ax_prev, 'Preview')
        self.button_preview.on_clicked(lambda e: self.controller.preview_smoothing())

        ax_apply = control_axes([0.53, 0.18, 0.42, 0.05])
        self.button_apply = Button(ax_apply, 'Apply')
        self.button_apply.on_clicked(lambda e: self.controller.apply_smoothing())

        # Right column (brightness/contrast, time, Z)
        y = 0.70

        ax_bright = control_axes([xL, y, w, 0.045])
        self.slider_bright = Slider(ax_bright, 'Brightness', -0.5, 0.5, valinit=0.0)
        self.slider_bright.on_changed(self.update_brightness)

        y = 0.62
        ax_contrast = control_axes([xL, y, w, 0.045])
        self.slider_contrast = Slider(ax_contrast, 'Contrast', 0.1, 3.0, valinit=1.0)
        self.slider_contrast.on_changed(self.update_contrast)

        y = 0.46
        ax_zthr = control_axes([xL, y, w, 0.045])
        self.slider_zthr = Slider(ax_zthr, 'Z thr', 0.0, 1.0, valinit=0.2)
        self.slider_zthr.on_changed(self.update_z_threshold)

        # Second left column: sensitivity + ROI + profiles
        y = 0.54
        ax_sens = control_axes([xL, y, w, 0.045])
        self.slider_sens = Slider(ax_sens, 'Sensitivity', 0.0, 1.0, valinit=0.5)
        self.slider_sens.on_changed(self.update_sensitivity)

        y = 0.38
        ax_roi = control_axes([0.05, y, 0.42, 0.05])
        self.button_roi = Button(ax_roi, 'ROI')
        self.button_roi.on_clicked(self.toggle_roi_mode)
        
        ax_roi_reset = control_axes([0.53, y, 0.42, 0.05])
        self.button_roi_reset = Button(ax_roi_reset, 'Reset ROI')
        self.button_roi_reset.on_clicked(self.reset_roi)

        # Second right column: detection, fit, picking, reset, export
        y = 0.32
        ax_detect = control_axes([0.05, y, 0.42, 0.05])
        self.button_detect = Button(ax_detect, 'Detect')
        self.button_detect.on_clicked(self.run_detection)

        ax_manualfit = control_axes([0.53, y, 0.42, 0.05])
        self.button_manualfit = Button(ax_manualfit, 'Manual fit')
        self.button_manualfit.on_clicked(self.toggle_manual_fit)

        y = 0.26
        ax_pick = control_axes([0.05, y, 0.42, 0.05])
        self.button_pick = Button(ax_pick, 'Manual picking')
        self.button_pick.on_clicked(self.toggle_manual_pick)

        ax_reset = control_axes([0.53, y, 0.42, 0.05])
        self.button_reset = Button(ax_reset, 'RESET')
        self.button_reset.on_clicked(self.reset_analysis)

        y = 0.1
        ax_export = control_axes([0.05, y, 0.90, 0.05])
        self.button_export = Button(ax_export, 'Export')
        self.button_export.on_clicked(self.export_lines)
    # -------------------------
    # UPDATES
    # -------------------------
    def update_brightness(self, val):
        self.controller.brightness = float(val)
        self.controller.apply_smoothing()

    def update_contrast(self, val):
        self.controller.contrast = float(val)
        self.controller.apply_smoothing()

    def update_time_scale_text(self, text):
        try:
            self.controller.time_per_frame = float(text)
            self.refresh_plots()
        except:
            print("Invalid value.")

    def update_z_scale(self, text):
        self.controller.set_z_max(text)

    def update_sensitivity(self, val):
        self.controller.edge_sensitivity = float(val)

    def update_z_threshold(self, val):
        self.controller.z_threshold = float(val)
        # Detection will use this value in the next run.

    # -------------------------
    # ROI
    # -------------------------
    def toggle_roi_mode(self, event=None):
        if self.roi_active:
            self.disable_roi_selector()
        else:
            self.enable_roi_selector()

    def enable_roi_selector(self):
        self._deactivate_modes(except_mode='roi')
        self.roi_active = True

        if self.roi_selector is not None:
            self.roi_selector.set_active(True)
            return

        def onselect(eclick, erelease):
            x0, y0 = eclick.xdata, erelease.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if None in (x0, y0, x1, y1):
                return
            image = self.controller.kymo_smooth
            if image is None:
                return
            height, width = image.shape
            px0 = int(np.clip(x0 / self.controller.pixel_size, 0, width - 1))
            px1 = int(np.clip(x1 / self.controller.pixel_size, 0, width - 1))
            py0 = int(np.clip(y0 / self.controller.time_per_frame, 0, height - 1))
            py1 = int(np.clip(y1 / self.controller.time_per_frame, 0, height - 1))
            self.controller.apply_roi(px0, px1, py0, py1)
            self.analyzer.draw_roi(x0, x1, y0, y1)
            self.fig.canvas.draw_idle()

        self.roi_selector = RectangleSelector(
            self.ax_img, onselect, useblit=True,
            button=[1], minspanx=5, minspany=5,
            interactive=True
        )

    def disable_roi_selector(self):
        self.roi_active = False
        if self.roi_selector is not None:
            self.roi_selector.set_active(False)

    def reset_roi(self, event=None):
        self.controller.reset_roi()
        self.fig.canvas.draw_idle()

    def run_detection(self, event=None):
        self.controller.profiles = self.profiles
        ys, xl, xr, values = self.controller.detect_polymer_edges()
        if ys is None or len(ys) == 0:
            self.detected_ys = None
            self.detected_values = None
            self.detected_segments = []
            self.accepted_segments = set()
            self.lines = []
            self._draw_slope_lists()
            return

        self.detected_ys = ys
        self.detected_values = values
        self.detected_segments = list(self.controller.detect_slope_segments(ys, values))
        self.accepted_segments = set()

        image = self.controller.apply_brightness_contrast(self.controller.kymo_roi)
        self.analyzer.draw_image(
            image, pixel_size=self.controller.pixel_size,
            time_per_frame=self.controller.time_per_frame,
        )
        self.analyzer.draw_edges(
            ys, xl, xr, self.controller.pixel_size, self.controller.time_per_frame,
        )
        self.analyzer.draw_length_plot(ys, values, self.controller.time_per_frame)
        self.analyzer.draw_slope_markers(
            ys, values, self.detected_segments, self.accepted_segments,
            self.controller.time_per_frame,
        )
        self._rebuild_slope_list()

    def toggle_manual_fit(self, event=None):
        self.manual_fit_mode = not self.manual_fit_mode
        if self.manual_fit_mode:
            self._deactivate_modes(except_mode='manual_fit')

    def toggle_manual_pick(self, event=None):
        self.manual_pick_mode = not self.manual_pick_mode
        if self.manual_pick_mode:
            self._deactivate_modes(except_mode='manual_pick')
        self.pick_start = None
        self.pick_end = None
        self.button_pick.label.set_text('Cancel picking' if self.manual_pick_mode else 'Manual picking')
        self.fig.canvas.draw_idle()

    def _deactivate_modes(self, except_mode=None):
        if except_mode != 'roi':
            self.disable_roi_selector()
        if except_mode != 'manual_fit':
            self.manual_fit_mode = False
        if except_mode != 'manual_pick':
            self.manual_pick_mode = False
            self.pick_start = None
            self.pick_end = None
            if hasattr(self, 'button_pick'):
                self.button_pick.label.set_text('Manual picking')

    def _draw_metadata(self):
        self.ax_metadata.clear()
        self.ax_metadata.set_axis_off()
        self.ax_metadata.set_title("Metadata", loc="left")
        metadata = self._kymo_metadata
        rows = (
            ("Frames", self.controller.kymo_raw.shape[0] if self.controller.kymo_raw is not None else "-"),
            ("Pixel size", metadata.get("pixel_size_nm") or metadata.get("pixel_size") or self.pixel_size or "-"),
            ("X range (nm)", metadata.get("x_range_nm", "-")),
            ("FPS", metadata.get("real_fps") or metadata.get("real_FPS") or metadata.get("fps") or metadata.get("frame_rate") or "-"),
            ("Channel", metadata.get("channel", "-")),
        )
        for row, (label, value) in enumerate(rows):
            self.ax_metadata.text(0.02, 0.88 - row * 0.18, f"{label}: {value}", transform=self.ax_metadata.transAxes, va='top')

    def _rebuild_slope_list(self):
        self.lines = []
        for index, (start, end) in enumerate(self.detected_segments):
            fit = self.controller.compute_fit_for_segment(self.detected_ys, self.detected_values, start, end)
            if fit is not None:
                self.lines.append({"index": index, "segment": (start, end), "slope": fit[0]})
        self.next_line_id = len(self.lines)
        self._draw_slope_lists()

    def _set_slope_list_offset(self, value):
        self.slope_list_offset = int(value)
        self._draw_slope_lists()

    def _set_saved_list_offset(self, value):
        self.saved_list_offset = int(value)
        self._draw_slope_lists()

    @staticmethod
    def _configure_scrollbar(slider, maximum, value):
        maximum = max(0, maximum)
        slider.valmax = maximum
        slider.ax.set_ylim(0, max(1, maximum))
        slider.eventson = False
        slider.set_val(min(value, maximum))
        slider.eventson = True

    def _draw_slope_lists(self):
        self.ax_list.clear()
        self.ax_list.set_axis_off()
        self.ax_list.set_title("Slopes")
        self.ax_pinned.clear()
        self.ax_pinned.set_axis_off()
        self.ax_pinned.set_title("Saved")
        available_lines = [line for line in self.lines if line['index'] not in self.accepted_segments]
        saved_lines = [line for line in self.lines if line['index'] in self.accepted_segments]
        visible_rows = 9
        self._configure_scrollbar(self.slider_list_scroll, len(available_lines) - visible_rows, self.slope_list_offset)
        self._configure_scrollbar(self.slider_pinned_scroll, len(saved_lines) - visible_rows, self.saved_list_offset)
        slope_start = self.slope_list_offset
        saved_start = self.saved_list_offset

        for row, line in enumerate(available_lines[slope_start:slope_start + visible_rows]):
            y = 0.94 - row * 0.10
            self.ax_list.text(0.02, y, '✓', color='green', transform=self.ax_list.transAxes, va='center', fontsize=12)
            self.ax_list.text(0.20, y, f"{line['slope']:.3g} nm/s", transform=self.ax_list.transAxes, va='center', fontsize=8)
            self.ax_list.text(0.88, y, '✗', color='red', transform=self.ax_list.transAxes, va='center', fontsize=12)
        for row, line in enumerate(saved_lines[saved_start:saved_start + visible_rows]):
            y = 0.94 - row * 0.10
            self.ax_pinned.text(0.05, y, f"{line['slope']:.3g} nm/s", transform=self.ax_pinned.transAxes, va='center', fontsize=8)
        self.fig.canvas.draw_idle()

    def reset_analysis(self, event=None):
        self.controller.apply_smoothing()
        self.fig.canvas.draw_idle()

    def export_lines(self, event=None):
        pass

    def refresh_plots(self):
        if self.controller.kymo_roi is not None:
            img = self.controller.apply_brightness_contrast(self.controller.kymo_roi)
            self.analyzer.draw_image(img, pixel_size=self.pixel_size, time_per_frame=self.controller.time_per_frame)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is self.ax_list and event.xdata is not None and event.ydata is not None:
            row = int(round((0.94 - event.ydata) / 0.10))
            available_lines = [line for line in self.lines if line['index'] not in self.accepted_segments]
            line_position = self.slope_list_offset + row
            if 0 <= row < 9 and line_position < len(available_lines):
                line_index = available_lines[line_position]['index']
                if event.xdata < 0.16:
                    self.accepted_segments.add(line_index)
                elif event.xdata > 0.80:
                    self.accepted_segments.discard(line_index)
                self._draw_slope_lists()
            return
        if not self.manual_pick_mode or event.inaxes is not self.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        point = (event.xdata, event.ydata)
        if self.pick_start is None:
            self.pick_start = point
            self.analyzer.draw_manual_line(point, point, color='w')
            return

        self.pick_end = point
        ys_pix, xs_pix = self.controller.manual_pick_profile(self.pick_start, self.pick_end)
        if len(ys_pix) > 0:
            values = self.controller.kymo_roi[ys_pix, xs_pix] * self.controller.pixel_size
            self.detected_ys = ys_pix
            self.detected_values = values
            self.analyzer.draw_manual_line(self.pick_start, self.pick_end, color='w')
            self.analyzer.draw_length_plot(ys_pix, values, self.controller.time_per_frame)
            if len(ys_pix) >= 2:
                fit = self.controller.compute_fit_for_segment(ys_pix, values, 0, len(ys_pix) - 1)
                if fit is not None:
                    self.lines.append({
                        "index": self.next_line_id,
                        "segment": (0, len(ys_pix) - 1),
                        "slope": fit[0],
                    })
                    self.next_line_id += 1
                    self._draw_slope_lists()

        self.pick_start = None
        self.pick_end = None
        self.fig.canvas.draw_idle()

    def on_pick(self, event):
        pass
