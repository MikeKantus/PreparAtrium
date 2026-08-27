import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox, RectangleSelector, RadioButtons

from .kymo_controller import KymoController
from .kymo_analyzer import KymoAnalyzer
from datetime import datetime


class KymoManager:
    def __init__(self, file_paths=None, pixel_size=None, kymo_array=None, metadata=None):
        """
        KymoManager supports two modes:
        - file-based: pass file_paths (list) and pixel_size (float) to load kymos from disk
        - memory-based: pass kymo_array (np.ndarray) and optional metadata (dict) to load a
          single in-memory kymogram without disk I/O.
        """
        # Normalize inputs
        self.file_paths = list(file_paths) if file_paths else []
        self.pixel_size = pixel_size
        self.current_index = 0

        # In-memory kymo support
        self._in_memory_kymo = None
        self._kymo_metadata = dict(metadata) if metadata is not None else {}
        if kymo_array is not None:
            self._in_memory_kymo = np.asarray(kymo_array, dtype=float)
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

        # Template profiles
        self.profiles = []
        self.active_profile = None
        self.profile_colors = ['yellow', 'cyan', 'magenta', 'orange', 'white']

        # Create main figure and axes
        self.fig = plt.figure(figsize=(15, 10))

        # Layout: controls, image, profile, list
        self.ax_controls = plt.axes([0.02, 0.05, 0.26, 0.90])
        self.ax_controls.set_xticks([])
        self.ax_controls.set_yticks([])
        self.ax_controls.set_title("Controls")

        self.ax_img = plt.axes([0.30, 0.30, 0.55, 0.65])
        self.ax_len = plt.axes([0.30, 0.05, 0.55, 0.20])

        # Panel for boxplot of saved slopes
        self.ax_boxplot = plt.axes([0.6, 0.5, 0.25, 0.4])
        self.ax_boxplot.set_xticks([])
        self.ax_boxplot.set_yticks([])
        self.ax_boxplot.set_title("Slope distribution", fontsize=10)

        self.ax_kymo_index = self.fig.add_axes([0.78, 0.96, 0.20, 0.03])
        self.ax_kymo_index.set_axis_off()

        # Slopes panel (top)
        self.ax_list = plt.axes([0.87, 0.55, 0.11, 0.40])
        self.ax_list.set_xticks([])
        self.ax_list.set_yticks([])
        self.ax_list.set_title("Slopes")

        # Saved slopes panel (bottom)
        self.ax_pinned = plt.axes([0.87, 0.05, 0.11, 0.40])
        self.ax_pinned.set_xticks([])
        self.ax_pinned.set_yticks([])
        self.ax_pinned.set_title("Saved")

        self.analyzer = KymoAnalyzer(self.ax_img, self.ax_len)
        self.controller = KymoController(self.analyzer)

        self.roi_selector = None
        self.roi_active = False

        self.build_controls()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

        # If an in-memory kymo was provided, load it directly and return (no file I/O)
        if self._in_memory_kymo is not None:
            try:
                time_per_frame = self._kymo_metadata.get('time_per_frame')
                if time_per_frame is None:
                    # try frame_rate -> time_per_frame
                    fr = self._kymo_metadata.get('frame_rate') or self._kymo_metadata.get('real_fps') or self._kymo_metadata.get('fps')
                    time_per_frame = 1.0 / fr if fr else 1.0

                psize = self.pixel_size or self._kymo_metadata.get('pixel_size') or self._kymo_metadata.get('pixel_size_nm') or 1.0

                self.controller.load_kymo_array(self._in_memory_kymo, pixel_size=psize, time_per_frame=time_per_frame)
                self.update_kymo_index_label()
                # Non-blocking show
                plt.show(block=False)
                return
            except Exception as e:
                print("KymoManager: failed to load in-memory kymo:", e)

        # Otherwise, continue file-based initialization
        self.load_current_kymo()
        plt.show()

    def update_kymo_index_label(self):
        total = len(self.file_paths)
        current = self.current_index + 1
        self.ax_kymo_index.clear()
        self.ax_kymo_index.set_axis_off()
        self.ax_kymo_index.text(
            0.5, 0.5,
            f"Kymo {current} of {total}",
            ha='center', va='center', fontsize=12
        )

    # -------------------------
    # CONTROL PANEL
    # -------------------------
    def build_controls(self):

        xL = 0.03
        w = 0.11
        xR = 0.15
        y = 0.88
        dy = 0.06

        # Navigation buttons between kymos
        ax_prev_kymo = plt.axes([0.35, 0.95, 0.10, 0.04])
        self.button_prev_kymo = Button(ax_prev_kymo, '← Previous')
        self.button_prev_kymo.on_clicked(self.prev_kymo)

        ax_next_kymo = plt.axes([0.65, 0.95, 0.10, 0.04])
        self.button_next_kymo = Button(ax_next_kymo, 'Next →')
        self.button_next_kymo.on_clicked(self.next_kymo)

        # Smoothing (left column)
        ax_smooth = plt.axes([xL, y, w, dy])
        self.radio_smooth = RadioButtons(ax_smooth, ('Gaussian', 'Median', 'None'))
        self.radio_smooth.on_clicked(self.controller.set_smoothing_mode)

        y -= dy
        ax_sigma = plt.axes([xL, y, w, dy])
        self.slider_sigma = Slider(ax_sigma, 'Filter', 1, 9, valinit=3, valstep=1)
        self.slider_sigma.on_changed(self.controller.set_smoothing_param)

        y -= dy
        ax_prev = plt.axes([xL, y, w, dy])
        self.button_preview = Button(ax_prev, 'Preview')
        self.button_preview.on_clicked(lambda e: self.controller.preview_smoothing())

        y -= dy
        ax_apply = plt.axes([xL, y, w, dy])
        self.button_apply = Button(ax_apply, 'Apply')
        self.button_apply.on_clicked(lambda e: self.controller.apply_smoothing())

        # Right column (brightness/contrast, time, Z)
        y = 0.88

        ax_bright = plt.axes([xR, y, w, dy])
        self.slider_bright = Slider(ax_bright, 'Brightness', -0.5, 0.5, valinit=0.0)
        self.slider_bright.on_changed(self.update_brightness)

        y -= dy
        ax_contrast = plt.axes([xR, y, w, dy])
        self.slider_contrast = Slider(ax_contrast, 'Contrast', 0.1, 3.0, valinit=1.0)
        self.slider_contrast.on_changed(self.update_contrast)

        y -= dy
        ax_time = plt.axes([xR, y, w, dy])
        self.text_time = TextBox(ax_time, 's/frame', initial="1.0")
        self.text_time.on_submit(self.update_time_scale_text)

        y -= dy
        ax_z = plt.axes([xR, y, w, dy])
        self.text_zmax = TextBox(ax_z, 'Z max', initial="1.0")
        self.text_zmax.on_submit(self.update_z_scale)
        
        y -= dy
        ax_zthr = plt.axes([xR, y, w, dy])
        self.slider_zthr = Slider(ax_zthr, 'Z thr', 0.0, 1.0, valinit=0.2)
        self.slider_zthr.on_changed(self.update_z_threshold)

        # Second left column: sensitivity + ROI + profiles
        y = 0.60
        ax_sens = plt.axes([xL, y, w, dy])
        self.slider_sens = Slider(ax_sens, 'Sensitivity', 0.0, 1.0, valinit=0.5)
        self.slider_sens.on_changed(self.update_sensitivity)

        y -= dy
        ax_roi = plt.axes([xL, y, w, dy])
        self.button_roi = Button(ax_roi, 'ROI')
        self.button_roi.on_clicked(self.toggle_roi_mode)
        
        y -= dy
        ax_open_roi = plt.axes([xL, y, w, dy])
        self.button_open_roi = Button(ax_open_roi, 'Open ROI')
        self.button_open_roi.on_clicked(lambda e: self.open_roi_window())

        y -= dy
        ax_roi_reset = plt.axes([xL, y, w, dy])
        self.button_roi_reset = Button(ax_roi_reset, 'Reset ROI')
        self.button_roi_reset.on_clicked(self.reset_roi)

        y -= dy
        ax_addprof = plt.axes([xL, y, w, dy])
        self.button_add_profile = Button(ax_addprof, '+ Profile')
        self.button_add_profile.on_clicked(self.add_profile)

        # Second right column: detection, fit, picking, reset, export
        y = 0.60
        ax_detect = plt.axes([xR, y, w, dy])
        self.button_detect = Button(ax_detect, 'Detect')
        self.button_detect.on_clicked(self.run_detection)

        y -= dy
        ax_manualfit = plt.axes([xR, y, w, dy])
        self.button_manualfit = Button(ax_manualfit, 'Manual fit')
        self.button_manualfit.on_clicked(self.toggle_manual_fit)

        y -= dy
        ax_pick = plt.axes([xR, y, w, dy])
        self.button_pick = Button(ax_pick, 'Manual picking')
        self.button_pick.on_clicked(self.toggle_manual_pick)

        y -= dy
        ax_reset = plt.axes([xR, y, w, dy])
        self.button_reset = Button(ax_reset, 'RESET')
        self.button_reset.on_clicked(self.reset_analysis)

        y -= dy
        ax_export = plt.axes([xR, y, w, dy])
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
    def toggle_roi_mode(self, event):
        if self.roi_active:
            self.disable_roi_selector()
        else:
            self.enable_roi_selector()

    def enable_roi_selector(self):
        self.roi_active = True

        if self.roi_selector is not None:
            self.roi_selector.set_active(True)
            return

        def onselect(eclick, erelease):
            x0, y0 = eclick.xdata, erelease.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if None in (x0, y0, x1, y1):
                return

        ...
