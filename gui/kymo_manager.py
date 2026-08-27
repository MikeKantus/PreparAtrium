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
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if None in (x0, y0, x1, y1):
                return

            img = self.controller.kymo_smooth
            if img is None:
                return

            H, W = img.shape

            px0 = int(np.clip(x0 / self.controller.pixel_size, 0, W - 1))
            px1 = int(np.clip(x1 / self.controller.pixel_size, 0, W - 1))
            py0 = int(np.clip(y0 / self.controller.time_per_frame, 0, H - 1))
            py1 = int(np.clip(y1 / self.controller.time_per_frame, 0, H - 1))

            self.controller.apply_roi(px0, px1, py0, py1)

            self.detected_ys = None
            self.detected_values = None
            self.detected_segments = []
            self.accepted_segments = set()
            self.lines = []
            self.next_line_id = 0
            self.profiles = []
            self.active_profile = None
            self.analyzer.draw_length_plot(None, None, self.controller.time_per_frame)
            self.update_line_list()

            self.analyzer.draw_roi(x0, x1, y0, y1)

        self.roi_selector = RectangleSelector(
            self.ax_img,
            onselect,
            useblit=True,
            button=[1],
            minspanx=1,
            minspany=1,
            spancoords='data',
            interactive=False
        )

    def disable_roi_selector(self):
        self.roi_active = False
        if self.roi_selector is not None:
            self.roi_selector.set_active(False)

    def reset_roi(self, event):
        self.controller.reset_roi()
        self.detected_ys = None
        self.detected_values = None
        self.detected_segments = []
        self.accepted_segments = set()
        self.lines = []
        self.next_line_id = 0
        self.profiles = []
        self.active_profile = None
        self.analyzer.draw_length_plot(None, None, self.controller.time_per_frame)
        self.update_line_list()

    # -------------------------
    # TEMPLATE PROFILES
    # -------------------------
    def add_profile(self, event):
        new_id = len(self.profiles)
        name = f"Profile {new_id + 1}"
        color = self.profile_colors[new_id % len(self.profile_colors)]

        profile = {
            "id": new_id,
            "name": name,
            "color": color,
            "ys": None,
            "xs": None,
            "line_handle": None
        }

        self.profiles.append(profile)
        self.active_profile = new_id

        print(f"Profile created: {name}. Now draw the line on the image.")

        self.manual_pick_mode = False
        self.manual_fit_mode = False
        self.profile_pick_mode = True
        self.pick_start = None
        self.pick_end = None

    def handle_profile_pick(self, event):
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return

        if self.pick_start is None:
            self.pick_start = (x, y)
            print("Template start:", self.pick_start)
            return

        self.pick_end = (x, y)
        print("Template end:", self.pick_end)

        ys, xs = self.controller.manual_pick_profile(self.pick_start, self.pick_end)
        if len(ys) == 0:
            self.pick_start = None
            self.pick_end = None
            self.profile_pick_mode = False
            return

        profile = self.profiles[self.active_profile]
        profile["ys"] = ys
        profile["xs"] = xs

        if profile["line_handle"] is not None:
            try:
                profile["line_handle"].remove()
            except:
                pass

        # Draw in physical coordinates
        lh, = self.ax_img.plot(
            xs * self.controller.pixel_size,
            ys * self.controller.time_per_frame,
            color=profile["color"],
            linewidth=2
        )
        profile["line_handle"] = lh

        self.ax_img.figure.canvas.draw_idle()

        self.pick_start = None
        self.pick_end = None
        self.profile_pick_mode = False

        # Update templates in controller
        self.controller.profiles = self.profiles

    # -------------------------
    # LOAD KYMO
    # -------------------------
    def load_current_kymo(self):
        if not self.file_paths:
            print("No files.")
            return

        path = self.file_paths[self.current_index]
        print(f"Loading: {path}")

        self.controller.load_kymo(path, self.pixel_size)
        self.load_state(path)
        self.update_kymo_index_label()

    # -------------------------
    # AUTOMATIC DETECTION
    # -------------------------
    def run_detection(self, event):
        # pass templates to controller
        self.controller.profiles = self.profiles

        ys, xs_left, xs_right, values = self.controller.detect_polymer_edges()
        if ys is None:
            print("Polymer could not be detected.")
            return

        self.detected_ys = ys
        self.detected_values = values

        slope_segments = self.controller.detect_slope_segments(ys, values)
        self.detected_segments = list(slope_segments)
        self.accepted_segments = set()

        img = self.controller.apply_brightness_contrast(self.controller.kymo_roi)
        self.analyzer.draw_image(img, pixel_size=self.controller.pixel_size,
                                 time_per_frame=self.controller.time_per_frame)

        self.analyzer.draw_edges(ys, xs_left, xs_right,
                                 pixel_size=self.controller.pixel_size,
                                 time_per_frame=self.controller.time_per_frame)

        self.analyzer.draw_length_plot(ys, values, self.controller.time_per_frame)

        self.analyzer.draw_slope_markers(
            ys, values, self.detected_segments, self.accepted_segments,
            self.controller.time_per_frame
        )

    # -------------------------
    # MANUAL FIT
    # -------------------------
    def toggle_manual_fit(self, event):
        self.manual_fit_mode = not self.manual_fit_mode
        self.manual_pick_mode = False
        self.profile_pick_mode = False
        self.pick_start = None
        self.pick_end = None
        print(f"Manual fit: {'ON' if self.manual_fit_mode else 'OFF'}")

        self.button_pick.ax.set_facecolor('0.85')
        self.fig.canvas.draw_idle()

    # -------------------------
    # MANUAL PICKING (length profile)
    # -------------------------
    def toggle_manual_pick(self, event):
        self.manual_pick_mode = not self.manual_pick_mode
        self.manual_fit_mode = False
        self.profile_pick_mode = False
        self.pick_start = None
        self.pick_end = None
        print(f"Manual picking: {'ON' if self.manual_pick_mode else 'OFF'}")

        if self.manual_pick_mode:
            self.button_pick.ax.set_facecolor('lightgreen')
        else:
            self.button_pick.ax.set_facecolor('0.85')
        self.fig.canvas.draw_idle()

    # -------------------------
    # CLICK ON IMAGE OR PROFILE
    # -------------------------
    def on_click(self, event):
        if event.inaxes == self.ax_len:
            self.on_click_slope(event)
            return

        if event.inaxes != self.ax_img:
            return

        if self.profile_pick_mode and self.active_profile is not None:
            self.handle_profile_pick(event)
            return

        if not self.manual_pick_mode:
            return

        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return

        if self.pick_start is None:
            self.pick_start = (x, y)
            print("Start point:", self.pick_start)
            return

        self.pick_end = (x, y)
        print("End point:", self.pick_end)

        self.analyzer.draw_manual_line(self.pick_start, self.pick_end, color='c')

        ys, values = self.controller.manual_pick_profile(self.pick_start, self.pick_end)
        values_um = values * self.controller.pixel_size

        self.detected_ys = ys
        self.detected_values = values_um
        self.detected_segments = []
        self.accepted_segments = set()

        self.analyzer.draw_length_plot(ys, values_um, self.controller.time_per_frame)

        if len(ys) > 1:
            fit = self.controller.compute_fit_for_segment(ys, values_um, 0, len(ys) - 1)
            if fit is not None:
                slope, intercept, r, p, stderr = fit
                t = ys * self.controller.time_per_frame
                t0 = t[0]
                t1 = t[-1]
                ax_len_line, = self.ax_len.plot(
                    [t0, t1],
                    [values_um[0], values_um[-1]],
                    'c-', linewidth=3
                )
                ax_line = None
                self.add_line_entry("manual_pick", slope, intercept, r, t0, t1,
                                    ax_line, ax_len_line)

        self.pick_start = None
        self.pick_end = None

    # -------------------------
    # CLICK ON PROFILE
    # -------------------------
    def on_click_slope(self, event):
        if self.detected_ys is None or self.detected_values is None:
            return

        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return

        if self.manual_fit_mode:
            if self.pick_start is None:
                self.pick_start = x
                print("Manual fit start:", x)
                return
            else:
                self.pick_end = x
                print("Manual fit end:", x)

                t = self.detected_ys * self.controller.time_per_frame

                i0 = int(np.argmin(np.abs(t - self.pick_start)))
                i1 = int(np.argmin(np.abs(t - self.pick_end)))
                i0, i1 = sorted((i0, i1))

                fit = self.controller.compute_fit_for_segment(
                    self.detected_ys,
                    self.detected_values,
                    i0,
                    i1
                )

                if fit is None:
                    print("Invalid manual fit")
                else:
                    slope, intercept, r, p, stderr = fit
                    t0 = t[i0]
                    t1 = t[i1]
                    ax_len_line, = self.ax_len.plot(
                        [t0, t1],
                        [self.detected_values[i0], self.detected_values[i1]],
                        'b-', linewidth=3
                    )
                    ax_line = None
                    self.add_line_entry("fit_manual", slope, intercept, r, t0, t1,
                                        ax_line, ax_len_line)

                self.refresh_plots()

                self.pick_start = None
                self.pick_end = None
                return

        if not self.detected_segments:
            return

        t = self.detected_ys * self.controller.time_per_frame

        best = None
        best_dist = 1e9

        for idx, (i0, i1) in enumerate(self.detected_segments):
            t0 = t[i0]
            v0 = self.detected_values[i0]
            dist = (x - t0) ** 2 + (y - v0) ** 2
            if dist < best_dist:
                best_dist = dist
                best = idx

        if best is not None:
            if best in self.accepted_segments:
                self.accepted_segments.remove(best)
            else:
                self.accepted_segments.add(best)

                i0, i1 = self.detected_segments[best]
                fit = self.controller.compute_fit_for_segment(
                    self.detected_ys,
                    self.detected_values,
                    i0,
                    i1
                )
                if fit is not None:
                    slope, intercept, r, p, stderr = fit
                    t0 = t[i0]
                    t1 = t[i1]
                    ax_len_line, = self.ax_len.plot(
                        [t0, t1],
                        [self.detected_values[i0], self.detected_values[i1]],
                        'g-', linewidth=3
                    )
                    ax_line = None
                    self.add_line_entry("auto", slope, intercept, r, t0, t1,
                                        ax_line, ax_len_line)

            self.refresh_plots()

    # -------------------------
    # LINE LIST
    # -------------------------
    def add_line_entry(self, tipo, slope, intercept, r, t0, t1, ax_line, ax_len_line):
        # Get X0 and X1 from detected values
        # (self.detected_values is in nm)
        if self.detected_values is not None:
            # Find closest indices to t0 and t1
            t = self.detected_ys * self.controller.time_per_frame
            i0 = int(np.argmin(np.abs(t - t0)))
            i1 = int(np.argmin(np.abs(t - t1)))
            x0 = self.detected_values[i0]
            x1 = self.detected_values[i1]
        else:
            x0 = None
            x1 = None

        phase = self.controller.classify_phase(slope)

        entry = {
            "id": self.next_line_id,
            "tipo": tipo,
            "phase": phase,
            "slope": slope,
            "intercept": intercept,
            "r": r,
            "t0": t0,
            "t1": t1,
            "x0": x0,
            "x1": x1,
            "kymo": os.path.basename(self.file_paths[self.current_index]) if self.file_paths else None,
            "ax_line": ax_line,
            "ax_len_line": ax_len_line,
        }

        self.lines.append(entry)
        self.next_line_id += 1
        self.update_line_list()


    def update_line_list(self):
        self.ax_list.clear()
        self.ax_list.set_xticks([])
        self.ax_list.set_yticks([])
        self.ax_list.set_title("Slopes")

        if not self.lines:
            self.ax_list.text(0.05, 0.95, "No lines", va='top')
            self.ax_list.figure.canvas.draw_idle()
            return

        y = 0.95
        dy = 0.06
        for entry in self.lines:
            txt = f"{entry['id']}: {entry['tipo']} [{entry['phase']}] m={entry['slope']:.3f}"
            self.ax_list.text(
                0.90, y, "★", va='top', ha='right',
                color='gold', fontsize=10,
                picker=True,
                gid=f"pin_{entry['id']}"
            )

            self.ax_list.text(
                0.98, y, "✕", va='top', ha='right',
                color='red', fontsize=10,
                picker=True,
                gid=f"del_{entry['id']}"
            )
            y -= dy

        self.ax_list.figure.canvas.draw_idle()


    def on_pick(self, event):
        artist = event.artist
        gid = None
        if hasattr(artist, "get_gid"):
            gid = artist.get_gid()
        if gid is None:
            return

        if gid.startswith("del_"):
            line_id = int(gid.split("_")[1])
            self.delete_line(line_id)
        elif gid.startswith("pin_"):
            line_id = int(gid.split("_")[1])
            self.pin_values(line_id)


    def delete_line(self, line_id):
        for i, entry in enumerate(self.lines):
            if entry["id"] == line_id:
                if entry["ax_line"] is not None:
                    try:
                        entry["ax_line"].remove()
                    except:
                        pass
                if entry["ax_len_line"] is not None:
                    try:
                        entry["ax_len_line"].remove()
                    except:
                        pass
                del self.lines[i]
                break

        self.ax_img.figure.canvas.draw_idle()
        self.ax_len.figure.canvas.draw_idle()
        self.update_line_list()


    def pin_values(self, line_id):
        # Find the slope in the normal list
        for entry in self.lines:
            if entry["id"] == line_id:
                saved = {
                    "id": entry["id"],
                    "tipo": entry["tipo"],
                    "phase": entry["phase"],
                    "slope": entry["slope"],
                    "intercept": entry["intercept"],
                    "r": entry["r"],
                    "t0": entry["t0"],
                    "t1": entry["t1"],
                    "x0": entry["x0"],
                    "x1": entry["x1"],
                    "kymo": entry["kymo"],
                }
                self.pinned_values.append(saved)
                break

        self.update_pinned_list()
        self.update_boxplot()


    def update_pinned_list(self):
        self.ax_pinned.clear()
        self.ax_pinned.set_xticks([])
        self.ax_pinned.set_yticks([])
        self.ax_pinned.set_title("Saved")

        self.ax_pinned_scroll = plt.axes([0.98, 0.05, 0.015, 0.40])
        self.slider_pinned_scroll = Slider(self.ax_pinned_scroll, '', 0, 1, valinit=1)
        self.slider_pinned_scroll.on_changed(self.update_pinned_list)

        if not self.pinned_values:
            self.ax_pinned.text(0.05, 0.95, "Empty", va='top', fontsize=8)
            self.ax_pinned.figure.canvas.draw_idle()
            return

        y = 0.95
        dy = 0.06
        for entry in self.pinned_values:
            txt = f"{entry['id']}: m={entry['slope']:.3f}"
            self.ax_pinned.text(0.05, y, txt, va='top', fontsize=8)
            y -= dy

        self.ax_pinned.figure.canvas.draw_idle()


    def update_boxplot(self):
        self.ax_boxplot.clear()
        self.ax_boxplot.set_title("Slope distribution", fontsize=10)

        if len(self.pinned_values) > 0:
            slopes = [e["slope"] for e in self.pinned_values]
            self.ax_boxplot.boxplot(slopes, vert=False)
        else:
            self.ax_boxplot.text(0.5, 0.5, "No data", ha='center', va='center')

        self.ax_boxplot.set_yticks([])
        self.ax_boxplot.figure.canvas.draw_idle()

        
    # -------------------------
    # EXPORT LINES
    # -------------------------
    def export_lines(self, event):
        base = os.path.dirname(self.controller.path) if self.controller.path else "."

        # Export normal slopes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(base, f"kymo_slopes_{timestamp}.csv")

        print(f"Exporting slopes to: {out_path}")
        self.ax_controls.text(0.5, 0.02, "Saved successfully", ha='center', color='green', fontsize=10)
        self.fig.canvas.draw_idle()

        with open(out_path, "w", newline="") as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(["id", "type", "phase", "slope", "intercept", "r",
                        "t0", "t1", "x0", "x1", "kymo"])
            for e in self.lines:
                w.writerow([
                    e["id"], e["tipo"], e["phase"], e["slope"], e["intercept"], e["r"],
                    e["t0"], e["t1"], e["x0"], e["x1"], e["kymo"]
                ])

        # Export saved slopes
        if self.pinned_values:
            out_path2 = os.path.join(base, f"saved_slopes_{timestamp}.csv")
            print(f"Exporting saved slopes to: {out_path2}")
            self.ax_controls.text(0.5, 0.02, "Saved successfully", ha='center', color='green', fontsize=10)
            self.fig.canvas.draw_idle()

            with open(out_path2, "w", newline="") as f:
                w = csv.writer(f, delimiter=';')
                w.writerow(["id", "type", "phase", "slope", "intercept", "r",
                            "t0", "t1", "x0", "x1", "kymo"])
                for e in self.pinned_values:
                    w.writerow([
                        e["id"], e["tipo"], e["phase"], e["slope"], e["intercept"], e["r"],
                        e["t0"], e["t1"], e["x0"], e["x1"], e["kymo"]
                    ])


    # -------------------------
    # DYNAMIC ZOOM
    # -------------------------
    def on_scroll(self, event):
        if event.inaxes not in [self.ax_img, self.ax_len]:
            return

        ax = event.inaxes
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        scale = 1.2 if event.button == 'up' else 1 / 1.2

        img = self.controller.kymo_roi
        if img is None:
            return

        H, W = img.shape
        width_nm = W * self.controller.pixel_size          # nm
        height_s = H * self.controller.time_per_frame

        x_min, x_max = 0, width_nm
        y_max, y_min = 0, height_s   # inverted axis

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        cx = xdata
        cy = ydata

        new_width = (x1 - x0) / scale
        new_height = (y1 - y0) / scale

        min_width = width_nm * 0.01
        min_height = height_s * 0.01

        new_width = max(new_width, min_width)
        new_height = max(new_height, min_height)

        nx0 = cx - new_width / 2
        nx1 = cx + new_width / 2
        ny0 = cy - new_height / 2
        ny1 = cy + new_height / 2

        # Left correction
        if nx0 < x_min:
            shift = x_min - nx0
            nx0 += shift
            nx1 += shift

        # Right correction
        if nx1 > x_max:
            shift = nx1 - x_max
            nx0 -= shift
            nx1 -= shift

        # Final correction in case zoom inversion happens
        nx0 = max(nx0, x_min)
        nx1 = min(nx1, x_max)

        # Avoid nx1 < nx0 (extreme case)
        if nx1 - nx0 < min_width:
            nx1 = nx0 + min_width
            if nx1 > x_max:
                nx1 = x_max
                nx0 = x_max - min_width

        if ny0 < y_min:
            ny1 += (y_min - ny0)
            ny0 = y_min
        if ny1 > y_max:
            ny0 -= (ny1 - y_max)
            ny1 = y_max

        ax.set_xlim(nx0, nx1)
        ax.set_ylim(max(ny0, ny1), min(ny0, ny1))
        ax.set_ylim(max(ny0, ny1), min(ny0, ny1))


    # -------------------------
    # REFRESH PLOTS
    # -------------------------
    def refresh_plots(self):
        if self.detected_ys is None or self.detected_values is None:
            return

        self.analyzer.draw_length_plot(self.detected_ys, self.detected_values,
                                    self.controller.time_per_frame)

        self.analyzer.draw_slope_markers(
            self.detected_ys,
            self.detected_values,
            self.detected_segments,
            self.accepted_segments,
            self.controller.time_per_frame
        )


    # -------------------------
    # RESET ANALYSIS
    # -------------------------
    def reset_analysis(self, event):
        self.detected_ys = None
        self.detected_values = None
        self.detected_segments = []
        self.accepted_segments = set()
        self.pick_start = None
        self.pick_end = None
        self.lines = []
        self.next_line_id = 0
        self.profiles = []
        self.active_profile = None
        self.profile_pick_mode = False
        self.manual_fit_mode = False
        self.manual_pick_mode = False

        img = self.controller.apply_brightness_contrast(self.controller.kymo_roi)
        self.analyzer.draw_image(img, pixel_size=self.controller.pixel_size,
                                time_per_frame=self.controller.time_per_frame)

        self.analyzer.draw_length_plot(None, None, self.controller.time_per_frame)
        self.update_line_list()

    # -------------------------
    # REDRAW EVERYTHING (for saved states)
    # -------------------------
    def redraw_everything(self):
        img = self.controller.apply_brightness_contrast(self.controller.kymo_roi)
        self.analyzer.draw_image(img, pixel_size=self.controller.pixel_size,
                                time_per_frame=self.controller.time_per_frame)

        # Redraw template profiles
        for profile in self.profiles:
            if profile.get("ys") is not None and profile.get("xs") is not None:
                ys = profile["ys"]
                xs = profile["xs"]
                lh, = self.ax_img.plot(
                    xs * self.controller.pixel_size,
                    ys * self.controller.time_per_frame,
                    color=profile["color"],
                    linewidth=2
                )
                profile["line_handle"] = lh

        # Redraw length profile and segments
        if self.detected_ys is not None and self.detected_values is not None:
            self.analyzer.draw_length_plot(self.detected_ys, self.detected_values,
                                        self.controller.time_per_frame)
            self.analyzer.draw_slope_markers(
                self.detected_ys,
                self.detected_values,
                self.detected_segments,
                self.accepted_segments,
                self.controller.time_per_frame
            )

        # Redraw lines on the length axis
        for entry in self.lines:
            t0 = entry["t0"]
            t1 = entry["t1"]
            slope = entry["slope"]
            intercept = entry["intercept"]
            y0 = slope * t0 + intercept
            y1 = slope * t1 + intercept

            ax_len_line, = self.ax_len.plot([t0, t1], [y0, y1], 'g-', linewidth=3)
            entry["ax_len_line"] = ax_len_line

        self.ax_img.figure.canvas.draw_idle()
        self.ax_len.figure.canvas.draw_idle()

    
    def pin_values(self, line_id):
        for entry in self.lines:
            if entry["id"] == line_id:
                saved = {
                    "id": entry["id"],
                    "tipo": entry["tipo"],
                    "phase": entry["phase"],
                    "slope": entry["slope"],
                    "intercept": entry["intercept"],
                    "r": entry["r"],
                    "t0": entry["t0"],
                    "t1": entry["t1"],
                    "x0": entry["x0"],
                    "x1": entry["x1"],
                    "kymo": entry["kymo"],
                }
                self.pinned_values.append(saved)
                break

        self.update_pinned_list()
        self.update_boxplot()


    def update_pinned_list(self):
        self.ax_pinned.clear()
        self.ax_pinned.set_xticks([])
        self.ax_pinned.set_yticks([])
        self.ax_pinned.set_title("Saved")

        self.ax_pinned_scroll = plt.axes([0.98, 0.05, 0.015, 0.40])
        self.slider_pinned_scroll = Slider(self.ax_pinned_scroll, '', 0, 1, valinit=1)
        self.slider_pinned_scroll.on_changed(self.update_pinned_list)

        if not self.pinned_values:
            self.ax_pinned.text(0.05, 0.95, "Empty", va='top', fontsize=8)
            self.ax_pinned.figure.canvas.draw_idle()
            return

        y = 0.95
        dy = 0.06
        for entry in self.pinned_values:
            txt = f"{entry['id']}: m={entry['slope']:.3f}"
            self.ax_pinned.text(0.05, y, txt, va='top', fontsize=8)
            y -= dy

        self.ax_pinned.figure.canvas.draw_idle()


    def update_boxplot(self):
        self.ax_boxplot.clear()
        self.ax_boxplot.set_title("Slope distribution", fontsize=10)

        if len(self.pinned_values) > 0:
            slopes = [e["slope"] for e in self.pinned_values]
            self.ax_boxplot.boxplot(slopes, vert=False)
        else:
            self.ax_boxplot.text(0.5, 0.5, "No data", ha='center', va='center')

        self.ax_boxplot.set_yticks([])
        self.ax_boxplot.figure.canvas.draw_idle()

        
    # -------------------------
    # Remaining methods unchanged...
    # -------------------------
