import numpy as np
import tifffile as tiff
import matplotlib.patches as patches


class KymoAnalyzer:
    def __init__(self, ax_img, ax_lenplot):
        self.ax_img = ax_img
        self.ax_len = ax_lenplot
        self.roi_patch = None

    def load_raw_kymo(self, path):
        return tiff.imread(path).astype(float)

    def load_raw_kymo_array(self, kymo):
        """Load a kymograph produced in memory by KymoPanel."""
        return np.asarray(kymo, dtype=float).copy()

    def draw_image(self, img, preview=False, pixel_size=1.0, time_per_frame=1.0):
        if img is None or self.ax_img is None:
            return

        H, W = img.shape
        width_nm = W * pixel_size          # pixel_size in nm/pixel
        total_time_s = H * time_per_frame

        self.ax_img.clear()
        self.ax_img.set_autoscale_on(False)
        self.ax_img.autoscale(False)

        title = "Preview" if preview else "Kymograph"
        self.ax_img.set_title(title)

        self.ax_img.imshow(
            img,
            cmap='viridis',
            extent=[0, width_nm, total_time_s, 0],
            aspect='equal',
            vmin=0,
            vmax=1
        )

        self.ax_img.set_xlim(0, width_nm)
        self.ax_img.set_ylim(total_time_s, 0)

        self.ax_img.set_xlabel("Distance (nm)")
        self.ax_img.set_ylabel("Time (s)")

        self.ax_img.figure.canvas.draw_idle()

    def draw_roi(self, x0, x1, y0, y1):
        if self.roi_patch is not None:
            try:
                self.roi_patch.remove()
            except:
                pass

        width = x1 - x0
        height = y1 - y0

        self.roi_patch = patches.Rectangle(
            (x0, y0),
            width,
            height,
            linewidth=2,
            edgecolor='cyan',
            facecolor='none'
        )
        self.ax_img.add_patch(self.roi_patch)
        self.ax_img.figure.canvas.draw_idle()

    def draw_edges(self, ys, xs_left, xs_right, pixel_size=1.0, time_per_frame=1.0):
        if ys is None:
            return

        t = ys * time_per_frame
        xl = xs_left * pixel_size   # nm
        xr = xs_right * pixel_size  # nm

        self.ax_img.plot(xl, t, 'c-', linewidth=2)
        self.ax_img.plot(xr, t, 'm-', linewidth=2)
        self.ax_img.figure.canvas.draw_idle()

    def draw_length_plot(self, ys, values, time_per_frame=1.0):
        self.ax_len.clear()
        if ys is not None and values is not None and len(ys) > 0:
            t = ys * time_per_frame
            self.ax_len.set_title("Edge profile")
            self.ax_len.set_xlabel("Time (s)")
            self.ax_len.set_ylabel("Height (nm)")
            self.ax_len.plot(t, values, 'k-')
        self.ax_len.figure.canvas.draw_idle()

    def draw_slope_markers(self, ys, values, segments, accepted_indices=None, time_per_frame=1.0):
        if ys is None or values is None:
            return
        if accepted_indices is None:
            accepted_indices = set()

        t = ys * time_per_frame

        for idx, (i0, i1) in enumerate(segments):
            t0 = t[i0]
            t1 = t[i1]
            v0 = values[i0]
            v1 = values[i1]

            color = 'g' if idx in accepted_indices else 'r'
            self.ax_len.plot([t0, t1], [v0, v1], color + '-', linewidth=3)
            self.ax_len.plot(t0, v0, color + 'o', markersize=8)

        self.ax_len.figure.canvas.draw_idle()

    def draw_manual_line(self, p0, p1, color='w'):
        x0, y0 = p0
        x1, y1 = p1
        # p0 and p1 are already in physical coordinates (nm, s) if passed that way
        self.ax_img.set_autoscale_on(False)
        self.ax_img.autoscale(False)

        self.ax_img.plot([x0, x1], [y0, y1], color + '--', linewidth=2)

        # Keep current limits
        self.ax_img.set_xlim(self.ax_img.get_xlim())
        self.ax_img.set_ylim(self.ax_img.get_ylim())

        self.ax_img.figure.canvas.draw_idle()
