import numpy as np
from scipy.interpolate import UnivariateSpline

class DriftModel:
    """
    Smooth drift model built on top of existing drift measurements.
    It does not modify the original tools; it only consumes their outputs.
    """
    def __init__(self, drifts, confidence=None):
        self.raw = np.asarray(drifts, dtype=float)
        if confidence is None:
            self.conf = np.ones(len(self.raw), dtype=float)
        else:
            self.conf = np.asarray(confidence, dtype=float)

    def interpolate(self, conf_threshold=0.3, smoothing=0.5):
        t = np.arange(len(self.raw))
        good = self.conf > conf_threshold

        if good.sum() < 2:
            return self.raw.copy()

        spline_y = UnivariateSpline(t[good], self.raw[good, 0], s=smoothing)
        spline_x = UnivariateSpline(t[good], self.raw[good, 1], s=smoothing)

        y_interp = spline_y(t)
        x_interp = spline_x(t)

        return np.column_stack([y_interp, x_interp])

    def detect_segments(self, jump_threshold=5.0, conf_threshold=0.3, smoothing=0.5):
        d = self.interpolate(conf_threshold=conf_threshold, smoothing=smoothing)
        jump = np.linalg.norm(np.diff(d, axis=0), axis=1)
        cut_indices = np.where(jump > jump_threshold)[0]

        segments = []
        start = 0
        for idx in cut_indices:
            segments.append((start, idx))
            start = idx + 1
        segments.append((start, len(d) - 1))

        return segments

    def predict(self, t, conf_threshold=0.3, smoothing=0.5):
        d = self.interpolate(conf_threshold=conf_threshold, smoothing=smoothing)
        t = int(np.clip(t, 0, len(d) - 1))
        return d[t]
