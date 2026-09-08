import numpy as np
import cv2


TEXT_SCALES = {"Small": 0.45, "Medium": 0.7, "Large": 1.0}
TEXT_COLORS = {
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Yellow": (0, 255, 255),
    "Cyan": (255, 255, 0),
    "Red": (0, 0, 255),
}
COLOR_PALETTES = {
    "Grayscale": None,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Turbo": cv2.COLORMAP_TURBO,
    "Hot": cv2.COLORMAP_HOT,
}


def annotate_frame(frame, index, metadata=None, *, show_timestamp=False,
                   show_frame_number=False, text_size="Medium", text_color="White",
                   show_scale_bar=False, scale_bar_color=None, color_palette="Grayscale"):
    """Return a display-normalized BGR frame with optional scientific annotations."""
    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Annotations require a 2D grayscale frame")

    finite = np.isfinite(arr)
    metadata = metadata or {}
    display_min = metadata.get("z_display_min_nm")
    display_max = metadata.get("z_display_max_nm")
    try:
        low = float(display_min)
        high = float(display_max)
        use_global_z_scale = np.isfinite(low) and np.isfinite(high) and high > low
    except (TypeError, ValueError):
        use_global_z_scale = False
    if not use_global_z_scale and finite.any():
        low, high = np.percentile(arr[finite], [0.5, 99.5])
        if high <= low:
            low, high = float(np.min(arr[finite])), float(np.max(arr[finite]))
    elif not use_global_z_scale:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0

    image = np.nan_to_num(arr, nan=low, posinf=high, neginf=low)
    image = np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    palette = COLOR_PALETTES.get(color_palette)
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if palette is None else cv2.applyColorMap(image, palette)

    color = TEXT_COLORS.get(text_color, TEXT_COLORS["White"])
    scale_color = TEXT_COLORS.get(scale_bar_color, color)
    scale_bar_text_scale = TEXT_SCALES.get(text_size, TEXT_SCALES["Medium"])
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = image.shape
    margin = max(8, int(round(min(height, width) * 0.025)))

    # Calibrate Hershey font scale so timestamp/frame glyphs are 1/10 of canvas height.
    target_text_height = max(1, int(round(height / 10)))
    _, reference_height = cv2.getTextSize("Hg", font, 1.0, 1)[0]
    overlay_text_scale = target_text_height / max(1, reference_height)
    overlay_text_thickness = max(1, int(round(overlay_text_scale * 2)))
    scale_bar_thickness = max(1, int(round(scale_bar_text_scale * 2)))

    labels = []
    if show_timestamp:
        fps = metadata.get("real_fps") or metadata.get("real_FPS") or metadata.get("fps") or metadata.get("frame_rate") or 10.0
        try:
            labels.append(f"{index / float(fps):.2f} s")
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if show_frame_number:
        labels.append(f"Frame {index}")

    y = margin
    for label in labels:
        (_, text_height), baseline = cv2.getTextSize(label, font, overlay_text_scale, overlay_text_thickness)
        y += text_height
        cv2.putText(bgr, label, (margin, y), font, overlay_text_scale, color, overlay_text_thickness, cv2.LINE_AA)
        y += baseline + margin // 2

    if show_scale_bar:
        bar_width = max(1, width // 5)
        y_bar = height - margin
        x_end = width - margin
        x_start = x_end - bar_width
        cv2.line(bgr, (x_start, y_bar), (x_end, y_bar), scale_color, scale_bar_thickness, cv2.LINE_AA)
        cv2.line(bgr, (x_start, y_bar - scale_bar_thickness * 2), (x_start, y_bar + scale_bar_thickness * 2), scale_color, scale_bar_thickness, cv2.LINE_AA)
        cv2.line(bgr, (x_end, y_bar - scale_bar_thickness * 2), (x_end, y_bar + scale_bar_thickness * 2), scale_color, scale_bar_thickness, cv2.LINE_AA)
        pixel_size = metadata.get("pixel_size_nm") or metadata.get("pixel_size")
        try:
            if pixel_size is not None:
                bar_length_nm = bar_width * float(pixel_size)
            else:
                bar_length_nm = bar_width * float(metadata["x_range_nm"]) / width
            label = f"{bar_length_nm:.3g} nm"
            (text_width, text_height), _ = cv2.getTextSize(label, font, scale_bar_text_scale, scale_bar_thickness)
            cv2.putText(bgr, label, (x_end - text_width, y_bar - text_height - margin // 3),
                font, scale_bar_text_scale, scale_color, scale_bar_thickness, cv2.LINE_AA)
        except (KeyError, TypeError, ValueError):
            pass

    return bgr
