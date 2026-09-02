import numpy as np
import cv2
from math import floor, ceil
from scipy.ndimage import shift as nd_shift

# ============================================================
#                   GAUSSIAN BLUR (recommended)
# ============================================================

def blur_frame(frame, ksize=5):
    return cv2.GaussianBlur(frame.astype(np.float32), (ksize, ksize), 0)


# ============================================================
#                   MULTI-SCALE ECC
# ============================================================

def ecc_multiscale(ref, img, warp_init):
    scales = [0.25, 0.5, 1.0]
    warp = warp_init.copy()

    for s in scales:
        ref_s = cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img_s = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        warp_s = warp.copy()
        warp_s[:,2] *= s

        try:
            cc, warp_s = cv2.findTransformECC(
                ref_s, img_s, warp_s,
                cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
            )
        except cv2.error:
            continue

        warp = warp_s.copy()
        warp[:,2] /= s

    return warp


# ============================================================
#                   ECC FIRST PASS — GLOBAL
# ============================================================

def ecc_align_first_global(frames, mask_frames):
    """
    Improved global ECC: align all frames to frame 0.
    """
    H, W = frames[0].shape
    pad = max(H, W)
    H_pad = H + pad
    W_pad = W + pad

    y0 = (H_pad - H) // 2
    x0 = (W_pad - W) // 2

    ref_canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
    ref_canvas[y0:y0+H, x0:x0+W] = blur_frame(frames[0])
    first_image_canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
    first_image_canvas[y0:y0+H, x0:x0+W] = frames[0].astype(np.float32)

    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    aligned = [first_image_canvas]
    transforms = [warp_matrix.copy()]
    masks_out = []

    mask0_canvas = np.zeros((H_pad, W_pad), dtype=np.uint8)
    mask0_canvas[y0:y0+H, x0:x0+W] = 1
    masks_out.append(mask0_canvas)

    for i in range(1, len(frames)):
        registration_canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
        registration_canvas[y0:y0+H, x0:x0+W] = blur_frame(frames[i])
        image_canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
        image_canvas[y0:y0+H, x0:x0+W] = frames[i].astype(np.float32)

        warp_new = ecc_multiscale(
            ref_canvas, registration_canvas, np.eye(2, 3, dtype=np.float32)
        )

        aligned_img = cv2.warpAffine(
            image_canvas, warp_new, (W_pad, H_pad),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        aligned.append(aligned_img)
        transforms.append(warp_new.copy())

        mask_canvas = np.zeros((H_pad, W_pad), dtype=np.uint8)
        mask_canvas[y0:y0+H, x0:x0+W] = 1
        aligned_mask = cv2.warpAffine(
            mask_canvas, warp_new, (W_pad, H_pad),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        masks_out.append(aligned_mask)

    return np.array(aligned), np.array(masks_out), transforms, H_pad, W_pad


# ============================================================
#                   ECC FIRST PASS — SEQUENTIAL
# ============================================================

def ecc_align_first_sequential(frames, mask_frames):
    """
    Sequential ECC: each frame aligned to the previous aligned frame.
    Extremely robust for AFM drift.
    """
    H, W = frames[0].shape
    pad = max(H, W)
    H_pad = H + pad
    W_pad = W + pad

    y0 = (H_pad - H) // 2
    x0 = (W_pad - W) // 2

    ref_canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
    ref_canvas[y0:y0+H, x0:x0+W] = blur_frame(frames[0])

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    first_image_canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
    first_image_canvas[y0:y0+H, x0:x0+W] = frames[0].astype(np.float32)
    aligned = [first_image_canvas]
    transforms = [warp_matrix.copy()]
    masks_out = []

    mask0_canvas = np.zeros((H_pad, W_pad), dtype=np.uint8)
    mask0_canvas[y0:y0+H, x0:x0+W] = 1
    masks_out.append(mask0_canvas)

    for i in range(1, len(frames)):
        registration_canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
        registration_canvas[y0:y0+H, x0:x0+W] = blur_frame(frames[i])
        image_canvas = np.zeros((H_pad, W_pad), dtype=np.float32)
        image_canvas[y0:y0+H, x0:x0+W] = frames[i].astype(np.float32)

        warp_init = transforms[-1].copy()
        warp_new = ecc_multiscale(ref_canvas, registration_canvas, warp_init)

        aligned_img = cv2.warpAffine(
            image_canvas, warp_new, (W_pad, H_pad),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        aligned.append(aligned_img)
        transforms.append(warp_new.copy())

        mask_canvas = np.zeros((H_pad, W_pad), dtype=np.uint8)
        mask_canvas[y0:y0+H, x0:x0+W] = 1
        aligned_mask = cv2.warpAffine(
            mask_canvas, warp_new, (W_pad, H_pad),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        masks_out.append(aligned_mask)

        ref_canvas = aligned_img.copy()

    return np.array(aligned), np.array(masks_out), transforms, H_pad, W_pad


# ============================================================
#                   TEMPLATE MATCHING UTILITIES
# ============================================================

def find_best_template(ref, template_size=64, margin=20):
    """
    Finds a square region of size `template_size` in `ref` (away from borders)
    that has the highest standard deviation (maximum feature contrast).
    Returns (template, top_left_y, top_left_x).
    """
    H, W = ref.shape
    half = template_size // 2

    y_min = margin + half
    y_max = H - margin - half
    x_min = margin + half
    x_max = W - margin - half

    if y_max <= y_min or x_max <= x_min:
        cy, cx = H // 2, W // 2
        top_y, top_x = max(0, cy - half), max(0, cx - half)
        return ref[top_y:top_y+template_size, top_x:top_x+template_size].astype(np.float32), top_y, top_x

    best_std = -1.0
    best_y, best_x = H // 2 - half, W // 2 - half

    step = max(1, template_size // 4)
    for y in range(y_min, y_max + 1, step):
        for x in range(x_min, x_max + 1, step):
            patch = ref[y-half:y+half, x-half:x+half]
            std = np.std(patch)
            if std > best_std:
                best_std = std
                best_y = y - half
                best_x = x - half

    template = ref[best_y:best_y+template_size, best_x:best_x+template_size].astype(np.float32)
    return template, best_y, best_x


def match_template_in_window(img, template, ref_y0, ref_x0, exp_dy=0.0, exp_dx=0.0, search_radius=80):
    """
    Matches `template` (extracted from `ref` at top-left `(ref_y0, ref_x0)`) inside `img`,
    constraining the search window around the expected position `(ref_y0 + exp_dy, ref_x0 + exp_dx)`.
    Applies 2nd-order quadratic subpixel refinement at the peak.
    Returns (dy, dx, max_val).
    """
    H, W = img.shape
    t_h, t_w = template.shape

    exp_y0 = ref_y0 + exp_dy
    exp_x0 = ref_x0 + exp_dx

    search_y0 = int(max(0, floor(exp_y0 - search_radius)))
    search_y1 = int(min(H, ceil(exp_y0 + t_h + search_radius)))
    search_x0 = int(max(0, floor(exp_x0 - search_radius)))
    search_x1 = int(min(W, ceil(exp_x0 + t_w + search_radius)))

    if (search_y1 - search_y0) < t_h or (search_x1 - search_x0) < t_w:
        search_y0, search_y1 = 0, H
        search_x0, search_x1 = 0, W

    crop = img[search_y0:search_y1, search_x0:search_x1].astype(np.float32)
    if crop.shape[0] < t_h or crop.shape[1] < t_w:
        crop = img.astype(np.float32)
        search_y0, search_x0 = 0, 0

    res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    match_x, match_y = max_loc

    # Quadratic subpixel refinement
    delta_x = 0.0
    delta_y = 0.0
    if 0 < match_y < res.shape[0] - 1 and 0 < match_x < res.shape[1] - 1:
        y_m1 = float(res[match_y, match_x - 1])
        y_0  = float(res[match_y, match_x])
        y_p1 = float(res[match_y, match_x + 1])
        denom_x = y_m1 - 2.0 * y_0 + y_p1
        if abs(denom_x) > 1e-7:
            delta_x = (y_m1 - y_p1) / (2.0 * denom_x)
            delta_x = max(-0.5, min(0.5, delta_x))

        x_m1 = float(res[match_y - 1, match_x])
        x_0  = float(res[match_y, match_x])
        x_p1 = float(res[match_y + 1, match_x])
        denom_y = x_m1 - 2.0 * x_0 + x_p1
        if abs(denom_y) > 1e-7:
            delta_y = (x_m1 - x_p1) / (2.0 * denom_y)
            delta_y = max(-0.5, min(0.5, delta_y))

    found_y0 = search_y0 + match_y + delta_y
    found_x0 = search_x0 + match_x + delta_x

    dy = found_y0 - ref_y0
    dx = found_x0 - ref_x0

    return dy, dx, float(max_val)


def drift_template_matching(ref, img, template_size=64):
    template, ref_y0, ref_x0 = find_best_template(ref, template_size=template_size)
    return match_template_in_window(img, template, ref_y0, ref_x0, search_radius=80)


def template_matching_sequential(frames, template_size=64):
    drifts = [[0.0, 0.0]]
    confidence = [1.0]

    for i in range(1, len(frames)):
        img_prev = blur_frame(frames[i-1])
        img_curr = blur_frame(frames[i])

        template, prev_y0, prev_x0 = find_best_template(img_prev, template_size=template_size)
        step_dy, step_dx, score = match_template_in_window(
            img_curr, template, prev_y0, prev_x0, exp_dy=0.0, exp_dx=0.0, search_radius=40
        )

        # Sanity check: reject extreme single-frame jumps (>25px)
        if score < 0.15 or abs(step_dy) > 25 or abs(step_dx) > 25:
            step_dy, step_dx = 0.0, 0.0
            score = max(0.01, score)

        total_dy = drifts[-1][0] + step_dy
        total_dx = drifts[-1][1] + step_dx

        drifts.append([total_dy, total_dx])
        confidence.append(score)

    return np.array(drifts), np.array(confidence)


# ============================================================
#                   TEMPLATE MATCHING — GLOBAL
# ============================================================
def pick_best_reference(frames, max_idx=20):
    scores = []
    limit = min(max_idx, len(frames)-1)
    for i in range(limit):
        scores.append((np.std(frames[i]), i))
    _, best_idx = max(scores)
    return best_idx


def template_matching_global(frames, template_size=64):
    # Keep frame 0 as the reference so drift coordinates match the stack.
    ref = blur_frame(frames[0])
    template, ref_y0, ref_x0 = find_best_template(ref, template_size=template_size)

    drifts = [[0.0, 0.0]]
    confidence = [1.0]

    last_dy, last_dx = 0.0, 0.0

    for i in range(1, len(frames)):
        img = blur_frame(frames[i])
        dy, dx, score = match_template_in_window(
            img, template, ref_y0, ref_x0, exp_dy=last_dy, exp_dx=last_dx, search_radius=80
        )

        if score < 0.15 or abs(dy - last_dy) > 40 or abs(dx - last_dx) > 40:
            dy, dx = last_dy, last_dx
            score = max(0.01, score)

        drifts.append([dy, dx])
        confidence.append(score)
        last_dy, last_dx = dy, dx

    return np.array(drifts), np.array(confidence)
def sample_mask_otsu(frame):
    # Ensure uint8.
    if frame.dtype != np.uint8:
        a = frame.astype(np.float32)
        a = a - np.nanmin(a)
        rng = np.nanmax(a)
        if rng == 0 or np.isnan(rng):
            rng = 1.0
        frame_u8 = (a / rng * 255.0).astype(np.uint8)
    else:
        frame_u8 = frame

    # Blur.
    blur = cv2.GaussianBlur(frame_u8, (5, 5), 0)

    # Otsu threshold.
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask // 255



def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ============================================================
#                   DRIFT METHODS (Optical Flow)
# ============================================================

# def drift_optical_flow(ref, img):
#     flow = cv2.calcOpticalFlowFarneback(
#         ref, img, None,
#         pyr_scale=0.5, levels=3, winsize=21,
#         iterations=5, poly_n=7, poly_sigma=1.5, flags=0
#     )
#     dy = np.mean(flow[..., 1])
#     dx = np.mean(flow[..., 0])
#     return np.array([dy, dx])

def compute_raw_drift(frames, template_size=64):
    drifts, _ = template_matching_global(frames, template_size=template_size)
    return drifts
def align_with_auto_canvas(frames, drifts):
    H, W = frames[0].shape
    H_pad, W_pad, top, left = compute_optimal_canvas(frames, drifts)

    aligned = []
    masks = []

    for i, f in enumerate(frames):
        dy, dx = drifts[i]

        canvas = np.zeros((H_pad, W_pad), dtype=f.dtype)
        mask = np.zeros((H_pad, W_pad), dtype=np.uint8)

        # Base position of the frame without drift.
        y0 = top
        x0 = left

        # Insert the frame.
        canvas[y0:y0+H, x0:x0+W] = f
        mask[y0:y0+H, x0:x0+W] = 1

        # The measured drift is the motion of the current frame relative to
        # the reference, so alignment applies its inverse.
        inverse_shift = (-dy, -dx)
        aligned.append(nd_shift(canvas, shift=inverse_shift, mode="constant", cval=0,
                    order=1, prefilter=False))
        masks.append(nd_shift(mask, shift=inverse_shift, mode="constant", cval=0,
                      order=0, prefilter=False).astype(np.uint8))

    return np.array(aligned), np.array(masks)


def crop_to_used_area(aligned, masks):
    """
    Crop unused padding while preserving all pixels required by any frame.
    """
    combined = np.any(np.asarray(masks, dtype=bool), axis=0)

    ys, xs = np.where(combined)
    if len(ys) == 0:
        raise ValueError("Masks do not contain any valid video pixels")
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    cropped_frames = aligned[:, y_min:y_max+1, x_min:x_max+1]
    cropped_masks  = masks[:,  y_min:y_max+1, x_min:x_max+1]

    return cropped_frames, cropped_masks
# ============================================================
#                   MASK PROPAGATION
# ============================================================

def propagate_mask(mask0, drifts, ecc_transforms=None, H_pad=None, W_pad=None):
    propagated = []

    H, W = mask0.shape

    if H_pad is not None and W_pad is not None:
        y0 = (H_pad - H) // 2
        x0 = (W_pad - W) // 2

    for i in range(len(drifts)):
        dy, dx = drifts[i]

        if H_pad is not None:
            mask_canvas = np.zeros((H_pad, W_pad), dtype=np.uint8)
            mask_canvas[y0:y0+H, x0:x0+W] = mask0
        else:
            mask_canvas = mask0.copy()

        mask_shifted = nd_shift(mask_canvas, shift=(dy, dx), mode="constant", cval=0)

        if ecc_transforms is not None:
            warp = ecc_transforms[i]
            mask_shifted = cv2.warpAffine(
                mask_shifted.astype(np.uint8),
                warp,
                (mask_shifted.shape[1], mask_shifted.shape[0]),
                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        propagated.append(mask_shifted)

    return np.array(propagated)


# ============================================================
#                   ECC FIRST (WITH PADDING)
# ============================================================

def compute_optimal_canvas(frames, drifts):
    """
    Calculate the smallest canvas containing all translated frames.
    This removes unused padding and keeps only what is required.
    """
    frames = np.asarray(frames)
    drifts = np.asarray(drifts, dtype=float)
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError("Frames must be a non-empty grayscale stack")
    if drifts.shape != (len(frames), 2) or not np.isfinite(drifts).all():
        raise ValueError("Drifts must contain one finite (dy, dx) pair per frame")

    H, W = frames.shape[1:]

    dy = drifts[:, 0]
    dx = drifts[:, 1]

    # Alignment applies -drift. Round outward so sub-pixel shifts cannot clip
    # an edge, then crop the union of valid masks after alignment.
    top = int(np.ceil(max(0.0, dy.max())))
    bottom = int(np.ceil(max(0.0, -dy.min())))
    left = int(np.ceil(max(0.0, dx.max())))
    right = int(np.ceil(max(0.0, -dx.min())))

    H_pad = H + top + bottom
    W_pad = W + left + right

    return H_pad, W_pad, top, left



# ============================================================
#                   ECC FINAL (WITHOUT PADDING)
# ============================================================

def ecc_align_final(frames, mask_frames):
    H_pad, W_pad = frames[0].shape

    ref_canvas = frames[0].astype(np.float32)

    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

    # Sanity check: mask_frames must exist, have the same length, and same frame shape
    if (mask_frames is None or len(mask_frames) != len(frames) or
            getattr(mask_frames[0], 'shape', None) != (H_pad, W_pad)):
        mask_frames = np.ones((len(frames), H_pad, W_pad), dtype=np.uint8)

    first_mask = np.asarray(mask_frames[0], dtype=np.uint8)
    if first_mask.shape != (H_pad, W_pad):
        first_mask = cv2.resize(first_mask, (W_pad, H_pad), interpolation=cv2.INTER_NEAREST)

    aligned = [ref_canvas.copy()]
    masks_out = [first_mask]
    ecc_transforms = [warp_matrix.copy()]

    for i in range(1, len(frames)):
        img = frames[i].astype(np.float32)

        frame_warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, frame_warp = cv2.findTransformECC(
                ref_canvas, img, frame_warp, warp_mode, criteria
            )
        except cv2.error:
            frame_warp = np.eye(2, 3, dtype=np.float32)

        aligned_img = cv2.warpAffine(
            img, frame_warp, (W_pad, H_pad),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        mask_i = np.asarray(mask_frames[i], dtype=np.uint8)
        if mask_i.shape != (H_pad, W_pad):
            mask_i = cv2.resize(mask_i, (W_pad, H_pad), interpolation=cv2.INTER_NEAREST)

        aligned_mask = cv2.warpAffine(
            mask_i, frame_warp, (W_pad, H_pad),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        aligned.append(aligned_img)
        masks_out.append(aligned_mask)
        ecc_transforms.append(frame_warp.copy())

    return np.array(aligned), np.array(masks_out), ecc_transforms

def ecc_transforms_to_drifts(ecc_transforms):
    """
    Convert ECC matrices (2x3) into translations (dy, dx).
    """
    drifts = []
    for M in ecc_transforms:
        dy = float(M[1, 2])
        dx = float(M[0, 2])
        drifts.append([dy, dx])
    return np.array(drifts, dtype=float)


