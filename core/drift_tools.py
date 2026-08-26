import numpy as np
import cv2
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
        masks_out[-1] = aligned_mask

        ref_canvas = aligned_img.copy()

    return np.array(aligned), np.array(masks_out), transforms, H_pad, W_pad

def drift_template_matching(ref, img, template_size=64):
    H, W = ref.shape
    cy, cx = H // 2, W // 2
    half = template_size // 2

    template = ref[cy-half:cy+half, cx-half:cx+half].astype(np.float32)

    res = cv2.matchTemplate(img.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    y, x = max_loc

    dy = y - (cy - half)
    dx = x - (cx - half)

    return dy, dx, max_val


def template_matching_sequential(frames, template_size=64):
    ref = blur_frame(frames[0])
    drifts = [[0.0, 0.0]]
    confidence = [1.0]

    for i in range(1, len(frames)):
        img = blur_frame(frames[i])
        dy, dx, score = drift_template_matching(ref, img, template_size)
        drifts.append([dy + drifts[-1][0], dx + drifts[-1][1]])
        confidence.append(score)

        # Update reference to aligned frame
        ref = nd_shift(img, shift=(-dy, -dx), mode="constant", cval=0)

    return np.array(drifts), np.array(confidence)

# ============================================================
#                   TEMPLATE MATCHING — SEQUENTIAL
# ============================================================
def pick_best_reference(frames, max_idx=20):
    scores = []
    limit = min(max_idx, len(frames)-1)
    for i in range(limit):
        scores.append((np.std(frames[i]), i))
    # elegir el frame con mayor contraste
    _, best_idx = max(scores)
    return best_idx

def template_matching_global(frames, template_size=64):
    # Keep frame 0 as the reference so drift coordinates match the stack.
    ref = blur_frame(frames[0])

    drifts = [[0.0, 0.0]]
    confidence = [1.0]

    for i in range(1, len(frames)):
        img = blur_frame(frames[i])
        dy, dx, score = drift_template_matching(ref, img, template_size)
        drifts.append([dy, dx])
        confidence.append(score)

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
    ref = frames[0]
    drifts = []

    for i in range(len(frames)):
        if i == 0:
            drifts.append([0.0, 0.0])
        else:
            dy, dx = drift_template_matching(ref, frames[i], template_size)
            drifts.append([dy, dx])

    return np.array(drifts)
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
    combined = np.sum(masks, axis=0)

    ys, xs = np.where(combined > 0)
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
    H, W = frames[0].shape

    dy = drifts[:, 0]
    dx = drifts[:, 1]

    # Minimum and maximum coordinates occupied by any frame.
    y_min = dy.min()
    y_max = dy.max()
    x_min = dx.min()
    x_max = dx.max()

    # Alignment applies the inverse drift, so calculate bounds for -drift.
    top    = int(max(0,  y_max))
    bottom = int(max(0, -y_min))
    left   = int(max(0,  x_max))
    right  = int(max(0, -x_min))

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

    aligned = [ref_canvas.copy()]
    masks_out = [mask_frames[0].astype(np.uint8)]
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

        mask_i = mask_frames[i].astype(np.uint8)
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


