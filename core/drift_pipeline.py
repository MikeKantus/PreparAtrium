import numpy as np
from scipy.ndimage import shift as nd_shift
import cv2

# IMPORTA AQUÍ tus funciones del documento:
# sample_mask_otsu, clean_mask, propagate_mask,
# ecc_align_first, ecc_align_final,
# compute_raw_drift, align_with_auto_canvas

class DriftPipeline:
    def __init__(self, stack):
        self.stack = stack

    def run_ecc1(self):
        frame0 = self.stack[0].astype(np.uint8)
        mask0 = sample_mask_otsu(frame0)
        mask0 = clean_mask(mask0)

        zero_drift = np.zeros((len(self.stack), 2))
        mask_drift = propagate_mask(mask0, zero_drift)

        ecc_frames, ecc_masks_raw, ecc_transforms, H_pad, W_pad = ecc_align_first(
            self.stack, mask_drift
        )

        mask_ecc = propagate_mask(
            mask0,
            zero_drift,
            ecc_transforms,
            H_pad=H_pad,
            W_pad=W_pad
        )

        mask_union = np.max(mask_ecc, axis=0)
        ys, xs = np.where(mask_union > 0)

        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        cropped_frames = ecc_frames[:, ymin:ymax+1, xmin:xmax+1]
        cropped_masks = mask_ecc[:, ymin:ymax+1, xmin:xmax+1]

        return cropped_frames, cropped_masks

    def run_optical_flow(self, ecc1_stack):
        drifts = compute_raw_drift(ecc1_stack)
        aligned, masks = align_with_auto_canvas(ecc1_stack, drifts)

        mask_union = np.max(masks, axis=0)
        ys, xs = np.where(mask_union > 0)

        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        aligned_crop = aligned[:, ymin:ymax+1, xmin:xmax+1]
        masks_crop = masks[:, ymin:ymax+1, xmin:xmax+1]

        return aligned_crop, masks_crop, drifts

    def run_ecc2(self, drift_stack, drift_masks, drift_vectors):
        frame0 = drift_stack[0].astype(np.uint8)
        mask0 = sample_mask_otsu(frame0)
        mask0 = clean_mask(mask0)

        mask_drift = propagate_mask(mask0, drift_vectors)

        ecc_frames, ecc_masks_raw, ecc_transforms = ecc_align_final(
            drift_stack, mask_drift
        )

        return ecc_frames, ecc_masks_raw
