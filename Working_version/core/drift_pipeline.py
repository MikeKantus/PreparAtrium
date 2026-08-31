import numpy as np
from core.drift_model import DriftModel
from core.drift_tools import (
    ecc_align_first_global,
    ecc_align_first_sequential,
    template_matching_global,
    template_matching_sequential,
    ecc_align_final,
    align_with_auto_canvas,
    crop_to_used_area,
    sample_mask_otsu,
    clean_mask,
    propagate_mask,
)

class DriftPipeline:
    def __init__(self, stack):
        self.stack = stack

    def run_ecc1_global(self):
        mask0 = clean_mask(sample_mask_otsu(self.stack[0]))
        zero = np.zeros((len(self.stack), 2))
        mask_drift = propagate_mask(mask0, zero)

        return ecc_align_first_global(self.stack, mask_drift)

    def run_ecc1_sequential(self):
        mask0 = clean_mask(sample_mask_otsu(self.stack[0]))
        zero = np.zeros((len(self.stack), 2))
        mask_drift = propagate_mask(mask0, zero)

        return ecc_align_first_sequential(self.stack, mask_drift)

    def run_tm_global(self, ecc1_stack):
        drifts, conf = template_matching_global(ecc1_stack)
        aligned, masks = align_with_auto_canvas(ecc1_stack, drifts)
        aligned, masks = crop_to_used_area(aligned, masks)
        return aligned, masks, drifts, conf

    def run_tm_sequential(self, ecc1_stack):
        drifts, conf = template_matching_sequential(ecc1_stack)
        aligned, masks = align_with_auto_canvas(ecc1_stack, drifts)
        aligned, masks = crop_to_used_area(aligned, masks)
        return aligned, masks, drifts, conf

    def run_ecc2(self, drift_stack, drift_vectors):
        mask0 = clean_mask(sample_mask_otsu(drift_stack[0]))
        mask_drift = propagate_mask(mask0, drift_vectors)
        ecc_frames, ecc_masks_raw, ecc_transforms = ecc_align_final(drift_stack, mask_drift)
        return ecc_frames, ecc_masks_raw, ecc_transforms


    @staticmethod
    def process_tm_drifts(drifts, confidence):
        model = DriftModel(drifts, confidence)
        return model.interpolate(), model.detect_segments()

