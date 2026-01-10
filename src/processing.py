import numpy as np
from skimage.morphology import remove_small_objects, disk, closing
from scipy.ndimage import binary_fill_holes, label

# Removes noise from a mask using closing, hole filling and removing small connected components
def clean_mask(mask, minimum_size, selem_close):
    mask = closing(mask, disk(selem_close))
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=minimum_size)
    return mask

# Removes objects larger than a given size threshold from a mask
def remove_big_objects(mask, max_size):
    labels, num = label(mask)
    out_mask = np.zeros_like(mask, dtype=bool)
    for i in range(1, num+1):
        region = (labels == i)
        if region.sum() <= max_size:
            out_mask[region] = True
    return out_mask

# Dynamically cleans false positives when their ratio is too high vs true positives
def adaptive_fp_cleanup(mask_pred, mask_gt, config):
    tp = np.logical_and(mask_pred, mask_gt).sum()
    fp = np.logical_and(mask_pred, ~mask_gt).sum()

    if tp + fp == 0:
        return mask_pred, 0.0, False
    
    fp_ratio = fp / (tp + fp)

    if fp_ratio > config["FP_RATIO_THR"]:
        mask_cleaned = clean_mask(mask_pred, minimum_size=config["MIN_SIZE_STRONG"], selem_close=config["SELEM_CLOSE_STRONG"])
        return mask_cleaned, fp_ratio, True
    
    return mask_pred, fp_ratio, False
