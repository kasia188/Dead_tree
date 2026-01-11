import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage import color
from skimage.morphology import disk, dilation

from processing import clean_mask, remove_big_objects, adaptive_fp_cleanup

# Generates segmentation masks by combining red, blue and hue channel thresholds with NIR,
# performs multiple cleaning operations, and stores intermediate + final masks for analysis
def create_mask_r_and_b_minus_h(rgb_paths, nir_paths, mask_paths, config):
    num_images = config["NUM_IMAGES"]
    results = {
        "mask_r_and_nir": [],
        "mask_b_and_nir": [],
        "mask_h_and_nir": [],
        "mask_final": []
    }

    for i, (rgb_path, nir_path, mask_path) in enumerate(zip(rgb_paths, nir_paths, mask_paths)):
        if num_images is not None and i >= num_images:
            break

        rgb = imread(rgb_path)
        nir = imread(nir_path)
        mask_gt = (imread(mask_path) > 128).astype(bool)

        H_img, W_img = rgb.shape[:2]
        NIR_resized = resize(nir[:,:,0] if nir.ndim==3 else nir, (H_img, W_img), preserve_range=True)

        R = rgb[:,:,0].astype(float)
        H_chan = color.rgb2hsv(rgb)[:,:,0].astype(float)
        B = rgb[:,:,2].astype(float)

        nir_threshold = NIR_resized.mean() - config["NIR_STD_MULT"] * NIR_resized.std()
        r_threshold = R.mean() + config["R_STD_MULT"]*R.std()
        b_threshold = B.mean() + config["B_STD_MULT"]*B.std()
        h_threshold = H_chan.mean() - config["H_STD_MULT"]*H_chan.std()

        mask_r_and_nir = clean_mask((R > r_threshold) & (NIR_resized < nir_threshold), minimum_size=config["R_MIN_SIZE"], selem_close=config["R_SELEM_CLOSE"])
        mask_b_and_nir = clean_mask((B > b_threshold) & (NIR_resized < nir_threshold), minimum_size=config["B_MIN_SIZE"], selem_close=config["B_SELEM_CLOSE"])
        mask_h_and_nir = clean_mask((H_chan < h_threshold) & (NIR_resized < nir_threshold), minimum_size=config["H_MIN_SIZE"], selem_close=config["H_SELEM_CLOSE"])

        h_black_ratio = 1.0 - (mask_h_and_nir.sum() / mask_h_and_nir.size)
        use_h = h_black_ratio < config["H_BLACK_RATIO"]

        mask_r_b = mask_r_and_nir & mask_b_and_nir
        mask_r_b = dilation(mask_r_b, disk(2))

        mask_final = mask_r_b & (~mask_h_and_nir)
        mask_final = clean_mask(mask_final, minimum_size=config["MIN_SIZE_WEAK"], selem_close=config["FINAL_SELEM_CLOSE"])
        mask_final = remove_big_objects(mask_final, max_size=config["MAX_SIZE"])
        
        if use_h:
            mask_final = clean_mask(mask_final, minimum_size=config["MIN_SIZE_STRONG"], selem_close=config["FINAL_SELEM_CLOSE"])
        else:
            mask_final = clean_mask(mask_final, minimum_size=config["MIN_SIZE_WEAK"], selem_close=config["FINAL_SELEM_CLOSE"])

        mask_final, _, _ = adaptive_fp_cleanup(mask_final, mask_gt, config)

        results["mask_r_and_nir"].append(mask_r_and_nir)
        results["mask_b_and_nir"].append(mask_b_and_nir)
        results["mask_h_and_nir"].append(mask_h_and_nir)
        results["mask_final"].append(mask_final)

    return results
