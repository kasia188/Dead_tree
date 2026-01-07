import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage import color
from skimage.morphology import disk, dilation

from processing import clean_mask, remove_big_objects, adaptive_fp_cleanup

def create_mask_r_and_b_minus_h(rgb_paths, nir_paths, mask_paths, num_images):
    results = {
        "mask_r_and_nir": [],
        "mask_b_and_nir": [],
        "mask_h_and_nir": [],
        "mask_final": []
    }

    for i, (rgb_path, nir_path, mask_path) in enumerate(zip(rgb_paths, nir_paths, mask_paths)):
        if i >= num_images:
            break

        rgb = imread(rgb_path)
        nir = imread(nir_path)
        mask_gt = (imread(mask_path) > 128).astype(bool)

        H_img, W_img = rgb.shape[:2]
        NIR_resized = resize(nir[:,:,0] if nir.ndim==3 else nir, (H_img, W_img), preserve_range=True)

        R = rgb[:,:,0].astype(float)
        H_chan = color.rgb2hsv(rgb)[:,:,0].astype(float)
        B = rgb[:,:,2].astype(float)

        nir_threshold = NIR_resized.mean() - 0.5 * NIR_resized.std()
        r_threshold = R.mean() + 0.1*R.std()
        b_threshold = B.mean() + 0.1*B.std()
        h_threshold = H_chan.mean() - 0.1*H_chan.std()

        mask_r_and_nir = clean_mask((R > r_threshold) & (NIR_resized < nir_threshold), minimum_size=5, selem_close=3)
        mask_b_and_nir = clean_mask((B > b_threshold) & (NIR_resized < nir_threshold), minimum_size=5, selem_close=3)
        mask_h_and_nir = clean_mask((H_chan < h_threshold) & (NIR_resized < nir_threshold), minimum_size=150, selem_close=2)

        h_black_ratio = 1.0 - (mask_h_and_nir.sum() / mask_h_and_nir.size)
        use_h = h_black_ratio < 0.9

        mask_r_b = mask_r_and_nir & mask_b_and_nir
        mask_r_b = dilation(mask_r_b, disk(2))

        if use_h:
            mask_final = mask_r_b & (~mask_h_and_nir)
            mask_final = clean_mask(mask_final, minimum_size=5, selem_close=3)
            mask_final = remove_big_objects(mask_final, max_size=1200)
            mask_final = clean_mask(mask_final, minimum_size=50, selem_close=3)

        else:
            mask_final = mask_r_b & (~mask_h_and_nir)
            mask_final = clean_mask(mask_final, minimum_size=5, selem_close=3)
            mask_final = remove_big_objects(mask_final, max_size=1200)
            mask_final = clean_mask(mask_final, minimum_size=5, selem_close=3)

        mask_final, _, _ = adaptive_fp_cleanup(mask_final, mask_gt, fp_ratio_thr=0.4)

        results["mask_r_and_nir"].append(mask_r_and_nir)
        results["mask_b_and_nir"].append(mask_b_and_nir)
        results["mask_h_and_nir"].append(mask_h_and_nir)
        results["mask_final"].append(mask_final)

    return results