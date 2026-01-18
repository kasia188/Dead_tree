from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Loads RGB, NIR and mask file paths from the dataset directory and performs sanity checks
def load_paths(config):
    rgb_folder = Path(config["RGB_FOLDER"])
    nir_folder = Path(config["NIR_FOLDER"])
    mask_folder = Path(config["MASK_FOLDER"])

    rgb_paths = sorted(rgb_folder.glob("*")) if rgb_folder.exists() else []
    nir_paths = sorted(nir_folder.glob("*")) if nir_folder.exists() else []
    mask_paths = sorted(mask_folder.glob("*")) if mask_folder.exists() else []

    if not rgb_paths:
        logger.warning(f"No files found in RGB folder: {rgb_folder}")
    if not nir_paths:
        logger.warning(f"No files found in NIR folder: {nir_folder}")
    if not mask_paths:
        logger.warning(f"No files found in MASK folder: {mask_folder}")

    return rgb_paths, nir_paths, mask_paths
