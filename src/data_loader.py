from pathlib import Path

def load_paths():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "USA_segmentation"

    rgb_dir = data_dir / "RGB_images"
    nir_dir = data_dir / "NRG_images"
    mask_dir = data_dir / "masks"

    print("Project root:", project_root)
    print("Data dir exists:", data_dir.exists())
    print("RGB exists:", rgb_dir.exists())
    print("NRG exists:", nir_dir.exists())
    print("Masks exists:", mask_dir.exists())

    rgb_paths = sorted(rgb_dir.glob("*")) if rgb_dir.exists() else []
    nir_paths = sorted(nir_dir.glob("*")) if nir_dir.exists() else []
    mask_paths = sorted(mask_dir.glob("*")) if mask_dir.exists() else []

    print("Number of RGB files:", len(rgb_paths))
    print("Number of NRG files:", len(nir_paths))
    print("Number of mask files:", len(mask_paths))

    if not rgb_paths or not nir_paths or not mask_paths:
        raise FileNotFoundError(
            f"Dataset not found or empty. "
            f"Check that your data folder is here: {data_dir} "
            f"and contains RGB_images, NIR_images, and masks folders with files."
        )

    return rgb_paths, nir_paths, mask_paths
