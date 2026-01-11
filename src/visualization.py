import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# Visualizes intermediate and final segmentation masks along with ground truth,
# and saves all images into a multipage PDF file
def visualize_final_mask(mask_paths, results, config):
    num_images = config["NUM_IMAGES"]

    output_folder = Path(config["OUTPUT_FOLDER"])
    output_folder.mkdir(parents=True, exist_ok=True)

    filename = output_folder / config["FINAL_MASKS_FILENAME"]

    total_available = len(results["mask_final"])
    num_images = min(num_images, total_available)

    with PdfPages(filename) as pdf:
        for i in range(num_images):
            fig, axes = plt.subplots(1, 5, figsize=(25,5))
            fig.suptitle(f"Image {i + 1}", fontsize=18)
            
            axes[0].imshow(results["mask_r_and_nir"][i], cmap="gray")
            axes[0].set_title("R and NIR")
            axes[0].axis("off")

            axes[1].imshow(results["mask_b_and_nir"][i], cmap="gray")
            axes[1].set_title("B and NIR")
            axes[1].axis("off")

            axes[2].imshow(results["mask_h_and_nir"][i], cmap="gray")
            axes[2].set_title("H and NIR")
            axes[2].axis("off")

            axes[3].imshow(results["mask_final"][i], cmap="gray")
            axes[3].set_title("Final mask")
            axes[3].axis("off")

            axes[4].imshow((imread(mask_paths[i]) > 128).astype(bool), cmap="gray")
            axes[4].set_title("Ground truth")
            axes[4].axis("off")

            pdf.savefig(fig)
            plt.close(fig)