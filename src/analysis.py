import numpy as np
import pandas as pd
from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import logging

logger = logging.getLogger(__name__)

#pre main analysis

# Converts a mask to boolean, handling both 0–1 and 0–255 formats
def mask_to_bool(mask):
    if mask.max() <= 1.0:
        return mask > 0.5
    else:
        return mask > 128

# Displays NIR, RGB and mask images and saves them into a multipage PDF file
def display_data(nir_paths, rgb_paths, mask_paths, config):
    num_images_requested = config.get("NUM_IMAGES")
    total_available = min(len(nir_paths), len(rgb_paths), len(mask_paths))

    if num_images_requested is None:
        num_images = total_available
    else:
        if num_images_requested > total_available:
            logger.warning(
                f"Requested number of images ({num_images_requested}) exceeds available images ({total_available}). "
                f"Displaying {total_available} images instead."
            )
        num_images = min(num_images_requested, total_available)

    output_file = Path(config["OUTPUT_FOLDER"]) / "data_display.pdf"
    logger.info(f"Saving example images PDF for {num_images} images to {output_file}")

    with PdfPages(output_file) as pdf:
        for i in range(num_images):
            nir = imread(nir_paths[i])
            rgb = imread(rgb_paths[i])
            mask = imread(mask_paths[i])

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(nir); ax[0].set_title("NIR"); ax[0].axis('off')
            ax[1].imshow(rgb); ax[1].set_title("RGB"); ax[1].axis('off')
            ax[2].imshow(mask, cmap='gray'); ax[2].set_title("Ground truth mask"); ax[2].axis('off')
            fig.suptitle(f"Image {i + 1}", fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    
    logger.info(f"Example images PDF saved: {output_file}")

# Computes means, standard deviations and separation between masked (tree) and background pixels 
def channel_statistics(channel, mask_bool, name):
    class_tree = channel[mask_bool]
    class_bg = channel[~mask_bool]

    mean_tree = np.mean(class_tree)
    mean_bg = np.mean(class_bg)
    std_tree = np.std(class_tree)
    std_bg = np.std(class_bg)
    separation = abs(mean_tree - mean_bg) / (std_tree + std_bg + 1e-8)

    logger.info(f"Channel: {name}")
    logger.info(f"  Dead trees: mean={mean_tree:.2f}, std={std_tree:.2f}")
    logger.info(f"  Background: mean={mean_bg:.2f}, std={std_bg:.2f}")
    logger.info(f"  Separation: {separation:.4f}")

# Plots pixel value histograms for masked foreground vs background
def plot_channel_hist(channel, mask_bool, title, config, pdf=None):
    class_tree = channel[mask_bool]
    class_bg = channel[~mask_bool]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(class_bg.flatten(), bins=config["HIST_BINS"], alpha=config["HIST_ALPHA"], label="Background", density=True)
    ax.hist(class_tree.flatten(), bins=config["HIST_BINS"], alpha=config["HIST_ALPHA"], label="Dead tree", density=True)
    ax.set_title(f"Histogram of a canal: {title}") 
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Probability density")
    ax.legend()
    plt.tight_layout()

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.show()

# Creates histograms for RGB + HSV channels and saves them to a multipage PDF file
def channel_histograms(rgb_paths, mask_paths, config):
    num_images = config.get("NUM_IMAGES") or len(rgb_paths)
    output_file = Path(config["OUTPUT_FOLDER"]) / "channel_histograms.pdf"
    logger.info(f"Saving channel histograms PDF for {num_images} images to {output_file}")

    with PdfPages(output_file) as pdf:
        for i in range(num_images):
            rgb = imread(rgb_paths[i])
            mask = mask_to_bool(imread(mask_paths[i]))
            hsv = color.rgb2hsv(rgb)

            channels = {
                "R": rgb[:, :, 0], "G": rgb[:, :, 1], "B": rgb[:, :, 2],
                "H": hsv[:, :, 0], "S": hsv[:, :, 1], "V": hsv[:, :, 2]
            }

            for name, ch in channels.items():
                plot_channel_hist(ch, mask, f"Image {i+1} - {name}", config, pdf=pdf)

    logger.info(f"Channel histograms PDF saved: {output_file}")

# Finds which channel provides the strongest separation between tree mask and background
def find_best_channels(rgb_paths, mask_paths, config):
    num_images = config.get("NUM_IMAGES") or len(rgb_paths)
    channels_names = ["R", "G", "B", "H","S", "V"]
    best_channel_counts = {name:0 for name in channels_names}

    for i in range(num_images):
        rgb = imread(rgb_paths[i])
        mask = mask_to_bool(imread(mask_paths[i]))
        hsv = color.rgb2hsv(rgb)

        channels = {name: rgb[:,:,idx] if name in ["R","G","B"] else hsv[:,:,["H","S","V"].index(name)]
                    for idx,name in enumerate(["R","G","B","H","S","V"])}

        separations = {}
        for name, ch in channels.items():
            class_tree = ch[mask]
            class_bg = ch[~mask]
            mean_tree, mean_bg = np.mean(class_tree), np.mean(class_bg)
            std_tree, std_bg = np.std(class_tree), np.std(class_bg)
            separations[name] = abs(mean_tree - mean_bg) / (std_tree + std_bg + 1e-8)

        max_sep = max(separations.values())
        for name, sep in separations.items():
            if sep == max_sep:
                best_channel_counts[name] += 1

    logger.info(f"Best channel counts: {best_channel_counts}")
    return best_channel_counts

# Plots how many images each channel was the best separator for
def plot_best_channels(best_channel_counts, config):
    output_file = Path(config["OUTPUT_FOLDER"]) / "best_channels.pdf"
    logger.info(f"Saving best channels PDF to {output_file}")

    with PdfPages(output_file) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(best_channel_counts.keys(), best_channel_counts.values(), color='skyblue')
        ax.set_ylabel("Number of images where channel was best")
        ax.set_title("Best channels based on class separation")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    logger.info(f"Best channels PDF saved. Counts: {best_channel_counts}")

   

#post main analysis

# Computes IoU (Intersection over Union) between predicted and ground-truth masks
def compute_iou(mask_pred, mask_gt, smooth=1e-8):
    mask_pred = mask_pred.astype(bool)
    mask_gt = mask_gt.astype(bool)
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return intersection / (union + smooth)

# Builds a confusion matrix: TP, FP, FN, TN
def confusion_matrix(mask_pred, mask_gt):
    tp = np.logical_and(mask_pred, mask_gt).sum()
    fp = np.logical_and(mask_pred, ~mask_gt).sum()
    fn = np.logical_and(~mask_pred, mask_gt).sum()
    tn = np.logical_and(~mask_pred, ~mask_gt).sum()
    return tp, fp, fn, tn

# Computes Dice coefficient for segmentation accuracy
def dice_score(mask_pred, mask_gt, smooth=1e-8):
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return (2.0*intersection + smooth) / (union + smooth)

# Computes segmentation precision (TP / (TP + FP))
def precision_score(mask_pred, mask_gt, smooth=1e-8):
    tp, fp, _, _ = confusion_matrix(mask_pred, mask_gt)
    return tp / (tp + fp + smooth)

# Computes segmentation recall (TP / (TP + FN))
def recall_score(mask_pred, mask_gt, smooth=1e-8):
    tp, _, fn, _ = confusion_matrix(mask_pred, mask_gt)
    return tp / (tp + fn + smooth)

# Evaluates segmentation metrics for many images and stores per-image statistics
def evaluate_segmentation(results, mask_paths, config):
    num_images = config["NUM_IMAGES"]
    if num_images is None:
        num_images = len(mask_paths)
    metrics = {
        "image_id": [],
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "tp": [],
        "fp": [],
        "fn":[],
        "tn": []    
        }
    
    for i in range(num_images):
        mask_pred = results["mask_final"][i]
        mask_gt = (imread(mask_paths[i]) > 128).astype(bool)

        tp, fp, fn, tn = confusion_matrix(mask_pred, mask_gt)

        metrics["image_id"].append(i)
        metrics["iou"].append(compute_iou(mask_pred, mask_gt))
        metrics["dice"].append(dice_score(mask_pred, mask_gt))
        metrics["precision"].append(precision_score(mask_pred, mask_gt))
        metrics["recall"].append(recall_score(mask_pred, mask_gt))
        metrics["tp"].append(tp)
        metrics["fp"].append(fp)
        metrics["fn"].append(fn)
        metrics["tn"].append(tn)

    return metrics

# Converts metric results stored in a dict into a pandas DataFrame
def metrics_to_df(metrics):
    return pd.DataFrame(metrics)

# Prints average IoU, Dice, precision and recall across all images
def print_average_metrics(df_metrics):
    metrics_to_avg = ["iou", "dice", "precision", "recall"]
    averages = df_metrics[metrics_to_avg].mean()

    logger.info("Average metrics across images:")
    for metric in metrics_to_avg:
        value = averages[metric]
        logger.info(f"  {metric}: {value:.4f}")

# Plots IoU distribution histogram for all images
def plot_iou_histogram(df_metrics, config):
    output_file = Path(config["OUTPUT_FOLDER"]) / "iou_histogram.pdf"

    with PdfPages(output_file) as pdf:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(df_metrics["iou"], bins=config["HIST_BINS"], color="skyblue")
        ax.set_xlabel("IoU")
        ax.set_ylabel("Number of images")
        ax.set_title("IoU distribution")
        ax.grid(True)
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

# Plots IoU score per image ID and saves to PDF
def plot_iou_per_image(df_metrics, config):
    output_file = Path(config["OUTPUT_FOLDER"]) / "iou_histogram.pdf"

    with PdfPages(output_file) as pdf:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df_metrics["image_id"], df_metrics["iou"], marker="o", linestyle="-", color="blue")
        ax.set_xlabel("Image ID")
        ax.set_ylabel("IoU")
        ax.set_title("IoU per image")
        ax.grid(True)
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)