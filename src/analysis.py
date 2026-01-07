import numpy as np
import pandas as pd
from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


#pre main analysis

def mask_to_bool(mask):
    if mask.max() <= 1.0:
        return mask > 0.5
    else:
        return mask > 128

def display_data(nir_paths, rgb_paths, mask_paths, filename="results/data_display.pdf", num_images=300):
  filename = Path(filename)
  with PdfPages(filename) as pdf:
    for i in range(num_images):
        nir = imread(nir_paths[i])
        rgb = imread(rgb_paths[i])
        mask = imread(mask_paths[i])

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(nir)
        ax[0].set_title("NIR")
        ax[0].grid(False)
        ax[0].axis('off')

        ax[1].imshow(rgb)
        ax[1].set_title("RGB")
        ax[1].grid(False)
        ax[1].axis('off')

        ax[2].imshow(mask, cmap='gray')
        ax[2].set_title("Ground truth mask")
        ax[2].grid(False)
        ax[2].axis('off')

        fig.suptitle(f"Image {i + 1}", fontsize=14)
        plt.suptitle(f"Example image {i + 1}", fontsize=14)
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)
    
def channel_statistics(channel, mask_bool, name):
    class_tree = channel[mask_bool]
    class_bg = channel[~mask_bool]

    mean_tree = np.mean(class_tree)
    mean_bg = np.mean(class_bg)
    std_tree = np.std(class_tree)
    std_bg = np.std(class_bg)

    separation = abs(mean_tree - mean_bg) / (std_tree + std_bg + 1e-8)

    print(f"\nChannel: {name}")
    print(f" Dead trees: mean={mean_tree: .2f}, std={std_tree:.2f}")
    print(f" Background: mean={mean_bg:.2f}, std={std_bg:.2f}")
    print(f" Separation: {separation:.4f}")
    
def plot_channel_hist(channel, mask_bool, title, pdf=None):
    class_tree = channel[mask_bool]
    class_bg = channel[~mask_bool]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(class_bg.flatten(), bins=50, alpha=0.5, label="Background", density=True)
    ax.hist(class_tree.flatten(), bins=50, alpha=0.5, label="Dead tree", density=True)
    ax.set_title(f"Histogram of a canal: {title}") 
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Probability density")
    ax.legend()
    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.show()

def channel_histograms(rgb_paths, mask_paths, filename="results/channel_histograms.pdf", num_images=300):
    filename = Path(filename)
    with PdfPages(filename) as pdf:
        for i in range(num_images):
            rgb = imread(rgb_paths[i])
            mask = mask_to_bool(imread(mask_paths[i]))
            hsv = color.rgb2hsv(rgb)

            channels = {
                "R": rgb[:, :, 0],
                "G": rgb[:, :, 1],
                "B": rgb[:, :, 2],
                "H": hsv[:, :, 0],
                "S": hsv[:, :, 1],
                "V": hsv[:, :, 2],
            }

            for name, ch in channels.items():
                plot_channel_hist(ch, mask, f"Image {i+1} - {name}", pdf=pdf)

def find_best_channels(rgb_paths, mask_paths, num_images=300):
    channels_names = ["R", "G", "B", "H","S", "V"]
    best_channel_counts = {name:0 for name in channels_names}

    for i in range(num_images):
        rgb = imread(rgb_paths[i])
        mask = mask_to_bool(imread(mask_paths[i]))
        hsv = color.rgb2hsv(rgb)

        channels = {"R": rgb[:,:,0],
                    "G": rgb[:,:,1],
                    "B": rgb[:,:,2],
                    "H": hsv[:, :, 0],
                    "S": hsv[:, :, 1],
                    "V": hsv[:, :, 2]
                    }
        
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
    return best_channel_counts

def plot_best_channels(best_channel_counts, filename="results/best_channels.pdf"):
    filename = Path(filename)
    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(best_channel_counts.keys(), best_channel_counts.values(), color='skyblue')
        ax.set_ylabel("Number of images, in which canal had the gratest separation")
        ax.set_title("Best channels based on class separation")
        plt.tight_layout()
        
        pdf.savefig(fig)
        plt.close(fig)

    print("Number of canals in which canal was best:", best_channel_counts) 

   

#post main analysis

def compute_iou(mask_pred, mask_gt, smooth=1e-8):
    mask_pred = mask_pred.astype(bool)
    mask_gt = mask_gt.astype(bool)
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return intersection / (union + smooth)

def confusion_matrix(mask_pred, mask_gt):
    tp = np.logical_and(mask_pred, mask_gt).sum()
    fp = np.logical_and(mask_pred, ~mask_gt).sum()
    fn = np.logical_and(~mask_pred, mask_gt).sum()
    tn = np.logical_and(~mask_pred, ~mask_gt).sum()
    return tp, fp, fn, tn

def dice_score(mask_pred, mask_gt, smooth=1e-8):
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return (2.0*intersection + smooth) / (union + smooth)

def precision_score(mask_pred, mask_gt, smooth=1e-8):
    tp, fp, _, _ = confusion_matrix(mask_pred, mask_gt)
    return tp / (tp + fp + smooth)

def recall_score(mask_pred, mask_gt, smooth=1e-8):
    tp, _, fn, _ = confusion_matrix(mask_pred, mask_gt)
    return tp / (tp + fn + smooth)

def evaluate_segmentation(results, mask_paths, num_images):
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

def metrics_to_df(metrics):
    return pd.DataFrame(metrics)

def print_average_metrics(df_metrics):
    metrics_to_avg = ["iou", "dice", "precision", "recall"]
    averages = df_metrics[metrics_to_avg].mean()

    print("\nAverage metrics across images:")
    for metric in metrics_to_avg:
        value = averages[metric]
        print(f" {metric}: {value:.4f}")

def plot_iou_histogram(df_metrics, filename, bin=20):
    filename = Path(filename)
    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(df_metrics["iou"], bins=bin, color="skyblue")
        ax.set_xlabel("IoU")
        ax.set_ylabel("Number of images")
        ax.set_title("IoU distribution")
        ax.grid(True)
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

def plot_iou_per_image(df_metrics, filename):
    filename = Path(filename)
    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df_metrics["image_id"], df_metrics["iou"], marker="o", linestyle="-", color="blue")
        ax.set_xlabel("Image ID")
        ax.set_ylabel("IoU")
        ax.set_title("IoU per image")
        ax.grid(True)
        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)