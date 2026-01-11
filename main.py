import argparse
import yaml
from pathlib import Path

from data_loader import load_paths
from segmentation import create_mask_r_and_b_minus_h
from visualization import visualize_final_mask
from analysis import (
    display_data,
    channel_histograms,
    find_best_channels,
    plot_best_channels,
    evaluate_segmentation,
    metrics_to_df,
    print_average_metrics,
    plot_iou_histogram,
    plot_iou_per_image
)

def main():
    default_config_file = Path("config/config_example.yaml")
    with open(default_config_file, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Pipeline for tree segmentation and analysis")

    # Key user parameters
    parser.add_argument("--config", type=str, default="config/config_example.yaml", help="Path to configuration file YAML")
    parser.add_argument("--rgb_folder", type=str, default=None, help="Folder with RGB images")
    parser.add_argument("--nir_folder", type=str, default=None, help="Folder z NIR images")
    parser.add_argument("--mask_folder", type=str, default=None, help="Folder z ground truth mask")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder wyjściowy dla wyników")
    parser.add_argument("--num_images", type=int, default=config["NUM_IMAGES"], help="Liczba obrazów do przetworzenia")
    
    args = parser.parse_args()

    if args.rgb_folder: config["RGB_FOLDER"] = args.rgb_folder
    if args.nir_folder: config["NIR_FOLDER"] = args.nir_folder
    if args.mask_folder: config["MASK_FOLDER"] = args.mask_folder
    if args.output_folder: config["OUTPUT_FOLDER"] = args.output_folder
    if args.num_images: config["NUM_IMAGES"] = args.num_images

    output_folder = Path(config["OUTPUT_FOLDER"])
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load file paths
    rgb_paths, nir_paths, mask_paths = load_paths(config)

    if not rgb_paths or not nir_paths or not mask_paths:
        print("No data to process. Check data folders.")

    results = create_mask_r_and_b_minus_h(rgb_paths, nir_paths, mask_paths, config)

    # Data analysis
    display_data(nir_paths, rgb_paths, mask_paths, config)
    channel_histograms(rgb_paths, mask_paths, config)
    best_channels = find_best_channels(rgb_paths, mask_paths, config)
    plot_best_channels(best_channels, config)

    # Visualization of final masks
    visualize_final_mask(mask_paths, results, config)
    print(f"File final_masks.pdf was saved in: {config['OUTPUT_FOLDER']}")

    # Segmentation metrics
    metrics = evaluate_segmentation(results, mask_paths, config)
    df_metrics = metrics_to_df(metrics)

    print("Segmentation metrics")
    print(df_metrics)
    print("Segmentation metrics per image:")
    print_average_metrics(df_metrics)

    # IoU plots
    plot_iou_histogram(df_metrics, config)
    plot_iou_per_image(df_metrics, config)
    print(f"IoU plots saved in: {config['OUTPUT_FOLDER']}")

if __name__ == "__main__":
    main()