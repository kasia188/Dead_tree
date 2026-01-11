import argparse
import yaml
from pathlib import Path

from src.data_loader import load_paths
from src.segmentation import create_mask_r_and_b_minus_h
from src.visualization import visualize_final_mask
from src.analysis import (
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
    parser = argparse.ArgumentParser(description="Pipeline for tree segmentation and analysis")

    # Key user parameters
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--rgb_folder", type=str, default=None, help="Folder with RGB images")
    parser.add_argument("--nir_folder", type=str, default=None, help="Folder with NIR images")
    parser.add_argument("--mask_folder", type=str, default=None, help="Folder with ground truth mask")
    parser.add_argument("--output_folder", type=str, default=None, help="Output folder for results")
    parser.add_argument("--num_images", type=int, default=config["NUM_IMAGES"], help="Number of images to process, omit to process all images")
    
    # Masking parameters
    parser.add_argument("--r_min_size", type=int, default=None, help="Minimum size of R-channel objects")
    parser.add_argument("--b_min_size", type=int, default=None, help="Minimum size of B-channel objects")
    parser.add_argument("--h_min_size", type=int, default=None, help="Minimum size of H-channel objects")

    parser.add_argument("--r_selem_close", type=int, default=None, help="Structuring element size for R-channel closing")
    parser.add_argument("--b_selem_close", type=int, default=None, help="Structuring element size for B-channel closing")
    parser.add_argument("--h_selem_close", type=int, default=None, help="Structuring element size for H-channel closing")
    parser.add_argument("--final_selem_close", type=int, default=None, help="Final mask closing size")

    parser.add_argument("--max_size", type=int, default=None, help="Maximum object size")

    # Adaptive mask parameters
    parser.add_argument("--min_size_weak", type=int, default=None, help="Minimum size for weak adaptive mask")
    parser.add_argument("--min_size_strong", type=int, default=None, help="Minimum size for strong adaptive mask")

    # False positive cleanup
    parser.add_argument("--fp_ratio_thr", type=float, default=None, help="False positive ratio threshold")
    parser.add_argument("--selem_close_strong", type=int, default=None, help="Structuring element for strong cleanup")

    # Channel thresholds
    parser.add_argument("--r_std_mult", type=float, default=None, help="R channel std multiplier")
    parser.add_argument("--b_std_mult", type=float, default=None, help="B channel std multiplier")
    parser.add_argument("--h_std_mult", type=float, default=None, help="H channel std multiplier")
    parser.add_argument("--nir_std_mult", type=float, default=None, help="NIR channel std multiplier")

    # Other parameters
    parser.add_argument("--h_black_ratio", type=float, default=None, help="H channel black ratio threshold")
    parser.add_argument("--hist_bins", type=int, default=None, help="Number of histogram bins")
    parser.add_argument("--hist_alpha", type=float, default=None, help="Histogram transparency")

    args = parser.parse_args()

    # Load configuration file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Override config with CLI arguments (only if provided)
    def override(key, value):
        if value is not None:
            config[key] = value

    override("INPUT_FOLDER", args.input_folder)
    override("RGB_FOLDER", args.rgb_folder)
    override("NIR_FOLDER", args.nir_folder)
    override("MASK_FOLDER", args.mask_folder)
    override("OUTPUT_FOLDER", args.output_folder)
    override("NUM_IMAGES", args.num_images)

    override("R_MIN_SIZE", args.r_min_size)
    override("B_MIN_SIZE", args.b_min_size)
    override("H_MIN_SIZE", args.h_min_size)

    override("R_SELEM_CLOSE", args.r_selem_close)
    override("B_SELEM_CLOSE", args.b_selem_close)
    override("H_SELEM_CLOSE", args.h_selem_close)
    override("FINAL_SELEM_CLOSE", args.final_selem_close)

    override("MAX_SIZE", args.max_size)

    override("MIN_SIZE_WEAK", args.min_size_weak)
    override("MIN_SIZE_STRONG", args.min_size_strong)

    override("FP_RATIO_THR", args.fp_ratio_thr)
    override("SELEM_CLOSE_STRONG", args.selem_close_strong)

    override("R_STD_MULT", args.r_std_mult)
    override("B_STD_MULT", args.b_std_mult)
    override("H_STD_MULT", args.h_std_mult)
    override("NIR_STD_MULT", args.nir_std_mult)

    override("H_BLACK_RATIO", args.h_black_ratio)
    override("HIST_BINS", args.hist_bins)
    override("HIST_ALPHA", args.hist_alpha)

    # Ensure output folder exists
    output_folder = Path(config["OUTPUT_FOLDER"])
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load data paths
    rgb_paths, nir_paths, mask_paths = load_paths(config)

    if not rgb_paths or not nir_paths or not mask_paths:
        print("No data to process. Check data folders.")

    # Segmentation
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
    print_average_metrics(df_metrics)

    # IoU plots
    plot_iou_histogram(df_metrics, config)
    plot_iou_per_image(df_metrics, config)
    print(f"IoU plots saved in: {config['OUTPUT_FOLDER']}")

if __name__ == "__main__":
    main()