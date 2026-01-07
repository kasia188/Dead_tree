from data_loader import load_paths
from segmentation import create_mask_r_and_b_minus_h
from visualization import visualize_final_mask
from analysis import evaluate_segmentation, metrics_to_df, print_average_metrics, plot_iou_histogram, plot_iou_per_image

EX_NUM = 300

def main():
    rgb_paths, nir_paths, mask_paths = load_paths()

    results = create_mask_r_and_b_minus_h(rgb_paths, nir_paths, mask_paths, num_images=EX_NUM)

    visualize_final_mask(mask_paths, results, filename="results/final_masks.pdf", num_images=EX_NUM)
    print("Plik final_masks.pdf został zapisany w katalogu projektu.")

    metrics = evaluate_segmentation(results, mask_paths, EX_NUM)
    df_metrics = metrics_to_df(metrics)

    print("Segmentation metrics")
    print(df_metrics)

    print_average_metrics(df_metrics)


    plot_iou_histogram(df_metrics, filename="results/iou_histogram.pdf", bin=20)
    plot_iou_per_image(df_metrics, filename="results/iou_per_image.pdf")

if __name__ == "__main__":
    main()