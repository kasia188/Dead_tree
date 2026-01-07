from data_loader import load_paths
from analysis import display_data, channel_histograms, find_best_channels, plot_best_channels

def main():
    rgb_paths, nir_paths, mask_paths = load_paths()

    display_data(nir_paths, rgb_paths, mask_paths, num_images=50)

    channel_histograms(rgb_paths, mask_paths, num_images=50)

    best_channels = find_best_channels(rgb_paths, mask_paths, num_images=300)
    plot_best_channels(best_channels)

if __name__ == "__main__":
    main()
