Dead Tree Segmentation using RGB & NIR Imagery

A classical computer vision pipeline for pixel-level segmentation of standing dead trees using multi-spectral aerial imagery.

The project combines RGB and NIR bands, supports YAML configuration, CLI overrides, and generates both visual reports and quantitative evaluation metrics.

Key Features

✔ Automatic loading of RGB, NIR and mask datasets

✔ Classical feature-engineering segmentation

✔ RGB- & NIR-based masking and fusion

✔ Per-image visualization & PDF summary

✔ YAML-driven configuration

✔ CLI overrides using argparse

✔ IoU metric computation & summary plots

✔ Auto-creation of output folders

✔ Reproducible environment via requirements.txt


Processing Pipeline

1️)Load data paths (RGB, NIR, masks)
2️) Explore dataset and compute statistics:

pixel histograms,

best color channels,

preview samples
3️) Generate segmentation masks using:

handcrafted RGB/NIR rules

morphological post-processing
4️) Compare predictions to ground-truth
5️) Compute:

per-image metrics

dataset averages
6️) Save all results into an output folder

Visual Outputs

The pipeline produces a side-by-side display of:

RGB image

Ground truth mask

Generated segmentation mask

NIR contribution / enhancement

Optional channel visualizations

A combined final_masks.pdf is also generated.

Dataset Structure

Expected layout:

data/
├── RGB/        # RGB images (PNG/JPG/TIF)
├── NIR/        # Near Infrared images (PNG/JPG/TIF)
└── mask/       # Binary ground truth segmentation masks

Configuration

Settings are stored in a YAML file, e.g.:

config/config_example.yaml


Contains:

paths to RGB, NIR, masks

output folder

number of processed images

thresholding parameters

plotting options

You can override any parameter with CLI flags (below).

Command-Line Interface (CLI)

Wyświetl dostępne opcje:

python main.py --help


Przykładowe uruchomienie:

python main.py \
   --config config/config_example.yaml \
   --rgb_folder data/RGB \
   --nir_folder data/NIR \
   --mask_folder data/mask \
   --output_folder output/run01 \
   --num_images 10


Argumenty z terminala mają pierwszeństwo nad wartościami w YAML.

Evaluation
Quantitative

IoU (Intersection over Union)

Average metrics

IoU per image plots

Histogram distributions

Qualitative

Visualizations saved for each image

Final multi-page PDF summary

Debug plots of color channels

Results are saved automatically under:

output/<run_name>/

Installation & Setup
1️) Create environment
python -m venv .venv

2️) Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1


If blocked:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

3️) Install dependencies
pip install -r requirements.txt

Running the Project

Default configuration:

python main.py


With overrides:

python main.py --num_images 5 --output_folder output/test

Project Structure
Dead-Tree/
│
├── main.py                     # Main pipeline script
├── requirements.txt            # Packages
├── README.md                   # This file
├── /config/
│   ├── config_example.yaml     # Default config (tracked)
│   └── config.yaml             # Local config (ignored)
├── /src/                       # Processing modules
│   ├── data_loader.py
│   ├── segmentation.py
│   ├── visualization.py
│   ├── analysis.py
│   └── processing.py
├── /data/                      
│   ├── data/                    # Default input folder (tracked)
│   │        ├── RGB/            # Local input folder (ignored)
│   │        ├── NRG/
│   │        ├── masks/
│   └── data/
└── /output/
    ├── output_examples/         # Default output folder (tracked)
    └── output/                  # Local output folder (ignored)


Design Philosophy

This project intentionally uses classical computer vision, not deep learning:

Transparent & interpretable algorithms

Easy to modify and debug

Good research baseline for future ML work

Fully reproducible & configurable

Reproducibility

All runtime choices in YAML

CLI overrides for experiments

Output folders never overwrite old runs

requirements.txt freezes dependencies

Final Notes

This repo is suitable as:

a research prototype,

coursework or thesis asset,

baseline for future DL segmentation.

Feel free to expand:

alternative spectral indices (NDVI, NBR),

adaptive thresholding,

deep learning models later.
