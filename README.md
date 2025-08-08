# Keinath Lab Calcium Imaging Analysis Pipeline

This repository contains a modular set of Python tools designed to process and analyze calcium imaging data recorded during rodent open-field navigation. It includes file handling utilities, spatial discretization, rate map construction, place cell quantification, and successor representation modeling. While tailored for the Keinath Lab's workflow and data structure (e.g., `.mat` files exported from MATLAB), the tools are flexible and adaptable to other formats with minimal changes.

## Repository Structure

```
.
├── Calcium.py           # Main class for calcium data analysis
├── Utils.py             # General-purpose helper functions
├── eval_dir.py          # File/folder discovery utilities
├── run_pipeline.py      # Example driver script for a single session
```

## Features

- Load and parse MATLAB-based calcium imaging datasets (`.mat`)
- Binarize and preprocess calcium traces
- Compute animal trajectory and occupancy maps
- Generate spatial rate maps (whole, split-half, smoothed/unsmoothed)
- Quantify place cell stability using split-half correlations and permutation testing
- Compute spatial information (bits/spike)
- Build transition matrices and successor representations
- Export figures for use in publications and lab notebooks

## Requirements

Tested in Python 3.9+. You can install required packages using:

```bash
pip install -r requirements.txt
```

**Dependencies**:

- numpy
- scipy
- h5py
- matplotlib
- tqdm
- termcolor

Optional for parallelism and performance:

- multiprocessing (built-in)

## Quickstart

### 1. Set up your directory

Your data should be organized like this:

```
~/Keinath Lab Dropbox/
└── MatlabData/
    └── OpenField/
        └── [MouseID]/
            ├── [Date1].mat
            ├── [Date2].mat
            └── ...
```

You can adapt the search paths in `run_pipeline.py` to your structure.

### 2. Run a full session analysis

```bash
python run_pipeline.py
```

This script will:

1. Locate a `.mat` file
2. Extract and store metadata
3. Compute rate maps, statistics, transitions, and SR
4. Save results to a `processed/` folder with figures and `.pkl` output

## Modules Overview

### Calcium.py

Core class for data loading, transformation, and analysis.

**Key methods:**

| Method                      | Description |
|----------------------------|-------------|
| `get_data_from_mat()`      | Parses MATLAB `.mat` structure |
| `set_params()`             | Defines binning & environment size |
| `grab_occupancy()`         | Computes 2D sampling heatmap |
| `get_maps()`               | Creates smoothed rate maps |
| `get_split_maps()`         | Generates first/second-half maps |
| `split_map_corr()`         | Computes stability via correlation |
| `permute_p()`              | Assesses significance via permutation |
| `get_state_transitions()`  | Builds transition matrix |
| `analytical_successor_representation()` | Computes SR (gamma-discounted) |
| `save_object()` / `load_object()` | Save/load full session object |

### Utils.py

Reusable functions for computation and structure handling.

Examples:

- `linear_position(position, env_size, bins)`  
  → Convert 2D coordinates to linear bin indices
- `filter_maps(maps, sigma)`  
  → Apply Gaussian smoothing
- `get_linear_velocity(position)`  
  → Return velocity in cm/sec
- `sub2ind(...)`  
  → MATLAB-style linear indexing

### eval_dir.py

Handles file system discovery and manipulation:

- `find_folder_os(start_path, folder_name)`
- `get_file_paths(...)`
- `mk_dir(...)`

Used for:

- Finding `MatlabData` and `OpenField` folders
- Creating `processed/MouseID/Date/` subfolders
- Pulling paths to `.mat` files

### run_pipeline.py

Example script showing how to run the full analysis for a single recording session.

**Steps include:**

1. Locate file
2. Initialize `Calcium(...)` object
3. Extract raw data
4. Run analysis pipeline
5. Export figures + store `.pkl` object

You can loop over multiple sessions by modifying the `i = 0` line.

## Outputs

Each session will generate the following:

### Folder: `fig/`
Contains visualizations for quick inspection and publication:

- `path.pdf` – Mouse trajectory path in the environment.
- `occupancy_matrix.pdf` – Heatmap of spatial sampling across the environment.
- `rotational_velocity.pdf` – Histogram of rotational velocity (deg/s).
- `linear_velocity.pdf` – Histogram of linear velocity (cm/s).
- `split_half.pdf` – Boxplot of split-half correlations for place cell stability.
- `spatial_information.pdf` – Histogram of spatial information (bits/spike).
- `split_half_null_hist.pdf` – Overlaid null distribution and actual split-half histogram.
- `cumulative_proportion.pdf` – Cumulative distribution of p-values (proportion of stable cells).
- `spike_plot_rate_map_examp.pdf` – Path with overlaid spikes for most stable cell, alongside rate map.

### File: `data.pkl`
A Python pickle file containing all computed attributes from the `Calcium` class instance:
- Position, traces, rate maps, split-half results, velocities, transitions, SR matrix, and more.

This makes it easy to reload and interact with analyzed sessions without reprocessing the raw data.

## Generalization Notes

- This pipeline assumes MATLAB `.mat` files with keys like `'processed'`, `'trace'`, and `'p'`
- If your data is organized differently, update:
  - `Calcium.get_data_from_mat()`
  - `run_pipeline.py` (file path logic)
- Binning parameters (`bins`, `env_size`) can be adjusted for spatial resolution

## Contributions

This repository is actively developed within the Keinath Lab. If you'd like to adapt this pipeline for your own lab or want help extending functionality, feel free to reach out or submit a pull request.

## License

MIT License — open to use, modify, and extend with attribution.

## Acknowledgments

This software was developed to support place cell analysis, successor representations, and exploratory behavior analysis in the Keinath Lab's calcium imaging experiments.
