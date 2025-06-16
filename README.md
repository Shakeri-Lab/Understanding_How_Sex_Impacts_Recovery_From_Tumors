# ICI Sex Project

This repository contains code for analyzing sex-based differences in immune checkpoint inhibitor (ICI) response.

## Project Structure

```
project_root/
├── docs/                                   # Documentation
├── output/                                 # Analysis outputs and results
├── pair_clinical_data_and_stages_of_tumors # Pipeline for pairing clinical data and stages of tumors as well as specification, description, dictionary, and testing
├── processed_data/                         # Processed data files
├── src/                                    # Source code
│   ├── cd8_analysis/                       # CD8+ T cell analysis
│   ├── data_processing/                    # Data processing utilities
│   ├── icb_analysis/                       # Immune checkpoint blockade analysis
│   ├── immune_analysis/                    # Immune signature analysis
│   ├── sex_stratified/                     # Sex-stratified analysis
│   └── utils/                              # Shared utility functions
├── requirements.txt                        # Python dependencies
└── setup.sh                                # Environment setup script
```

## Setup

### Environment Setup

1. Make the setup script executable and run it:

```bash
chmod +x setup.sh
./setup.sh
```

2. Activate the conda environment:

```bash
source miniconda3/etc/profile.d/conda.sh
conda activate ici_sex
```

### Dependencies

Dependencies are installed by `setup.sh`.

## Modules

### CD8 Analysis

Analysis of CD8+ T cell signatures in RNA-seq data. See [CD8 Analysis README](docs/readme_cd8.md) for details.

### ICB Analysis

Analysis of immune checkpoint blockade medications and their relationship with CD8+ T cell signatures. See [ICB Analysis README](src/icb_analysis/README.md) for details.

### Immune Analysis

Analysis of immune signatures and tumor microenvironment. See [Immune Analysis README](docs/readme_microenv.md) for details.

### Sex-Stratified Analysis

Sex-stratified analysis of immune signatures and treatment response.

## Usage

Each module can be run independently. For example:

```bash
# Run ICB analysis
python -m src.icb_analysis.icb_main

# Run CD8 group analysis
python src/cd8_analysis/cd8_groups_analysis.py
```

See individual module READMEs for more details.

## Processed Data

The processed data directory contains processed data files used in the analyses. Raw data is not included in this repository.

## Results

Analysis results are saved to the output directory, organized by analysis type. 