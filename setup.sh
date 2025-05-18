#!/bin/bash
set -euo pipefail

export EDITOR=${EDITOR:-vi}

# Set the project directory
PROJECT_DIR="/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors"
CONDA_DIR="$PROJECT_DIR/miniconda3"

# Download and install Miniconda if not already installed
if [ ! -d "$CONDA_DIR" ]; then
    echo "Downloading Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    
    echo "Installing Miniconda..."
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR
    
    # Initialize conda for bash shell
    echo "Initializing conda..."
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    
    # Add conda initialization to .bashrc if not already present
    if ! grep -q "conda initialize" ~/.bashrc; then
        $CONDA_DIR/bin/conda init bash
    fi
    
    # Clean up
    rm Miniconda3-latest-Linux-x86_64.sh
fi

# Make sure conda is initialized
source "$CONDA_DIR/etc/profile.d/conda.sh"

conda config --env --set channel_priority strict
conda config --env --prepend channels conda-forge
conda config --env --append channels bioconda

# Create new environment if it doesn't exist
if ! conda env list | grep -q "ici_sex"; then
    echo "Creating ici_sex environment..."
    conda create -y -n ici_sex python=3.12
fi

# Activate the environment
conda activate ici_sex

# Install required packages
echo "Installing required packages..."
conda install -y 'libblas=*=*openblas' lifelines numpy matplotlib pandas scipy scikit-learn seaborn statsmodels 'r-base>=4.4,<4.5' r-essentials r-devtools r-remotes 'r-htmltools>=0.5.7' r-pracma r-quadprog 'bioconductor-genomeinfodbdata=1.2.13' bioconductor-gsva bioconductor-gseabase bioconductor-biobase bioconductor-summarizedexperiment bioconductor-biomart bioconductor-org.hs.eg.db

pip install --no-cache-dir rpy2==3.5.17

# Install R and Bioconductor packages through conda
echo "Installing R packages..."

# Install xCell and prepare data
conda run -n ici_sex R -e '
  options(repos = c(CRAN = "https://cloud.r-project.org"))
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
  BiocManager::install(version = "3.20", ask = FALSE, update = FALSE)
  
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes");
  
  # Install xCell from GitHub
  remotes::install_github("dviraran/xCell", dependencies = FALSE, upgrade = "never");
  
  # Download and save xCell data
  library(xCell);
  data_url <- "https://raw.githubusercontent.com/dviraran/xCell/master/data/xCell.data.rda";
  data_path <- file.path(.libPaths()[1], "xCell", "data", "xCell.data.rda");
  dir.create(dirname(data_path), recursive=TRUE, showWarnings=FALSE);
  download.file(data_url, data_path);
  
  # Verify data
  load(data_path);
  if (exists("xCell.data")) {
    print("xCell data loaded successfully");
    print(paste("Number of reference genes:", length(xCell.data$genes)));
  } else {
    stop("Failed to load xCell data");
  }
'

echo "Environment setup complete!" 
