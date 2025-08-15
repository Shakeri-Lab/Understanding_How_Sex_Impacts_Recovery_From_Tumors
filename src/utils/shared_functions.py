"""
Shared Functions Module
Common utility functions used across multiple modules
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback

from src.config import paths


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s – %(levelname)s – %(message)s"
)
logger = logging.getLogger(__name__)


def load_expression_matrix():
    list_of_paths_of_expression_data = list(paths.gene_and_transcript_expression_results.glob("*.genes.results"))
    
    logger.info(f"{len(list_of_paths_of_expression_data)} expression files were found.")

    list_of_expression_data_frames = []
    for path_of_expression_data in list_of_paths_of_expression_data:
        sample_id = os.path.basename(str(path_of_expression_data)).split('.')[0]
        expression_data_frame = pd.read_csv(path_of_expression_data, sep = '\t')
        list_of_expression_data_frames.append(
            expression_data_frame[['gene_id', 'TPM']]
            .set_index('gene_id')['TPM']
            .rename(sample_id)
        )
    # Expression data frame has rows corresponding to genes and columns corresponding to samples.
    expression_matrix = pd.concat(list_of_expression_data_frames, axis = 1)

    logger.info(f"Expression matrix has shape {expression_matrix.shape}.")

    twenty_percent_of_number_of_samples = 0.2 * expression_matrix.shape[1]
    expression_matrix = expression_matrix.loc[
        (expression_matrix > 1).sum(axis = 1) >= twenty_percent_of_number_of_samples
    ]

    logger.info(f"Expression matrix after filtering has shape {expression_matrix.shape}.")

    return expression_matrix


def filter_by_primary_diagnosis_site(data_frame_of_clinical_data_and_CD8_signature_scores):
    list_of_sites_to_exclude = ["Prostate gland", "Vulva, NOS"]
    initial_number_of_rows = len(data_frame_of_clinical_data_and_CD8_signature_scores)
    filtered_data_frame = data_frame_of_clinical_data_and_CD8_signature_scores[
        ~data_frame_of_clinical_data_and_CD8_signature_scores["PrimaryDiagnosisSite"].isin(list_of_sites_to_exclude)
    ]
    final_number_of_rows = len(filtered_data_frame)
    number_of_excluded_rows = initial_number_of_rows - final_number_of_rows
    
    logger.info(f"Rows of data frame of clinical data and CD8 signatures scores with specific primary diagnosis sites will be filtered out.")
    
    return filtered_data_frame


def calculate_survival_months(df, age_at_diagnosis_col='AGE_AT_DIAGNOSIS', 
                             age_at_last_contact_col='AGE_AT_LAST_CONTACT',
                             age_at_death_col='AGE_AT_DEATH',
                             vital_status_col='VITAL_STATUS'):
    """Calculate survival months from age columns"""
    try:
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Check if required columns exist
        required_cols = [age_at_diagnosis_col, vital_status_col]
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        
        if missing_cols:
            print(f"Warning: Missing required columns for survival calculation: {missing_cols}")
            return result_df
        
        # Calculate survival months
        for col in [age_at_diagnosis_col, age_at_last_contact_col, age_at_death_col]:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors = "coerce")
                
        if vital_status_col in result_df.columns:
            result_df[vital_status_col] = result_df[vital_status_col].astype(str).str.strip().str.capitalize()
        
        result_df['survival_months'] = np.nan
        
        # For deceased patients
        deceased_mask = result_df[vital_status_col] == 'Dead'
        if age_at_death_col in result_df.columns:
            result_df.loc[deceased_mask, 'survival_months'] = (
                result_df.loc[deceased_mask, age_at_death_col] - 
                result_df.loc[deceased_mask, age_at_diagnosis_col]
            ) * 12
        
        # For living patients
        living_mask = result_df[vital_status_col] == 'Alive'
        if age_at_last_contact_col in result_df.columns:
            result_df.loc[living_mask, 'survival_months'] = (
                result_df.loc[living_mask, age_at_last_contact_col] - 
                result_df.loc[living_mask, age_at_diagnosis_col]
            ) * 12
        
        # Add event indicator (1 for death, 0 for censored)
        result_df['event'] = (result_df[vital_status_col] == 'Dead').astype(int)

        result_df.loc[result_df["survival_months"] <= 0, "survival_months"] = np.nan
        
        # Count patients with usable survival data
        survival_count = result_df['survival_months'].notna().sum()
        print(f"Calculated survival months for {survival_count} patients")
        
        return result_df
        
    except Exception as e:
        print(f"Error calculating survival months: {e}")
        print(traceback.format_exc())
        return df


def normalize_gene_expression(expr_data, method='log2'):
    """Normalize gene expression data"""
    try:
        # Make a copy to avoid modifying the original
        norm_data = expr_data.copy()
        
        if method == 'log2':
            # Add small value to avoid log(0)
            norm_data = np.log2(norm_data + 1)
            print(f"Applied log2(x+1) normalization to expression data")
        
        elif method == 'zscore':
            # Z-score normalization (gene-wise)
            norm_data = (norm_data - norm_data.mean(axis=1).values.reshape(-1, 1)) / norm_data.std(axis=1).values.reshape(-1, 1)
            print(f"Applied Z-score normalization to expression data")
        
        elif method == 'minmax':
            # Min-max normalization (gene-wise)
            min_vals = norm_data.min(axis=1).values.reshape(-1, 1)
            max_vals = norm_data.max(axis=1).values.reshape(-1, 1)
            norm_data = (norm_data - min_vals) / (max_vals - min_vals)
            print(f"Applied min-max normalization to expression data")
        
        else:
            print(f"Warning: Unknown normalization method '{method}'. Returning original data.")
        
        return norm_data
        
    except Exception as e:
        print(f"Error normalizing gene expression: {e}")
        print(traceback.format_exc())
        return expr_data


def save_results(df, output_dir, filename, index=True):
    """Save results to CSV file"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file path
        output_file = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(output_file, index=index)
        print(f"Saved results to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error saving results: {e}")
        print(traceback.format_exc())
        return False


def load_gene_signatures(signature_file):
    """Load gene signatures from file"""
    try:
        # Check if file exists
        if not os.path.exists(signature_file):
            print(f"Signature file not found: {signature_file}")
            return None
        
        # Load signatures
        signatures = {}
        
        with open(signature_file, 'r') as f:
            current_signature = None
            
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check if line is a signature name
                if line.startswith('>'):
                    current_signature = line[1:].strip()
                    signatures[current_signature] = []
                
                # Otherwise, add gene to current signature
                elif current_signature is not None:
                    signatures[current_signature].append(line)
        
        # Print summary
        print(f"Loaded {len(signatures)} gene signatures:")
        for sig_name, genes in signatures.items():
            print(f"- {sig_name}: {len(genes)} genes")
        
        return signatures
        
    except Exception as e:
        print(f"Error loading gene signatures: {e}")
        print(traceback.format_exc())
        return None


def save_plot(fig, filename, output_dir):
    """
    Save a matplotlib figure to the specified output directory
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Name of the file (without extension)
    output_dir : str
        Directory to save the plot
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the figure
        plot_path = os.path.join(output_dir, f"{filename}.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {plot_path}")
        
    except Exception as e:
        print(f"Error saving plot: {e}")
        print(traceback.format_exc())


def create_id_mapping(base_path):
    """
    Create mapping between lab IDs and ORIEN Avatar IDs
    
    Parameters:
    -----------
    base_path : str
        Base directory containing QC files
        
    Returns:
    --------
    dict
        Mapping from lab IDs to ORIEN Avatar IDs
    """
    try:
        qc_file = os.path.join(base_path, "Manifest_and_QC_Files/24PRJ217UVA_20250130_RNASeq_QCMetrics.csv")
        qc_data = pd.read_csv(qc_file)
        
        # Create mapping
        id_mapping = {}
        for _, row in qc_data.iterrows():
            lab_id = row['SLID'].replace('-RNA', '')
            orien_id = row['ORIENAvatarKey']
            id_mapping[lab_id] = orien_id
        
        print(f"Created ID mapping for {len(id_mapping)} samples")
        return id_mapping
        
    except Exception as e:
        print(f"Error creating ID mapping: {e}")
        print(traceback.format_exc())
        return {}


def clean_ID(ID: str) -> str:
    ID = str(ID)
    return ID.replace("-RNA", "").replace("FT-", "").replace("SA", "SL")


def map_sample_IDs_to_patient_IDs(data_frame_of_sample_IDs_CD8_signatures_and_scores: pd.DataFrame) -> pd.DataFrame:
    
    logger.info("Sample IDs will be mapped to patient IDs.")

    QC_data = pd.read_csv(paths.QC_data)
    dictionary_of_sample_IDs_and_patient_IDs = {
        clean_ID(row["SLID"]): row["ORIENAvatarKey"] for _, row in QC_data.iterrows()
    }
    index_of_sample_IDs = data_frame_of_sample_IDs_CD8_signatures_and_scores.index
    data_frame_of_sample_IDs_CD8_signatures_and_scores.index = data_frame_of_sample_IDs_CD8_signatures_and_scores.index.map(clean_ID)
    data_frame_of_sample_IDs_CD8_signatures_and_scores.index = data_frame_of_sample_IDs_CD8_signatures_and_scores.index.map(
        lambda sample_ID: dictionary_of_sample_IDs_and_patient_IDs.get(sample_ID, sample_ID)
    )
    return data_frame_of_sample_IDs_CD8_signatures_and_scores