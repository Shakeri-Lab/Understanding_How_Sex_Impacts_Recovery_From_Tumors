"""
Shared Functions Module
Common utility functions used across multiple modules
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import traceback
from datetime import datetime

def load_clinical_data(base_path):
    """Load clinical data from CSV"""
    try:
        # Define possible locations for clinical data
        '''
        possible_paths = [
            os.path.join(base_path, "processed_data/processed_clinical_molecular.csv"),
            os.path.join(base_path, "../processed_data/processed_clinical_molecular.csv"),
            os.path.join(base_path, "Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20240112_UnifiedClinicallData_V4.csv"),
            os.path.join(base_path, "../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20240112_UnifiedClinicallData_V4.csv")
        ]
        '''
        possible_paths = [ "/sfs/gpfs/tardis/project/orien/data/aws/24PRJ217UVA_IORIG/output/melanoma_clinical_data.csv" ]
        
        # Check each possible path
        clinical_file = None
        for path in possible_paths:
            if os.path.exists(path):
                clinical_file = path
                break
                
        # If not found in standard locations, search for it
        if clinical_file is None:
            print("\nClinical file not found in standard locations. Searching...")
            # First check the current base_path
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if "clinical" in file.lower() and file.endswith(".csv"):
                        clinical_file = os.path.join(root, file)
                        break
                if clinical_file:
                    break
            
            # If still not found, try one level up
            if clinical_file is None:
                parent_path = os.path.dirname(base_path)
                for root, dirs, files in os.walk(parent_path):
                    for file in files:
                        if "clinical" in file.lower() and file.endswith(".csv"):
                            clinical_file = os.path.join(root, file)
                            break
                    if clinical_file:
                        break
        
        # If still not found, return None
        if clinical_file is None:
            print("Clinical file not found.")
            return None
            
        print(f"\nLoading clinical data from {clinical_file}")
        clinical_data = pd.read_csv(clinical_file)
        print(f"Loaded clinical data for {len(clinical_data)} patients")
        print(f"Clinical data columns: {clinical_data.columns.tolist()}")
        
        return clinical_data
        
    except Exception as e:
        print(f"Error loading clinical data: {e}")
        print(traceback.format_exc())
        return None

def load_rnaseq_data(base_path):
    """
    Load and merge RNAseq expression data from genes.results files
    
    Parameters:
    -----------
    base_path : str
        Base path to the project directory
        
    Returns:
    --------
    pd.DataFrame
        Expression matrix (genes x samples) with TPM values
    """
    try:
        # Define possible locations for RNA-seq data
        possible_paths = [
            os.path.join(base_path, "RNAseq/gene_and_transcript_expression_results"),
            os.path.join(base_path, "../RNAseq/gene_and_transcript_expression_results"),
            os.path.join(base_path, "RNA_Seq_Data/gene_and_transcript_expression_results"),
            os.path.join(base_path, "../RNA_Seq_Data/gene_and_transcript_expression_results"),
            os.path.join(base_path, "RNA_Seq_Data")
        ]
        
        # Try each path
        expression_files = []
        rnaseq_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                rnaseq_path = Path(path)
                expression_files = list(rnaseq_path.glob("*.genes.results"))
                if expression_files:
                    print(f"Found RNA-seq data at {rnaseq_path}")
                    break
        
        # If not found in standard locations, search for it
        if not expression_files:
            print("\nRNA-seq files not found in standard locations. Searching...")
            # First check the current base_path
            for root, dirs, files in os.walk(base_path):
                results_files = [os.path.join(root, f) for f in files if f.endswith('.genes.results')]
                if results_files:
                    expression_files = [Path(f) for f in results_files]
                    rnaseq_path = Path(root)
                    print(f"Found RNA-seq data at {rnaseq_path}")
                    break
            
            # If still not found, try one level up
            if not expression_files:
                parent_path = os.path.dirname(base_path)
                for root, dirs, files in os.walk(parent_path):
                    results_files = [os.path.join(root, f) for f in files if f.endswith('.genes.results')]
                    if results_files:
                        expression_files = [Path(f) for f in results_files]
                        rnaseq_path = Path(root)
                        print(f"Found RNA-seq data at {rnaseq_path}")
                        break
        
        print(f"Found {len(expression_files)} expression files")
        
        if not expression_files:
            print("No RNA-seq expression files found.")
            return None
        
        # Create expression matrix
        expr_dfs = []
        for f in expression_files:
            sample_id = os.path.basename(str(f)).split('.')[0]
            df = pd.read_csv(f, sep='\t')
            expr_dfs.append(df[['gene_id', 'TPM']].set_index('gene_id')['TPM'].rename(sample_id))
        
        expr_df = pd.concat(expr_dfs, axis=1)
        
        # Quality control
        print(f"\nInitial expression matrix shape: {expr_df.shape}")
        
        # Filter lowly expressed genes
        min_samples = 0.2 * expr_df.shape[1]  # At least 20% of samples
        expr_df = expr_df.loc[(expr_df > 1).sum(axis=1) >= min_samples]
        
        print(f"Expression matrix after filtering: {expr_df.shape}")
        
        return expr_df  # Return genes x samples
        
    except Exception as e:
        print(f"Error loading RNAseq data: {e}")
        print(traceback.format_exc())
        return None

def filter_by_diagnosis(merged_data):
    """
    Filter out specific cancer types
    
    Parameters:
    -----------
    merged_data : pd.DataFrame
        Merged data with clinical information
        
    Returns:
    --------
    pd.DataFrame
        Filtered data
    """
    if 'DIAGNOSIS' not in merged_data.columns:
        print("Warning: Cannot filter by diagnosis - DIAGNOSIS column not found")
        return merged_data
    
    # Sites to exclude
    exclude_sites = ['Prostate gland', 'Vulva, NOS']
    
    # Count before filtering
    total_before = len(merged_data)
    
    # Filter out specified sites
    filtered_data = merged_data[~merged_data['DIAGNOSIS'].isin(exclude_sites)]
    
    # Count after filtering
    total_after = len(filtered_data)
    excluded_count = total_before - total_after
    
    print(f"\nFiltering out specific cancer types:")
    print(f"- Excluded: {exclude_sites}")
    print(f"- Removed {excluded_count} patients ({excluded_count/total_before:.1%} of cohort)")
    print(f"- Remaining: {total_after} patients")
    
    # Count by excluded site
    if excluded_count > 0:
        for site in exclude_sites:
            site_count = len(merged_data[merged_data['DIAGNOSIS'] == site])
            if site_count > 0:
                print(f"  - {site}: {site_count} patients")
    
    return filtered_data

# TODO: Integrate function `filter_by_primary_diagnosis_site` with function `filter_by_diagnosis`.
def filter_by_primary_diagnosis_site(merged_data):
    """
    Filter out specific cancer types
    
    Parameters:
    -----------
    merged_data : pd.DataFrame
        Merged data with clinical information
        
    Returns:
    --------
    pd.DataFrame
        Filtered data
    """
    if 'PrimaryDiagnosisSite' not in merged_data.columns:
        print("Warning: Cannot filter by diagnosis - PrimaryDiagnosisSite column not found")
        return merged_data
    
    # Sites to exclude
    exclude_sites = ['Prostate gland', 'Vulva, NOS']
    
    # Count before filtering
    total_before = len(merged_data)
    
    # Filter out specified sites
    filtered_data = merged_data[~merged_data['PrimaryDiagnosisSite'].isin(exclude_sites)]
    
    # Count after filtering
    total_after = len(filtered_data)
    excluded_count = total_before - total_after
    
    print(f"\nFiltering out specific cancer types:")
    print(f"- Excluded: {exclude_sites}")
    print(f"- Removed {excluded_count} patients ({excluded_count/total_before:.1%} of cohort)")
    print(f"- Remaining: {total_after} patients")
    
    # Count by excluded site
    if excluded_count > 0:
        for site in exclude_sites:
            site_count = len(merged_data[merged_data['PrimaryDiagnosisSite'] == site])
            if site_count > 0:
                print(f"  - {site}: {site_count} patients")
    
    return filtered_data

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
        
        # Count patients with survival data
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

def _clean_id(x: str) -> str:
    '''
    Remove run-specific affixes and standardize lab IDs.
    '''
    x = str(x)
    return x.replace("-RNA", '').replace("FT-", '').replace("SA", "SL")
    
def map_sample_ids(scores: pd.DataFrame, base_path: str) -> pd.DataFrame:
    """
    Map RNA-seq sample IDs to patient IDs
    
    Parameters:
    -----------
    scores : pd.DataFrame
        DataFrame with RNA-seq sample IDs as index
    base_path : str
        Base path to the project directory
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with patient IDs as index
    """
    try:
        print("\nMapping RNA-seq sample IDs to patient IDs...")
        
        # Load QC metrics file with correct name
        qc_file = os.path.join(base_path, "Manifest_and_QC_Files/24PRJ217UVA_20250130_RNASeq_QCMetrics.csv")
        qc_data = pd.read_csv(qc_file)
        
        # Print sample of QC data
        print("\nQC data sample:")
        print(qc_data[['SLID', 'ORIENAvatarKey']].head())
        
        # Create mapping dictionary
        id_map = {_clean_id(row["SLID"]): row["ORIENAvatarKey"] for _, row in qc_data.iterrows()}
        
        # Print sample of mapping
        print("\nSample ID mappings:")
        for k, v in list(id_map.items())[:5]:
            print(f"  {k} -> {v}")
        
        # Clean score IDs
        orig_ids = scores.index.copy()
        scores.index = scores.index.map(_clean_id)
        
        # Map to patient IDs
        scores.index = scores.index.map(lambda x: id_map.get(x, x))
        
        # Print mapping results
        print("\nID mapping results:")
        print("Original IDs:", orig_ids[:5].tolist())
        print("Mapped IDs:", scores.index[:5].tolist())
        print(f"Total unique patients: {len(scores.index.unique())}")
        
        # Check for unmapped IDs
        unmapped = [x for x in scores.index if x not in id_map.values()]
        if unmapped:
            print(f"\nWarning: {len(unmapped)} IDs could not be mapped:")
            print(unmapped[:5])
        
        # Try additional ID formats
        for orig_id in unmapped:
            # Try without -RNA
            clean_id = orig_id.replace('-RNA', '')
            if clean_id in id_map:
                scores.index = scores.index.map(lambda x: id_map.get(clean_id) if x == orig_id else x)
                unmapped.remove(orig_id)
        
        if unmapped:
            print("\nChecking QC file for unmapped IDs:")
            print(qc_data[qc_data['SLID'].str.contains('|'.join(unmapped), na=False)])
        
        return scores
        
    except Exception as e:
        print(f"Error mapping sample IDs: {e}")
        print(traceback.format_exc())
        return scores 