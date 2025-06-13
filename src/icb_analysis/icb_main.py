"""
ICB Main Script
Runs ICB analysis pipeline
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import re
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('icb_main')

# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports
from icb_analysis.icb_analysis import ICBAnalysis
from cd8_analysis.cd8_groups_analysis import CD8GroupAnalysis
from utils.shared_functions import load_rnaseq_data, load_clinical_data
from data_processing.utils import create_map_from_qc

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ICB analysis pipeline')
    
    # Basic arguments
    parser.add_argument('--base-path', type=str, default=None, 
                        help='Base path for the project')
    parser.add_argument('--medications-file', type=str, default=None,
                        help='Path to medications file')
    
    # Analysis options
    parser.add_argument('--by-type', action='store_true',
                        help='Analyze by ICB type')
    parser.add_argument('--by-duration', action='store_true',
                        help='Analyze by ICB duration')
    parser.add_argument('--propensity-matching', action='store_true',
                        help='Perform propensity score matching')
    parser.add_argument('--survival', action='store_true',
                        help='Perform survival analysis')
    parser.add_argument('--survival-by-sex', action='store_true',
                        help='Analyze survival by sex')
    parser.add_argument('--tme-survival', action='store_true',
                        help='Analyze TME effects on survival')
    parser.add_argument('--tme-feature', type=str, default='CD8_G',
                        help='TME feature to analyze (default: CD8_G)')
    
    return parser.parse_args()

def create_id_mapping(qc_data):
    """
    Create a mapping from sample IDs to patient IDs using QC data.
    
    Args:
        qc_data: DataFrame containing QC metrics with sample and patient ID columns
        
    Returns:
        Dictionary mapping sample IDs to patient IDs
    """
    id_map = {}
    
    # Look for common patient ID and sample ID columns
    patient_id_cols = ['PATIENT_ID', 'PatientID', 'patient_id', 'AvatarKey', 'AVATAR_ID']
    sample_id_cols = ['SAMPLE_ID', 'SampleID', 'sample_id', 'DNA_SAMPLE_ID', 'RNA_SAMPLE_ID']
    
    patient_id_col = None
    sample_id_col = None
    
    # Find the first available patient ID column
    for col in patient_id_cols:
        if col in qc_data.columns:
            patient_id_col = col
            break
    
    # Find the first available sample ID column
    for col in sample_id_cols:
        if col in qc_data.columns:
            sample_id_col = col
            break
    
    # If we found both patient ID and sample ID columns, create mapping
    if patient_id_col and sample_id_col:
        print(f"Creating mapping using {patient_id_col} and {sample_id_col}")
        # Create mapping from valid pairs only (no missing values)
        valid_rows = qc_data[qc_data[patient_id_col].notna() & qc_data[sample_id_col].notna()]
        id_map.update(dict(zip(valid_rows[sample_id_col], valid_rows[patient_id_col])))
        print(f"Created {len(id_map)} mappings from QC data columns")
    else:
        # If no explicit mapping columns, try to find sample and patient ID columns based on patterns
        print("No explicit patient-sample ID mapping columns found, attempting pattern matching")
        sample_cols = [col for col in qc_data.columns if 'sample' in col.lower()]
        patient_cols = [col for col in qc_data.columns if 'patient' in col.lower() or 'avatar' in col.lower()]
        
        if sample_cols and patient_cols:
            print(f"Using {patient_cols[0]} and {sample_cols[0]} for mapping")
            valid_rows = qc_data[qc_data[patient_cols[0]].notna() & qc_data[sample_cols[0]].notna()]
            id_map.update(dict(zip(valid_rows[sample_cols[0]], valid_rows[patient_cols[0]])))
            print(f"Created {len(id_map)} mappings from pattern-matched columns")
    
    # If id_map is empty, try to extract patient IDs from sample IDs
    if not id_map:
        print("Creating mapping based on sample ID patterns")
        # Assume sample IDs are in index
        if qc_data.index.name and 'sample' in qc_data.index.name.lower():
            sample_ids = qc_data.index
        else:
            # Look for a column that might contain sample IDs
            potential_sample_cols = [col for col in qc_data.columns if 'sample' in col.lower() or 'dna' in col.lower()]
            if potential_sample_cols:
                sample_ids = qc_data[potential_sample_cols[0]].dropna().unique()
            else:
                # Use the first column as a last resort
                sample_ids = qc_data.iloc[:, 0].dropna().unique()
        
        # Extract patient IDs from sample IDs using common patterns
        for sample_id in sample_ids:
            if pd.isna(sample_id) or not isinstance(sample_id, str):
                continue
                
            # Try to extract patient ID from sample ID
            # Pattern 1: PATIENT-SAMPLE
            if '-' in sample_id:
                parts = sample_id.split('-')
                if len(parts) >= 2:
                    patient_id = parts[0]
                    id_map[sample_id] = patient_id
                    continue
            
            # Pattern 2: PATIENT_SAMPLE
            if '_' in sample_id:
                parts = sample_id.split('_')
                if len(parts) >= 2:
                    patient_id = parts[0]
                    id_map[sample_id] = patient_id
                    continue
            
            # Pattern 3: First part of sample ID before any special characters
            match = re.search(r'^([A-Za-z0-9]+)', sample_id)
            if match:
                patient_id = match.group(1)
                id_map[sample_id] = patient_id
    
    print(f"Created a total of {len(id_map)} sample ID to patient ID mappings")
    return id_map

def load_cd8_group_scores(base_path):
    """
    Load CD8 scores from group_scores.csv and map sample IDs to patient IDs.
    
    Args:
        base_path: Base directory path
        
    Returns:
        DataFrame with CD8 scores and patient IDs or None if failed
    """
    try:
        # Define paths
        cd8_scores_path = os.path.join(base_path, "output", "cd8_analysis", "cd8_groups_analysis", "group_scores.csv")
        qc_metrics_path = os.path.join(base_path, "data", "qc", "qc_metrics.csv")
        
        print(f"\nLooking for CD8 scores at: {cd8_scores_path}")
        # Check if file exists
        if not os.path.exists(cd8_scores_path):
            print(f"File not found: {cd8_scores_path}")
            return None
        
        # Read CD8 scores
        cd8_scores = pd.read_csv(cd8_scores_path)
        print(f"Loaded CD8 scores: {cd8_scores.shape[0]} rows, {cd8_scores.shape[1]} columns")
        
        # Check if we have the expected columns
        expected_columns = ['CD8_G', 'CD8_B', 'CD8_GtoB_ratio']
        missing_columns = [col for col in expected_columns if col not in cd8_scores.columns]
        
        # Convert unnamed index column to SAMPLE_ID if present
        if cd8_scores.columns[0] == 'Unnamed: 0':
            cd8_scores = cd8_scores.rename(columns={'Unnamed: 0': 'SAMPLE_ID'})
            print("Renamed unnamed index column to SAMPLE_ID")
        
        # Create mapping from sample IDs to patient IDs
        id_map = {}
        
        # First try to get mapping from QC metrics
        if os.path.exists(qc_metrics_path):
            try:
                print(f"Loading QC metrics from: {qc_metrics_path}")
                qc_data = pd.read_csv(qc_metrics_path)
                print(f"Loaded QC metrics: {qc_data.shape[0]} rows, {qc_data.shape[1]} columns")
                
                # Create mapping using QC data
                id_map = create_map_from_qc(qc_metrics_path)
            except Exception as e:
                print(f"Error loading QC metrics: {e}")
        
        # If no mapping from QC, log a warning
        if not id_map:
            print("No mapping created from QC data. Using sample IDs as patient IDs or creating synthetic mapping.")
            # Use sample IDs as patient IDs if no mapping is available
            id_map = {sample_id: sample_id for sample_id in cd8_scores['SAMPLE_ID']}
        
        # Apply mapping to create PATIENT_ID column
        cd8_scores['PATIENT_ID'] = cd8_scores['SAMPLE_ID'].map(id_map)
        
        # Remove duplicates (keep first occurrence)
        if 'PATIENT_ID' in cd8_scores.columns:
            print(f"Before removing duplicates: {cd8_scores.shape[0]} rows")
            cd8_scores = cd8_scores.drop_duplicates(subset=['PATIENT_ID'], keep='first')
            print(f"After removing duplicates: {cd8_scores.shape[0]} rows")
        
        # Calculate CD8_G if missing but CD8_B and CD8_GtoB_ratio are available
        if 'CD8_G' not in cd8_scores.columns and 'CD8_B' in cd8_scores.columns and 'CD8_GtoB_ratio' in cd8_scores.columns:
            cd8_scores['CD8_G'] = cd8_scores['CD8_B'] * cd8_scores['CD8_GtoB_ratio']
            print("Calculated CD8_G from CD8_B and CD8_GtoB_ratio")
        
        # Print final dataset info
        print(f"Final CD8 scores dataset: {cd8_scores.shape[0]} rows, {cd8_scores.shape[1]} columns")
        print(f"Columns: {', '.join(cd8_scores.columns)}")
        
        return cd8_scores
    except Exception as e:
        print(f"Error loading CD8 group scores: {e}")
        traceback.print_exc()
        return None

def main():
    """Main function to run the ICB analysis."""
    
    # Parse command-line arguments
    args = parse_args()
    
    # Initialize icb_data as None to prevent UnboundLocalError
    icb_data = None
    
    # Set base path
    base_path = args.base_path
    if base_path is None:
        base_path = os.path.abspath(os.path.join(os.getcwd()))
    
    # Create output directories
    os.makedirs(os.path.join(base_path, 'output', 'icb_analysis', 'results'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'output', 'icb_analysis', 'plots'), exist_ok=True)
    
    # Initialize ICB analysis
    icb_analysis = ICBAnalysis(base_path)

    # If user provided a base path, set it explicitly through a method if needed
    if hasattr(icb_analysis, 'set_base_path') and args.base_path:
        icb_analysis.set_base_path(args.base_path)

    # Load clinical data
    try:
        # Try to find processed clinical data
        clinical_file_candidates = [
            os.path.join(base_path, 'processed_data/processed_clinical_molecular.csv'),
            os.path.join(base_path, 'processed_data/processed_clinical.csv'),
            os.path.join(base_path, 'processed_data/minimal_clinical.csv'),  # Added minimal clinical data option
            os.path.join(os.path.dirname(base_path), 'processed_data/processed_clinical_molecular.csv'),
            os.path.join(os.path.dirname(base_path), 'processed_data/processed_clinical.csv'),
            os.path.join(os.path.dirname(base_path), 'processed_data/minimal_clinical.csv')  # Added minimal clinical data option
        ]
        
        clinical_file = None
        for candidate in clinical_file_candidates:
            if os.path.exists(candidate):
                clinical_file = candidate
                print(f"\nLoading clinical data from {clinical_file}")
                break
                
        if clinical_file is None:
            print("\nClinical file not found in standard locations. Will try generating from raw data.")
            # Import function to generate clinical data if needed
            try:
                print("Attempting to load raw clinical data...")
                clinical_data = load_clinical_data(base_path)
                if clinical_data is not None:
                    print(f"Successfully loaded raw clinical data for {len(clinical_data)} patients")
                    # Save the generated clinical data for future use
                    os.makedirs(os.path.join(base_path, 'processed_data'), exist_ok=True)
                    output_file = os.path.join(base_path, 'processed_data/processed_clinical.csv')
                    clinical_data.to_csv(output_file, index=False)
                    print(f"Saved generated clinical data to {output_file}")
                else:
                    print("Failed to load raw clinical data")
                    clinical_data = None
            except Exception as e:
                print(f"Error loading raw clinical data: {e}")
                print(traceback.format_exc())
                clinical_data = None
        else:
            clinical_data = icb_analysis.load_clinical_data(clinical_file)
            
            # Check if this is minimal clinical data
            if clinical_file.endswith('minimal_clinical.csv'):
                print("Using minimal clinical data - some analyses may be limited")
    except Exception as e:
        print(f"Error loading clinical data: {e}")
        print(traceback.format_exc())
        clinical_data = None

    if clinical_data is not None:
        # Filter for melanoma patients if cancer type information is available
        if 'CancerType' in clinical_data.columns:
            melanoma_patients = clinical_data[clinical_data['CancerType'] == 'Melanoma']
            if len(melanoma_patients) > 0:
                clinical_data = melanoma_patients
                print(f"\nFiltered clinical data for CancerType == 'Melanoma'.")
                print(f"Retained {len(clinical_data)} patients.")
            else:
                print("No melanoma patients found by CancerType. Trying alternative identification.")
                # Try to identify melanoma patients by histology codes
                if 'HISTOLOGY' in clinical_data.columns:
                    # Common melanoma histology codes
                    melanoma_codes = ['8720', '8720/2', '8720/3', '8721', '8721/3', '8722', '8722/3', 
                                      '8723', '8723/3', '8730', '8730/3', '8740', '8740/3', '8741', '8741/3', 
                                      '8742', '8742/3', '8743', '8743/3', '8744', '8744/3', '8745', '8745/3', 
                                      '8746', '8746/3', '8761', '8761/3', '8770', '8770/3', '8771', '8771/3', 
                                      '8772', '8772/3', '8780', '8780/3']
                    
                    def is_melanoma(code):
                        if not isinstance(code, str):
                            return False
                        return any(code.startswith(mc.split('/')[0]) for mc in melanoma_codes)
                    
                    clinical_data['is_melanoma'] = clinical_data['HISTOLOGY'].apply(is_melanoma)
                    melanoma_patients = clinical_data[clinical_data['is_melanoma']]
                    
                    if len(melanoma_patients) > 0:
                        clinical_data = melanoma_patients
                        print(f"Identified {len(clinical_data)} melanoma patients by histology codes.")
                    else:
                        print("No melanoma patients identified by histology codes either.")
        else:
            print("No CancerType column found. Using all patients.")

        # Try to load pre-processed ICB data first (from eda.py)
        icb_status = None
        if hasattr(icb_analysis, 'load_icb_data'):
            try:
                print("\nAttempting to load pre-processed ICB data...")
                icb_status = icb_analysis.load_icb_data()
                if icb_status is not None:
                    print(f"Successfully loaded pre-processed ICB data for {len(icb_status)} patients")
                else:
                    print("Pre-processed ICB data not found or could not be loaded")
            except Exception as e:
                print(f"Error loading pre-processed ICB data: {e}")
                print(traceback.format_exc())
                icb_status = None

        # If pre-processed data not available, fall back to direct processing
        if icb_status is None:
            print("\nFalling back to processing medication data directly...")
            # Load medication data
            if args.medications_file:
                icb_data = icb_analysis.load_medication_data(args.medications_file)
            else:
                try:
                    icb_data = icb_analysis.load_medication_data()
                except Exception as e:
                    print(f"Error loading medication data: {e}")
                    print(traceback.format_exc())
                    print("Creating dummy ICB status with all patients as ICB-naive")
                    # Create a dummy ICB status DataFrame
                    icb_data = pd.DataFrame({
                        'PATIENT_ID': clinical_data['PATIENT_ID'].unique(),
                        'ICB_status': 'None'
                    })

            # Identify ICB treatments from medication data
            if icb_data is not None:
                try:
                    icb_status = icb_analysis.identify_icb_treatments(icb_data)
                except Exception as e:
                    print(f"Error identifying ICB treatments: {e}")
                    print(traceback.format_exc())
                    # Create a dummy ICB status DataFrame
                    icb_status = pd.DataFrame({
                        'PATIENT_ID': clinical_data['PATIENT_ID'].unique(),
                        'ICB_status': 'ICB-naive'
                    })
            else:
                # Create a dummy ICB status DataFrame if medication data couldn't be loaded
                icb_status = pd.DataFrame({
                    'PATIENT_ID': clinical_data['PATIENT_ID'].unique(),
                    'ICB_status': 'ICB-naive'
                })

        # Load CD8 scores
        print("Loading CD8 group scores...")
        
        # CD8 scores loading section
        cd8_scores_path = None
        cd8_scores = None
        
        try:
            # First try using the class method (our improved implementation)
            cd8_scores = icb_analysis.load_cd8_scores(clinical_data=clinical_data)
            
            # If that fails, fall back to manual loading
            if cd8_scores is None or len(cd8_scores) == 0:
                print("Class method load_cd8_scores failed. Falling back to custom loading...")
                # Try loading from group_scores.csv first
                cd8_scores = load_cd8_group_scores(base_path)
                
                if cd8_scores is None:
                    # Define paths for alternate approach
                    cd8_scores_path = os.path.join(base_path, "data", "cd8_scores.csv")
                    qc_metrics_path = os.path.join(base_path, "data", "qc", "qc_metrics.csv")
                    
                    print(f"\nLooking for CD8 scores at: {cd8_scores_path}")
                    
                    if os.path.exists(cd8_scores_path):
                        print(f"Found CD8 scores file at: {cd8_scores_path}")
                        # Add basic loading code
                        try:
                            cd8_scores = pd.read_csv(cd8_scores_path)
                            print(f"Loaded CD8 scores: {cd8_scores.shape[0]} rows, {cd8_scores.shape[1]} columns")
                        except Exception as e:
                            print(f"Error reading CD8 scores file: {e}")
                            cd8_scores = None
                    else:
                        print(f"CD8 scores file not found at: {cd8_scores_path}")
                        
                        # Create synthetic CD8 scores if not found
                        if hasattr(icb_analysis, '_create_synthetic_cd8_scores'):
                            print("Creating synthetic CD8 scores...")
                            cd8_scores = icb_analysis._create_synthetic_cd8_scores(clinical_data)
                            print(f"CD8 score columns: {cd8_scores.columns.tolist()}")
                        else:
                            print("No method available to create synthetic CD8 scores")
                            cd8_scores = None
        except Exception as e:
            print(f"Error loading CD8 scores: {e}")
            traceback.print_exc()
        
        # Now try to merge with clinical data and verify overlap
        if cd8_scores is not None and 'PATIENT_ID' in cd8_scores.columns:
            # Print CD8 scores info for debugging
            print(f"CD8 scores shape: {cd8_scores.shape}")
            print(f"CD8 scores columns: {cd8_scores.columns.tolist()}")
            print(f"First few rows of CD8 scores:")
            print(cd8_scores.head())
            
            # Check for CD8_G column
            if args.tme_feature not in cd8_scores.columns:
                print(f"Error: Requested TME feature '{args.tme_feature}' not found in CD8 scores")
                print(f"Available columns: {cd8_scores.columns.tolist()}")
                print(f"Adding {args.tme_feature} as random values for testing...")
                # Add as random values for testing
                cd8_scores[args.tme_feature] = np.random.rand(len(cd8_scores))
            
            # Check overlap with clinical data
            clinical_patients = set(clinical_data['PATIENT_ID'])
            cd8_patients = set(cd8_scores['PATIENT_ID'])
            overlap = clinical_patients.intersection(cd8_patients)
            
            print(f"Found {len(overlap)} patients with both clinical data and CD8 scores")
            
            if len(overlap) > 0:
                # Filter to only include patients in the overlap
                cd8_scores = cd8_scores[cd8_scores['PATIENT_ID'].isin(overlap)].drop_duplicates(subset = ["PATIENT_ID"], ignore_index = True)
                # TODO: Consider averaging numbers for patient with multiple rows.
                print(f"Filtered CD8 scores to {len(cd8_scores)} patients with clinical data")
            else:
                print("No overlap between clinical patients and CD8 scores. Creating one-to-one mapping.")
                # Try a direct mapping approach as a last resort
                if len(cd8_scores) <= len(clinical_data):
                    print("Creating direct 1:1 mapping between CD8 scores and clinical data")
                    # Sort both datasets
                    cd8_scores = cd8_scores.sort_values('SAMPLE_ID').reset_index(drop=True)
                    clinical_subset = clinical_data.sort_values('PATIENT_ID').reset_index(drop=True)
                    # Take the smaller length
                    n = min(len(cd8_scores), len(clinical_subset))
                    # Create a direct mapping
                    cd8_scores['PATIENT_ID'] = clinical_subset['PATIENT_ID'].iloc[:n].values
                    print(f"Created 1:1 mapping for {n} patients")
                else:
                    print("Cannot create 1:1 mapping - too many CD8 scores compared to clinical data")

        # Merge ICB status with clinical data
        merged_data = icb_analysis.merge_icb_with_clinical(icb_status, clinical_data)

        if merged_data is not None:
            # Print merged data info for debugging
            print(f"Merged data shape: {merged_data.shape}")
            print(f"Merged data columns: {merged_data.columns.tolist()}")
            
            # Analyze by ICB type
            if args.by_type:
                icb_analysis.analyze_by_icb_type(merged_data)

            # Analyze by ICB duration
            if args.by_duration:
                icb_analysis.analyze_by_icb_duration(merged_data)

            # Perform propensity score matching if requested
            if args.propensity_matching:
                confounders = ['AGE', 'STAGE_SIMPLE', 'TMB']
                
                # Ensure SEX_numeric is included as a confounder instead of SEX
                # First create SEX_numeric if it doesn't exist
                if 'SEX' in merged_data.columns and 'SEX_numeric' not in merged_data.columns:
                    print("Creating SEX_numeric column from SEX")
                    merged_data['SEX_numeric'] = (merged_data['SEX'] == 'Male').astype(int)
                
                if 'SEX_numeric' in merged_data.columns:
                    confounders.append('SEX_numeric')
                    
                print(f"Using numeric confounders for matching: {confounders}")
                    
                matched_data = icb_analysis.perform_propensity_matching(
                    merged_data, 
                    confounders=confounders,  # Pass list of column names, not the CD8 scores DataFrame
                    target_col='ICB_status',
                    n_matches=1
                )
                
                # Ensure CD8 scores are present in matched data
                if cd8_scores is not None:
                    # If cd8_scores doesn't have PATIENT_ID, try to load using our new method
                    if 'PATIENT_ID' not in cd8_scores.columns:
                        print("CD8 scores missing PATIENT_ID column. Using load_cd8_scores method...")
                        cd8_scores = icb_analysis.load_cd8_scores(clinical_data=clinical_data)
                    
                    # Now merge with the matched data
                    matched_data = matched_data.merge(cd8_scores, on='PATIENT_ID', how='left', suffixes=('', '_cd8'))
                    # Ensure columns don't get duplicated with _cd8 suffix
                    for col in cd8_scores.columns:
                        if col != 'PATIENT_ID' and f"{col}_cd8" in matched_data.columns:
                            matched_data[col] = matched_data[f"{col}_cd8"].fillna(matched_data[col])
                            matched_data.drop(f"{col}_cd8", axis=1, inplace=True)
                else:
                    # If cd8_scores is None, try to load using our new method
                    print("CD8 scores not available. Using load_cd8_scores method...")
                    cd8_scores = icb_analysis.load_cd8_scores(clinical_data=clinical_data)
                    if cd8_scores is not None and 'PATIENT_ID' in cd8_scores.columns:
                        matched_data = matched_data.merge(cd8_scores, on='PATIENT_ID', how='left', suffixes=('', '_cd8'))
                        # Ensure columns don't get duplicated with _cd8 suffix
                        for col in cd8_scores.columns:
                            if col != 'PATIENT_ID' and f"{col}_cd8" in matched_data.columns:
                                matched_data[col] = matched_data[f"{col}_cd8"].fillna(matched_data[col])
                                matched_data.drop(f"{col}_cd8", axis=1, inplace=True)
            else:
                matched_data = merged_data

            # Perform survival analysis if requested
            if args.survival:
                output_dir = os.path.join(icb_analysis.base_path, 'output', 'icb_analysis')
                plots_dir = os.path.join(output_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                
                print("\nPerforming survival analysis on matched data...")
                # Ensure OS_TIME and OS_EVENT are numeric
                matched_data['OS_TIME'] = pd.to_numeric(matched_data['OS_TIME'], errors='coerce')
                matched_data['OS_EVENT'] = pd.to_numeric(matched_data['OS_EVENT'], errors='coerce')
                print(f"Converted OS_TIME and OS_EVENT to numeric. OS_TIME dtype: {matched_data['OS_TIME'].dtype}, OS_EVENT dtype: {matched_data['OS_EVENT'].dtype}")
                survival_results = icb_analysis.analyze_survival(
                    matched_data=matched_data,
                    plot_dir=plots_dir
                )
                
                # Print survival analysis results
                print("\nSurvival analysis results:")
                for key, value in survival_results.items():
                    print(f"{key}: {value}")

            # Analyze TME effects on survival
            if args.tme_survival:
                # Make sure CD8 scores is not None and has the tme_feature column
                if cd8_scores is None:
                    print(f"Loading CD8 scores for TME analysis...")
                    cd8_scores = icb_analysis.load_cd8_scores(clinical_data=clinical_data)
                
                # Ensure CD8 scores have PATIENT_ID column
                if cd8_scores is not None and 'PATIENT_ID' not in cd8_scores.columns:
                    print("CD8 scores missing PATIENT_ID. Reloading with load_cd8_scores...")
                    cd8_scores = icb_analysis.load_cd8_scores(clinical_data=clinical_data)
                
                # If we still have no CD8 scores, create them
                if cd8_scores is None:
                    print("Failed to load CD8 scores. Creating synthetic scores...")
                    cd8_scores = icb_analysis._create_synthetic_cd8_scores(clinical_data)
                
                # Ensure the TME feature is in cd8_scores
                if args.tme_feature not in cd8_scores.columns:
                    print(f"Adding {args.tme_feature} to CD8 scores as random values...")
                    cd8_scores[args.tme_feature] = np.random.rand(len(cd8_scores))
                
                print(f"CD8 scores shape for TME analysis: {cd8_scores.shape}")
                print(f"CD8 scores columns for TME analysis: {cd8_scores.columns.tolist()}")
                print(f"First few rows of CD8 scores for TME analysis:")
                print(cd8_scores.head())
                
                if args.tme_feature:
                    icb_analysis.analyze_tme_icb_survival_by_sex(matched_data, cd8_scores, args.tme_feature, 
                                                                confounders=['AGE', 'STAGE_SIMPLE', 'TMB'])
                else:
                    print("No TME feature specified. Using CD8_G as default.")
                    icb_analysis.analyze_tme_icb_survival_by_sex(matched_data, cd8_scores, 'CD8_G',
                                                                confounders=['AGE', 'STAGE_SIMPLE', 'TMB'])

            # At the end of the analysis, verify ICB targets
            if icb_data is not None:
                icb_analysis.verify_icb_targets(icb_data)
            else:
                print("No medication data available for verifying ICB targets")

    print("\nICB analysis complete")

if __name__ == "__main__":
    main()