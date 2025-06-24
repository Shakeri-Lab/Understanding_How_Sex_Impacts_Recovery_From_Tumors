'''
Usage:
./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.microenv
'''

import pandas as pd
import numpy as np
import os
import glob
import logging
from datetime import datetime
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import traceback

# Import necessary function from data_loading
from src.immune_analysis.data_loading import load_rnaseq_data, identify_melanoma_samples, load_melanoma_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Activate converters for rpy2
numpy2ri.activate()
pandas2ri.activate()

# Define the standard xCell cell types/scores in their typical output order
# Corrected 67 items list including Keratinocytes and Hepatocytes (kept for reference)
# XCELL_CELL_TYPES_ORDERED = [
#     'Adipocytes', 'Astrocytes', 'B-cells', 'Basophils', 'CD4+ memory T-cells',
#     'CD4+ naive T-cells', 'CD4+ T-cells', 'CD4+ Tcm', 'CD4+ Tem', 'CD8+ naive T-cells',
#     'CD8+ T-cells', 'CD8+ Tcm', 'CD8+ Tem', 'Chondrocytes', 'Class-switched memory B-cells',
#     'CLP', 'CMP', 'cDC', 'DC', 'Endothelial cells', 'Eosinophils', 'Epithelial cells',
#     'Erythrocytes', 'Fibroblasts', 'GMP', 'HSC', 'iDC', 'Keratinocytes',  # Added
#     'ly Endothelial cells', 'Macrophages', 'Macrophages M1', 'Macrophages M2', 
#     'Mast cells', 'Megakaryocytes', 'Memory B-cells', 'MEP', 'Mesangial cells', 
#     'Monocytes', 'MPP', 'mv Endothelial cells', 'naive B-cells', 'Neutrophils', 
#     'NK cells', 'NKT', 'Osteoblast', 'pDC', 'Pericytes', 'Plasma cells', 'Platelets', 
#     'Preadipocytes', 'pro B-cells', 'Sebocytes', 'Skeletal muscle', 'Smooth muscle', 
#     'Tgd cells', 'Th1 cells', 'Th2 cells', 'Tregs', 'aDC', 'Neurons', 'Hepatocytes',  # Added
#     'MSC', 'common myeloid progenitor', 'melanocyte', 'ImmuneScore', 'StromaScore', 
#     'MicroenvironmentScore'
# ]  # Now 67 items

# Define the Focused Panel for ICB Response Analysis (14 items)
FOCUSED_XCELL_PANEL = [
    'CD8+ T-cells',
    'CD4+ memory T-cells', # Assuming this covers general helper/memory
    'Tgd cells', 
    'Macrophages M2',
    'Tregs',
    'cDC',
    'pDC',
    'Memory B-cells',
    'Plasma cells',
    'Endothelial cells', # For covariate analysis
    'Fibroblasts', # For covariate analysis
    'ImmuneScore',
    'StromaScore', # For covariate analysis
    'MicroenvironmentScore'
]

def process_melanoma_immune_data(base_path, output_dir=None):
    """
    Process melanoma RNA-seq data to extract immune microenvironment information.
    Enhanced to utilize SURGERYBIOPSY_V4 table to determine biopsy origins.
    
    Parameters:
    -----------
    base_path : str
        Base path to the project directory
    output_dir : str, optional
        Directory to save the output files, default is to create a directory in base_path
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with melanoma samples and immune features
    """
    try:
        # Set default output directory
        if output_dir is None:
            output_dir = os.path.join(base_path, "codes/output/melanoma_analysis")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        log_file = os.path.join(output_dir, 'microenv_processing.log')
        setup_logging(log_file)
        
        # Load clinical data and ID mapping
        map_file_path = os.path.join(output_dir, "sample_to_patient_map.csv")
        clinical_file_path = os.path.join(output_dir, "melanoma_patients_with_sequencing.csv")
        
        # Check if needed files exist
        for file_path in [map_file_path, clinical_file_path]:
            if not os.path.exists(file_path):
                logger.error(f"Required file not found: {file_path}")
                return None
        
        # Load RNA-seq data for melanoma samples using the enhanced identify_melanoma_samples function
        # which now returns sample details alongside the list of SLIDs
        immune_df, clinical_data = load_melanoma_data(base_path, map_file_path)
        if immune_df is None or clinical_data is None:
            logger.error("Failed to load immune or clinical data.")
            return None
        
        # Get the sample details that were returned by identify_melanoma_samples
        try:
            logger.info("Attempting to access enhanced sample details")
            from src.immune_analysis.data_loading import identify_melanoma_samples
            melanoma_slids, sample_details = identify_melanoma_samples(base_path, clinical_data)
            logger.info(f"Successfully retrieved enhanced sample details for {len(sample_details)} samples")
        except Exception as e:
            logger.error(f"Failed to retrieve enhanced sample details: {e}")
            sample_details = {}
            
        # Get the columns in the immune dataframe
        logger.info(f"Immune dataframe shape: {immune_df.shape}")
        logger.info(f"Immune dataframe columns: {immune_df.columns[:5]}...")
        
        # Add patient IDs and clinical information to immune data
        sample_to_patient = {}
        for slid, details in sample_details.items():
            if 'patient_id' in details:
                sample_to_patient[slid] = details['patient_id']
        
        # If the sample_details dictionary is empty or doesn't have patient_id mappings,
        # use the old approach to get the mappings
        if not sample_to_patient:
            logger.warning("No sample details found with patient mappings. Using backup approach.")
            try:
                map_df = pd.read_csv(map_file_path)
                for _, row in map_df.iterrows():
                    sample_to_patient[row['SampleID']] = row['PatientID']
            except Exception as e:
                logger.error(f"Error loading map file: {e}")
                return None
        
        # Add PATIENT_ID to the immune data
        immune_df['PATIENT_ID'] = immune_df.index.map(lambda x: sample_to_patient.get(x, None))
        
        # Drop samples without a PATIENT_ID mapping
        missing_mapping = immune_df['PATIENT_ID'].isna().sum()
        if missing_mapping > 0:
            logger.warning(f"Dropping {missing_mapping} samples without patient ID mapping")
            immune_df = immune_df.dropna(subset=['PATIENT_ID'])
        
        # Merge clinical info with immune data
        clinical_cols = ['PATIENT_ID', 'Sex', 'Race', 'AgeAtClinicalRecordCreation', 
                        'EarliestMelanomaDiagnosisAge', 'HAS_ICB', 'ICB_START_AGE', 'STAGE_AT_ICB']
        available_cols = [col for col in clinical_cols if col in clinical_data.columns]
        if len(available_cols) < len(clinical_cols):
            missing_cols = set(clinical_cols) - set(available_cols)
            logger.warning(f"Missing clinical columns: {missing_cols}")
        
        # Merge with clinical data
        if available_cols:
            try:
                immune_clinical = pd.merge(immune_df, clinical_data[available_cols], on='PATIENT_ID', how='left')
                logger.info(f"Merged dataframe shape: {immune_clinical.shape}")
            except Exception as e:
                logger.error(f"Error merging clinical data: {e}")
                immune_clinical = immune_df.copy()
        else:
            logger.warning("No clinical columns available for merging")
            immune_clinical = immune_df.copy()
        
        # Add biopsy information from sample_details dictionary
        logger.info("Adding biopsy information from SURGERYBIOPSY_V4 to the analysis")
        # New columns for biopsy information
        immune_clinical['SpecimenSite'] = immune_clinical.index.map(
            lambda x: sample_details.get(x, {}).get('specimen_site', None))
        immune_clinical['ProcedureType'] = immune_clinical.index.map(
            lambda x: sample_details.get(x, {}).get('procedure_type', None))
        immune_clinical['IsConfirmedMelanoma'] = immune_clinical.index.map(
            lambda x: sample_details.get(x, {}).get('is_confirmed_melanoma', None))
        immune_clinical['HistologyCode'] = immune_clinical.index.map(
            lambda x: sample_details.get(x, {}).get('histology_code', None))
        
        # Add metastatic status based on specimen site information
        # Define keywords that might indicate metastatic sites
        metastatic_keywords = ['metast', 'lymph node', 'brain', 'lung', 'liver', 'distant']
        
        def determine_metastatic_status(site):
            """Determine if a specimen site indicates metastatic disease."""
            if pd.isna(site) or site is None:
                return None
            site_lower = str(site).lower()
            if any(keyword in site_lower for keyword in metastatic_keywords):
                return True
            # List of common primary melanoma sites
            primary_sites = ['skin', 'cutaneous', 'dermal', 'epidermis', 'primary']
            if any(keyword in site_lower for keyword in primary_sites):
                return False
            return None  # Unknown/uncertain
        
        # Add metastatic status column
        immune_clinical['IsMetastatic'] = immune_clinical['SpecimenSite'].apply(determine_metastatic_status)
        logger.info(f"Metastatic status determined for {immune_clinical['IsMetastatic'].count()} samples")
        logger.info(f"Metastatic samples: {(immune_clinical['IsMetastatic'] == True).sum()}")
        logger.info(f"Primary samples: {(immune_clinical['IsMetastatic'] == False).sum()}")
        
        # Save processed data
        output_file = os.path.join(output_dir, "melanoma_sample_immune_clinical.csv")
        immune_clinical.to_csv(output_file)
        logger.info(f"Saved processed data to {output_file}")
        
        # Create a summary of biopsy origins
        site_summary = immune_clinical['SpecimenSite'].value_counts().reset_index()
        site_summary.columns = ['SpecimenSite', 'Count']
        site_summary_file = os.path.join(output_dir, "specimen_site_summary.csv")
        site_summary.to_csv(site_summary_file, index=False)
        logger.info(f"Saved specimen site summary to {site_summary_file}")
        
        # Create a summary of procedure types
        proc_summary = immune_clinical['ProcedureType'].value_counts().reset_index()
        proc_summary.columns = ['ProcedureType', 'Count']
        proc_summary_file = os.path.join(output_dir, "procedure_type_summary.csv")
        proc_summary.to_csv(proc_summary_file, index=False)
        logger.info(f"Saved procedure type summary to {proc_summary_file}")
        
        # Create a summary of metastatic status
        meta_summary = immune_clinical['IsMetastatic'].value_counts().reset_index()
        meta_summary.columns = ['IsMetastatic', 'Count']
        meta_summary_file = os.path.join(output_dir, "metastatic_status_summary.csv")
        meta_summary.to_csv(meta_summary_file, index=False)
        logger.info(f"Saved metastatic status summary to {meta_summary_file}")
        
        return immune_clinical
        
    except Exception as e:
        logger.error(f"Error processing melanoma immune data: {e}", exc_info=True)
        return None

def main():
    """Main execution function"""
    base_path = "/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors"
    # Use the output file from data_loading.py as the input clinical data
    processed_clinical_file = os.path.join(base_path, "output/eda", "melanoma_patients_with_sequencing.csv")

    if not os.path.exists(processed_clinical_file):
        logger.error(f"Input clinical file not found: {processed_clinical_file}. Run data_loading.py first.")
        return

    clinical_data_processed = pd.read_csv(processed_clinical_file)
    logger.info(f"Loaded clinical data for {clinical_data_processed['PATIENT_ID'].nunique()} unique melanoma patients from {processed_clinical_file}.") # Log unique patients

    expr_matrix = load_rnaseq_data(base_path)
    if expr_matrix is None:
        logger.error("Failed to load RNA-seq data.")
        return

    # Run the analysis (which now includes R execution and file writing)
    success = run_xcell_analysis(expr_matrix, clinical_data_processed, base_path)

    if success:
        logger.info("microenv.py script completed successfully (R finished execution).")
    else:
        logger.error("microenv.py script failed during processing.")

if __name__ == "__main__":
    main()