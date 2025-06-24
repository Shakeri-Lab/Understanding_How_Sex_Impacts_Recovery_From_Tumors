import os
import pandas as pd
import logging
import glob  # Added to resolve NameError

# Configure logging to help debug
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_clinical_data(base_path):
    try:
        # Define file paths
        norm_files_dir = os.path.join(base_path, "Clinical_Data/24PRJ217UVA_NormalizedFiles")
        diag_file = os.path.join(norm_files_dir, "24PRJ217UVA_20241112_Diagnosis_V4.csv")
        patient_file = os.path.join(norm_files_dir, "24PRJ217UVA_20241112_PatientMaster_V4.csv")
        
        # Load diagnosis and patient data
        diag_df = pd.read_csv(diag_file).rename(columns={'AvatarKey': 'PATIENT_ID'})
        patient_df = pd.read_csv(patient_file).rename(columns={'AvatarKey': 'PATIENT_ID'})
        
        # Log total records
        logger.info(f"Total records in diagnosis dataframe: {len(diag_df)}")
        
        # Ensure HistologyCode is a string
        diag_df['HistologyCode'] = diag_df['HistologyCode'].astype(str)
        
        # Log unique histology codes
        unique_codes = diag_df['HistologyCode'].unique()
        logger.info(f"Unique histology codes: {unique_codes}")
        
        # Define melanoma histology code prefixes (based on log output)
        melanoma_prefixes = ['8720/', '8721/', '8742/', '8743/', '8744/', '8730/', '8745/', '8723/', '8740/', '8746/', '8771/', '8772/']
        
        # Filter for melanoma patients using string matching
        melanoma_diag = diag_df[diag_df['HistologyCode'].apply(lambda x: any(x.startswith(prefix) for prefix in melanoma_prefixes))]
        
        # Log the number of melanoma patients found
        logger.info(f"Found {len(melanoma_diag)} melanoma patients with prefixes {melanoma_prefixes}")
        
        # Merge with patient data
        clinical_data = melanoma_diag.merge(patient_df, on='PATIENT_ID', how='left')
        logger.info(f"Loaded clinical data with {len(clinical_data)} melanoma patients.")
        return clinical_data
    except Exception as e:
        logger.error(f"Error loading clinical data: {e}")
        return None
def load_rnaseq_data(base_path):
    try:
        rnaseq_path = os.path.join(base_path, "../RNAseq", "gene_and_transcript_expression_results")
        expr_files = glob.glob(os.path.join(rnaseq_path, "*.genes.results"))
        if not expr_files:
            print(f"No .genes.results files found in {rnaseq_path}. Check the directory contents.")
            return None
        
        # Initialize a dictionary to store expression data
        expr_data = {}
        for file in expr_files:
            sample_id = os.path.basename(file).replace(".genes.results", "")
            df = pd.read_csv(file, sep="\t")  # Assuming tab-delimited files
            # Use 'gene_id' as index and 'TPM' (or another metric) as expression values
            expr_data[sample_id] = df.set_index("gene_id")["TPM"]
        
        # Combine into a single DataFrame
        expr_matrix = pd.DataFrame(expr_data)
        print(f"Loaded expression matrix with {expr_matrix.shape} (genes x samples).")
        return expr_matrix
    
    except Exception as e:
        print(f"Error loading RNA-Seq data: {e}")
        return None

def load_sample_to_patient_map(map_file_path):
    """Loads the mapping from sample ID (SLID) to patient ID (PATIENT_ID)."""
    try:
        if not os.path.exists(map_file_path):
            logger.error(f"Sample-to-patient mapping file not found at: {map_file_path}")
            return {}
            
        map_df = pd.read_csv(map_file_path)
        logger.info(f"Columns found in mapping file ({map_file_path}): {map_df.columns.tolist()}")
        
        # Define expected column names based on the actual file content
        slid_col = 'SampleID' # Corrected based on log output
        patient_col = 'PatientID' # Corrected based on log output
        
        # Check if required columns exist using the corrected names
        if slid_col not in map_df.columns:
            logger.error(f"Required column '{slid_col}' not found in mapping file.")
            return {} # Return empty if essential SLID column is missing

        if patient_col not in map_df.columns:
             logger.error(f"Required column '{patient_col}' not found in mapping file.")
             return {}

        # Create the dictionary using the identified column names
        sample_to_patient = dict(zip(map_df[slid_col], map_df[patient_col]))
        logger.info(f"Loaded sample-to-patient mapping with {len(sample_to_patient)} entries.")
        return sample_to_patient
        
    except KeyError as e:
        # This catch might be redundant now with explicit checks, but good practice
        logger.error(f"KeyError loading sample-to-patient map: Column '{e}' not found. Please check the mapping file format.")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading sample-to-patient map: {e}")
        return {}

def identify_melanoma_samples(base_path, clinical_data):
    """
    Identify melanoma tumor sample IDs (SLIDs) using clinical and QC data.
    Enhanced to use SURGERYBIOPSY_V4 data to clarify biopsy origins and filter samples.
    
    Parameters:
    -----------
    base_path : str
        Base path to the project directory containing QC and clinical files.
    clinical_data : pd.DataFrame
        DataFrame containing clinical data with melanoma patient IDs.
    
    Returns:
    --------
    list
        List of unique SLIDs corresponding to melanoma tumor samples.
    dict
        Dictionary with additional sample information including site, procedure, and diagnosis.
    """
    try:
        # Define actual file paths
        qc_file = os.path.join(base_path, "Manifest_and_QC_Files", "24PRJ217UVA_20250130_RNASeq_QCMetrics.csv")
        biopsy_file = os.path.join(base_path, "Clinical_Data", "24PRJ217UVA_NormalizedFiles", "24PRJ217UVA_20241112_SurgeryBiopsy_V4.csv")
        
        # Log the file paths being used
        logger.info(f"QC file path: {qc_file}")
        logger.info(f"Biopsy file path: {biopsy_file}")
        
        # Check if files exist
        if not os.path.exists(qc_file):
            logger.error(f"QC file not found at: {qc_file}")
            return [], {}
        if not os.path.exists(biopsy_file):
            logger.error(f"Biopsy file not found at: {biopsy_file}")
            return [], {}
        
        # Load dataframes
        qc_df = pd.read_csv(qc_file).rename(columns={'ORIENAvatarKey': 'PATIENT_ID'})
        biopsy_df = pd.read_csv(biopsy_file).rename(columns={'AvatarKey': 'PATIENT_ID'})
        
        # Log dataframe columns for debugging
        logger.info(f"QC dataframe columns: {qc_df.columns.tolist()}")
        logger.info(f"Biopsy dataframe columns: {biopsy_df.columns.tolist()}")
        
        # Get unique melanoma patient IDs from clinical_data
        melanoma_patient_ids = clinical_data['PATIENT_ID'].unique()
        logger.info(f"Found {len(melanoma_patient_ids)} melanoma patients in clinical data.")
        
        # If no melanoma patients are found, return an empty list
        if len(melanoma_patient_ids) == 0:
            logger.warning("No melanoma patients found in clinical data.")
            return [], {}
            
        # Initialize dictionary to store detailed sample information
        sample_details = {}
        
        # First get all biopsies for melanoma patients from SURGERYBIOPSY_V4
        melanoma_biopsies = biopsy_df[biopsy_df['PATIENT_ID'].isin(melanoma_patient_ids)].copy()
        logger.info(f"Found {len(melanoma_biopsies)} total biopsies for melanoma patients.")
        
        # Check if relevant columns exist in the biopsy data
        relevant_cols = ['PATIENT_ID', 'SLID', 'SpecimenSite', 'DiagnosisID', 'ProcedureType']
        missing_cols = [col for col in relevant_cols if col not in melanoma_biopsies.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns in SURGERYBIOPSY_V4: {missing_cols}")
            # Create empty columns for missing fields
            for col in missing_cols:
                if col != 'PATIENT_ID':  # We know PATIENT_ID exists
                    melanoma_biopsies[col] = None
        
        # Now check for histology and cancer type information in the biopsy data
        if 'DiagnosisID' in melanoma_biopsies.columns and 'DiagnosisID' in clinical_data.columns:
            # Try to match biopsies to specific diagnosis records
            logger.info("Matching biopsies to diagnosis records using DiagnosisID")
            
            # Create a diagnosis map for quick lookup
            diagnosis_map = {}
            if 'DiagnosisID' in clinical_data.columns and 'HistologyCode' in clinical_data.columns:
                for _, row in clinical_data.iterrows():
                    if pd.notna(row['DiagnosisID']):
                        diagnosis_map[row['DiagnosisID']] = {
                            'HistologyCode': row.get('HistologyCode', ''),
                            'PrimaryDiagnosisSite': row.get('PrimaryDiagnosisSite', ''),
                            'DiagnosisAge': row.get('DiagnosisAge', None)
                        }
            
            # Match diagnosis information to biopsies
            melanoma_biopsies['MatchedHistologyCode'] = melanoma_biopsies['DiagnosisID'].map(
                lambda x: diagnosis_map.get(x, {}).get('HistologyCode', '') if pd.notna(x) else ''
            )
            
            # Identify confirmed melanoma biopsies based on histology code
            melanoma_histology_prefixes = ['8720/', '8721/', '8742/', '8743/', '8744/', '8730/', '8745/', '8723/', '8740/', '8746/', '8771/', '8772/']
            melanoma_biopsies['IsConfirmedMelanoma'] = melanoma_biopsies['MatchedHistologyCode'].apply(
                lambda x: any(str(x).startswith(prefix) for prefix in melanoma_histology_prefixes) if pd.notna(x) else False
            )
            
            # Count confirmed melanoma biopsies
            confirmed_count = melanoma_biopsies['IsConfirmedMelanoma'].sum()
            logger.info(f"Identified {confirmed_count} biopsies with confirmed melanoma histology.")
        else:
            # If we can't match diagnoses, assume all biopsies are potentially melanoma
            logger.warning("Cannot match biopsies to specific diagnosis records. Treating all biopsies as potential melanoma samples.")
            melanoma_biopsies['IsConfirmedMelanoma'] = True
        
        # Create detailed sample information
        for _, row in melanoma_biopsies.iterrows():
            if pd.notna(row.get('SLID')) and row['SLID'] != '':
                sample_details[row['SLID']] = {
                    'patient_id': row['PATIENT_ID'],
                    'specimen_site': row.get('SpecimenSite', None),
                    'procedure_type': row.get('ProcedureType', None),
                    'is_confirmed_melanoma': row.get('IsConfirmedMelanoma', True),
                    'histology_code': row.get('MatchedHistologyCode', None),
                    'diagnosis_id': row.get('DiagnosisID', None)
                }
        
        # Now match with QC data to get all RNA-seq samples
        # Filter qc_df for samples from melanoma patients
        melanoma_qc = qc_df[qc_df['PATIENT_ID'].isin(melanoma_patient_ids)]
        
        # Get SLIDs from QC data
        melanoma_slids_from_qc = melanoma_qc['SLID'].unique().tolist()
        logger.info(f"Found {len(melanoma_slids_from_qc)} total RNA-seq SLIDs for melanoma patients.")
        
        # If we have biopsy information, prioritize samples with confirmed melanoma
        if 'IsConfirmedMelanoma' in melanoma_biopsies.columns:
            # Get SLIDs from confirmed melanoma biopsies
            confirmed_melanoma_slids = melanoma_biopsies[melanoma_biopsies['IsConfirmedMelanoma']]['SLID'].dropna().unique().tolist()
            
            # Get the overlap between QC data and confirmed melanoma biopsies
            final_slids = list(set(melanoma_slids_from_qc) & set(confirmed_melanoma_slids))
            
            # If we have a good number of confirmed samples, use only those
            if len(final_slids) >= 10:  # Arbitrary threshold, adjust as needed
                logger.info(f"Using {len(final_slids)} SLIDs from confirmed melanoma biopsies.")
            else:
                # If too few confirmed samples, use all potential melanoma samples
                logger.warning(f"Only {len(final_slids)} confirmed melanoma samples found. Using all potential melanoma samples.")
                final_slids = melanoma_slids_from_qc
        else:
            # If we don't have confirmation, use all samples from melanoma patients
            final_slids = melanoma_slids_from_qc
        
        # Add QC information to sample details if not already present
        for slid in final_slids:
            if slid not in sample_details:
                # Get patient ID from QC data
                qc_rows = melanoma_qc[melanoma_qc['SLID'] == slid]
                if not qc_rows.empty:
                    patient_id = qc_rows.iloc[0]['PATIENT_ID']
                    sample_details[slid] = {
                        'patient_id': patient_id,
                        'specimen_site': None,
                        'procedure_type': None,
                        'is_confirmed_melanoma': None,
                        'histology_code': None,
                        'diagnosis_id': None,
                        'source': 'QC only (no biopsy data)'
                    }
        
        logger.info(f"Identified {len(final_slids)} unique SLIDs for melanoma patients.")
        return final_slids, sample_details
    
    except Exception as e:
        logger.error(f"Error identifying melanoma samples: {e}", exc_info=True)
        return [], {}

def load_melanoma_data(base_path, map_file_path):
    clinical_data = load_clinical_data(base_path)
    if clinical_data is None:
        return None, None
    expr_matrix = load_rnaseq_data(base_path)
    if expr_matrix is None:
        return None, None
        
    # Load sample-to-patient mapping *before* identifying melanoma samples that require it
    sample_to_patient = load_sample_to_patient_map(map_file_path)
    if not sample_to_patient: # Check if mapping loaded successfully
        logger.error("Failed to load sample-to-patient map. Cannot proceed with mapping sample IDs.")
        # Decide how to proceed: return None or continue without mapping?
        # For now, returning None as mapping is critical
        return None, None 
        
    melanoma_slids, sample_details = identify_melanoma_samples(base_path, clinical_data)
    if not melanoma_slids:
        logger.warning("No melanoma SLIDs identified. Cannot filter expression matrix.")
        # Return potentially unfiltered clinical data, or None? Depends on desired behavior.
        # Returning None for consistency as expression data won't be filtered correctly.
        return None, None
    
    # Filter expression matrix to melanoma tumor samples
    common_slids = list(set(expr_matrix.columns) & set(melanoma_slids))
    if not common_slids:
        logger.warning("No common SLIDs found between expression matrix columns and identified melanoma samples.")
        return None, None
        
    expr_matrix_filtered = expr_matrix[common_slids]
    logger.info(f"Filtered expression matrix to {len(common_slids)} melanoma tumor samples.")
    
    # Map sample IDs (SLIDs) in the filtered expression matrix to patient IDs
    # Handle cases where a SLID might not be in the map (though should be rare if map is comprehensive)
    expr_matrix_filtered.columns = [sample_to_patient.get(col, f"UNMAPPED_{col}") for col in expr_matrix_filtered.columns]
    
    # Filter clinical data to patients with sequencing data *after* successful mapping
    sequenced_patients = [pid for pid in expr_matrix_filtered.columns if not pid.startswith("UNMAPPED_")]
    sequenced_patients = list(set(sequenced_patients)) # Get unique patient IDs
    
    if not sequenced_patients:
        logger.warning("No patients remain after mapping SLIDs to PATIENT_IDs.")
        return expr_matrix_filtered, pd.DataFrame() # Return empty clinical df

    clinical_data_filtered = clinical_data[clinical_data['PATIENT_ID'].isin(sequenced_patients)].copy() # Use .copy() to avoid SettingWithCopyWarning
    logger.info(f"Filtered clinical data to {len(clinical_data_filtered)} patients with valid sequencing data mapping.")
    
    return expr_matrix_filtered, clinical_data_filtered

if __name__ == "__main__":
    base_path = "/project/orien/data/aws/24PRJ217UVA_IORIG"
    # Ensure the map_file_path is correct
    map_file_path = os.path.join(base_path, "codes/output/sample_to_patient_map.csv") 
    output_dir = os.path.join(base_path, "codes/output")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Attempting to load sample map from: {map_file_path}") # Add log for map path confirmation
    
    expr_matrix, clinical_data = load_melanoma_data(base_path, map_file_path)
    
    # Check if dataframes are valid and not empty before saving
    if expr_matrix is not None and not expr_matrix.empty and clinical_data is not None and not clinical_data.empty:
        logger.info("Saving filtered expression matrix and clinical data.")
        expr_matrix.to_csv(os.path.join(output_dir, "melanoma_expression_matrix.csv"))
        clinical_data.to_csv(os.path.join(output_dir, "melanoma_clinical_data.csv"))
    else:
         logger.warning("Failed to generate or save output files due to errors in data loading or processing.")