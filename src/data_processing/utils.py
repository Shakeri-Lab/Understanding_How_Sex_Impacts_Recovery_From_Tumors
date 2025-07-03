'''
Usage:
./miniconda3/envs/ici_sex/bin/python src/data_processing/utils.py ../Manifest_and_QC_Files/24PRJ217UVA_20250130_RNASeq_QCMetrics.csv
'''


import argparse
import logging
import os
import pandas as pd


def create_map_from_qc(qc_file_path, sample_col=None, patient_col=None, clean_ids=True):
    """
    Create a mapping between sample IDs and patient IDs from a QC metrics file.
    
    Parameters:
    -----------
    qc_file_path : str
        Path to the QC metrics file (CSV)
    sample_col : str, optional
        Column name for sample IDs. If None, will attempt to detect automatically.
    patient_col : str, optional
        Column name for patient IDs. If None, will attempt to detect automatically.
    clean_ids : bool, default=True
        Whether to clean and normalize sample IDs by removing common suffixes
        and creating multiple entries to improve mapping success
        
    Returns:
    --------
    dict
        Dictionary mapping sample IDs to patient IDs
    """
    logger.info(f"Loading QC file from: {qc_file_path}")
    
    if not os.path.exists(qc_file_path):
        logger.error(f"QC metrics file not found at {qc_file_path}")
        return {}
    
    try:
        qc_data = pd.read_csv(qc_file_path)
        logger.info(f"Loaded QC data with {qc_data.shape[0]} rows and columns: {qc_data.columns.tolist()}")
        
        # Auto-detect columns if not provided
        if sample_col is None or patient_col is None:
            # Common column names for sample IDs
            sample_col_options = ['SLID', 'SampleID', 'Sample_ID', 'Sample', 'LabID', 'Lab_ID']
            # Common column names for patient IDs
            patient_col_options = ['ORIENAvatarKey', 'PATIENT_ID', 'PatientID', 'Patient_ID', 'AvatarKey']
            
            # Try to find sample column
            if sample_col is None:
                for col in sample_col_options:
                    if col in qc_data.columns:
                        sample_col = col
                        logger.info(f"Auto-detected sample ID column: {sample_col}")
                        break
                
                # If still not found, look for columns containing key terms
                if sample_col is None:
                    potential_cols = [col for col in qc_data.columns if any(
                        term in col.lower() for term in ['sample', 'lab', 'slid', 'id'])]
                    if potential_cols:
                        sample_col = potential_cols[0]
                        logger.info(f"Using column containing sample ID term: {sample_col}")
            
            # Try to find patient column
            if patient_col is None:
                for col in patient_col_options:
                    if col in qc_data.columns:
                        patient_col = col
                        logger.info(f"Auto-detected patient ID column: {patient_col}")
                        break
                
                # If still not found, look for columns containing key terms
                if patient_col is None:
                    potential_cols = [col for col in qc_data.columns if any(
                        term in col.lower() for term in ['patient', 'avatar', 'orien', 'key'])]
                    if potential_cols and sample_col != potential_cols[0]:
                        patient_col = potential_cols[0]
                        logger.info(f"Using column containing patient ID term: {patient_col}")
        
        # Ensure we found both columns
        if sample_col is None or patient_col is None:
            logger.error("Could not identify sample and/or patient ID columns")
            logger.info(f"Available columns: {qc_data.columns.tolist()}")
            return {}
        
        logger.info(f"Using {sample_col} as sample ID and {patient_col} as patient ID")
        
        # Show sample values to verify
        sample_values = qc_data[sample_col].dropna().head(5).tolist()
        patient_values = qc_data[patient_col].dropna().head(5).tolist()
        logger.info(f"Sample {sample_col} values: {sample_values}")
        logger.info(f"Sample {patient_col} values: {patient_values}")
        
        # Create basic mapping dictionary
        id_map = dict(zip(qc_data[sample_col], qc_data[patient_col]))
        
        # If requested, clean and normalize sample IDs
        if clean_ids:
            clean_map = {}
            for lab_id, orien_id in id_map.items():
                # Skip if either ID is NaN
                if pd.isna(lab_id) or pd.isna(orien_id):
                    continue
                    
                # Convert IDs to strings
                lab_id = str(lab_id)
                orien_id = str(orien_id)
                
                # Additional cleaning steps for lab IDs
                # Remove any common prefixes/suffixes that might be in expression data
                cleaned_lab_id = lab_id
                for suffix in ['-RNA', '-DNA', '-T', '-N', '-PRIMARY', '-METASTATIC']:
                    cleaned_lab_id = cleaned_lab_id.replace(suffix, '')
                
                # Some IDs might have dots as separators
                dot_cleaned_lab_id = cleaned_lab_id.split('.')[0] if '.' in cleaned_lab_id else cleaned_lab_id
                
                # Add both original and cleaned versions to the mapping
                clean_map[lab_id] = orien_id
                clean_map[cleaned_lab_id] = orien_id
                clean_map[dot_cleaned_lab_id] = orien_id
            
            id_map = clean_map
        
        logger.info(f"Created mapping for {len(id_map)} sample IDs to {len(set(id_map.values()))} patient IDs")
        
        # Check if we have duplicated patient IDs
        patient_counts = {}
        for patient_id in id_map.values():
            patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1
        
        duplicated_patients = {k: v for k, v in patient_counts.items() if v > 1}
        if duplicated_patients:
            logger.info(f"Found {len(duplicated_patients)} patients with multiple samples")
            
        return id_map
        
    except Exception as e:
        logger.error(f"Error creating ID mapping: {e}")
        logger.error(traceback.format_exc())
        return {}


# This allows the module to be run directly for testing
if __name__ == "__main__":
    
    # Configure logging for direct execution.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('data_processing.utils')
    
    parser = argparse.ArgumentParser(description = "Test QC file mapping functionality.")
    parser.add_argument("qc_file", help = "Path to QC metrics file")
    parser.add_argument("--sample-col", help = "Column name for sample IDs")
    parser.add_argument("--patient-col", help = "Column name for patient IDs")
    args = parser.parse_args()
    
    # Test the mapping function
    id_map = create_map_from_qc(
        args.qc_file,
        sample_col=args.sample_col,
        patient_col=args.patient_col
    )
    
    print(f"Created mapping with {len(id_map)} entries")
    print("First 5 mappings:")
    for i, (k, v) in enumerate(id_map.items()):
        if i >= 5: break
        print(f"  {k} -> {v}") 