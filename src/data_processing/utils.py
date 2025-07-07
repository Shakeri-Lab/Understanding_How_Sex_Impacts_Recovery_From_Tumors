'''
Usage:
./miniconda3/envs/ici_sex/bin/python src/data_processing/utils.py ../Manifest_and_QC_Files/24PRJ217UVA_20250130_RNASeq_QCMetrics.csv
'''


import argparse
import logging
import os
import pandas as pd


logger = logging.getLogger('data_processing.utils')


SAMPLE_COL_CANDIDATES: list[str] = ["SLID", "SampleID", "Sample_ID", "Sample", "LabID", "Lab_ID"]
PATIENT_COL_CANDIDATES: list[str] = ["ORIENAvatarKey", "PATIENT_ID", "PatientID", "Patient_ID", "AvatarKey"]


def first_match(cols, candidates) -> str | None:
    return next((c for c in candidates if c in cols), None)


def heuristic_col_match(cols, keywords) -> str | None:
    lowered_cols = {col.lower(): col for col in cols}
    for key in keywords:
        for lcol, orig in lowered_cols.items():
            if key in lcol:
                return orig
    return None


def create_map_from_qc(qc_file_path: str, sample_col: str | None, patient_col: str | None = None, clean_ids: bool = True) -> dict[str, str]:
    """
    Create a mapping between sample IDs and patient IDs from a QC metrics file.
    
    Parameters:
    -----------
    qc_file_path: str
        Path to the QC metrics file (CSV)
    sample_col: str | None
        Column name for sample IDs. If None, will attempt to detect automatically.
    patient_col: str | None
        Column name for patient IDs. If None, will attempt to detect automatically.
    clean_ids: bool, default True
        Whether to clean and normalize sample IDs by removing common suffixes and creating multiple entries to improve mapping success
        
    Returns:
    --------
    dict
        Dictionary mapping sample IDs to patient IDs
    """
    qc_data = pd.read_csv(qc_file_path)
    
    logger.info(f"QC data with {qc_data.shape[0]} rows and the following columns were loaded.\n{qc_data.columns.tolist()}")

    # Auto-detect columns if not provided
    cols: list[str] = qc_data.columns.tolist()
    
    if sample_col is None:
        sample_col = first_match(cols, SAMPLE_COL_CANDIDATES) or heuristic_col_match(cols, ("sample", "lab", "slid", "id"))
        logger.info(f"Sample ID column {sample_col} was detected automatically.")
        
    if patient_col is None:
        patient_col = first_match(cols, PATIENT_COL_CANDIDATES) or heuristic_col_match(cols, ("patient", "avatar", "orien", "key"))
        logger.info(f"Patient ID column {patient_col} was detected automatically.")
    
    if sample_col is None or patient_col is None:
        logger.error(f"Sample and patient ID columns both not be both identified. Available columns are {cols}.")
        return {}

    logger.info(f"Using {sample_col} as sample ID and {patient_col} as patient ID.")
    
    # Create basic mapping dictionary
    id_map: dict[str, str] = dict(zip(qc_data[sample_col], qc_data[patient_col]))

    # If requested, clean and normalize sample IDs
    if clean_ids:
        clean_map = {}
        suffixes = ("-RNA", "-DNA", "-T", "-N", "-PRIMARY", "-METASTATIC")
        
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
            for suffix in suffixes:
                cleaned_lab_id = cleaned_lab_id.replace(suffix, '')

            # Some IDs might have dots as separators
            dot_cleaned_lab_id = cleaned_lab_id.split('.')[0]

            # Add both original and cleaned versions to the mapping
            variants: list[str] = []
            for v in (lab_id, cleaned_lab_id, dot_cleaned_lab_id):
                if v not in variants:
                    variants.append(v)
                    
            for v in variants:
                clean_map[v] = orien_id

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


# This allows the module to be run directly for testing
if __name__ == "__main__":
    
    # Configure logging for direct execution.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
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
        if i >= 5:
            break
        print(f"  {k} -> {v}") 