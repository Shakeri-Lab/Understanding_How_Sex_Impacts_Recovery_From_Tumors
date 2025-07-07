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
SUFFIXES = ("-RNA", "-DNA", "-T", "-N", "-PRIMARY", "-METASTATIC")


def pick_column(
    cols,
    preferred: list[str],
    heuristics: tuple[str, ...] = (),
) -> str | None:
    '''
    Return the first matching column name.

    Priority:
    1. Exact match against preferred (case-sensitive).
    2. First column whose lowercase name contains any keyword in heuristics.
    '''
    cols = list(cols) # in case an Index object was passed
    lower_map = {c.lower(): c for c in cols}

    # 1) exact match
    for cand in preferred:
        if cand in cols:
            return cand

    # 2) keyword heuristic
    for kw in heuristics:
        found = next(
            (orig for low, orig in lower_map.items() if kw in low),
            None,
        )
        if found:
            return found

    return None


def strip_suffixes(string: str) -> str:
    '''
    Remove common suffixes and anything starting with the first period.
    '''
    for suffix in SUFFIXES:
        string = string.removesuffix(suffix)
    return string.split('.', 1)[0]


def create_map_from_qc(
    qc_file_path: str,
    sample_col: str | None = None,
    patient_col: str | None = None,
    clean_ids: bool = True
) -> dict[str, str]:
    '''
    Create a mapping between sample IDs and patient IDs from a QC metrics CSV file.
    
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
    dict[str, str] -- dictionary mapping sample IDs to patient IDs
    '''
    qc_data = pd.read_csv(qc_file_path)
    
    logger.info(f"QC data with {qc_data.shape[0]} rows and the following columns were loaded.\n{qc_data.columns.tolist()}")

    # Auto-detect columns if not provided
    cols: list[str] = qc_data.columns.tolist()
    
    sample_col = sample_col or pick_column(cols, SAMPLE_COL_CANDIDATES, ("sample", "lab", "slid", "id"))
    patient_col = patient_col or pick_column(cols, PATIENT_COL_CANDIDATES, ("patient", "avatar", "orien", "key"))
    
    if sample_col is None or patient_col is None:
        logger.error(f"Sample and patient ID columns both not be both identified. Available columns are {cols}.")
        return {}

    logger.info(f"Using {sample_col} as sample ID and {patient_col} as patient ID.")
    
    df = qc_data[[sample_col, patient_col]].dropna()
    
    # Create basic mapping dictionary
    id_map: dict[str, str] = dict(zip(qc_data[sample_col].astype(str), qc_data[patient_col].astype(str)))

    # If requested, clean and normalize sample IDs
    if clean_ids:
        cleaned = df[sample_col].astype(str).map(strip_suffixes)
        id_map.update(dict(zip(cleaned, df[patient_col].astype(str))))
        
    logger.info(f"Created mapping for {len(id_map)} sample IDs to {len(set(id_map.values()))} patient IDs")
        
    counts = pd.Series(id_map.values()).value_counts()
    dups = counts[counts.gt(1)]
    if not dups.empty:
        logger.info(f"Found {len(dups)} patients with multiple samples")

    return id_map


# This allows the module to be run directly for testing
if __name__ == "__main__":
    
    # Configure logging for direct execution.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description = "Create sample to patient map from QC CSV file.")
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
    
    print(f"Created mapping with {len(id_map)} entries. First five:")
    for i, (k, v) in enumerate(id_map.items()):
        if i == 5:
            break
        print(f"  {k} -> {v}") 