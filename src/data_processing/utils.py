'''
Usage:
./miniconda3/envs/ici_sex/bin/python src/data_processing/utils.py ../Manifest_and_QC_Files/24PRJ217UVA_20250130_RNASeq_QCMetrics.csv
'''


import argparse
import logging
import os
import pandas as pd


logger = logging.getLogger(__name__)


SAMPLE_COL_CANDIDATES: list[str] = ["SLID", "SampleID", "Sample_ID", "Sample", "LabID", "Lab_ID"]
PATIENT_COL_CANDIDATES: list[str] = ["ORIENAvatarKey", "PATIENT_ID", "PatientID", "Patient_ID", "AvatarKey"]
SUFFIXES = ("-RNA", "-DNA", "-T", "-N", "-PRIMARY", "-METASTATIC")


def pick_column(cols, preferred: list[str], heuristics: tuple[str, ...] = ()) -> str | None:
    cols = list(cols) # in case an Index object was passed
    lower_map = {c.lower(): c for c in cols}

    for cand in preferred:
        if cand in cols:
            return cand

    for kw in heuristics:
        found = next((orig for low, orig in lower_map.items() if kw in low), None)
        if found:
            return found

    return None


def strip_suffixes(string: str) -> str:
    for suffix in SUFFIXES:
        string = string.removesuffix(suffix)
    return string.split('.', 1)[0]


def create_map_from_qc(qc_file_path: str, sample_col: str | None = None, patient_col: str | None = None, clean_ids: bool = True) -> dict[str, str]:
    qc_data = pd.read_csv(qc_file_path)
    
    logger.info(f"QC data with {qc_data.shape[0]} rows and the following columns were loaded.\n{qc_data.columns.tolist()}")

    cols: list[str] = qc_data.columns.tolist()    
    sample_col = sample_col or pick_column(cols, SAMPLE_COL_CANDIDATES, ("sample", "lab", "slid", "id"))
    patient_col = patient_col or pick_column(cols, PATIENT_COL_CANDIDATES, ("patient", "avatar", "orien", "key"))
    
    if not sample_col or not patient_col:
        raise Exception(f"Sample and patient ID columns both not be both identified. Available columns are {cols}.")
    
    logger.info(f"Using {sample_col} as sample ID and {patient_col} as patient ID.")
    
    pairs = qc_data.loc[:, [sample_col, patient_col]].dropna().astype(str)
    if clean_ids:
        pairs[sample_col] = pairs[sample_col].map(strip_suffixes)
    id_map = dict(pairs.values)
        
    logger.info(f"Created mapping for {len(id_map)} sample IDs to {len(set(id_map.values()))} patient IDs")
        
    dups = {p for p in id_map.values() if list(id_map.values()).count(p) > 1}
    if dups:
        logger.info("Found %d patients with multiple samples", len(dups))

    return id_map


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description = "Create sample to patient map from QC CSV file.")
    parser.add_argument("qc_file", help = "Path to QC metrics file")
    parser.add_argument("--sample-col", help = "Column name for sample IDs")
    parser.add_argument("--patient-col", help = "Column name for patient IDs")
    args = parser.parse_args()
    
    id_map = create_map_from_qc(args.qc_file, sample_col = args.sample_col, patient_col = args.patient_col)
    
    print(f"Created mapping with {len(id_map)} entries. First five:")
    for k, v in list(id_map.items())[:5]:
        print(f"  {k} -> {v}")