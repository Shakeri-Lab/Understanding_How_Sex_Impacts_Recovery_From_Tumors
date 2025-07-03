'''
Usage:
./miniconda3/envs/ici_sex/bin/python src/immune_analysis/data_loading.py
'''

from pathlib import Path
import glob
import logging
import numpy as np
import os
import pandas as pd


PATH_OF_ROOT = Path("/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors")
PATH_OF_NORMALIZED_CLINICAL_DATA = PATH_OF_ROOT / "../Clinical_Data/24PRJ217UVA_NormalizedFiles"
PATH_OF_MANIFEST_AND_QC_FILES = PATH_OF_ROOT / "../Manifest_and_QC_Files"
PATH_OF_GENE_AND_TRANSCRIPT_EXPRESSION_RESULTS = PATH_OF_ROOT / "../RNAseq/gene_and_transcript_expression_results"
PATH_OF_OUTPUTS = PATH_OF_ROOT / "output/data_loading"

FILENAME_OF_DIAGNOSIS_DATA = PATH_OF_NORMALIZED_CLINICAL_DATA / "24PRJ217UVA_20241112_Diagnosis_V4.csv" # Used 1 time
FILENAME_OF_PATIENT_DATA = PATH_OF_NORMALIZED_CLINICAL_DATA / "24PRJ217UVA_20241112_PatientMaster_V4.csv" # Used 1 time
FILENAME_OF_MEDICATIONS_DATA = PATH_OF_NORMALIZED_CLINICAL_DATA / "24PRJ217UVA_20241112_Medications_V4.csv" # Used 1 time
FILENAME_OF_SURGERY_BIOPSY_DATA = PATH_OF_NORMALIZED_CLINICAL_DATA / "24PRJ217UVA_20241112_SurgeryBiopsy_V4.csv"
FILENAME_OF_QC_DATA = PATH_OF_MANIFEST_AND_QC_FILES / "24PRJ217UVA_20250130_RNASeq_QCMetrics.csv"
FILENAME_OF_MAP_FROM_SAMPLE_TO_PATIENT = PATH_OF_ROOT / "output/eda/sample_to_patient_map.csv" # Used 1 time
FILENAME_OF_MELANOMA_EXPRESSION_MATRIX = PATH_OF_OUTPUTS / "melanoma_expression_matrix.csv"
FILENAME_OF_MELANOMA_CLINICAL_DATA = PATH_OF_OUTPUTS / "melanoma_clinical_data.csv"

LIST_OF_MELANOMA_HISTOLOGY_PREFIXES = ["8720/", "8721/", "8742/", "8743/", "8744/", "8730/", "8745/", "8723/", "8740/", "8746/", "8771/", "8772/"]

ICB_AGENTS = {
    # PD‑1
    "Pembrolizumab", "Nivolumab", "Cemiplimab", "Toripalimab", "Sintilimab", "Camrelizumab",
    # PD‑L1
    "Atezolizumab", "Durvalumab", "Avelumab",
    # CTLA‑4
    "Ipilimumab", "Tremelimumab",
    # combinations
    "nivolumab + ipilimumab",
}


MAP_OF_STRINGS_REPRESENTING_AGES_TO_AGES = {
    "Age 90 or older": 90.0,
    "Age Unknown/Not Recorded": np.nan
}

SET_OF_NAMES_OF_ICB_RELATED_COLUMNS_AND_DEFAULT_VALUES = {
    "HAS_ICB": False,
    "ICB_START_AGE": np.nan,
    "STAGE_AT_ICB": pd.NA
}


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

        
def add_earliest_melanoma_diagnosis_age(clinical_data: pd.DataFrame, melanoma_diag: pd.DataFrame) -> pd.DataFrame:
    if "AgeAtDiagnosis" not in melanoma_diag.columns:
        return clinical_data.assign(EarliestMelanomaDiagnosisAge = np.nan)
    clinical_data["EarliestMelanomaDiagnosisAge"] = clinical_data.groupby("PATIENT_ID")["AgeAtDiagnosis"].transform("min")
    return clinical_data


def add_icb_info(clinical_data: pd.DataFrame, diag_df: pd.DataFrame) -> pd.DataFrame:
    med_df = pd.read_csv(FILENAME_OF_MEDICATIONS_DATA)
    med_df["is_icb"] = med_df["Medication"].str.lower().isin({x.lower() for x in ICB_AGENTS})
    
    icb_rows = med_df[med_df["is_icb"]].copy()
    
    logger.info("Medications rows flagged as ICB: %d (of %d total rows)", len(icb_rows), len(med_df))

    if icb_rows.empty:
        return clinical_data.assign(**SET_OF_NAMES_OF_ICB_RELATED_COLUMNS_AND_DEFAULT_VALUES)
    
    icb_rows["AgeAtMedStart_num"] = icb_rows["AgeAtMedStart"].replace(MAP_OF_STRINGS_REPRESENTING_AGES_TO_AGES).pipe(pd.to_numeric, errors = "raise")

    #  ICB summary per patient (HAS_ICB, ICB_START_AGE)
    icb_summary = (
        icb_rows.groupby("AvatarKey")
        .agg(
            HAS_ICB = ("is_icb", "any"),
            ICB_START_AGE = ("AgeAtMedStart_num", "min")
        )
        .reset_index()
        .rename(
            columns = {"AvatarKey": "PATIENT_ID"}
        )
    )

    # Stage-at-ICB — compare to Diagnosis rows
    list_of_names_of_columns_of_stages = ["PathGroupStage", "ClinGroupStage"]
    stage_cols_diag = [c for c in list_of_names_of_columns_of_stages if c in diag_df.columns]
    if stage_cols_diag and "AgeAtDiagnosis" in diag_df.columns:
        diag_stage = diag_df[["PATIENT_ID", "AgeAtDiagnosis"] + stage_cols_diag].dropna(subset = stage_cols_diag, how = "all")

        def _stage_at_icb(row):
            patient_ID = row["PATIENT_ID"]
            icb_age = row["ICB_START_AGE"]
            if pd.isna(icb_age):
                return pd.NA
            cand = diag_stage[diag_stage["PATIENT_ID"] == patient_ID]
            if cand.empty or pd.isna(icb_age):
                return pd.NA
            cand = cand.assign(diff = (cand["AgeAtDiagnosis"] - icb_age).abs())
            best = cand.sort_values("diff").iloc[0]
            # preference order: pathologic → clinical
            for c in list_of_names_of_columns_of_stages:
                if c in best and pd.notna(best[c]):
                    return best[c]
            return pd.NA

        icb_summary["STAGE_AT_ICB"] = icb_summary.apply(_stage_at_icb, axis = 1)
    else:
        # no usable staging info available
        icb_summary["STAGE_AT_ICB"] = pd.NA

    clinical_data = clinical_data.merge(icb_summary, on = "PATIENT_ID", how = "left")
    
    for col in ["HAS_ICB", "ICB_START_AGE", "STAGE_AT_ICB"]:
        filled = clinical_data[col].notna().sum()
        logger.info("%s non-null in clinical_data: %d", col, filled)

    # Quick peek at unusual blanks
    blank_stage = clinical_data[clinical_data["HAS_ICB"] & clinical_data["STAGE_AT_ICB"].isna()]
    if not blank_stage.empty:
        logger.info("Patients with HAS_ICB=True but missing STAGE_AT_ICB:\n%s", blank_stage[["PATIENT_ID", "ICB_START_AGE"]].head())
    
    return clinical_data


def identify_melanoma_samples(clinical_data):
    '''
    Identify melanoma tumor Sample Level IDs (SLIDs) using clinical and QC data.
    Use surgery biopsy data to clarify biopsy origins and filter samples.
    
    Parameter:
    clinical_data: pd.DataFrame -- data frame containing clinical data with melanoma patient IDs
    
    This function returns a tuple of a list of unique SLIDs corresponding to melanoma tumor samples and a dictionary with additional sample information including site, procedure, and diagnosis.
    '''

    logger.info(f"Filename of QC data is {FILENAME_OF_QC_DATA}.")
    logger.info(f"Filename of surgery biopsy data is {FILENAME_OF_SURGERY_BIOPSY_DATA}.")

    # Check if files exist
    if not os.path.exists(FILENAME_OF_QC_DATA):
        logger.error(f"File of QC data was not found at {FILENAME_OF_QC_DATA}.")
        return [], {}

    if not os.path.exists(FILENAME_OF_SURGERY_BIOPSY_DATA):
        logger.error(f"File of surgery biopsy was not found at {FILENAME_OF_SURGERY_BIOPSY_DATA}.")
        return [], {}

    # Load dataframes
    qc_df = pd.read_csv(FILENAME_OF_QC_DATA).rename(columns = {"ORIENAvatarKey": "PATIENT_ID"})
    biopsy_df = pd.read_csv(FILENAME_OF_SURGERY_BIOPSY_DATA).rename(columns = {"AvatarKey": "PATIENT_ID"})

    logger.info(f"Columns of data frame of QC data are {qc_df.columns.tolist()}.")
    logger.info(f"Columns of data frame of surgery biopsy data are {biopsy_df.columns.tolist()}.")

    # Get unique melanoma patient IDs from clinical_data
    melanoma_patient_ids = clinical_data['PATIENT_ID'].unique()

    logger.info(f"{len(melanoma_patient_ids)} melanoma patients in clinical data were found.")

    # If no melanoma patients are found, return an empty list.
    if len(melanoma_patient_ids) == 0:
        logger.warning("No melanoma patients were found in clinical data.")
        return [], {}

    # Initialize dictionary to store detailed sample information.
    sample_details = {}

    '''
    Derive the procedure used for each biopsy.
    Any column whose header starts with "Method" describes a possible procedure (e.g. `MethodExcisional` = "Yes", `MethodPunch` = "Yes").
    '''
    method_cols = [c for c in biopsy_df.columns if c.startswith("Method")]

    if not method_cols:
        logger.warning("No columns beginning with \"Method\" were found in the biopsy file.")
        biopsy_df["ProcedureType"] = None

    else:

        # TODO: Evaluate keeping only first header in `24PRJ217UVA_20241112_SurgeryBiopsy_V4.csv` per row.

        biopsy_df["ProcedureType"] = biopsy_df.apply(
            lambda row: next((col for col in method_cols if str(row[col]).strip().lower() == "yes"), None),
            axis = 1
        )

    # Keep only melanoma-patient biopsies and give SpecimenSite a shorter name. ProcedureType now comes along for free.
    melanoma_biopsies = (
        biopsy_df[biopsy_df["PATIENT_ID"].isin(melanoma_patient_ids)]
        .copy()
        .rename(columns = {"SurgeryBiopsyLocation": "SpecimenSite"})
    )

    logger.info(f"{len(melanoma_biopsies)} total biopsies for melanoma patients were found.")

    # TODO: Evaluate keeping only first row in `24PRJ217UVA_20250130_RNASeq_QCMetrics.csv` per patient.

    # Attach exactly one SLID per patient from the QC metrics file.
    qc_first_slid = qc_df.sort_values(['PATIENT_ID', 'SLID']).drop_duplicates("PATIENT_ID")[['PATIENT_ID', 'SLID']].rename(columns = {"SLID": "SLID_QC"})

    # Merge onto the biopsy table
    melanoma_biopsies = melanoma_biopsies.merge(
        qc_first_slid,
        on = "PATIENT_ID",
        how = "left",
        validate = "m:1" # many biopsies ↔︎ one QC-row
    )

    # If the biopsy already had an SLID, keep it; otherwise fill from QC
    if "SLID" in melanoma_biopsies.columns:
        melanoma_biopsies["SLID"] = melanoma_biopsies["SLID"].fillna(melanoma_biopsies["SLID_QC"])
        melanoma_biopsies.drop(columns = ['SLID_QC'], inplace = True)
    else:
        melanoma_biopsies.rename(columns = {"SLID_QC": "SLID"}, inplace = True)

    # TODO: Evaluate keeping only first row in `24PRJ217UVA_20241112_Diagnosis_V4.csv` per patient.

    # Attach DiagnosisID (row index) to every biopsy for the same patient
    if os.path.exists(FILENAME_OF_DIAGNOSIS_DATA):
        diag_idx_df = (
            pd.read_csv(FILENAME_OF_DIAGNOSIS_DATA)
            .reset_index() # make 0-based idx
            .rename(
                columns = {
                    "index": 'DiagnosisID',
                    "AvatarKey": "PATIENT_ID"
                }
            )
            .drop_duplicates("PATIENT_ID", keep = 'first')[["PATIENT_ID", "DiagnosisID"]]
        )
        melanoma_biopsies = melanoma_biopsies.merge(
            diag_idx_df,
            on = "PATIENT_ID",
            how = "left",
            validate = "m:1" # many biopsies ↔︎ one diagnosis row
        )
    else:
        logger.warning(f"Diagnosis file was not found at {diag_file_path}.")
        melanoma_biopsies["DiagnosisID"] = None # keep column for downstream code

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
                        'AgeAtDiagnosis': row.get('AgeAtDiagnosis', None)
                    }

        # Match diagnosis information to biopsies
        melanoma_biopsies['MatchedHistologyCode'] = melanoma_biopsies['DiagnosisID'].map(
            lambda x: diagnosis_map.get(x, {}).get('HistologyCode', '') if pd.notna(x) else ''
        )

        # Identify confirmed melanoma biopsies based on histology code
        melanoma_biopsies["IsConfirmedMelanoma"] = melanoma_biopsies['MatchedHistologyCode'].apply(
            lambda x: any(str(x).startswith(prefix) for prefix in LIST_OF_MELANOMA_HISTOLOGY_PREFIXES) if pd.notna(x) else False
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
    melanoma_qc = qc_df[qc_df["PATIENT_ID"].isin(melanoma_patient_ids)]

    # Get SLIDs from QC data
    melanoma_slids_from_qc = qc_first_slid["SLID_QC"].tolist()

    logger.info(f"{len(melanoma_slids_from_qc)} total RNA-seq SLIDs for melanoma patients were found.")

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

    logger.info(f"{len(final_slids)} unique SLIDs for melanoma patients were identified.")
    return final_slids, sample_details


def load_clinical_data():
    
    diag_df = pd.read_csv(FILENAME_OF_DIAGNOSIS_DATA)        
    # Create a zero-based row index to serve as a synthetic DiagnosisID.
    diag_df.reset_index(drop = False, inplace = True)
    diag_df.rename(
        columns = {
            "index": "DiagnosisID",
            "AvatarKey": "PATIENT_ID"
        },
        inplace = True
    )

    if "AgeAtDiagnosis" in diag_df.columns:
        diag_df["AgeAtDiagnosis"] = diag_df["AgeAtDiagnosis"].replace(MAP_OF_STRINGS_REPRESENTING_AGES_TO_AGES).pipe(pd.to_numeric, errors = "raise")

    diag_df["HistologyCode"] = diag_df["HistologyCode"].astype(str)

    patient_df = pd.read_csv(FILENAME_OF_PATIENT_DATA).rename(columns = {"AvatarKey": "PATIENT_ID"})

    # Filter for melanoma patients.
    melanoma_diag = diag_df[diag_df["HistologyCode"].apply(lambda x: any(x.startswith(prefix) for prefix in LIST_OF_MELANOMA_HISTOLOGY_PREFIXES))]

    logger.info(f"The number of records in data frame of diagnosis data is {len(diag_df)}.")
    logger.info("The following list is a list of unique histology codes. %s", diag_df["HistologyCode"].unique())
    logger.info(f"{len(melanoma_diag)} melanoma patients were found.")

    clinical_data = melanoma_diag.merge(patient_df, on = "PATIENT_ID", how = "left")

    if os.path.exists(FILENAME_OF_MEDICATIONS_DATA):
        clinical_data = add_icb_info(clinical_data, diag_df)
    else:
        logger.warning("File of medications data was not found at %s.", med_file)

    clinical_data = add_earliest_melanoma_diagnosis_age(clinical_data, melanoma_diag)


    non_null_age = clinical_data["EarliestMelanomaDiagnosisAge"].notna().sum()
    logger.info(
        "EarliestMelanomaDiagnosisAge populated for "
        f"{non_null_age}/{clinical_data['PATIENT_ID'].nunique()} patients"
    )

    '''
    Fill ICB-related columns of data frame of clinical data with values or placeholders.
    Column HAS_ICB contains boolean indicators of whether there are records of ICB for patients and defaults to False.
    Column ICB_START_AGE contains age at first ICB dose and defaults to NaN.
    Column STAGE_AT_ICB contains stage when ICB started and defaults to NA.
    '''
    for col, default in SET_OF_NAMES_OF_ICB_RELATED_COLUMNS_AND_DEFAULT_VALUES.items():
        clinical_data[col] = clinical_data.get(col, default).fillna(default)

    logger.info("Data frame of clinical data has columns [%s].", ", ".join(sorted(clinical_data.columns.tolist())))
    logger.info(f"Clinical data for {len(clinical_data)} melanoma patients was loaded.")

    return clinical_data

    
def load_melanoma_data():

    clinical_data = load_clinical_data()
    if clinical_data is None:
        return None, None
    
    expr_matrix = load_rnaseq_data()
    if expr_matrix is None:
        return None, None
        
    # Load sample-to-patient mapping before identifying melanoma samples that require it.
    sample_to_patient = load_sample_to_patient_map()
    if not sample_to_patient: # Check if mapping loaded successfully
        logger.error("Load sample-to-patient map failed.")
        return None, None
        
    melanoma_slids, sample_details = identify_melanoma_samples(clinical_data)
    if not melanoma_slids:
        logger.warning("No melanoma SLIDs were identified.")
        return None, None
    
    # Filter expression matrix to melanoma tumor samples
    common_slids = [c for c in expr_matrix.columns if c in melanoma_slids]
    if not common_slids:
        logger.warning("No common SLIDs were found between expression matrix columns and identified melanoma samples.")
        return None, None
        
    expr_matrix_filtered = expr_matrix[common_slids]
    
    logger.info(f"Expression matrix was filtered to {len(common_slids)} melanoma tumor samples.")
    
    # Map sample IDs (SLIDs) in the filtered expression matrix to patient IDs
    # Handle cases where a SLID might not be in the map.
    expr_matrix_filtered.columns = [sample_to_patient.get(col, f"UNMAPPED_{col}") for col in expr_matrix_filtered.columns]
    
    # Filter clinical data to patients with sequencing data after successful mapping.
    sequenced_patients = [pid for pid in expr_matrix_filtered.columns if not pid.startswith("UNMAPPED_")]
    sequenced_patients = list(set(sequenced_patients)) # Get unique patient IDs
    
    if not sequenced_patients:
        logger.warning("No patients remain after mapping SLIDs to PATIENT_IDs.")
        return expr_matrix_filtered, pd.DataFrame() # Return empty clinical df

    clinical_data_filtered = clinical_data[clinical_data['PATIENT_ID'].isin(sequenced_patients)].copy()
    logger.info(f"Clinical data was filtered to {len(clinical_data_filtered)} patients with valid sequencing data mapping.")
    
    return expr_matrix_filtered, clinical_data_filtered
    

def load_rnaseq_data():
    
    expr_files = list(PATH_OF_GENE_AND_TRANSCRIPT_EXPRESSION_RESULTS.glob("*.genes.results"))
    if not expr_files:
        logger.error(f"No .genes.results files were found in {PATH_OF_GENE_AND_TRANSCRIPT_EXPRESSION_RESULTS}.")
        return None

    # Initialize a dictionary to store expression data
    expr_data = {
        os.path.basename(f).removesuffix(".genes.results"): pd.read_csv(f, sep = "\t").set_index("gene_id")["TPM"]
        for f in expr_files
    }

    # Combine into a single DataFrame
    expr_matrix = pd.DataFrame(expr_data)
    print(f"Expression matrix with {expr_matrix.shape} (genes x samples) was loaded.")
    return expr_matrix

    
def load_sample_to_patient_map() -> dict[str, str]:
    '''
    This function loads the mapping from sample ID (SLID) to patient ID (PATIENT_ID).
    '''

    if not os.path.exists(FILENAME_OF_MAP_FROM_SAMPLE_TO_PATIENT):
        logger.error(f"Sample-to-patient mapping file was not found at {FILENAME_OF_MAP_FROM_SAMPLE_TO_PATIENT}.")
        return {}

    map_df = pd.read_csv(FILENAME_OF_MAP_FROM_SAMPLE_TO_PATIENT)

    logger.info(f"Columns found in mapping file ({FILENAME_OF_MAP_FROM_SAMPLE_TO_PATIENT}): {map_df.columns.tolist()}")
    
    if "SampleID" not in map_df.columns:
        logger.error(f"Required column \"{slid_col}\" was not found in mapping file.")
        return {}
    
    if "PatientID" not in map_df.columns:
         logger.error(f"Required column \"{patient_col}\" was not found in mapping file.")
         return {}

    # Create the dictionary using the identified column names
    sample_to_patient = dict(zip(map_df["SampleID"], map_df["PatientID"]))
    logger.info(f"Sample-to-patient mapping with {len(sample_to_patient)} entries was loaded.")
    return sample_to_patient


if __name__ == "__main__": 

    os.makedirs(PATH_OF_OUTPUTS, exist_ok = True)
    
    logger.info(f"Sample map will be loaded from {FILENAME_OF_MAP_FROM_SAMPLE_TO_PATIENT}.")
    
    expr_matrix, clinical_data = load_melanoma_data()
    
    if expr_matrix is not None and not expr_matrix.empty and clinical_data is not None and not clinical_data.empty:
        logger.info("Saving filtered expression matrix and clinical data.")
        expr_matrix.to_csv(FILENAME_OF_MELANOMA_EXPRESSION_MATRIX, index_label = "Ensembl ID")
        clinical_data.to_csv(FILENAME_OF_MELANOMA_CLINICAL_DATA, index = False)
    else:
         logger.warning("Generating or saving output files failed due to errors in data loading or processing.")