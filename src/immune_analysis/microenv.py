'''
Usage:
conda activate ici_sex
./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.microenv
'''

from rpy2.rinterface import NULLType
from pathlib import Path
from datetime import datetime
import glob
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import logging
import numpy as np
from rpy2.robjects import numpy2ri, pandas2ri
import os
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.vectors import StrVector
import traceback

from src.config import FOCUSED_XCELL_PANEL
from src.config import paths
from src.immune_analysis.data_loading import identify_melanoma_samples
from src.immune_analysis.data_loading import load_RNA_sequencing_data_for_melanoma_samples


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Activate converters for rpy2.
numpy2ri.activate()
pandas2ri.activate()


def clean_expression_df(df: pd.DataFrame) -> pd.DataFrame:
    df.index = df.index.astype(str)
    dictionary_of_Ensembl_IDs_and_HGNC_symbols = (
        pd.read_csv(
            paths.data_frame_of_Ensembl_IDs_and_HGNC_symbols,
            usecols = ["gene_id", "gene_symbol"]
        )
        .dropna(subset = ["gene_symbol"])
        .drop_duplicates(subset = ["gene_id"])
        .set_index("gene_id")["gene_symbol"]
        .to_dict()
    )
    df = df.loc[df.index.isin(dictionary_of_Ensembl_IDs_and_HGNC_symbols.keys())].copy()
    df.index = df.index.map(dictionary_of_Ensembl_IDs_and_HGNC_symbols.get)
    df = df.groupby(level = 0, sort = False).sum(numeric_only = True)
    df = df.loc[df.index.notna() & (df.index != "")]
    df = df.apply(pd.to_numeric, errors = "raise").fillna(0.0)
    return df


def metastatic_from_primary_met(flag: str | float | None):
    """
    Translate the Diagnosis-level "Primary/Met" column into a Boolean.

    • "Metastatic"  → True  
    • "Primary"     → False  
    • "Not applicable (germline)" or NA/blank → None
    """
    if pd.isna(flag):
        return None
    flag = str(flag).strip().lower()
    if flag == "metastatic":
        return True
    if flag == "primary":
        return False
    return None


def main():
    
    paths.ensure_dependencies_for_microenv_exist()

    expr_matrix, clinical_data = load_RNA_sequencing_data_for_melanoma_samples()
    process_melanoma_immune_data(clinical_data)

    clinical_data_processed = pd.read_csv(paths.data_frame_of_melanoma_patient_and_sequencing_data)
    logger.info(f"Clinical data for {clinical_data_processed['PATIENT_ID'].nunique()} unique melanoma patients was loaded from {paths.data_frame_of_melanoma_patient_and_sequencing_data}.")

    success = run_xcell_analysis(expr_matrix, clinical_data_processed)
    if success:
        logger.info("xCell analysis succeeded.")
    else:
        raise Exception("xCell analysis failed.")


def process_melanoma_immune_data(clinical_data):
    '''
    Process expression matrix to extract immune Micro-Environment information.
    TODO: What is immune Micro-Environment information?
        
    Returns:
    --------
    immune_clinical: pd.DataFrame -- data frame with melanoma samples and immune features
    TODO: Describe the data frame with melanoma samples and immune features.
    '''
    
    if clinical_data is None:
        logger.error("Failed to load clinical data.")
        return None

    logger.info("Sample details will be collected.")
    
    melanoma_slids, sample_details = identify_melanoma_samples(clinical_data)
    
    logger.info(f"Sample details for {len(sample_details)} samples were retrieved.")

    # Build a tidy "one row per tumour sample" data-frame from `sample_details`.
    sample_rows = (pd.DataFrame.from_dict(sample_details, orient = "index")
        .rename_axis("SLID") # make SLID an index label
        .reset_index() # turn SLID into an ordinary column
        .rename(columns = {"patient_id": "PATIENT_ID"})
    )

    sample_rows = sample_rows.rename(
        columns = {
            "specimen_site": "SpecimenSite",
            "procedure_type": "ProcedureType",
            "is_confirmed_melanoma": "IsConfirmedMelanoma",
            "histology_code": "HistologyCode",
            "diagnosis_id": "DiagnosisID"
        }
    )

    clinical_cols = ["PATIENT_ID", "Sex", "Race", "AgeAtClinicalRecordCreation", "EarliestMelanomaDiagnosisAge", "HAS_ICB", "ICB_START_AGE", "STAGE_AT_ICB", "Primary/Met"]
    available = [c for c in clinical_cols if c in clinical_data.columns]
    
    immune_clinical = sample_rows.merge(clinical_data[available], on = "PATIENT_ID", how = "left").set_index("SLID")
    
    logger.info("Per-sample data-frame shape: %s", immune_clinical.shape)

    # Merge clinical info with immune data
    available_cols = [col for col in clinical_cols if col in clinical_data.columns]
    if len(available_cols) < len(clinical_cols):
        missing_cols = set(clinical_cols) - set(available_cols)
        
        logger.warning(f"Missing clinical columns: {missing_cols}")

    immune_clinical["IsMetastatic"] = immune_clinical["Primary/Met"].apply(metastatic_from_primary_met)
    
    logger.info(f"Metastatic status was determined for {immune_clinical['IsMetastatic'].count()} samples.")
    logger.info(f"There are {(immune_clinical['IsMetastatic'] == True).sum()} samples that are part of metastatic disease.")
    logger.info(f"There are {(immune_clinical['IsMetastatic'] == False).sum()} primary samples.")

    age_non_null = immune_clinical["EarliestMelanomaDiagnosisAge"].notna().sum()
    stage_non_null = immune_clinical["STAGE_AT_ICB"].notna().sum()

    logger.info(
        "Column `EarliestMelanomaDiagnosisAge` in data frame `immune_clinical` was populated for %d / %d samples corresponding to %d unique patients.",
        age_non_null,
        len(immune_clinical),
        immune_clinical["PATIENT_ID"].nunique()
    )
    logger.info(
        "Column `STAGE_AT_ICB` in data frame `immune_clinical` was populated for %d / %d samples.",
        stage_non_null,
        len(immune_clinical)
    )
    if stage_non_null == 0:
        logger.info(
            "The first 3 rows in data frame `immune_clinical` lacking STAGE_AT_ICB are the following.\n%s",
            immune_clinical[["PATIENT_ID", "ICB_START_AGE"]].head(n = 3)
        )

    logger.info(
        "Column completeness just before write:\n%s",
        immune_clinical[
            ["EarliestMelanomaDiagnosisAge", "STAGE_AT_ICB"]
        ].isna().mean().rename(lambda x: f"{x}_null_fraction"),
    )
    
    #pd.set_option("display.max_columns", None)
    immune_clinical = immune_clinical.reset_index()
    dupes = immune_clinical[
        immune_clinical["SLID"].duplicated(keep = False)
    ]
    logger.info(dupes.sort_values("SLID"))
    immune_clinical = immune_clinical.set_index("SLID")
    
    # All groups of duplicate rows have equal values in all fields.
    immune_clinical = (
        immune_clinical
            .reset_index()
            .drop_duplicates("SLID", keep="first")
            .set_index("SLID")
    )

    immune_clinical.to_csv(paths.melanoma_sample_immune_clinical_data)
    
    logger.info(f"Saved data frame `immune_clinical` to {paths.melanoma_sample_immune_clinical_data}.")

    # Create a summary of biopsy locations.
    site_summary = immune_clinical["SpecimenSite"].value_counts().reset_index()
    site_summary.columns = ["SpecimenSite", "Count"]
    site_summary.to_csv(paths.map_of_biopsy_locations_to_counts, index = False)
    
    logger.info(f"Map of biopsy locations to counts was saved to {paths.map_of_biopsy_locations_to_counts}.")

    # Create a summary of procedure types.
    proc_summary = immune_clinical["ProcedureType"].dropna().str.split('|').explode().str.strip().value_counts().reset_index(name = "Count").rename(columns = {"index": "ProcedureType"})
    proc_summary.to_csv(paths.map_of_procedure_types_to_counts, index = False)
    
    logger.info(f"Map of procedure type to counts was saved to {paths.map_of_procedure_types_to_counts}.")

    # Create a summary of metastatic status.
    meta_summary = immune_clinical['IsMetastatic'].value_counts().reset_index()
    meta_summary.columns = ['IsMetastatic', 'Count']
    meta_summary.to_csv(paths.map_of_indicators_that_specimens_are_part_of_metastatic_disease_to_counts, index = False)
    
    logger.info(f"Map of indicators that specimens are part of metastatic disease to counts was saved to {paths.map_of_indicators_that_specimens_are_part_of_metastatic_disease_to_counts}.")

    return immune_clinical


def run_xcell_analysis(expr_df: pd.DataFrame, clinical_df: pd.DataFrame) -> bool:
    '''
    Run xCell on a gene-by-sample matrix and write `xcell_scores_raw.csv` and `xcell_scores_focused_panel.csv`.
    '''

    # 1. Tidy the matrix so xCell will recognise the genes.
    expr_df = clean_expression_df(expr_df)
    if expr_df.empty:
        raise Exception("Expression matrix is empty after cleaning — aborting.")
    
    logger.info(
        "The matrix of gene and sample information after cleaning has %s genes and %s samples.",
        expr_df.shape[0],
        expr_df.shape[1]
    )

    # 2. Convert matrix of gene and sample information to R.
    
    # TODO: What is the type of `expr_r`?
    
    logger.info("The matrix of gene and sample information will be converted to R.")

    with localconverter(ro.default_converter + pandas2ri.converter):
        expr_r = ro.conversion.py2rpy(expr_df)

    # 3. Run xCell.
    xcell = importr("xCell")
    xcell2 = importr("xCell2")
    
    scores_xcell = ro.r('as.data.frame')(xcell.xCellAnalysis(expr_r, rnaseq = True))
    
    ro.r('data(PanCancer.xCell2Ref, package = "xCell2")')
    ro.r('data(TMECompendium.xCell2Ref, package = "xCell2")')
    pan_cancer_reference = ro.r('PanCancer.xCell2Ref')
    tme_ref = ro.r('TMECompendium.xCell2Ref')
    r_data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer = ro.r('as.data.frame')(
        xcell2.xCell2Analysis(mix = expr_r, xcell2object = pan_cancer_reference)
    )
    scores_xcell2 = ro.r('as.data.frame')(xcell2.xCell2Analysis(mix = expr_r, xcell2object = tme_ref))
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        scores_df_xcell = ro.conversion.rpy2py(scores_xcell)
        data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer = ro.conversion.rpy2py(r_data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer)
        scores_df_xcell2 = ro.conversion.rpy2py(scores_xcell2)
        
    scores_df_xcell.columns = expr_df.columns
    data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer.columns = expr_df.columns
    scores_df_xcell2.columns = expr_df.columns
        
    ro.r("library(xCell); data(xCell.data)")
    cell_types_xcell = [str(numpy_string_representing_cell_type) for numpy_string_representing_cell_type in ro.r("rownames(xCell.data$spill$K)")]
    # See https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1349-1#Sec24 .
    
    cell_types_xcell = cell_types_xcell + ["ImmuneScore", "StromaScore", "MicroenvironmentScore"]
    # See line 219 of https://github.com/dviraran/xCell/blob/master/R/xCell.R .
    
    get_signatures = ro.r('xCell2::getSignatures')
    names_fn = ro.r['names']
    pan_cancer_cell_types = names_fn(get_signatures(pan_cancer_reference))
    cell_types_r = names_fn(get_signatures(tme_ref))
    with localconverter(ro.default_converter):
        pan_cancer_cell_types_raw = list(map(str, ro.conversion.rpy2py(pan_cancer_cell_types)))
        cell_types_raw = list(map(str, ro.conversion.rpy2py(cell_types_r)))
        list_of_pan_cancer_cell_types = []
        cell_types_xcell2 = []
        for s in pan_cancer_cell_types_raw:
            cell_type = s.split("#")[0]
            if cell_type not in list_of_pan_cancer_cell_types:
                list_of_pan_cancer_cell_types.append(cell_type)
        for s in cell_types_raw:
            cell_type = s.split("#")[0]
            if cell_type not in cell_types_xcell2:
                cell_types_xcell2.append(cell_type)

    scores_df_xcell = scores_df_xcell.T
    scores_df_xcell.index.name = "SampleID"
    data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer = data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer.T
    data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer.index.name = "SampleID"
    scores_df_xcell2 = scores_df_xcell2.T
    scores_df_xcell2.index.name = "SampleID"

    scores_df_xcell.columns = cell_types_xcell
    scores_df_xcell.to_csv(paths.data_frame_of_scores_by_sample_and_cell_type, index_label = "SampleID")
    
    data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer.columns = list_of_pan_cancer_cell_types
    path_pan_cancer = paths.data_frame_of_scores_by_sample_and_cell_type.with_name(f"{paths.data_frame_of_scores_by_sample_and_cell_type.stem}_pan_cancer{paths.data_frame_of_scores_by_sample_and_cell_type.suffix}")
    data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer.to_csv(path_pan_cancer)
    
    scores_df_xcell2.columns = cell_types_xcell2
    path_tme = paths.data_frame_of_scores_by_sample_and_cell_type.with_name(f"{paths.data_frame_of_scores_by_sample_and_cell_type.stem}_tme{paths.data_frame_of_scores_by_sample_and_cell_type.suffix}")
    scores_df_xcell2.to_csv(path_tme)
    
    logger.info("Full xCell score matrix was written to %s.", paths.data_frame_of_scores_by_sample_and_cell_type)
    logger.info("Full enrichment score matrix per xCell2 and Pan Cancer was written to %s.", path_pan_cancer)
    logger.info("Full enrichment score matrix per xCell2 and TME Compendium was written to %s.", path_tme)

    panel_cols = [c for c in FOCUSED_XCELL_PANEL if c in scores_df_xcell.columns]
    missing = sorted(set(FOCUSED_XCELL_PANEL) - set(panel_cols))
    if missing:
        logger.warning(
            "Focused panel is missing columns in the following list. [%s]",
            ", ".join(sorted(missing))
        )

    focused_df = scores_df_xcell[panel_cols]
    focused_df.to_csv(paths.focused_data_frame_of_scores_by_sample_and_cell_type)
    
    logger.info("Focused panel was written to %s", paths.focused_data_frame_of_scores_by_sample_and_cell_type)

    return True
    
    
def safe_r_names(r_call_result) -> list[str]:
    '''
    Return row/col names or an empty list when the R value is NULL.
    '''
    return [] if isinstance(r_call_result, NULLType) else list(map(str, r_call_result))


if __name__ == "__main__":
    main()