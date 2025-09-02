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
    if "gene_id" in df.columns:
        df = df.set_index("gene_id")
    df.index = df.index.astype(str)
    dictionary_of_Ensembl_IDs_and_HGNC_symbols = (
        pd.read_csv(
            paths.series_of_Ensembl_IDs_and_HGNC_symbols,
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

    #expr_matrix, clinical_data = load_RNA_sequencing_data_for_melanoma_samples()
    #process_melanoma_immune_data(clinical_data)
    expr_matrix = pd.read_csv(paths.melanoma_expression_matrix)

    success = run_xcell_analysis(expr_matrix)
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

    logger.info("Sample details will be collected.")
    
    melanoma_slids, sample_details = identify_melanoma_samples(clinical_data)
    
    logger.info(f"Sample details for {len(sample_details)} samples were retrieved.")

    # Build a tidy "one row per tumour sample" data-frame from `sample_details`.
    sample_rows = (
        pd.DataFrame.from_dict(sample_details, orient = "index")
        .rename_axis("SLID")
        .reset_index()
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

    clinical_cols = [
        "PATIENT_ID",
        "Sex",
        "Race",
        "AgeAtClinicalRecordCreation",
        "EarliestMelanomaDiagnosisAge",
        "HAS_ICB",
        "ICB_START_AGE",
        "STAGE_AT_ICB",
        "Primary/Met"
    ]
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
        immune_clinical[["EarliestMelanomaDiagnosisAge", "STAGE_AT_ICB"]].isna().mean().rename(lambda x: f"{x}_null_fraction")
    )
    
    immune_clinical = immune_clinical.reset_index()
    dupe_mask = immune_clinical["SLID"].duplicated(keep = False)
    if dupe_mask.any():
        duplicate_rows = immune_clinical.loc[dupe_mask].copy()
        n_groups = duplicate_rows["SLID"].nunique()
        n_all_germline = duplicate_rows.groupby("SLID")["Primary/Met"].apply(
            lambda s: all(metastatic_from_primary_met(v) is None for v in s)
        ).sum()
        logger.info("Found %d SLIDs with multiple clinical matches; %d groups are all germline/NA.", n_groups, n_all_germline)
        immune_clinical["keep_rank"] = immune_clinical["Primary/Met"].apply(
            lambda v: 0 if metastatic_from_primary_met(v) is not None else 1
        )
        immune_clinical = (
            immune_clinical
            .sort_values(["SLID", "keep_rank"])
            .drop_duplicates(subset = ["SLID"], keep = "first")
            .drop(columns = "keep_rank")
        )
    else:
        logger.info("No duplicate SLIDs after merge.")
    immune_clinical = immune_clinical.set_index("SLID")

    immune_clinical.to_csv(paths.melanoma_sample_immune_clinical_data)
    
    logger.info(f"Saved data frame `immune_clinical` to {paths.melanoma_sample_immune_clinical_data}.")

    # Create a summary of biopsy locations.
    site_summary = immune_clinical["SpecimenSite"].value_counts().reset_index()
    site_summary.columns = ["SpecimenSite", "Count"]
    site_summary.to_csv(paths.map_of_biopsy_locations_to_counts, index = False)
    
    logger.info(f"Map of biopsy locations to counts was saved to {paths.map_of_biopsy_locations_to_counts}.")

    # Create a summary of procedure types.
    proc_summary = (
        immune_clinical["ProcedureType"]
        .dropna()
        .str
        .split('|')
        .explode()
        .str
        .strip()
        .value_counts()
        .reset_index(name = "Count")
        .rename(columns = {"index": "ProcedureType"})
    )
    proc_summary.to_csv(paths.map_of_procedure_types_to_counts, index = False)
    
    logger.info(f"Map of procedure type to counts was saved to {paths.map_of_procedure_types_to_counts}.")

    # Create a summary of metastatic status.
    meta_summary = immune_clinical['IsMetastatic'].value_counts().reset_index()
    meta_summary.columns = ['IsMetastatic', 'Count']
    meta_summary.to_csv(paths.map_of_indicators_that_specimens_are_part_of_metastatic_disease_to_counts, index = False)
    
    logger.info(f"Map of indicators that specimens are part of metastatic disease to counts was saved to {paths.map_of_indicators_that_specimens_are_part_of_metastatic_disease_to_counts}.")

    return immune_clinical


def run_xcell_analysis(expr_df: pd.DataFrame) -> bool:
    '''
    Run xCell on a gene-by-sample matrix.
    Remove cell types that are not present in any sample based on significance analysis.
    Write enrichment matrices.
    '''

    print(expr_df.shape)

    expr_df = clean_expression_df(expr_df)
    
    logger.info(
        "The matrix of gene and sample information after cleaning has %s genes and %s samples.",
        expr_df.shape[0],
        expr_df.shape[1]
    )
    logger.info("The matrix of gene and sample information will be converted to R.")

    with localconverter(ro.default_converter + pandas2ri.converter):
        expression_data_frame = ro.conversion.py2rpy(expr_df)

    xCell = importr("xCell")
    xCell2 = importr("xCell2")
    
    r_data_frame_of_enrichment_scores_per_xCell = ro.r('as.data.frame')(xCell.xCellAnalysis(expression_data_frame, rnaseq = True))
    ro.r('data(PanCancer.xCell2Ref, package = "xCell2")')
    ro.r('data(TMECompendium.xCell2Ref, package = "xCell2")')
    reference_Pan_Cancer = ro.r('PanCancer.xCell2Ref')
    reference_TME_Compendium = ro.r('TMECompendium.xCell2Ref')
    r_data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer = ro.r('as.data.frame')(
        xCell2.xCell2Analysis(mix = expression_data_frame, xcell2object = reference_Pan_Cancer)
    )
    r_data_frame_of_enrichment_scores_per_xCell_and_TME_Compendium = ro.r('as.data.frame')(
        xCell2.xCell2Analysis(mix = expression_data_frame, xcell2object = reference_TME_Compendium)
    )
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        data_frame_of_enrichment_scores_per_xCell = ro.conversion.rpy2py(r_data_frame_of_enrichment_scores_per_xCell)
        data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer = ro.conversion.rpy2py(r_data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer)
        data_frame_of_enrichment_scores_per_xCell2_and_reference_TME_Compendium = ro.conversion.rpy2py(r_data_frame_of_enrichment_scores_per_xCell_and_TME_Compendium)
        
    data_frame_of_enrichment_scores_per_xCell.columns = expr_df.columns
    data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer.columns = expr_df.columns
    data_frame_of_enrichment_scores_per_xCell2_and_reference_TME_Compendium.columns = expr_df.columns
        
    ro.r("library(xCell)")
    ro.r("data(xCell.data)")
    list_of_cell_types_per_xCell = [
        str(numpy_string_representing_cell_type)
        for numpy_string_representing_cell_type
        in ro.r("rownames(xCell.data$spill$K)")
    ] # See https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1349-1#Sec24 .
    list_of_composite_scores = ["ImmuneScore", "StromaScore", "MicroenvironmentScore"]
    list_of_cell_types_and_composite_scores_per_xCell = list_of_cell_types_per_xCell + list_of_composite_scores # See line 219 of https://github.com/dviraran/xCell/blob/master/R/xCell.R .
    
    function_get_signatures = ro.r('xCell2::getSignatures')
    function_names = ro.r['names']
    array_of_cell_types_per_xCell2_and_Pan_Cancer = function_names(function_get_signatures(reference_Pan_Cancer))
    array_of_cell_types_per_xCell2_and_TME_Compendium = function_names(function_get_signatures(reference_TME_Compendium))
    with localconverter(ro.default_converter):
        list_of_raw_cell_types_per_xCell2_and_Pan_Cancer = list(
            map(
                str,
                ro.conversion.rpy2py(array_of_cell_types_per_xCell2_and_Pan_Cancer)
            )
        )
        list_of_raw_cell_types_per_xCell2_and_TME_Compendium = list(
            map(
                str,
                ro.conversion.rpy2py(array_of_cell_types_per_xCell2_and_TME_Compendium)
            )
        )
        list_of_cell_types_per_xCell2_and_Pan_Cancer = []
        list_of_cell_types_per_xCell2_and_TME_Compendium = []
        for raw_cell_type in list_of_raw_cell_types_per_xCell2_and_Pan_Cancer:
            cell_type = raw_cell_type.split("#")[0]
            if cell_type not in list_of_cell_types_per_xCell2_and_Pan_Cancer:
                list_of_cell_types_per_xCell2_and_Pan_Cancer.append(cell_type)
        for raw_cell_type in list_of_raw_cell_types_per_xCell2_and_TME_Compendium:
            cell_type = raw_cell_type.split("#")[0]
            if cell_type not in list_of_cell_types_per_xCell2_and_TME_Compendium:
                list_of_cell_types_per_xCell2_and_TME_Compendium.append(cell_type)

    data_frame_of_enrichment_scores_per_xCell = data_frame_of_enrichment_scores_per_xCell.T
    data_frame_of_enrichment_scores_per_xCell.index.name = "SampleID"
    data_frame_of_enrichment_scores_per_xCell.columns = list_of_cell_types_and_composite_scores_per_xCell
    data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer = data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer.T
    data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer.index.name = "SampleID"
    data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer.columns = list_of_cell_types_per_xCell2_and_Pan_Cancer
    data_frame_of_enrichment_scores_per_xCell2_and_reference_TME_Compendium = data_frame_of_enrichment_scores_per_xCell2_and_reference_TME_Compendium.T
    data_frame_of_enrichment_scores_per_xCell2_and_reference_TME_Compendium.index.name = "SampleID"
    data_frame_of_enrichment_scores_per_xCell2_and_reference_TME_Compendium.columns = list_of_cell_types_per_xCell2_and_TME_Compendium

    data_frame_of_enrichment_scores_per_xCell_without_composite_scores = data_frame_of_enrichment_scores_per_xCell[list_of_cell_types_per_xCell]
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data_frame_of_enrichment_scores_per_xCell_without_composite_scores = ro.conversion.py2rpy(data_frame_of_enrichment_scores_per_xCell_without_composite_scores.T)
    
    significance_function = xCell.xCellSignifcanceBetaDist
    p_values = significance_function(r_data_frame_of_enrichment_scores_per_xCell_without_composite_scores)
    r_data_frame_of_p_values = ro.r('as.data.frame')(p_values)
    with localconverter(ro.default_converter + pandas2ri.converter):
        data_frame_of_p_values = ro.conversion.rpy2py(r_data_frame_of_p_values)
    data_frame_of_p_values.index = list_of_cell_types_per_xCell
    data_frame_of_p_values.columns = expr_df.columns
    significance_level = 0.05
    list_of_detected_columns = data_frame_of_p_values.index[(data_frame_of_p_values < significance_level).any(axis = 1)].tolist()

    list_of_columns_to_keep = list_of_detected_columns + list_of_composite_scores
    list_of_columns_to_drop = sorted(set(data_frame_of_enrichment_scores_per_xCell.columns) - set(list_of_columns_to_keep))
    if list_of_columns_to_drop:
        logger.info("%d cell types not detected in any sample will be removed from the list [%s]", len(list_of_columns_to_drop), ", ".join(list_of_columns_to_drop))
    
    data_frame_of_enrichment_scores_per_xCell = data_frame_of_enrichment_scores_per_xCell[list_of_columns_to_keep]

    data_frame_of_enrichment_scores_per_xCell.to_csv(paths.enrichment_data_frame_per_xCell)
    data_frame_of_enrichment_scores_per_xCell2_and_reference_Pan_Cancer.to_csv(
        paths.enrichment_data_frame_per_xCell2_and_Pan_Cancer
    )
    data_frame_of_enrichment_scores_per_xCell2_and_reference_TME_Compendium.to_csv(
        paths.enrichment_data_frame_per_xCell2_and_TME_Compendium
    )
    
    logger.info(
        "Full xCell score matrix was written to %s.",
        paths.enrichment_data_frame_per_xCell
    )
    logger.info(
        "Full enrichment score matrix per xCell2 and Pan Cancer was written to %s.", 
        paths.enrichment_data_frame_per_xCell2_and_Pan_Cancer
    )
    logger.info(
        "Full enrichment score matrix per xCell2 and TME Compendium was written to %s.",
        paths.enrichment_data_frame_per_xCell2_and_TME_Compendium
    )

    panel_cols = [c for c in FOCUSED_XCELL_PANEL if c in data_frame_of_enrichment_scores_per_xCell.columns]
    missing = sorted(set(FOCUSED_XCELL_PANEL) - set(panel_cols))
    if missing:
        logger.warning(
            "Focused panel is missing columns in the following list. [%s]",
            ", ".join(sorted(missing))
        )
    focused_df = data_frame_of_enrichment_scores_per_xCell[panel_cols]
    focused_df.to_csv(paths.focused_enrichment_data_frame)
    
    logger.info("Focused panel was written to %s", paths.focused_enrichment_data_frame)

    return True


if __name__ == "__main__":
    main()