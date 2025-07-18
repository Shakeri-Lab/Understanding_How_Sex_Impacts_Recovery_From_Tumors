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
from src.immune_analysis.data_loading import identify_melanoma_samples, load_melanoma_data, load_rnaseq_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Activate converters for rpy2.
numpy2ri.activate()
pandas2ri.activate()


# Define a list of standard xCell cell types in their typical output order.
XCELL_CELL_TYPES_ORDERED = [
    'Adipocytes', 'Astrocytes', 'B-cells', 'Basophils', 'CD4+ memory T-cells',
    'CD4+ naive T-cells', 'CD4+ T-cells', 'CD4+ Tcm', 'CD4+ Tem', 'CD8+ naive T-cells',
    'CD8+ T-cells', 'CD8+ Tcm', 'CD8+ Tem', 'Chondrocytes', 'Class-switched memory B-cells',
    'CLP', 'CMP', 'cDC', 'DC', 'Endothelial cells',
    'Eosinophils', 'Epithelial cells', 'Erythrocytes', 'Fibroblasts', 'GMP',
    'HSC', 'iDC', 'Keratinocytes', 'ly Endothelial cells', 'Macrophages',
    'Macrophages M1', 'Macrophages M2', 'Mast cells', 'Megakaryocytes', 'Memory B-cells',
    'MEP', 'Mesangial cells', 'Monocytes', 'MPP', 'mv Endothelial cells',
    'naive B-cells', 'Neutrophils', 'NK cells', 'NKT', 'Osteoblast',
    'pDC', 'Pericytes', 'Plasma cells', 'Platelets', 'Preadipocytes',
    'pro B-cells', 'Sebocytes', 'Skeletal muscle', 'Smooth muscle', 'Tgd cells',
    'Th1 cells', 'Th2 cells', 'Tregs', 'aDC', 'Neurons',
    'Hepatocytes', 'MSC', 'common myeloid progenitor', 'melanocyte', 'ImmuneScore',
    'StromaScore', 'MicroenvironmentScore'
]


def clean_expression_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    - 1. Strip versions (e.g., ".15") from Ensembl IDs, resulting in Ensembl IDs like ENSG00000000003.
    - 2. Convert Ensembl IDs to HGNC symbols.
    - 3. Keep rows with HGNC symbols and make the index the HGNC symbols.
    - 4. Drop duplicates / unmapped rows and convert to numeric.
    
    This function returns a tidy gene by sample matrix ready to be analyzed by xCell.
    '''

    df.index = df.index.astype(str)

    # 1. Strip versions from Ensembl IDs.
    df.index = df.index.str.replace(r"\.\d+$", "", regex = True)

    # 2. Convert Ensembl IDs to HGNC symbols.
    #ro.r("library(org.Hs.eg.db)")
    ro.r("suppressPackageStartupMessages(library(org.Hs.eg.db))")
    org_Hs_eg_db = ro.r("org.Hs.eg.db")
    annotation_dbi = importr("AnnotationDbi")

    keys = df.index.unique().to_list()

    with localconverter(ro.default_converter + pandas2ri.converter):
        res_r = annotation_dbi.select(
            org_Hs_eg_db,
            keys = StrVector(keys),
            columns = StrVector(["SYMBOL"]),
            keytype = "ENSEMBL"
        )
        res_df: pd.DataFrame = ro.conversion.rpy2py(res_r)

    # TODO: Evaluate selecting first row of multiple rows per Ensembl ID.
    
    logger.info("First row of multiple rows per Ensembl ID will be selected.")

    res_df = (res_df[["ENSEMBL", "SYMBOL"]]
        .dropna()
        .query("SYMBOL != ''")
        .drop_duplicates("ENSEMBL")
    )
    ens2sym = dict(
        zip(
            res_df["ENSEMBL"],
            res_df["SYMBOL"])
    )

    # 3. Keep rows with HGNC symbols and make the index the HGNC symbols.
    df = df.loc[df.index.isin(ens2sym)].copy()
    df.index = df.index.map(ens2sym.get)

    # 4. Drop duplicates / unmapped rows and convert to numeric.
    df = df[~df.index.duplicated(keep = "first")]          # drop duplicate symbols
    
    # TODO: Don't coerce.
    
    df = df.apply(pd.to_numeric, errors = "coerce").fillna(0.0)

    return df


def determine_metastatic_status(site):
    '''
    Determine if a specimen site is part of metastatic disease.
    '''
    if pd.isna(site) or site is None:
        return None
    site_lower = str(site).lower()
    # Add metastatic status based on specimen site information.
    # Define keywords that might indicate metastatic sites.
    metastatic_keywords = ['metast', 'lymph node', 'brain', 'lung', 'liver', 'distant']
    if any(keyword in site_lower for keyword in metastatic_keywords):
        return True
    primary_sites = ['skin', 'cutaneous', 'dermal', 'epidermis', 'primary'] # `primary_sites` is a list of common primary melanoma sites.
    if any(keyword in site_lower for keyword in primary_sites):
        return False
    return None # It is unknown / uncertain that the site is part of metastatic disease.


def main():
    
    paths.ensure_dependencies_for_microenv_exist()
    
    process_melanoma_immune_data()

    clinical_data_processed = pd.read_csv(paths.data_frame_of_melanoma_patient_and_sequencing_data)
    
    logger.info(f"Clinical data for {clinical_data_processed['PATIENT_ID'].nunique()} unique melanoma patients was loaded from {paths.data_frame_of_melanoma_patient_and_sequencing_data}.")

    expr_matrix = load_rnaseq_data()
    if expr_matrix is None:
        raise Exception("Load RNA sequencing data failed.")

    success = run_xcell_analysis(expr_matrix, clinical_data_processed)

    if success:
        logger.info("xCell analysis succeeded.")
    else:
        logger.error("xCell analysis failed.")


def process_melanoma_immune_data():
    """
    Process melanoma RNA sequencing data to extract immune Micro-Environment information.
    TODO: What is immune Micro-Environment information?
        
    Returns:
    --------
    immune_clinical: pd.DataFrame -- data frame with melanoma samples and immune features
    TODO: Describe the data frame with melanoma samples and immune features.
    """

    immune_df, clinical_data = load_melanoma_data()
    
    if immune_df is None or clinical_data is None:
        logger.error("Failed to load immune or clinical data.")
        return None

    logger.info("Sample details will be collected.")
    
    melanoma_slids, sample_details = identify_melanoma_samples(clinical_data)
    
    logger.info(f"Sample details for {len(sample_details)} samples were retrieved.")

    logger.info(f"The shape of the immune dataframe is {immune_df.shape}.")
    logger.info(f"A sublist of the list of columns of the immune dataframe is [{immune_df.columns[:5]}...].")

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

    clinical_cols = ["PATIENT_ID", "Sex", "Race", "AgeAtClinicalRecordCreation", "EarliestMelanomaDiagnosisAge", "HAS_ICB", "ICB_START_AGE", "STAGE_AT_ICB"]
    available = [c for c in clinical_cols if c in clinical_data.columns]

    immune_clinical = sample_rows.merge(clinical_data[available], on = "PATIENT_ID", how = "left").set_index("SLID")
    
    logger.info("Per-sample data-frame shape: %s", immune_clinical.shape)

    # Merge clinical info with immune data
    available_cols = [col for col in clinical_cols if col in clinical_data.columns]
    if len(available_cols) < len(clinical_cols):
        missing_cols = set(clinical_cols) - set(available_cols)
        
        logger.warning(f"Missing clinical columns: {missing_cols}")

    # Add a column of indicators that corresponding specimen sites are part of metastatic disease.
    immune_clinical['IsMetastatic'] = immune_clinical['SpecimenSite'].apply(determine_metastatic_status)
    
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
    xcell  = importr("xCell")
    scores_r = xcell.xCellAnalysis(expr_r, rnaseq = True) # scores_r is a matrix of cell type and sample information.
    
    logger.info("HGNC symbols that are elements of the index of the matrix of gene and sample information were matched with gene sets.")

    # Convert `scores_r` to `scores_np`.
    with localconverter(ro.default_converter + pandas2ri.converter):
        scores_np = ro.conversion.rpy2py(scores_r)
    r_samp_ids = safe_r_names(ro.r("colnames")(scores_r))

    # Retrieve cell-type names (i.e., labels of rows of the R matrix).
    cell_types = safe_r_names(ro.r("rownames")(scores_r))

    if not cell_types: # Our first search for cell-types names failed.
        tmp_df_r   = ro.r("as.data.frame")(scores_r)
        cell_types = safe_r_names(ro.r("rownames")(tmp_df_r))

    if cell_types and all(ct.strip().isdigit() for ct in cell_types): # Assume xCell lost cell-type names.
        
        logger.warning("xCell returned numeric row-names; the canonical xCell cell-type list will be used.")
        
        cell_types = XCELL_CELL_TYPES_ORDERED.copy()

    if len(cell_types) != scores_np.shape[0]:
        raise Exception(f"We expected {scores_np.shape[0]} cell-type names but got {len(cell_types)}.")

    '''
    Align sample IDs.
    Prefer the order coming from xCell;
    If sizes mismatch, fall back to the cleaned expression matrix (xCell may drop duplicate / empty columns).
    '''
    if r_samp_ids and len(r_samp_ids) == scores_np.shape[1]:
        sample_ids = r_samp_ids
    else:
        sample_ids = list(expr_df.columns.astype(str))[: scores_np.shape[1]]
        logger.warning(
            "Recovered %d sample IDs from expression matrix because xCell returned %d columns.",
            len(sample_ids),
            scores_np.shape[1]
        )

    # Build the DataFrame with correct labels on both axes.
    scores_df = pd.DataFrame(
        scores_np,
        index = pd.Index(cell_types, name = "CellType"),
        columns = pd.Index(sample_ids, name="SampleID")
    )
    # Clean.
    scores_df.index = scores_df.index.str.strip()
    scores_df.columns = scores_df.columns.str.strip()

    # Transpose to a data frame of sample and cell type information.
    scores_df = scores_df.T
    scores_df.index.name = "SampleID"

    # 5 : Write results.
    scores_df.to_csv(paths.data_frame_of_scores_by_sample_and_cell_type, index_label = "SampleID")
    
    logger.info("Full xCell score matrix was written to %s.", paths.data_frame_of_scores_by_sample_and_cell_type)

    panel_cols = [c for c in FOCUSED_XCELL_PANEL if c in scores_df.columns]
    missing = sorted(set(FOCUSED_XCELL_PANEL) - set(panel_cols))
    if missing:
        logger.warning(
            "Focused panel is missing columns in the following list. [%s]",
            ", ".join(sorted(missing))
        )

    focused_df = scores_df[panel_cols]
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