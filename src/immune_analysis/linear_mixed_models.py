#!/usr/bin/env python3
'''
Usage:
./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.linear_mixed_models

Mixed-effects association between xCell enrichment scores and sex

For every xCell cell-type score we fit a linear mixed model:

    score  ~  Sex
             + Age
             + Stage
             + PriorTreatment           # ICB ever / yes-no
             + C(Batch)                 # categorical
             + SequencingDepth          # log10-scaled Total reads
           +  (1 | PatientID)           # random intercept per patient

P-values for the Sex coefficient are adjusted across all cell types with the
Benjamini-Hochberg procedure (FDR 10 %).

Outputs
-------
1.  «mixed_model_results.csv» – one row per cell type with β, SE, raw- and
    FDR-adjusted p-values for Sex.
2.  «mixed_model_results_significant.csv» – subset with FDR < 0.10.

Run from the project root after xCell has finished:

    conda activate ici_sex
    ./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.mixed_models
'''

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src.config import paths


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s – %(levelname)s – %(message)s"
)
logger = logging.getLogger(__name__)


def assemble_modelling_dataframe() -> tuple[pd.DataFrame, list[str]]:
    
    data_frame_of_enrichment_scores = pd.read_csv(paths.data_frame_of_scores_by_sample_and_cell_type)
    logger.info(
        "Data frame of sample IDs, cell types, and enrichment scores has %d sample IDs and %d cell types.", 
        *data_frame_of_enrichment_scores.shape
    )
    list_of_names_of_columns_of_enrichment_scores = data_frame_of_enrichment_scores.columns.tolist()[1:]
    
    data_frame_of_sample_IDs_and_clinical_data = pd.read_csv(paths.melanoma_sample_immune_clinical_data).rename(columns = {"SLID": "SampleID"})
    logger.info("Data frame of sample IDs and clinical data has %d samples and %d columns.", *data_frame_of_sample_IDs_and_clinical_data.shape)
    
    QC_data = pd.read_csv(paths.QC_data).rename(columns = {"SLID": "SampleID"})
    QC_data["SequencingDepth"] = np.log10(QC_data["TotalReads"])
    QC_data = QC_data[["SampleID", "NexusBatch", "SequencingDepth"]]
    logger.info("QC data has %d samples.", len(QC_data))

    df = (
        data_frame_of_enrichment_scores
        .merge(data_frame_of_sample_IDs_and_clinical_data, how = "inner", left_on = "SampleID", right_on = "SampleID") # Enrichment scores and sexes must both be present.
        .merge(QC_data, how = "left", left_on = "SampleID", right_on = "SampleID")
    )

    # Clean data frame of enrichment scores, clinical data, and QC data.
    df["Sex"]  = df["Sex"].str.upper().map({"MALE": 1, "M": 1, "FEMALE": 0, "F": 0})
    df["AgeAtClinicalRecordCreation"]  = pd.to_numeric(
        df["AgeAtClinicalRecordCreation"].apply(
            lambda age: 90.0 if age == "Age 90 or older" else float(age)
        ),
        errors = "raise"
    )
    df["STAGE_AT_ICB"] = df["STAGE_AT_ICB"].fillna("Unknown")
    df["HAS_ICB"] = df["HAS_ICB"].fillna(0).astype(int)
    df = df.dropna(subset = ["Sex"]) # Sex is the key predictor and must be present.
    
    logger.info("Data frame of enrichment scores, clinical data, and QC data has %d samples.", len(df))
    
    list_of_essential_predictors = ["Sex", "AgeAtClinicalRecordCreation", "STAGE_AT_ICB", "HAS_ICB", "NexusBatch", "SequencingDepth"]
    for predictor in list_of_essential_predictors:
        if df[predictor].empty:
            raise Exception(f"Column {predictor} is empty.")
        if df[predictor].isna().any():
            raise Exception(f"Column {predictor} has some values of NA for samples.")

    return df, list_of_names_of_columns_of_enrichment_scores


def _fit_models(df: pd.DataFrame, cell_types: list[str]) -> pd.DataFrame:
    '''
    Loop over cell types, fit mixed models, and collect coefficients.
    Returns tidy results.
    '''
    results = []
    for cell in cell_types:
        logger.info(f"Cell type is {cell}.")
        dat = df[[
            cell, "Sex", "AgeAtClinicalRecordCreation", "STAGE_AT_ICB", "HAS_ICB", "NexusBatch", "SequencingDepth", "PATIENT_ID"
        ]].rename(columns = {cell: "Score"})
        dat["Score"] = pd.to_numeric(dat["Score"], errors = "raise")
        dat = dat.dropna(subset = ["Score", "Sex"])
        if dat["Sex"].nunique() < 2:
            logger.debug("Skipping %s – only one sex present", cell)
            continue

        # Build formula; Batch as categorical, Stage as categorical
        formula = (
            f"Score ~ Sex + AgeAtClinicalRecordCreation + C(STAGE_AT_ICB) + HAS_ICB + C(NexusBatch) + SequencingDepth"
        )
        model = smf.mixedlm(
            formula,
            dat,
            groups = dat["PATIENT_ID"],
            missing = "drop"
        )
        fit = model.fit(reml = False)
        beta, se, p = fit.params["Sex"], fit.bse["Sex"], fit.pvalues["Sex"]
        results.append(
            (
                cell,
                beta,
                se,
                p,
                len(dat["PATIENT_ID"].unique())
            )
        )

    res_df = pd.DataFrame(
        results,
        columns = ["CellType", "Beta_Sex", "SE_Sex", "Pval_Sex", "N_Patients"]
    )
    logger.info("Mixed models fitted for %d cell types", len(res_df))
    return res_df


def _add_fdr_adjustment(res_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Benjamini–Hochberg (FDR 10 %) on p-values for Sex.
    '''
    reject, qvals, _, _ = multipletests(res_df["Pval_Sex"], method = "fdr_bh", alpha = 0.10)
    res_df["Qval_Sex"] = qvals
    res_df["Significant"] = reject
    return res_df


def main():
    
    paths.ensure_dependencies_for_linear_mixed_models_exist()
    
    df, list_of_names_of_columns_of_enrichment_scores = assemble_modelling_dataframe()
    logging.info(f"Columns in enrichment matrix of sample IDs, cell types, and enrichment scores are {list_of_names_of_columns_of_enrichment_scores}.")
    res = _fit_models(df, list_of_names_of_columns_of_enrichment_scores)
    res = _add_fdr_adjustment(res)

    res.sort_values("Qval_Sex").to_csv(paths.mixed_model_results, index = False)
    res.query("Significant").sort_values("Qval_Sex").to_csv(paths.mixed_model_results_significant, index = False)

    logger.info("Results → %s", paths.mixed_model_results)
    logger.info("Significant (FDR < 0.10) → %s", paths.mixed_model_results_significant)
    logger.info("%d / %d cell types significant at FDR 10 %%", res["Significant"].sum(), len(res))


if __name__ == "__main__":
    main()