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


def _load_xcell_scores() -> pd.DataFrame:
    '''
    Load matrix of samples IDs, cell types, and enrichment scores.
    '''
    df = pd.read_csv(paths.data_frame_of_scores_by_sample_and_cell_type, index_col = "SampleID")
    logger.info("Matrix of sample IDs, cell types, and enrichment scores has %d sample IDs and %d cell types.", *df.shape)
    return df


def _load_sample_level_clinical() -> pd.DataFrame:
    '''
    One row per sample with clinical & technical covariates
    (`process_melanoma_immune_data` output).
    '''
    df = pd.read_csv(paths.melanoma_sample_immune_clinical_data, index_col = "SLID")
    logger.info("Sample-level clinical data: %d samples × %d columns", *df.shape)
    return df


def _load_qc_metrics() -> pd.DataFrame:
    '''
    QC file gives sequencing depth and batch.
    If a 'Batch' column is missing, create a single-level batch.
    '''
    df = pd.read_csv(paths.QC_data).rename(columns = {"SLID": "SampleID"})
    depth_col = "TotalReads" if "TotalReads" in df else df.columns[0]
    df["SequencingDepth"] = np.log10(df[depth_col])
    if "Batch" not in df:
        df["Batch"] = "Batch1"
    qc = df[["SampleID", "Batch", "SequencingDepth"]].set_index("SampleID")
    logger.info("QC metrics: %d samples", len(qc))
    return qc


def _assemble_modelling_dataframe() -> tuple[pd.DataFrame, list[str]]:
    '''
    Join scores with covariates required for the mixed models.
    All variables are kept as generic as possible so that the script works even
    when some optional columns are missing.
    '''
    scores = _load_xcell_scores()
    xcell_cols = scores.columns.tolist()
    clin = _load_sample_level_clinical()
    qc = _load_qc_metrics()

    scores.index = scores.index.astype(str).str.strip()
    clin.index = clin.index.astype(str).str.strip()
    qc.index = qc.index.astype(str).str.strip()

    df = (
        scores
        .join(clin, how = "inner")
        .join(qc, how = "left")
    )
    df.rename(
        columns = {
            "Sex": "Sex",
            "AgeAtClinicalRecordCreation": "Age",
            "STAGE_AT_ICB": "Stage",
            "HAS_ICB": "PriorTreatment",
        },
        inplace = True
    )

    # Minimal cleaning / type coercion
    df["Sex"]  = df["Sex"].str.upper().map({"MALE": 1, "M": 1, "FEMALE": 0, "F": 0})
    df["Age"]  = pd.to_numeric(df["Age"].apply(lambda age: 90.0 if age == "Age 90 or older" else float(age)), errors = "raise")
    df["Stage"] = df["Stage"].fillna("Unknown")
    if "PriorTreatment" in df:
        df["PriorTreatment"] = df["PriorTreatment"].fillna(0).astype(int)
    else:
        df["PriorTreatment"] = 0

    # Do any rows still lack essential predictors?
    essential = ["Sex", "Age", "Stage", "PriorTreatment", "Batch", "SequencingDepth"]
    keep = df.dropna(subset = ["Sex"]) # Sex is key predictor and must be present.
    logger.info("After merges %d samples remain.", len(keep))
    missing_predictors = [c for c in essential if keep[c].isna().any()]
    if missing_predictors:
        logger.warning("Some samples missing values for %s – affected rows will be skipped", ", ".join(missing_predictors))
    return keep, xcell_cols


def _fit_models(df: pd.DataFrame, cell_types: list[str]) -> pd.DataFrame:
    '''
    Loop over cell types, fit mixed models, and collect coefficients.
    Returns tidy results.
    '''
    #cell_types = [
    #    c for c in df.columns if c not in {"Sex", "Age", "Stage", "PriorTreatment", "Batch", "SequencingDepth", "PATIENT_ID"}
    #]

    results = []
    for cell in cell_types:
        logger.info(f"Cell type is {cell}.")
        dat = df[[
            cell, "Sex", "Age", "Stage", "PriorTreatment", "Batch", "SequencingDepth", "PATIENT_ID"
        ]].rename(columns = {cell: "Score"})
        dat["Score"] = pd.to_numeric(dat["Score"], errors = "raise")
        dat = dat.dropna(subset = ["Score", "Sex"])
        if dat["Sex"].nunique() < 2:
            logger.debug("Skipping %s – only one sex present", cell)
            continue

        # Build formula; Batch as categorical, Stage as categorical
        formula = (
            f"Score ~ Sex + Age + C(Stage) + PriorTreatment + C(Batch) + SequencingDepth"
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
    
    df, xcell_cols = _assemble_modelling_dataframe()
    logging.info(f"Columns in enrichment matrix of sample IDs, cell types, and enrichment scores are {xcell_cols}.")
    res = _fit_models(df, xcell_cols)
    res = _add_fdr_adjustment(res)

    res.sort_values("Qval_Sex").to_csv(paths.mixed_model_results, index = False)
    res.query("Significant").sort_values("Qval_Sex").to_csv(paths.mixed_model_results_significant, index = False)

    logger.info("Results → %s", paths.mixed_model_results)
    logger.info("Significant (FDR < 0.10) → %s", paths.mixed_model_results_significant)
    logger.info("%d / %d cell types significant at FDR 10 %%", res["Significant"].sum(), len(res))


if __name__ == "__main__":
    main()