#!/usr/bin/env python3
'''
Usage:
./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.linear_mixed_models

`linear_mixed_models.py` fits for every patient and cell type a linear mixed model of enrichment score vs. categorical predictor sex with values 0 and 1 for female and male, continuous predictor age at clinical record creation, categorical predictor stage at start of ICB therapy, categorical predictor of whether the patient has received ICB therapy with values 0 and 1 for false and true, categorical predictor batch, and logarithm of total number of RNA fragments in a sample. Batch is a label that tags every sample with the corresponding batch of samples that were prepared for sequencing by fragmenting the RNA of the samples. Including batch as a predictor accounts for systematic differences between batches.

Let a null hypothesis be that the coefficient for Sex is 0 for the best linear mixed model with our variables and many samples. A p value for Sex is a probability of observing a coefficient for Sex in a linear mixed model with our variables that is at least as large in magnitude as the estimated coefficient for Sex when assuming the null hypothesis is true. A small p value indicates that the coefficient for Sex for the best linear mixed model with our variables and many samples is unlikely to be 0. p values for Sex are adjusted across all 64 cell types into q values with the Benjamini-Hochberg procedure and a False Discovery Rate of 10%.

Outputs
-------
1. `mixed_model_results.csv` – one row per cell type with parameter for Sex, standard error of parameter for sex, p value for Sex, number of patients, q value for Sex, and an indicator that the best linear mixed model for that cell type with our variables and many samples has a coefficient of Sex that is not 0.
2. `mixed_model_results_significant.csv` – a CSV file of rows of `mixed_model_results.csv` where the best linear mixed model for the corresponding cell type with our variables and many samples has a coefficient of Sex that is not 0.
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


def create_data_frame_of_enrichment_scores_clinical_data_and_QC_data(path) -> tuple[pd.DataFrame, list[str]]:
    
    data_frame_of_enrichment_scores = pd.read_csv(path)
    list_of_cell_types = data_frame_of_enrichment_scores.columns.tolist()[1:-3]
    logger.info(f"Data frame of sample IDs, cell types, and enrichment scores has {len(data_frame_of_enrichment_scores)} sample IDs and {len(list_of_cell_types)} cell types.")
    
    data_frame_of_sample_IDs_and_clinical_data = pd.read_csv(paths.melanoma_sample_immune_clinical_data).rename(columns = {"SLID": "SampleID"})
    logger.info("Data frame of sample IDs and clinical data has %d samples and %d columns.", *data_frame_of_sample_IDs_and_clinical_data.shape)
    
    missing = set(data_frame_of_enrichment_scores.SampleID) - set(data_frame_of_sample_IDs_and_clinical_data.SampleID)
    logger.info(f"{len(missing)} enrichment rows with the following sample IDs lack clinical data: {sorted(missing)}")
    
    QC_data = pd.read_csv(paths.QC_data).rename(columns = {"SLID": "SampleID"})
    QC_data["SequencingDepth"] = np.log10(QC_data["TotalReads"])
    QC_data = QC_data[["SampleID", "NexusBatch", "SequencingDepth"]]
    logger.info("QC data has %d samples.", len(QC_data))

    df = data_frame_of_enrichment_scores.merge(
        data_frame_of_sample_IDs_and_clinical_data,
        how = "inner",
        left_on = "SampleID",
        right_on = "SampleID"
    ) # Enrichment scores and sexes must both be present.
    
    logger.info("After merging, data frame of enrichment scores and clinical data has %d samples.", len(df))
    
    df = df.merge(QC_data, how = "left", left_on = "SampleID", right_on = "SampleID")
    df["Sex"]  = df["Sex"].str.upper().map({"FEMALE": 0, "MALE": 1})
    df["AgeAtClinicalRecordCreation"]  = pd.to_numeric(
        df["AgeAtClinicalRecordCreation"].apply(
            lambda age: 90.0 if age == "Age 90 or older" else float(age)
        ),
        errors = "raise"
    )
    df["STAGE_AT_ICB"] = df["STAGE_AT_ICB"].fillna("Unknown")
    df["HAS_ICB"] = df["HAS_ICB"].fillna(0).astype(int)
    df = df.dropna(subset = ["Sex"]) # Sex is the key predictor and must be present.
    
    logger.info("After merging, data frame of enrichment scores, clinical data, and QC data has %d samples.", len(df))
    
    list_of_essential_predictors = ["Sex", "AgeAtClinicalRecordCreation", "STAGE_AT_ICB", "HAS_ICB", "NexusBatch", "SequencingDepth"]
    for predictor in list_of_essential_predictors:
        if df[predictor].empty:
            raise Exception(f"Column {predictor} is empty.")
        if df[predictor].isna().any():
            raise Exception(f"Column {predictor} has some values of NA for samples.")

    return df, list_of_cell_types


def fit_linear_mixed_models(df: pd.DataFrame, list_of_cell_types: list[str]) -> pd.DataFrame:
    list_of_results = []
    for cell_type in list_of_cell_types:
        logger.info(f"A linear mixed model will be fit for cell type {cell_type}.")
        
        data_frame = df[[
            cell_type, "Sex", "AgeAtClinicalRecordCreation", "STAGE_AT_ICB", "HAS_ICB", "NexusBatch", "SequencingDepth", "PATIENT_ID"
        ]].rename(columns = {cell_type: "Score"})
        data_frame["Score"] = pd.to_numeric(data_frame["Score"], errors = "raise")
        data_frame = data_frame.dropna(subset = ["Score"])
        if data_frame["Sex"].nunique() < 2:
            raise Exception(f"Only sex {data_frame['Sex'].iloc[0]} is present after dropping enrichment scores of NA for cell type {cell_type}.")

        '''
        # 1. Within-patient replication
        replication_counts = data_frame.groupby("PATIENT_ID").size()
        frac_singletons     = (replication_counts <= 1).mean()
        logger.info(
            "Diagnostic - %s: %.1f%% of patients contribute only one sample.",
            cell_type, 100 * frac_singletons
        )

        # 3. Spread of the response
        score_std = data_frame["Score"].std(ddof = 0)
        logger.info("Diagnostic - %s: response std = %.4f", cell_type, score_std)

        # 4. Categorical level sparsity
        sparsity_msgs = []
        for cat in ["STAGE_AT_ICB", "NexusBatch"]:
            lvl_counts = data_frame[cat].value_counts()
            rare_lvls  = lvl_counts[lvl_counts < 3]
            if not rare_lvls.empty:
                sparsity_msgs.append(f"{cat}: {len(rare_lvls)} sparse levels")
        if sparsity_msgs:
            logger.info("Diagnostic - %s: %s", cell_type, "; ".join(sparsity_msgs))

        # 6. Scaling of numeric predictors
        scale_summary = data_frame[["AgeAtClinicalRecordCreation", "SequencingDepth"]].describe().loc[["mean", "std"]]
        logger.info("Diagnostic - %s: numeric predictor scale\n%s", cell_type, scale_summary.to_string())
        '''
        
        formula = (
            f"Score ~ Sex + AgeAtClinicalRecordCreation + C(STAGE_AT_ICB) + C(HAS_ICB) + C(NexusBatch) + SequencingDepth"
        ) # C means "make categorical".
        mixed_linear_model = smf.mixedlm(
            formula,
            data_frame,
            groups = data_frame["PATIENT_ID"]
        )
        mixed_linear_model_results_wrapper = mixed_linear_model.fit(reml = False)
        
        '''
        # 2. Random-effect variance & boundary fit indicator
        try:
            rand_var = mixed_linear_model_results_wrapper.cov_re.iloc[0, 0]
        except Exception:
            rand_var = np.nan
        logger.info("Diagnostic - %s: random-effect variance = %.6g", cell_type, rand_var)

        # 5. Design-matrix condition number (fixed effects only)
        try:
            cond_num = matrix_condition_number(mixed_linear_model.exog)
            logger.info("Diagnostic - %s: design-matrix condition number = %.2e", cell_type, cond_num)
        except Exception as exc:
            logger.info("Diagnostic - %s: could not compute condition number (%s)", cell_type, exc)

        # 7. AIC comparison with OLS (no random intercept)
        try:
            ols_res = smf.ols(formula.replace("Score ~", "Score ~"), data_frame).fit()
            logger.info(
                "Diagnostic - %s: AIC (mixed) = %.1f ; AIC (OLS) = %.1f",
                cell_type, mixed_linear_model_results_wrapper.aic, ols_res.aic
            )
        except Exception as exc:
            logger.info("Diagnostic - %s: could not fit OLS for AIC comparison (%s)", cell_type, exc)
        '''
        
        parameter_for_Sex = mixed_linear_model_results_wrapper.params["Sex"]
        standard_error_of_parameter_for_Sex = mixed_linear_model_results_wrapper.bse["Sex"]
        p_value_for_Sex = mixed_linear_model_results_wrapper.pvalues["Sex"]
        number_of_patients = len(data_frame["PATIENT_ID"].unique())
        
        list_of_results.append(
            (
                cell_type,
                parameter_for_Sex,
                standard_error_of_parameter_for_Sex,
                p_value_for_Sex,
                number_of_patients
            )
        )

    data_frame_of_results = pd.DataFrame(
        list_of_results,
        columns = [
            "cell type",
            "parameter for Sex",
            "standard error of parameter for Sex",
            "p value for Sex",
            "number of patients"
        ]
    )
    logger.info("Linear mixed models were fitted for %d cell types.", len(data_frame_of_results))
    return data_frame_of_results


def add_fdr_adjustment(data_frame_of_results: pd.DataFrame) -> pd.DataFrame:
    '''
    p values for the Sex coefficient are adjusted across all cell types with the Benjamini-Hochberg procedure and a False Discovery Rate of 10%.
    '''
    indicator_of_whether_to_reject_null_hypothesis, array_of_q_values, _, _ = multipletests(data_frame_of_results["p value for Sex"], method = "fdr_bh", alpha = 0.10)
    data_frame_of_results["q value for Sex"] = array_of_q_values
    data_frame_of_results["significant"] = indicator_of_whether_to_reject_null_hypothesis
    return data_frame_of_results


def main():
    
    paths.ensure_dependencies_for_linear_mixed_models_exist()
    
    dictionary_of_paths_of_enrichment_data_frames_and_tuples_of_paths_of_mixed_model_results = {
        paths.enrichment_data_frame_per_xCell: (
            paths.mixed_model_results_per_xCell,
            paths.mixed_model_results_significant_per_xCell
        ),
        paths.enrichment_data_frame_per_xCell2_and_Pan_Cancer: (
            paths.mixed_model_results_per_xCell2_and_Pan_Cancer,
            paths.mixed_model_results_significant_per_xCell2_and_Pan_Cancer
        ),
        paths.enrichment_data_frame_per_xCell2_and_TME_Compendium: (
            paths.mixed_model_results_per_xCell2_and_TME_Compendium,
            paths.mixed_model_results_significant_per_xCell2_and_TME_Compendium
        )
    }
    
    for path_of_enrichment_data_frame, tuple_of_paths_of_mixed_model_results in dictionary_of_paths_of_enrichment_data_frames_and_tuples_of_paths_of_mixed_model_results.items():
        path_of_mixed_model_results = tuple_of_paths_of_mixed_model_results[0]
        path_of_significant_mixed_model_results = tuple_of_paths_of_mixed_model_results[1]
        
        data_frame_of_enrichment_scores_clinical_data_and_QC_data, list_of_cell_types = create_data_frame_of_enrichment_scores_clinical_data_and_QC_data(path_of_enrichment_data_frame)

        data_frame_of_results = fit_linear_mixed_models(data_frame_of_enrichment_scores_clinical_data_and_QC_data, list_of_cell_types)
        data_frame_of_results = add_fdr_adjustment(data_frame_of_results)

        data_frame_of_results.sort_values("q value for Sex").to_csv(path_of_mixed_model_results, index = False)
        data_frame_of_results.query("significant").sort_values("q value for Sex").to_csv(path_of_significant_mixed_model_results, index = False)
        logger.info("Data frame of all results were saved to %s.", path_of_mixed_model_results)
        logger.info("Data frame of significant results were saved to %s.", path_of_significant_mixed_model_results)

        logger.info(
            "%d out of %d cell types were significant at False Discovery Rate 10%%",
            data_frame_of_results["significant"].sum(),
            len(data_frame_of_results)
        )


if __name__ == "__main__":
    main()