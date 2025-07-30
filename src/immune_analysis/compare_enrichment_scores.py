#!/usr/bin/env python3
'''
This script compares enrichment scores produced by xCell or xCell2 for
- tumors of females (0) and males (1),
- ICB-naive tumors (0) and ICB-experienced (1) tumors of females, and
- ICB-naive tumors and ICB-experienced tumors of males and
various cell types.
This script for each family of tumors and cell type
performs a 2 sided Mann-Whitney U Test / Wilcoxon Rank Sum Test.

This script optionally removes variability due to age at clinical record creation and stage at start of ICB therapy before comparing.
This script optionally isolates differences between samples of patients with different sexes or
between samples for patients of a certain sex with different experiences with ICB therapy
by discounting age at clinical record creation and stage at start of ICB therapy.

A Mann-Whitney U Test / Wilcoxon Rank Sum Test tests a null hypothesis that
2 independent samples come from the same continuous distribution.
The test compares ranks.
TODO: What does it mean for 2 independent samples to come from the same continuous distribution?
TODO: What is a rank?
The test provides a U test statistic equal to
the number of times a value in a group A precedes a value in a group B
when all observations are ranked together.
TODO: What does "precedes" mean?
TODO: What does "ranked" mean?
The test also provides a p value associated with the U statistic.
This script adjusts p values into False Discovery Rates with the Benjamini-Hochberg procedure.
A Benjamini-Hochberg adjusted False Discovery Rate is a p value multiple by a number of hypotheses, divided by the index of the p value in ascending list of p values, clipped to 1, and "monotone-decreasingly corrected".
TODO: What does "monotone-decreasingly corrected" mean?

This script outputs 1 CSV filer per comparison that flags sigificant and suggestive cell types.

TODO: What is a p value?
TODO: What does significant mean?
TODO: What does suggestive mean?

Usage
-----
conda activate ici_sex
./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.compare_enrichment_scores --adjust-covariates --matrix xCell

Pass flag `--adjust-covariates` to remove variability due to age at clinical record creation and stage at start of ICB therapy before comparing.
Otherwise raw enrichment scores are used.

Outputs
-------
icb_wilcoxon_female.csv
icb_wilcoxon_male.csv
sex_wilcoxon.csv

Each file contains one row per cell type with columns
    - n_F and n_M or n_Naive and n_Exp containing
      numbers of samples corresponding to female and male patients or
      numbers of ICB-naive and ICB-experienced samples,
    - U-stat containing U statistics,
    - p value containing p values,
    - FDR containing Benjamini Hochberg adjusted p values / False Discovery Rates,
    - significant containing indicators of significance / FDR less than or equal to 0.05, and
    - suggestive containing indicators of suggestiveness / FDR between 0.05 and 0.20.
'''

from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd
import re
import scipy.stats as ss
from src.config import paths
from src.immune_analysis.linear_mixed_models import (
    create_data_frame_of_enrichment_scores_clinical_data_and_QC_data,
)
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s – %(levelname)s – %(message)s"
)
logger = logging.getLogger(__name__)


def bh_fdr(pvals: pd.Series) -> pd.Series:
    '''
    Provide a series of False Discovery Rates based on a series of p values.
    '''
    _, q, _, _ = multipletests(pvals, method = "fdr_bh")
    return pd.Series(q, index = pvals.index)


def make_result_table(stat_df: pd.DataFrame, family: str, out_path: Path) -> None:
    '''
    Save a data frame with added columns FDR, significant, and suggestive.
    Log numbers of cell types that are significant / suggestive for each family of tests.
    '''
    stat_df["FDR"] = bh_fdr(stat_df["pval"])
    stat_df["significant"] = stat_df["FDR"] <= 0.05
    stat_df["suggestive"] = (stat_df["FDR"] > 0.05) & (stat_df["FDR"] <= 0.20)
    stat_df.sort_values("FDR").to_csv(out_path, index = False)
    logger.info(
        "[%s] %d/%d significant (FDR ≤ 0.05); %d additional suggestive (0.05 < FDR ≤ 0.20)",
        family,
        stat_df["significant"].sum(),
        len(stat_df),
        stat_df["suggestive"].sum()
    )


def wilcoxon_table(
    df: pd.DataFrame,
    list_cell_types: list[str],
    group_var: str,
    group_a,
    group_b,
    adjust_covariates: bool
) -> pd.DataFrame:
    '''
    Perform Mann-Whitney U Tests / Wilcoxon Rank Sum Tests for every cell type.
    If covariates will be adjusted, residuals of a linear model of enrichment scores
    vs. age at clinical record creation and stage at start of ICB therapy will be compared
    instead of raw enrichment scores.
    
    Returns
    -------
    a data frame with 
    - numbers of samples corresponding to female and male patients or
    - numbers of ICB-naive and ICB-experienced samples,
    - U statistic, and
    - p value

    For each cell type,
    Choose either the raw enrichment scores for that cell type or the residuals of a linear model of
    enrichment score vs. age at clinical record creation and stage at start of ICB therapy.
    In the first case, split enrichment scores by sex or experience of ICB therapy.
    In the second case, split residuals.
    Find a U statistic and a p value using the Mann–Whitney U Test / Wilcoxon Rank Sum Test.
    Record in a row corresponding to a cell type of a data frame
    - numbers of samples corresponding to female and male patients or
    - numbers of ICB-naive and ICB-experienced samples,
    - U statistic, and
    - p value.
    '''
    rows = []
    mask = df[group_var].isin([group_a, group_b]).copy()
    for ct in list_cell_types:
        sub = df.loc[mask, [ct, group_var]].dropna()
        if adjust_covariates:
            sub_for_lm = df.loc[mask, [ct, "AgeAtClinicalRecordCreation", "STAGE_AT_ICB"]].dropna()
            model = smf.ols(f"{ct} ~ AgeAtClinicalRecordCreation + C(STAGE_AT_ICB)", data = sub_for_lm).fit()
            residuals = model.resid
            sub = sub.assign(resid = residuals.loc[sub.index]).dropna(subset = ["resid"])
            values_a = sub.loc[sub[group_var] == group_a, "resid"]
            values_b = sub.loc[sub[group_var] == group_b, "resid"]
        else:
            values_a = sub.loc[sub[group_var] == group_a, ct]
            values_b = sub.loc[sub[group_var] == group_b, ct]

        if len(values_a) < 3 or len(values_b) < 3:
            raise Exception("Enrichment scores for a group are too few to estimate U statistic.")

        U, p = ss.mannwhitneyu(values_a, values_b, alternative = "two-sided")
        rows.append(
            dict(
                cell_type = ct,
                n_a = len(values_a),
                n_b = len(values_b),
                U_stat = U,
                pval = p
            )
        )
    return pd.DataFrame(rows)


def main():
    '''
    Compare enrichment scores by sex.
    Compare enrichment scores for females by experience of ICB therapy.
    Compare enrichment scores for males by experience of ICB therapy.
    '''
    parser = argparse.ArgumentParser(
        description = "Compare enrichment scores by family and cell type."
    )
    parser.add_argument(
        "--adjust-covariates",
        action = "store_true",
        help = "Regress out Age and Stage before rank tests."
    )
    args = parser.parse_args()

    dictionary_of_engines_and_paths_of_enrichment_data = {
        "xCell": paths.enrichment_data_frame_per_xCell,
        "xCell2_and_Pan_Cancer": paths.enrichment_data_frame_per_xCell2_and_Pan_Cancer,
        "xCell2_and_TME_Compendium": paths.enrichment_data_frame_per_xCell2_and_TME_Compendium
    }
    
    for engine, path_to_enrichment_data in dictionary_of_engines_and_paths_of_enrichment_data.items():
        logger.info(f"Enrichment scores in {path_to_enrichment_data} will be compared.")
    
        df, cell_types = create_data_frame_of_enrichment_scores_clinical_data_and_QC_data(
            path_to_enrichment_data
        )
        df = df.rename(columns = lambda cell_type: cell_type.replace(' ', '_').replace('-', '_').replace('+', "plus").replace(',', ''))
        cell_types = [cell_type.replace(' ', '_').replace('-', '_').replace('+', "plus").replace(',', '') for cell_type in cell_types]

        sex_tbl = wilcoxon_table(
            df,
            cell_types,
            group_var = "Sex",
            group_a = 0,
            group_b = 1,
            adjust_covariates = args.adjust_covariates
        ).rename(columns = {"n_a": "n_F", "n_b": "n_M"})
        make_result_table(sex_tbl, "Sex F vs M", Path(f"sex_wilcoxon_{engine}.csv"))

        out_map = {
            0: Path(f"icb_wilcoxon_female_{engine}.csv"),
            1: Path(f"icb_wilcoxon_male_{engine}.csv"),
        }
        for sex_code, out_path in out_map.items():
            sex_df = df[df["Sex"] == sex_code]
            icb_tbl = wilcoxon_table(
                sex_df,
                cell_types,
                group_var = "HAS_ICB",
                group_a = 0,
                group_b = 1,
                adjust_covariates = args.adjust_covariates
            ).rename(columns = {"n_a": "n_Naive", "n_b": "n_Exp"})
            make_result_table(icb_tbl, f"ICB within sex={sex_code}", out_path)


if __name__ == "__main__":
    main()
