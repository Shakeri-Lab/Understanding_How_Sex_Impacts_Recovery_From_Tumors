#!/usr/bin/env python3
'''
This script compares enrichment scores produced by xCell or xCell2 for
- tumors of females (0) and males (1),
- ICB-naive tumors (0) and ICB-experienced (1) tumors of females, and
- ICB-naive tumors and ICB-experienced tumors of males
across various cell types.
This script for each family of tumors and cell type
performs a 2 sided Mann-Whitney U Test / Wilcoxon Rank Sum Test.

This script optionally removes variability due to age at clinical record creation and stage at start of ICB therapy before comparing.

Enrichment scores from group A (e.g., samples of females) and group B (e.g., samples of males)
may be combined in a list.
The list may be sorted from smallest to largest.
A rank is the position of an enrichment score in the sorted list.
If multiple enrichment scores are equal, their ranks are averaged and
assigned to each equal enrichment score.
When an enrichment score precedes another enrichment score,
the former occurs earlier in the sorted list.

A Mann-Whitney U Test / Wilcoxon Rank Sum Test tests a null hypothesis that every enrichment score in either group A or B is drawn from the same population distribution of enrichment scores.
The U test statistic of a Mann-Whitney U Test / Wilcoxon Rank Sum Test is
the number of pairs of enrichment scores in which
the enrichment score for group A precedes the enrichment score for group B.
For a given U test statistic, the p value associated with that U statistic is
the probability, when assuming the null hypothesis is true,
of observing a U statistic at least as extreme as the calculated U statistic.
Smaller p values provide stronger evidence against the null hypothesis.

This script adjusts p values into False Discovery Rates (FDRs) with the Benjamini-Hochberg procedure.
p values p_i and any associated information are sorted in ascending order.
Each p value p_i is multiplied by the number of p values, divided by i, and clipped to 1.
The resulting series is corrected so that each value is at least as large as the preceding value.
The resulting series contains FDRs and may be added as a column next to the p values.

A cell type is significant if the FDR associated with that cell type is less than or equal to 0.05.
A cell type being significant indicates that enrichment scores for group A and that cell type
were not drawn from the same population distribution as
enrichment scores for group B and that cell type.
A cell type is suggestive if the FDR associated with that cell type is greater than 0.05 and
less than or equal to 0.20.
A cell type being suggestive suggests that enrichment scores for group A and that cell type
were not drawn from the same population distribution as
enrichment scores for group B and that cell type.


Usage
-----
conda activate ici_sex
./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.compare_enrichment_scores --adjust-covariates

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
    - FDR containing False Discovery Rates,
    - significant containing indicators of significance, and
    - suggestive containing indicators of suggestiveness.
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
    mask = df[group_var].isin([group_a, group_b])
    for ct in list_cell_types:
        sub = df.loc[mask, [ct, group_var]].dropna()
        if adjust_covariates:
            sub_for_lm = df.loc[mask, [ct, "AgeAtClinicalRecordCreation", "STAGE_AT_ICB"]].dropna()
            if not (sub_for_lm.index == sub.index).all():
                raise ValueError("sub_for_lm has fewer rows than sub.")
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
    paths.ensure_dependencies_for_compare_enrichment_scores_exist()
    
    parser = argparse.ArgumentParser(
        description = "Compare enrichment scores by family and cell type."
    )
    parser.add_argument(
        "--adjust-covariates",
        action = "store_true",
        help = "Regress out Age and Stage before rank tests."
    )
    args = parser.parse_args()

    dictionary_of_paths = {
        paths.enrichment_data_frame_per_xCell: (
            paths.comparisons_for_females_and_males_and_xCell,
            paths.comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell,
            paths.comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell
        ),
        paths.enrichment_data_frame_per_xCell2_and_Pan_Cancer: (
            paths.comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer,
            paths.comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer,
            paths.comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_Pan_Cancer
        ),
        paths.enrichment_data_frame_per_xCell2_and_TME_Compendium: (
            paths.comparisons_for_females_and_males_and_xCell2_and_TME_Compendium,
            paths.comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_TME_Compendium,
            paths.comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_TME_Compendium
        )
    }
    
    for path_to_enrichment_data, tuple_of_paths_to_comparisons in dictionary_of_paths.items():
        logger.info(f"Enrichment scores in {path_to_enrichment_data} will be compared.")
        
        path_to_comparisons_for_females_and_males = tuple_of_paths_to_comparisons[0]
        path_to_comparisons_for_ICB_naive_and_experienced_samples_of_females = tuple_of_paths_to_comparisons[1]
        path_to_comparisons_for_ICB_naive_and_experienced_samples_of_males = tuple_of_paths_to_comparisons[2]
    
        df, cell_types = create_data_frame_of_enrichment_scores_clinical_data_and_QC_data(
            path_to_enrichment_data
        )
        df = df.rename(columns = lambda cell_type: cell_type.replace(' ', '_').replace('-', '_').replace('+', "plus").replace(',', ''))
        dupes = df.columns[df.columns.duplicated()]
        if len(dupes):
            raise ValueError(f"Duplicate column names after cleaning: {sorted(dupes)}")
        cell_types = [cell_type.replace(' ', '_').replace('-', '_').replace('+', "plus").replace(',', '') for cell_type in cell_types]
        set_of_cell_types = set(cell_types)
        if len(set_of_cell_types) != len(cell_types):
            raise ValueError("Set and list of cell types are different.")

        sex_tbl = wilcoxon_table(
            df,
            cell_types,
            group_var = "Sex",
            group_a = 0,
            group_b = 1,
            adjust_covariates = args.adjust_covariates
        ).rename(columns = {"n_a": "n_F", "n_b": "n_M"})
        make_result_table(sex_tbl, "Sex F vs M", path_to_comparisons_for_females_and_males)

        out_map = {
            0: path_to_comparisons_for_ICB_naive_and_experienced_samples_of_females,
            1: path_to_comparisons_for_ICB_naive_and_experienced_samples_of_males,
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
