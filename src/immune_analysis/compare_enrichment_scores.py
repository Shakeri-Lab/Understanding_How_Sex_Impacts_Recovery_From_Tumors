#!/usr/bin/env python3
'''
Usage
-----
conda activate ici_sex
./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.compare_enrichment_scores --adjust-covariates

Pass flag `--adjust-covariates` to regress out Age and Stage before the rank tests.
Otherwise raw enrichment scores are used.

Outputs
-------
icb_wilcoxon_female.csv
icb_wilcoxon_male.csv
sex_wilcoxon.csv
Each file contains one row per cell type with columns
    - n_F and n_M or n_Naive and n_Exp
    - U-stat (U statistic)
    - p value
    - FDR (Benjamini Hochberg adjusted False Discovery Rate)
    - significant (FDR ≤ 0.05)
    - suggestive (0.05 < FDR ≤ 0.20)
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
    Benjamini–Hochberg across a 1-D array; returns FDRs with original index.
    '''
    _, q, _, _ = multipletests(pvals, method="fdr_bh")
    return pd.Series(q, index=pvals.index)


def make_result_table(stat_df: pd.DataFrame, family: str, out_path: Path) -> None:
    '''
    Add FDR columns + write CSV.
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
        stat_df["suggestive"].sum(),
    )


def wilcoxon_table(
    df: pd.DataFrame,
    list_cell_types: list[str],
    group_var: str,
    group_a,
    group_b,
    adjust_covariates: bool,
) -> pd.DataFrame:
    '''
    Run two-sided Wilcoxon rank-sum for each cell type between two groups.
    '''
    rows = []
    mask = df[group_var].isin([group_a, group_b]).copy()
    for ct in list_cell_types:
        sub = df.loc[mask, [ct, group_var]].dropna()
        if adjust_covariates:
            sub_for_lm = df.loc[mask, [ct, "AgeAtClinicalRecordCreation", "STAGE_AT_ICB"]].dropna()
            model = smf.ols(f"{ct} ~ AgeAtClinicalRecordCreation + C(STAGE_AT_ICB)", data = sub_for_lm).fit()
            residuals = model.resid
            sub = sub.assign(resid = residuals.loc[sub.index])
            sub = sub.dropna(subset = ["resid"])
            values_a = sub.loc[sub[group_var] == group_a, "resid"].dropna()
            values_b = sub.loc[sub[group_var] == group_b, "resid"].dropna()
        else:
            values_a = sub.loc[sub[group_var] == group_a, ct]
            values_b = sub.loc[sub[group_var] == group_b, ct]

        if len(values_a) < 3 or len(values_b) < 3:
            raise Exception("Enrichment scores for a group are too few.")

        U, p = ss.mannwhitneyu(values_a, values_b, alternative = "two-sided")
        rows.append(
            dict(
                cell_type = ct,
                n_a = len(values_a),
                n_b = len(values_b),
                U_stat = U,
                pval = p,
            )
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description = "Wilcoxon tests on xCell scores.")
    parser.add_argument(
        "--adjust-covariates",
        action = "store_true",
        help = "Regress out Age and Stage before rank tests."
    )
    args = parser.parse_args()

    df, cell_types = create_data_frame_of_enrichment_scores_clinical_data_and_QC_data(
        paths.enrichment_data_frame_per_xCell
    )
    df = df.rename(columns = lambda cell_type: cell_type.replace(' ', '_').replace('-', '_').replace('+', "plus"))
    cell_types = [cell_type.replace(' ', '_').replace('-', '_').replace('+', "plus") for cell_type in cell_types]

    sex_tbl = wilcoxon_table(
        df,
        cell_types,
        group_var = "Sex",
        group_a = 0,
        group_b = 1,
        adjust_covariates = args.adjust_covariates
    ).rename(columns = {"n_a": "n_F", "n_b": "n_M"})
    make_result_table(sex_tbl, "Sex F vs M", Path("sex_wilcoxon.csv"))

    out_map = {
        0: Path("icb_wilcoxon_female.csv"),
        1: Path("icb_wilcoxon_male.csv"),
    }
    for sex_code, out_path in out_map.items():
        sex_df = df[df["Sex"] == sex_code]
        icb_tbl = wilcoxon_table(
            sex_df,
            cell_types,
            group_var = "HAS_ICB",
            group_a = 0,
            group_b = 1,
            adjust_covariates = args.adjust_covariates,
        ).rename(columns = {"n_a": "n_Naive", "n_b": "n_Exp"})
        make_result_table(icb_tbl, f"ICB within sex={sex_code}", out_path)


if __name__ == "__main__":
    main()
