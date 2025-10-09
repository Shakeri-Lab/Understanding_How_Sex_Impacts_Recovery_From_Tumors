'''
Aim 1.2 — Evaluate sex differences in phenotypes of T cell populations (CD8 TILs)

Pipeline:
  1) Load RNA-seq expression matrix (genes x samples) and metadata with 'sex'.
  2) Pre-rank genes by point-biserial correlation (Pearson vs. binary sex).
  3) Run GSEA using fgsea (via rpy2) on Sade-Feldman CD8 TIL signatures
     (especially partially exhausted cytotoxic T cells; CD8_1..CD8_6).
  4) Compute sample-level module scores for each fine cluster (CD8_1..CD8_6)
     and for combined groups CD8_B = {CD8_1,2,3}, CD8_G = {CD8_4,5,6}.
  5) Compare CD8_G vs CD8_B by sex (Mann-Whitney U, Cliff's delta, BH-FDR).
  6) Save tidy CSV outputs and figures.

Notes:
  - Include all tumor specimens with RNA-seq data. If your metadata contains a
    tumor indicator, pass --tumor-only with column/value filters; otherwise all
    samples in the expression/metadata intersection are used.
  - Expression should be gene symbols (case-insensitive). Gene sets are matched
    case-insensitively.

Gene sets are derived from Table S4 from https://pmc.ncbi.nlm.nih.gov/articles/PMC6641984/#SD5 .
'''

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# stats / FDR
from statsmodels.stats.multitest import multipletests

# rpy2 / fgsea
from rpy2.robjects import r, pandas2ri, ListVector, FloatVector, StrVector
from rpy2.robjects.packages import importr

from ORIEN_analysis.config import paths

from rpy2.robjects.conversion import localconverter
from rpy2 import robjects as ro
from rpy2.robjects import vectors


def load_metadata(meta_path: str) -> pd.DataFrame:
    """
    Load metadata table with at least:
      - 'sample_id'
      - sex column (name provided via --sex-col)
    """
    if meta_path.endswith(".tsv") or meta_path.endswith(".txt"):
        df = pd.read_csv(meta_path, sep="\t")
    else:
        df = pd.read_csv(meta_path, sep=None, engine="python")
    # enforce str for sample ids
    if "sample_id" not in df.columns:
        raise ValueError("Metadata must contain a 'sample_id' column.")
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def load_gene_sets() -> Dict[str, List[str]]:
    '''
    Load gene sets from .json.
    Returns dict: {set_name: [GENE1, GENE2, ...]} with genes upper-cased.
    '''
    obj = json.loads(paths.gene_sets.read_text(encoding = "utf-8"))
    return {k: [g.upper() for g in v] for k, v in obj.items()}


def compute_point_biserial_correlation_for_each_gene(
    expression_matrix: pd.DataFrame,
    series_of_indicators_of_sex: pd.Series
) -> pd.DataFrame:
    series_of_centered_indicators_of_sex = series_of_indicators_of_sex - series_of_indicators_of_sex.mean()
    number_of_samples = len(expression_matrix.columns)
    array_of_centered_indicators_of_sex = series_of_centered_indicators_of_sex.to_numpy()
    standard_deviation_of_centered_indicators_of_sex = array_of_centered_indicators_of_sex.std()
    series_of_means_of_expression_values_across_samples_for_each_gene = expression_matrix.mean(axis = 1)
    data_frame_of_centered_expression_values = expression_matrix.subtract(
        series_of_means_of_expression_values_across_samples_for_each_gene,
        axis = 0
    )
    array_of_covariances = (
        (data_frame_of_centered_expression_values.values @ array_of_centered_indicators_of_sex) /
        (number_of_samples - 1)
    )
    array_of_standard_deviations_of_expression_values_across_samples_for_each_gene = (
        data_frame_of_centered_expression_values.std(axis = 1).to_numpy()
    )
    product_that_would_be_divisor = (
        array_of_standard_deviations_of_expression_values_across_samples_for_each_gene *
        standard_deviation_of_centered_indicators_of_sex
    )
    point_biserial_correlation = np.divide(
        array_of_covariances,
        product_that_would_be_divisor,
        out = np.zeros_like(array_of_covariances),
        where = (product_that_would_be_divisor != 0.0)
    )
    data_frame = (
        pd.DataFrame(
            {
                "gene": expression_matrix.index.values,
                "point_biserial_correlation": point_biserial_correlation,
                "number_of_samples": number_of_samples,
                "standard_deviation_of_expression_values_across_samples_for_each_gene": array_of_standard_deviations_of_expression_values_across_samples_for_each_gene,
                "standard_deviation_of_centered_indicators_of_sex": standard_deviation_of_centered_indicators_of_sex,
            }
        )
        .set_index("gene")
    )
    return data_frame


def build_preranked_vector(data_frame_of_genes_and_statistics: pd.DataFrame, break_ties: bool = True) -> pd.Series:
    """
    Build a pre-ranked Series for fgsea from 'r' values.
    Name = gene, value = correlation coefficient.

    If break_ties=True, add a tiny deterministic tie-breaker
    based on the "first" rank to avoid fgsea tie warnings while preserving order.
    """
    s = data_frame_of_genes_and_statistics["point_biserial_correlation"].astype(float).copy()
    if break_ties:
        s = s + (s.rank(method="first") * 1e-12)
    s = s.sort_values(ascending=False)
    s.name = "stat"
    return s


def run_fgsea(
    preranked: pd.Series,
    gene_sets: dict[str, list[str]],
    nperm: int | None = None,
    min_size: int = 10,
    max_size: int = 1_000,
    seed: int = 0
) -> pd.DataFrame:
    '''
    Run fgsea on a pre-ranked vector and Python dict of gene sets.
    
    - If nperm is None (default), run fgseaMultilevel (recommended).
    - If nperm is an int, run fgsea (Simple) with nperm permutations.
    '''
    base = importr("base")
    fgsea = importr("fgsea")

    stats_r = vectors.FloatVector(preranked.values.astype(float))
    stats_r.names = vectors.StrVector(preranked.index.tolist())
    pathways_r = ro.ListVector({
        k: vectors.StrVector(sorted(set(v)))
        for k, v in gene_sets.items() if v
    })
    ro.r(f"set.seed({int(seed)})")
    
    if nperm is None:
        # Recommended method
        res_r = fgsea.fgseaMultilevel(
            pathways = pathways_r,
            stats = stats_r,
            minSize = min_size,
            maxSize = max_size
        )
    else:
        res_r = fgsea.fgsea(
            pathways = pathways_r,
            stats = stats_r,
            nperm = int(nperm),
            minSize = min_size
        )

    sanitize_r = ro.r('''
        function(d) {
            d <- as.data.frame(d, stringsAsFactors = FALSE)
            if (nrow(d) == 0L) {
                return(data.frame(
                    pathway=character(), pvalue=double(), padj=double(),
                    ES=double(), NES=double(), size=integer(),
                    leadingEdge=character(), stringsAsFactors=FALSE
                ))
            }
            for (nm in names(d)) {
                if (is.list(d[[nm]])) {
                    d[[nm]] <- vapply(
                        d[[nm]],
                        function(x) {
                            if (is.null(x) || length(x) == 0L) return("")
                            if (is.atomic(x)) return(paste(as.character(x), collapse = ","))
                            if (is.list(x)) return(paste(as.character(unlist(x, use.names = FALSE)), collapse = ","))
                            as.character(x)
                        },
                        FUN.VALUE = character(1L)
                    )
                }
            }
            d
        }
    ''')
    res_r_sanitized = sanitize_r(res_r)
    with localconverter(ro.default_converter + pandas2ri.converter):
        res_df = ro.conversion.rpy2py(res_r_sanitized)
    res_df = res_df.rename(columns = {
        "pathway": "term"
    })
    cols = [c for c in ["term", "pval", "padj", "ES", "NES", "size", "leadingEdge"] if c in res_df.columns]
    return res_df.loc[:, cols] if cols else res_df


# ------------------ Sample-level module scoring ------------------ #

def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each gene (row) across samples (columns).
    """
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=1).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
    return z


def module_score(
    expr: pd.DataFrame,               # genes x samples
    genes: List[str]
) -> pd.Series:
    """
    Mean z-scored expression across given genes present in expr.
    """
    genes = [g for g in genes if g in expr.index]
    if len(genes) == 0:
        return pd.Series(0.0, index=expr.columns, name="score")
    z = zscore_rows(expr.loc[genes])
    return z.mean(axis=0).rename("score")


def robust_log2_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    '''
    Visualization-only log2 ratio between two centered scores.
    Ensures strictly positive arguments to log2 by:
    1) shifting both series so the global minimum is > 0
    2) adding a data-scaled pseudocount to avoid tiny denominators
    '''
    base = pd.concat([numer, denom])
    min_val = float(base.min())
    # shift so the smallest value becomes ~1e-6
    shift_pos = (-min_val + 1e-6) if min_val <= 0 else 1e-6
    # pseudocount based on typical scale of the data
    pseudo = max(float(base.abs().median()), 0.25)
    num_pos = numer + shift_pos + pseudo
    den_pos = denom + shift_pos + pseudo
    # strictly positive by construction; no invalid log2
    return np.log2(num_pos / den_pos)


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta effect size: P(X>Y)-P(X<Y).
    Computed via ranks to avoid O(n*m).
    """
    nx, ny = len(x), len(y)
    allv = np.concatenate([x, y])
    ranks = rankdata(allv, method="average")
    rx = ranks[:nx]
    ry = ranks[nx:]
    # Equivalent formula using U statistic:
    # U = sum_{i} rank(x_i) - nx*(nx+1)/2
    U = rx.sum() - nx * (nx + 1) / 2.0
    delta = (2*U) / (nx*ny) - 1
    return float(delta)


def compare_by_sex(
    scores: pd.Series,         # indexed by sample_id
    series_of_indicators_of_sex: pd.Series,          # 1=male, 0=female aligned to sample_id
    label: str
) -> pd.DataFrame:
    """
    Mann-Whitney test of score by sex, with Cliff's delta.
    series_of_indicators_of_sex coding: 0=female, 1=male.
    """
    common = scores.index.intersection(series_of_indicators_of_sex.index)
    s = scores.loc[common]
    y = series_of_indicators_of_sex.loc[common]

    f = s[y == 0].to_numpy()  # females
    m = s[y == 1].to_numpy()  # males
    if len(f) == 0 or len(m) == 0:
        raise ValueError("One of the sex groups is empty after alignment.")
    u_stat, pval = mannwhitneyu(f, m, alternative="two-sided")
    delta = cliff_delta(m, f)  # positive -> higher in males
    return pd.DataFrame({
        "contrast": [label],
        "n_female": [len(f)],
        "n_male": [len(m)],
        "mw_u": [u_stat],
        "pval": [pval],
        "cliffs_delta_male_minus_female": [delta],
        "mean_female": [float(np.mean(f))],
        "mean_male": [float(np.mean(m))]
    }).set_index("contrast")


# ----------------------------- Plots ----------------------------- #

def volcano_from_fgsea(res_df: pd.DataFrame, out_png: str):
    '''
    Volcano-like plot: NES vs. FDR using 'padj' from fgsea.
    '''
    df = res_df.copy()
    if df.empty:
        print("[WARN] fgsea result is empty; skipping volcano plot.")
        return
    if "padj" not in df.columns:
        raise ValueError("fgsea results are missing 'padj' (FDR) column.")
    df["FDR"] = df["padj"].replace(0, np.nextafter(0, 1))
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(
        data=df, x="NES", y="FDR",
        hue=(df["padj"] < 0.1), style=(df["padj"] < 0.05),
        s=60
    )
    ax.axhline(0.1, ls="--", lw=1)
    ax.axvline(0, ls=":", lw=1)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_ylabel("FDR")
    ax.set_title("fgsea results (sex pre-ranked)")
    if ax.get_legend() is not None:
        ax.legend(title = "FDR thresholds")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def box_by_sex(scores: pd.Series, series_of_indicators_of_sex: pd.Series, title: str, out_png: str):
    common = scores.index.intersection(series_of_indicators_of_sex.index)
    df = pd.DataFrame({
        "score": scores.loc[common].values,
        "series_of_indicators_of_sex": series_of_indicators_of_sex.loc[common].values
    })
    df["sex"] = np.where(df["series_of_indicators_of_sex"] == 0, "Female", "Male")
    plt.figure(figsize=(5, 5))
    ax = sns.boxplot(data=df, x="sex", y="score")
    sns.stripplot(data=df, x="sex", y="score", color="black", alpha=0.5)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    expression_matrix = pd.read_csv(
        paths.expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest,
        index_col = 0
    )
    index_of_sample_IDs = expression_matrix.columns
    metadata_frame = pd.DataFrame(
        {"sample_id": index_of_sample_IDs}
    )
    clinical_molecular_linkage_data = pd.read_csv(paths.clinical_molecular_linkage_data)
    patient_data = pd.read_csv(paths.patient_data)
    data_frame_of_sample_IDs_and_patient_IDs = clinical_molecular_linkage_data[["RNASeq", "ORIENAvatarKey"]]
    data_frame_of_sample_IDs_and_patient_IDs = (
        data_frame_of_sample_IDs_and_patient_IDs
        .sort_values(by = ["RNASeq", "ORIENAvatarKey"])
        .drop_duplicates(subset = ["RNASeq"], keep = "first")
    )
    metadata_frame = (
        metadata_frame
        .merge(
            data_frame_of_sample_IDs_and_patient_IDs,
            how = "left",
            left_on = "sample_id",
            right_on = "RNASeq",
            validate = "one_to_one"
        )
        .drop(columns = "RNASeq")
        .merge(
            patient_data[["AvatarKey", "Sex"]],
            how = "left",
            left_on = "ORIENAvatarKey",
            right_on = "AvatarKey",
            validate = "one_to_one"
        )
        .drop(columns = ["ORIENAvatarKey", "AvatarKey"])
        .set_index("sample_id")
    )
    series_of_indicators_of_sex = vals = metadata_frame["Sex"].map(
        {
            "Female": 0,
            "Male": 1
        }
    )
    data_frame_of_genes_and_statistics = compute_point_biserial_correlation_for_each_gene(
        expression_matrix,
        series_of_indicators_of_sex
    )
    data_frame_of_genes_and_statistics.to_csv(paths.data_frame_of_genes_and_statistics)
    
    preranked = build_preranked_vector(data_frame_of_genes_and_statistics, break_ties=True)
    gene_sets = load_gene_sets()

    # 5) fgsea
    fgsea_res = run_fgsea(
        preranked,
        gene_sets,
        nperm = None,
        min_size = 10,
        max_size = 1_000,
        seed = 0
    )
    fgsea_res.to_csv(paths.outputs_of_completing_Aim_1_2 / "fgsea_results.csv", index=False)

    # volcano-like plot
    try:
        volcano_from_fgsea(fgsea_res, str(paths.outputs_of_completing_Aim_1_2 / "fgsea_volcano.png"))
    except Exception as e:
        print(f"[WARN] Volcano plot failed: {e}")

    # 6) Sample-level module scores: CD8_1..CD8_6
    zscores = zscore_rows(expression_matrix)  # genes x samples (z per gene)
    fine_labels = [f"CD8_{i}" for i in range(1, 7)]
    missing = [lb for lb in fine_labels if lb not in gene_sets]
    if missing:
        print(f"[WARN] Missing expected fine clusters in gene sets: {missing}")

    fine_scores = {}
    for lb in fine_labels:
        genes = gene_sets.get(lb, [])
        fine_scores[lb] = module_score(expression_matrix, genes)

    fine_df = pd.DataFrame(fine_scores)
    fine_df.index.name = "sample_id"
    fine_df.to_csv(paths.outputs_of_completing_Aim_1_2 / "sample_module_scores_CD8_1_to_6.csv")

    # Combined groups
    cd8_b_names = [f"CD8_{i}" for i in (1, 2, 3)]
    cd8_g_names = [f"CD8_{i}" for i in (4, 5, 6)]
    cd8_b_genes = sorted(set(sum((gene_sets.get(n, []) for n in cd8_b_names), [])))
    cd8_g_genes = sorted(set(sum((gene_sets.get(n, []) for n in cd8_g_names), [])))

    score_b = module_score(expression_matrix, cd8_b_genes).rename("CD8_B_score")
    score_g = module_score(expression_matrix, cd8_g_genes).rename("CD8_G_score")
    # Difference (primary metric) and robust log-ratio (for visualization only)
    diff = (score_g - score_b).rename("CD8_G_minus_CD8_B")
    base = pd.concat([score_b, score_g])
    c = max(float(base.abs().median()), 0.25)
    ratio_robust = robust_log2_ratio(score_g, score_b).rename("log2_CD8_G_over_CD8_B_robust")

    # Save only B, G, and difference; ratio is visualization-only
    combined = pd.concat([score_b, score_g, diff], axis = 1)
    combined.index.name = "sample_id"
    combined.to_csv(paths.outputs_of_completing_Aim_1_2 / "sample_module_scores_CD8_B_vs_CD8_G.csv")

    # 7) Stats by sex for key contrasts
    tests = []
    tests.append(compare_by_sex(diff, series_of_indicators_of_sex, "CD8_G_minus_CD8_B"))
    # (No ratio in hypothesis testing; visualization-only)

    # Add each fine cluster as well
    for lb in fine_labels:
        tests.append(compare_by_sex(fine_df[lb], series_of_indicators_of_sex, lb))

    stats_tbl = pd.concat(tests, axis=0)
    # BH-FDR across all tests
    stats_tbl["fdr"] = multipletests(stats_tbl["pval"].values, method="fdr_bh")[1]
    stats_tbl.to_csv(paths.outputs_of_completing_Aim_1_2 / "by_sex_tests_CD8_scores.csv")

    # 8) Plots: box by sex for CD8_G-CD8_B and (robust) log2 ratio (visualization only)
    try:
        box_by_sex(diff, series_of_indicators_of_sex, "CD8_G - CD8_B (module score) by sex",
                   str(paths.outputs_of_completing_Aim_1_2 / "box_CD8_G_minus_CD8_B_by_sex.png"))
        # Clip extremes for display so a few near-zero denominators don't dominate the axis.
        rlow, rhigh = np.nanpercentile(ratio_robust, [1, 99])
        ratio_for_plot = ratio_robust.clip(lower=rlow, upper=rhigh)
        box_by_sex(
            ratio_for_plot,
            series_of_indicators_of_sex,
            "log2(CD8_G / CD8_B) (module score) by sex - robust (display only)",
            str(paths.outputs_of_completing_Aim_1_2 / "box_log2_CD8G_over_CD8B_by_sex.png")
        )
    except Exception as e:
        print(f"[WARN] Boxplot failed: {e}")

    # 9) Write a compact README
    with open(paths.outputs_of_completing_Aim_1_2 / "README_Aim1_2.txt", "w", encoding="utf-8") as fh:
        fh.write(
            "Aim 1.2 — CD8 TIL phenotypes by sex\n"
            "-----------------------------------\n"
            "Files:\n"
            "  - preranked_stats_point_biserial.csv: gene-level r vs sex (female=0, male=1).\n"
            "  - fgsea_results.csv: fgsea on pre-ranked stats (NES, ES, pval, padj [BH-FDR], leading edge).\n"
            "  - fgsea_volcano.png: NES vs FDR scatter (FDR = padj).\n"
            "  - sample_module_scores_CD8_1_to_6.csv: per-sample module scores for fine clusters.\n"
            "  - sample_module_scores_CD8_B_vs_CD8_G.csv: per-sample CD8_B, CD8_G, and difference (CD8_G_minus_CD8_B) (ratio is visualization-only).\n"
            "  - by_sex_tests_CD8_scores.csv: Mann-Whitney + Cliff's delta + FDR across contrasts (no ratio tests). Cliff's delta: positive = higher in males\n"
            "  - box_*.png: box/strip plots by sex for key contrasts.\n"
            "\n"
            "Methods:\n"
            "  - Pre-ranking uses point-biserial (Pearson vs binary sex), with tiny deterministic tie-breaking.\n"
            "  - fgsea: Multilevel by default (recommended); min size filter applied.\n"
            "  - Module scores = mean z-scored expression across genes in the set.\n"
            "  - CD8_B = union of CD8_1,2,3; CD8_G = union of CD8_4,5,6.\n"
            "  - Robust log2 ratio uses a shift-to-positive plus data-scaled pseudocount and is shown for visualization (clipped to 1st-99th percentiles).\n"
        )
    print(f"Outputs were written to {paths.outputs_of_completing_Aim_1_2}.")


if __name__ == "__main__":
    main()