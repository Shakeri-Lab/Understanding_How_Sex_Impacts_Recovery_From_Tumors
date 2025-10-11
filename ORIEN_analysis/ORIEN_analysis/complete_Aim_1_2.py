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


def load_dictionary_of_names_of_sets_of_genes_and_lists_of_genes() -> Dict[str, List[str]]:
    string_representing_dictionary_of_names_of_sets_of_genes_and_lists_of_genes = (
        paths.dictionary_of_names_of_sets_of_genes_and_lists_of_genes.read_text(encoding = "utf-8")
    )
    JSON_representing_dictionary_of_names_of_sets_of_genes_and_lists_of_genes = json.loads(
        string_representing_dictionary_of_names_of_sets_of_genes_and_lists_of_genes
    )
    return {
        name_of_set_of_genes: list_of_genes
        for name_of_set_of_genes, list_of_genes
        in JSON_representing_dictionary_of_names_of_sets_of_genes_and_lists_of_genes.items()
    }


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


def run_fgsea(
    series_of_ranked_point_biserial_correlations: pd.Series,
    dictionary_of_names_of_sets_of_genes_and_lists_of_genes: dict[str, list[str]]
) -> pd.DataFrame:
    fgsea = importr("fgsea")
    named_vector_of_ranked_point_biserial_correlations = vectors.FloatVector(series_of_ranked_point_biserial_correlations)
    named_vector_of_ranked_point_biserial_correlations.names = vectors.StrVector(
        series_of_ranked_point_biserial_correlations.index
    )
    named_list_of_names_of_sets_of_genes_and_lists_of_genes = ro.ListVector(
        {
            name_of_set_of_genes: vectors.StrVector(list_of_genes)
            for name_of_set_of_genes, list_of_genes
            in dictionary_of_names_of_sets_of_genes_and_lists_of_genes.items()
        }
    )
    ro.r(f"set.seed(0)")
    r_data_frame_of_names_of_sets_of_genes_statistics_and_vectors_of_genes = fgsea.fgseaMultilevel(
        pathways = named_list_of_names_of_sets_of_genes_and_lists_of_genes,
        stats = named_vector_of_ranked_point_biserial_correlations
    )
    serialize_vectors_of_genes = ro.r(
        '''function(data_frame) {
    for (name_of_column in names(data_frame)) {
        if (is.list(data_frame[[name_of_column]])) {
            data_frame[[name_of_column]] <- vapply(
                data_frame[[name_of_column]],
                function(element_of_column) {
                    if (is.atomic(element_of_column)) return(paste(as.character(element_of_column), collapse = ","))
                },
                FUN.VALUE = character(1)
            )
        }
    }
    data_frame
}'''
    )
    r_data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes = serialize_vectors_of_genes(
        r_data_frame_of_names_of_sets_of_genes_statistics_and_vectors_of_genes
    )
    with localconverter(ro.default_converter + pandas2ri.converter):
        data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes = ro.conversion.rpy2py(
            r_data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes
        )
    return data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes


def z_score_expressions_for_gene(expression_matrix: pd.DataFrame) -> pd.DataFrame:
    series_of_mean_expressions = expression_matrix.mean(axis = 1)
    series_of_standard_deviations = expression_matrix.std(axis = 1)
    z_scored_expression_matrix = (
        expression_matrix
        .sub(series_of_mean_expressions, axis = 0)
        .div(series_of_standard_deviations, axis = 0)
    )
    return z_scored_expression_matrix


def create_series_of_module_scores(expression_matrix: pd.DataFrame, list_of_genes: List[str]) -> pd.Series:
    list_of_genes = [gene for gene in list_of_genes if gene in expression_matrix.index]
    expression_submatrix_corresponding_to_provided_genes = expression_matrix.loc[list_of_genes]
    z_scored_expression_submatrix = z_score_expressions_for_gene(expression_submatrix_corresponding_to_provided_genes)
    return z_scored_expression_submatrix.mean(axis = 0).rename("module_score")


def create_series_of_logs_of_ratios_of_CD8_G_module_scores_to_CD8_B_module_scores(
    series_of_CD8_G_module_scores_for_samples: pd.Series,
    series_of_CD8_B_module_scores_for_samples: pd.Series
) -> pd.Series:
    series_of_module_scores = pd.concat(
        [
            series_of_CD8_G_module_scores_for_samples,
            series_of_CD8_B_module_scores_for_samples
        ]
    )
    minimum_module_score = series_of_module_scores.min()
    shift = 1e-6 - minimum_module_score
    series_of_shifted_CD8_G_module_scores_for_samples = series_of_CD8_G_module_scores_for_samples + shift
    series_of_shifted_CD8_B_module_scores_for_samples = series_of_CD8_B_module_scores_for_samples + shift
    return np.log2(series_of_shifted_CD8_G_module_scores_for_samples / series_of_shifted_CD8_B_module_scores_for_samples)


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


def plot_FDR_vs_Normalized_Enrichment_Score(
    data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes: pd.DataFrame,
    plot_of_FDR_vs_Normalized_Enrichment_Score: str
):
    data_frame = data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes.copy()
    data_frame["FDR"] = data_frame["padj"]
    series_of_indicators_that_FDRs_are_suggestive = data_frame["FDR"] < 0.1
    series_of_indicators_that_FDRs_are_significant = data_frame["FDR"] < 0.05
    ax = sns.scatterplot(
        data = data_frame,
        x = "NES",
        y = "FDR",
        hue = series_of_indicators_that_FDRs_are_suggestive,
        style = series_of_indicators_that_FDRs_are_significant
    ) 
    ax.set_ylim(bottom = 0, top = 1)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_ylabel("False Discovery Rate (FDR)")
    ax.set_title("False Discovery Rate vs. Normalized Enrichment Score")
    plt.savefig(paths.plot_of_FDR_vs_Normalized_Enrichment_Score)
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
    series_of_point_biserial_correlations = data_frame_of_genes_and_statistics["point_biserial_correlation"].copy()
    series_of_ranks = series_of_point_biserial_correlations.rank(method = "first")
    # Point biserial correlations are ranked in ascending order.
    # When 2 point biserial correlations are equal, the first receives a lower rank and the second receives a higher rank.
    series_of_point_biserial_correlations = series_of_point_biserial_correlations + (series_of_ranks * 1e-12)
    series_of_ranked_point_biserial_correlations = series_of_point_biserial_correlations.sort_values(ascending = False)
    dictionary_of_names_of_sets_of_genes_and_lists_of_genes = load_dictionary_of_names_of_sets_of_genes_and_lists_of_genes()
    data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes = run_fgsea(
        series_of_ranked_point_biserial_correlations,
        dictionary_of_names_of_sets_of_genes_and_lists_of_genes
    )
    data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes.to_csv(
        paths.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes,
        index = False
    )
    plot_FDR_vs_Normalized_Enrichment_Score(
        data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes,
        paths.plot_of_FDR_vs_Normalized_Enrichment_Score
    )
    list_of_names_of_sets_of_genes = [f"CD8_{i}" for i in range(1, 6 + 1)]
    dictionary_of_names_of_sets_of_genes_and_series_of_module_scores = {}
    for name_of_set_of_genes in list_of_names_of_sets_of_genes:
        list_of_genes = dictionary_of_names_of_sets_of_genes_and_lists_of_genes.get(name_of_set_of_genes)
        dictionary_of_names_of_sets_of_genes_and_series_of_module_scores[name_of_set_of_genes] = create_series_of_module_scores(
            expression_matrix,
            list_of_genes
        )
    data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes = pd.DataFrame(
        dictionary_of_names_of_sets_of_genes_and_series_of_module_scores
    )
    data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes.index.name = "sample_id"
    data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes.to_csv(
        paths.data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes
    )
    list_of_names_of_sets_of_genes_in_set_CD8_B = [f"CD8_{i}" for i in (1, 2, 3)]
    list_of_names_of_sets_of_genes_in_set_CD8_G = [f"CD8_{i}" for i in (4, 5, 6)]
    sorted_list_of_genes_in_set_CD8_B = sorted(
        set(
            sum(
                (
                    dictionary_of_names_of_sets_of_genes_and_lists_of_genes.get(name_of_set_of_genes)
                    for name_of_set_of_genes in list_of_names_of_sets_of_genes_in_set_CD8_B
                ), # tuple of lists of genes
                []
            ) # list of genes
        ) # set of genes
    ) # sorted list of genes
    sorted_list_of_genes_in_set_CD8_G = sorted(
        set(
            sum(
                (
                    dictionary_of_names_of_sets_of_genes_and_lists_of_genes.get(name_of_set_of_genes)
                    for name_of_set_of_genes in list_of_names_of_sets_of_genes_in_set_CD8_G
                ),
                []
            )
        )
    )
    series_of_CD8_B_module_scores_for_samples = create_series_of_module_scores(
        expression_matrix,
        sorted_list_of_genes_in_set_CD8_B
    ).rename("CD8_B_module_score_for_sample")
    series_of_CD8_G_module_scores_for_samples = create_series_of_module_scores(
        expression_matrix,
        sorted_list_of_genes_in_set_CD8_G
    ).rename("CD8_G_module_score_for_sample")
    series_of_differences = (
        series_of_CD8_G_module_scores_for_samples - series_of_CD8_B_module_scores_for_samples
    ).rename("difference_between_CD8_G_and_CD8_B_module_scores")
    series_of_logs_of_ratios_of_CD8_G_module_scores_to_CD8_B_module_scores = create_series_of_logs_of_ratios_of_CD8_G_module_scores_to_CD8_B_module_scores(
        series_of_CD8_G_module_scores_for_samples,
        series_of_CD8_B_module_scores_for_samples
    ).rename("log_of_ratio_of_CD8_G_module_score_to_CD8_B_module_score")
    data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences = pd.concat(
        [
            series_of_CD8_B_module_scores_for_samples,
            series_of_CD8_G_module_scores_for_samples,
            series_of_differences
        ],
        axis = 1
    )
    data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences.index.name = "sample_id"
    data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences.to_csv(
        paths.data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences
    )
    list_of_data_frames_of_categories_of_module_scores_and_statistics = []
    data_frame_of_category_of_module_score_CD8_G_minus_CD8_B_and_statistics = compare_by_sex(
        series_of_differences,
        series_of_indicators_of_sex,
        "CD8_G_minus_CD8_B"
    )
    list_of_data_frames_of_categories_of_module_scores_and_statistics.append(
        data_frame_of_category_of_module_score_CD8_G_minus_CD8_B_and_statistics
    )
    for name_of_set_of_genes in list_of_names_of_sets_of_genes:
        series_of_module_scores = data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes[name_of_set_of_genes]
        list_of_data_frames_of_categories_of_module_scores_and_statistics.append(
            compare_by_sex(
                series_of_module_scores,
                series_of_indicators_of_sex,
                name_of_set_of_genes
            )
        )
    data_frame_of_categories_of_module_scores_and_statistics = pd.concat(
        list_of_data_frames_of_categories_of_module_scores_and_statistics,
        axis = 0
    )
    data_frame_of_categories_of_module_scores_and_statistics["FDR"] = multipletests(
        data_frame_of_categories_of_module_scores_and_statistics["pval"],
        method = "fdr_bh"
    )[1]
    data_frame_of_categories_of_module_scores_and_statistics.to_csv(
        paths.data_frame_of_categories_of_module_scores_and_statistics
    )

    # 8) Plots: box by sex for CD8_G-CD8_B and (robust) log2 ratio (visualization only)
    box_by_sex(
        series_of_differences,
        series_of_indicators_of_sex,
        "CD8_G - CD8_B (module score) by sex",
        str(paths.outputs_of_completing_Aim_1_2 / "box_CD8_G_minus_CD8_B_by_sex.png")
    )
    # Clip extremes for display so a few near-zero denominators don't dominate the axis.
    rlow, rhigh = np.nanpercentile(series_of_logs_of_ratios_of_CD8_G_module_scores_to_CD8_B_module_scores, [1, 99])
    ratio_for_plot = series_of_logs_of_ratios_of_CD8_G_module_scores_to_CD8_B_module_scores.clip(lower=rlow, upper=rhigh)
    box_by_sex(
        ratio_for_plot,
        series_of_indicators_of_sex,
        "log2(CD8_G / CD8_B) (module score) by sex - robust (display only)",
        str(paths.outputs_of_completing_Aim_1_2 / "box_log2_CD8G_over_CD8B_by_sex.png")
    )

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


if __name__ == "__main__":
    main()