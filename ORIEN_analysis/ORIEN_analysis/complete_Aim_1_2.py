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

from cliffs_delta import cliffs_delta


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
    #pseudocount = series_of_module_scores.abs().median()
    series_of_shifted_CD8_G_module_scores_for_samples = series_of_CD8_G_module_scores_for_samples + shift# + pseudocount
    series_of_shifted_CD8_B_module_scores_for_samples = series_of_CD8_B_module_scores_for_samples + shift# + pseudocount
    return np.log2(series_of_shifted_CD8_G_module_scores_for_samples / series_of_shifted_CD8_B_module_scores_for_samples)


def create_data_frame_of_category_of_module_score_and_statistics(
    series_of_values: pd.Series,
    series_of_indicators_of_sex: pd.Series,
    category_of_module_score: str
) -> pd.DataFrame:
    series_of_values_for_females = series_of_values[series_of_indicators_of_sex == 0]
    series_of_values_for_males = series_of_values[series_of_indicators_of_sex == 1]
    number_of_females = len(series_of_values_for_females)
    number_of_males = len(series_of_values_for_males)
    mean_value_for_females = np.mean(series_of_values_for_females)
    mean_value_for_males = np.mean(series_of_values_for_males)
    value_of_cliffs_delta, _ = cliffs_delta(series_of_values_for_males, series_of_values_for_females)
    U_statistic, p_value = mannwhitneyu(series_of_values_for_females, series_of_values_for_males, alternative = "two-sided")
    value_of_cliffs_delta, _ = cliffs_delta(series_of_values_for_males, series_of_values_for_females)
    return pd.DataFrame(
        {
            "category_of_module_score": [category_of_module_score],
            "number_of_females": [number_of_females],
            "number_of_males": [number_of_males],
            "U_statistic": [U_statistic],
            "p_value": [p_value],
            "Cliffs_delta": [value_of_cliffs_delta],
            "mean_value_for_females": [mean_value_for_females],
            "mean_value_for_males": [mean_value_for_males]
        }
    ).set_index("category_of_module_score")


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


def plot_value_vs_sex(
    series_of_values: pd.Series,
    series_of_indicators_of_sex: pd.Series,
    title: str,
    path_to_plot: str
):
    data_frame_of_values_and_indicators_of_sex = pd.DataFrame(
        {
            "value": series_of_values,
            "indicator_of_sex": series_of_indicators_of_sex
        }
    )
    data_frame_of_values_and_indicators_of_sex["sex"] = np.where(
        data_frame_of_values_and_indicators_of_sex["indicator_of_sex"] == 0,
        "Female",
        "Male"
    )
    plt.figure()
    ax = sns.boxplot(data = data_frame_of_values_and_indicators_of_sex, x = "sex", y = "value")
    sns.stripplot(data = data_frame_of_values_and_indicators_of_sex, x = "sex", y = "value", color = "black")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path_to_plot)
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
    data_frame_of_category_of_module_score_CD8_G_minus_CD8_B_and_statistics = create_data_frame_of_category_of_module_score_and_statistics(
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
            create_data_frame_of_category_of_module_score_and_statistics(
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
        data_frame_of_categories_of_module_scores_and_statistics["p_value"],
        method = "fdr_bh"
    )[1]
    data_frame_of_categories_of_module_scores_and_statistics.to_csv(
        paths.data_frame_of_categories_of_module_scores_and_statistics
    )
    plot_value_vs_sex(
        series_of_differences,
        series_of_indicators_of_sex,
        "Difference between CD8 G Module Score and CD8 B Module Score\nvs. Sex",
        paths.plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_sex
    )
    first_percentile, ninety_ninth_percentile = np.nanpercentile(
        series_of_logs_of_ratios_of_CD8_G_module_scores_to_CD8_B_module_scores,
        [1, 99]
    )
    series_of_clipped_logs_of_ratios = series_of_logs_of_ratios_of_CD8_G_module_scores_to_CD8_B_module_scores.clip(
        lower = first_percentile,
        upper = ninety_ninth_percentile
    )
    plot_value_vs_sex(
        series_of_clipped_logs_of_ratios,
        series_of_indicators_of_sex,
        "Log of Ratio of CD8 G Module Score and CD8 B Module Score\nvs. Sex",
        paths.plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_sex
    )


if __name__ == "__main__":
    main()