'''
`complete_Aim_1_2.py`

Evaluate differences in phenotypes of CD8+ T cells that are Tumor Infiltrating Lymphocytes by sex and ICB status.

Pipeline:
1. Load an expression matrix with HGNC symbols and SLIDs approved by manifest and metadata.
2. Pre-rank genes by point biserial correlation.
3. Run GSEA using fgsea.
4. Compute module scores for samples and 6 sets of genes CD8 1, CD8 2, CD8 3, CD8 4, CD8 5, and CD8 6.
5. Compute module scores for set of genes CD8 B that is the union of sets CD8 1, CD8 2, and CD8 3.
6. Compute module scores for set of genes CD8 G that is the union of sets CD8 4, CD8 5, and CD8 6.
7. Compare module scores by sex and ICB status.
8. Create files of CSVs and plots.

ICB status is derived from medication data and age at specimen collection.
An sample is experienced if any age at start of ICB medication is at most age at specimen collection.

Gene sets are derived from Table S4 from https://pmc.ncbi.nlm.nih.gov/articles/PMC6641984/#SD5 .
'''

from rpy2.robjects import FloatVector
from rpy2.robjects import ListVector
from rpy2.robjects import StrVector
from cliffs_delta import cliffs_delta
from ORIEN_analysis.fit_linear_models import (
    create_data_frame_of_output_and_clinical_molecular_linkage_data,
    create_data_frame_of_specimen_IDs_and_indicators_that_patient_received_ICB_therapy
)
from rpy2.robjects.packages import importr
import json
from rpy2.robjects.conversion import localconverter
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np
from ORIEN_analysis.fit_linear_models import numericize_age
from rpy2.robjects import pandas2ri
from ORIEN_analysis.config import paths
import pandas as pd
import matplotlib.pyplot as plt
from rpy2 import robjects as ro
import seaborn as sns
from rpy2.robjects import vectors

from pathlib import Path


def load_dictionary_of_names_of_sets_of_genes_and_lists_of_genes() -> dict[str, list[str]]:
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
    series_of_indicators: pd.Series
) -> pd.DataFrame:
    number_of_samples = len(expression_matrix.columns)
    series_of_centered_indicators = series_of_indicators - series_of_indicators.mean()
    array_of_centered_indicators = series_of_centered_indicators.to_numpy()
    standard_deviation_of_centered_indicators = array_of_centered_indicators.std(ddof = 1)
    series_of_means_of_expression_values_across_samples_for_each_gene = expression_matrix.mean(axis = 1)
    data_frame_of_centered_expression_values = expression_matrix.subtract(
        series_of_means_of_expression_values_across_samples_for_each_gene,
        axis = 0
    )
    array_of_covariances = (
        (data_frame_of_centered_expression_values.values @ array_of_centered_indicators) /
        (number_of_samples - 1)
    )
    array_of_standard_deviations_of_expression_values_across_samples_for_each_gene = (
        data_frame_of_centered_expression_values.std(axis = 1, ddof = 1).to_numpy()
    )
    product_that_would_be_divisor = (
        array_of_standard_deviations_of_expression_values_across_samples_for_each_gene *
        standard_deviation_of_centered_indicators
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
                "standard_deviation_of_centered_indicators": standard_deviation_of_centered_indicators,
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
        stats = named_vector_of_ranked_point_biserial_correlations,
        nPermSimple = 10_000
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
    series_of_standard_deviations = expression_matrix.std(axis = 1, ddof = 1)
    z_scored_expression_matrix = (
        expression_matrix
        .sub(series_of_mean_expressions, axis = 0)
        .div(series_of_standard_deviations, axis = 0)
    )
    return z_scored_expression_matrix


def create_series_of_module_scores(expression_matrix: pd.DataFrame, list_of_genes: list[str]) -> pd.Series:
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
    series_of_indicators: pd.Series,
    name_of_group_0: str,
    name_of_group_1: str,
    category_of_module_score: str
) -> pd.DataFrame:
    series_of_values_for_group_0 = series_of_values[series_of_indicators == 0]
    series_of_values_for_group_1 = series_of_values[series_of_indicators == 1]
    number_of_samples_for_group_0 = len(series_of_values_for_group_0)
    number_of_samples_for_group_1 = len(series_of_values_for_group_1)
    mean_value_for_group_0 = np.mean(series_of_values_for_group_0)
    mean_value_for_group_1 = np.mean(series_of_values_for_group_1)
    U_statistic, p_value = mannwhitneyu(series_of_values_for_group_0, series_of_values_for_group_1, alternative = "two-sided")
    value_of_cliffs_delta, _ = cliffs_delta(series_of_values_for_group_1, series_of_values_for_group_0)
    return pd.DataFrame(
        {
            "category_of_module_score": [category_of_module_score],
            f"number_of_samples_for_{name_of_group_0}": [number_of_samples_for_group_0],
            f"number_of_samples_for_{name_of_group_1}": [number_of_samples_for_group_1],
            "U_statistic": [U_statistic],
            "p_value": [p_value],
            "Cliffs_delta": [value_of_cliffs_delta],
            f"mean_value_for_{name_of_group_0}": [mean_value_for_group_0],
            f"mean_value_for_{name_of_group_1}": [mean_value_for_group_1]
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
    plt.savefig(plot_of_FDR_vs_Normalized_Enrichment_Score)
    plt.close()


def plot_value_vs_indicator(
    series_of_values: pd.Series,
    series_of_indicators: pd.Series,
    name_of_group_0: str,
    name_of_group_1: str,
    title: str,
    path_to_plot: str
):
    data_frame_of_values_and_indicators = pd.DataFrame(
        {
            "value": series_of_values,
            "indicator": series_of_indicators
        }
    )
    data_frame_of_values_and_indicators["indicator"] = np.where(
        data_frame_of_values_and_indicators["indicator"] == 0,
        name_of_group_0,
        name_of_group_1
    )
    plt.figure()
    ax = sns.boxplot(data = data_frame_of_values_and_indicators, x = "indicator", y = "value")
    sns.stripplot(data = data_frame_of_values_and_indicators, x = "indicator", y = "value", color = "black")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path_to_plot)
    plt.close()


def create_series_of_indicators_of_ICB_status(
    clinical_molecular_linkage_data: pd.DataFrame,
    expression_matrix: pd.DataFrame
) -> pd.Series:
    medications_data = pd.read_csv(paths.medications_data)
    output_of_pairing_clinical_data_and_stages_of_tumors = pd.read_csv(
        paths.output_of_pairing_clinical_data_and_stages_of_tumors
    )
    data_frame_of_output_and_clinical_molecular_linkage_data = create_data_frame_of_output_and_clinical_molecular_linkage_data(
        clinical_molecular_linkage_data,
        output_of_pairing_clinical_data_and_stages_of_tumors
    )
    data_frame_of_specimen_IDs_and_indicators_that_patient_received_ICB_therapy = (
        create_data_frame_of_specimen_IDs_and_indicators_that_patient_received_ICB_therapy(
            clinical_molecular_linkage_data,
            data_frame_of_output_and_clinical_molecular_linkage_data,
            medications_data,
            output_of_pairing_clinical_data_and_stages_of_tumors
        )
    )
    data_frame_of_clinical_data = (
        data_frame_of_output_and_clinical_molecular_linkage_data
        [data_frame_of_output_and_clinical_molecular_linkage_data["RNASeq"].notna()]
        .merge(
            pd.DataFrame(
                {"SampleID": expression_matrix.columns}
            ),
            how = "inner",
            left_on = "RNASeq",
            right_on = "SampleID",
            validate = "one_to_one"
        )
        .drop(columns = "SampleID")
        .merge(
            data_frame_of_specimen_IDs_and_indicators_that_patient_received_ICB_therapy[
                ["ORIENSpecimenID", "patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"]
            ],
            how = "left",
            left_on = "ORIENSpecimenID",
            right_on = "ORIENSpecimenID",
            validate = "one_to_one"
        )
    )
    data_frame_of_clinical_data["patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"] = (
        data_frame_of_clinical_data["patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"]
        .astype("boolean")
        .fillna(False)
    )
    series_of_indicators_of_ICB_status = (
        data_frame_of_clinical_data
        [["RNASeq", "patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"]]
        .set_index("RNASeq")
        ["patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"]
        .astype(int)
    )
    return series_of_indicators_of_ICB_status


def create_scatterplot_of_module_score_vs_ICB_status_by_sex(
    series_of_module_scores: pd.Series,
    series_of_indicators_of_ICB_status: pd.Series,
    series_of_indicators_of_sex: pd.Series,
    title: str,
    path_to_plot: str
):
    data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex = pd.DataFrame(
        {
            "module_score": series_of_module_scores,
            "ICB_indicator": series_of_indicators_of_ICB_status,
            "sex_indicator": series_of_indicators_of_sex
        }
    ).dropna()
    data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex["ICB_status"] = np.where(
        data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex["ICB_indicator"] == 0,
        "Naive",
        "Experienced"
    )
    data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex["Sex"] = np.where(
        data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex["sex_indicator"] == 0,
        "Female",
        "Male"
    )
    plt.figure()
    ax = sns.stripplot(
        data = data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex,
        x = "ICB_status",
        y = "module_score",
        hue = "Sex",
        jitter = 0.25
    )
    ax.set_xlabel("ICB status")
    ax.set_ylabel("module score")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path_to_plot)
    plt.close()


def create_plot_of_mean_module_score_vs_ICB_status_by_sex(
    series_of_module_scores: pd.Series,
    series_of_indicators_of_ICB_status: pd.Series,
    series_of_indicators_of_sex: pd.Series,
    title: str,
    path_to_plot: str
):
    data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex = pd.DataFrame(
        {
            "module_score": series_of_module_scores,
            "ICB_indicator": series_of_indicators_of_ICB_status,
            "sex_indicator": series_of_indicators_of_sex
        }
    ).dropna()
    data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex["ICB_status"] = np.where(
        data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex["ICB_indicator"] == 0,
        "Naive",
        "Experienced"
    )
    data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex["Sex"] = np.where(
        data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex["sex_indicator"] == 0,
        "Female",
        "Male"
    )
    data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex["ICB_status"] = pd.Categorical(
        data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex["ICB_status"],
        categories = ["Naive", "Experienced"],
        ordered = True
    )
    data_frame_of_sexes_ICB_statuses_and_mean_module_scores = (
        data_frame_of_module_scores_and_indicators_of_ICB_status_and_sex
        .groupby(
            ["Sex", "ICB_status"],
            observed = True
        )
        ["module_score"]
        .mean()
        .reset_index()
    )
    plt.figure()
    ax = sns.lineplot(
        data = data_frame_of_sexes_ICB_statuses_and_mean_module_scores,
        x = "ICB_status",
        y = "module_score",
        hue = "Sex",
        marker = "o"
    )
    ax.set_xlabel("ICB status")
    ax.set_ylabel("mean module score")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path_to_plot)
    plt.close()


def create_data_frame_of_genes_log_FCs_p_values_and_FDRs(
    expression_submatrix: pd.DataFrame,
    series_of_indicators: pd.Series
) -> pd.DataFrame:
    expression_submatrix_for_indicator_0 = expression_submatrix.loc[:, series_of_indicators == 0]
    expression_submatrix_for_indicator_1 = expression_submatrix.loc[:, series_of_indicators == 1]
    pseudocount: float = 1e-6
    series_of_mean_expressions_for_indicator_0 = expression_submatrix_for_indicator_0.mean(axis = 1) + pseudocount
    series_of_mean_expressions_for_indicator_1 = expression_submatrix_for_indicator_1.mean(axis = 1) + pseudocount
    series_of_log_FCs = np.log2(series_of_mean_expressions_for_indicator_1 / series_of_mean_expressions_for_indicator_0)
    list_of_p_values = []
    for gene in expression_submatrix.index:
        series_of_expressions_for_gene_and_indicator_0 = expression_submatrix_for_indicator_0.loc[gene].to_numpy()
        series_of_expressions_for_gene_and_indicator_1 = expression_submatrix_for_indicator_1.loc[gene].to_numpy()
        _, p_value = mannwhitneyu(
            series_of_expressions_for_gene_and_indicator_0,
            series_of_expressions_for_gene_and_indicator_1,
            alternative = "two-sided"
        )
        list_of_p_values.append(p_value)
    series_of_p_values = pd.Series(list_of_p_values, index = expression_submatrix.index, name = "p_value")
    series_of_FDRs = multipletests(series_of_p_values, method = "fdr_bh")[1]
    data_frame_of_genes_log_FCs_p_values_and_FDRs = pd.DataFrame(
        {
            "log_FC": series_of_log_FCs,
            "p_value": list_of_p_values,
            "FDR": series_of_FDRs
        }
    )
    data_frame_of_genes_log_FCs_p_values_and_FDRs.index.name = "gene"
    return data_frame_of_genes_log_FCs_p_values_and_FDRs


def create_volcano_plot(
    data_frame_of_genes_log_FCs_p_values_and_FDRs: pd.DataFrame,
    title: str,
    path_of_plot: str
):
    data_frame = data_frame_of_genes_log_FCs_p_values_and_FDRs.copy()
    data_frame["negative_log_base_10_of_p_value"] = -np.log10(data_frame["p_value"])
    significance_level = 0.05
    data_frame["indicator_of_whether_differential_expression_is_significant"] = (data_frame["FDR"] < significance_level)
    plt.figure()
    ax = sns.scatterplot(
        data = data_frame,
        x = "log_FC",
        y = "negative_log_base_10_of_p_value",
        hue = "indicator_of_whether_differential_expression_is_significant",
        s = 10
    )
    log_fold_change_of_1 = 1.0
    ax.axvline(-log_fold_change_of_1, lw = 1, c = "black")
    ax.axvline(log_fold_change_of_1, lw = 1, c = "black")
    log_base_10_of_significance_level = -np.log10(significance_level)
    ax.axhline(log_base_10_of_significance_level, lw = 1, c = "black")
    ax.set_xlabel("log_2(fold change)")
    ax.set_ylabel("-log_10(p value)")
    ax.set_title(title)
    ax.legend(title = f"FDR < {significance_level}")
    data_frame_of_scores = data_frame["negative_log_base_10_of_p_value"] * data_frame["log_FC"].abs()
    number_of_labels = 3
    for gene in data_frame_of_scores.nlargest(number_of_labels).index:
        horizontal_coordinate = data_frame.at[gene, "log_FC"]
        vertical_coordinate = data_frame.at[gene, "negative_log_base_10_of_p_value"]
        ax.text(horizontal_coordinate, vertical_coordinate, gene, fontsize = 7)
    plt.tight_layout()
    plt.savefig(path_of_plot)
    plt.close()


def create_expression_heatmap(
    expression_submatrix: pd.DataFrame,
    series_of_indicators: pd.Series,
    name_of_indicator_0: str,
    name_of_indicator_1: str,
    title: str,
    path_of_plot: str,
    data_frame_of_genes_log_FCs_p_values_and_FDRs: pd.DataFrame
):
    number_of_genes = 100
    data_frame = data_frame_of_genes_log_FCs_p_values_and_FDRs.copy()
    data_frame["volcano_score"] = -np.log10(data_frame["p_value"]) * data_frame["log_FC"].abs()
    series_of_indicators_that_differential_expression_is_significant = data_frame["FDR"] < 0.05
    data_frame_of_significant_genes_and_statistics = (
        data_frame[series_of_indicators_that_differential_expression_is_significant].copy()
    )
    data_frame_of_significant_or_insignificant_genes_and_statistics = (
        data_frame_of_significant_genes_and_statistics
        if not data_frame_of_significant_genes_and_statistics.empty
        else data_frame
    )
    index_of_significant_or_insignificant_genes_with_positive_log_FCs_and_highest_volcano_scores_and_log_FCs = (
        data_frame_of_significant_or_insignificant_genes_and_statistics
        [data_frame_of_significant_or_insignificant_genes_and_statistics["log_FC"] > 0]
        .sort_values(
            ["volcano_score", "log_FC"],
            ascending = [False, False]
        )
        .head(number_of_genes // 2)
        .index
    )
    index_of_significant_or_insignificant_genes_with_negative_log_FCs_highest_volcano_scores_and_lowest_log_FCs = (
        data_frame_of_significant_or_insignificant_genes_and_statistics
        [data_frame_of_significant_or_insignificant_genes_and_statistics["log_FC"] < 0]
        .sort_values(
            ["volcano_score", "log_FC"],
            ascending = [False, True]
        )
        .head(number_of_genes // 2)
        .index
    )
    union_of_indices = (
        pd.Index(index_of_significant_or_insignificant_genes_with_positive_log_FCs_and_highest_volcano_scores_and_log_FCs)
        .union(
            pd.Index(
                index_of_significant_or_insignificant_genes_with_negative_log_FCs_highest_volcano_scores_and_lowest_log_FCs
            )
        )
    )
    if len(union_of_indices) < number_of_genes:
        index_of_significant_or_insignificant_genes_other_than_genes_in_union_of_indices = (
            data_frame_of_significant_or_insignificant_genes_and_statistics
            .drop(union_of_indices)
            .sort_values("volcano_score", ascending = False)
            .head(number_of_genes - len(union_of_indices))
            .index
        )
        union_of_indices = union_of_indices.union(
            index_of_significant_or_insignificant_genes_other_than_genes_in_union_of_indices
        )
    if len(union_of_indices) < number_of_genes:
        raise Exception("Union of indices still does not have 100 genes.")
    expression_submatrix = expression_submatrix.loc[union_of_indices]
    z_scored_expression_matrix = z_score_expressions_for_gene(expression_submatrix)
    list_of_genes_ordered_by_indicator = list(series_of_indicators[series_of_indicators == 0].index) + list(series_of_indicators[series_of_indicators == 1].index)
    z_scored_expression_matrix = z_scored_expression_matrix.loc[:, list_of_genes_ordered_by_indicator]
    series_of_genes_and_colors = series_of_indicators.loc[list_of_genes_ordered_by_indicator].map(
        {
            0: "#1f77b4", # blue
            1: "#d62728" # red
        }
    )
    series_of_genes_and_colors.name = "Sex" if name_of_indicator_0 == "Female" else "ICB status"
    _ = sns.clustermap(
        z_scored_expression_matrix,
        row_cluster = False,
        col_cluster = False,
        col_colors = series_of_genes_and_colors,
        center = 0,
        cmap = "RdBu_r",
        vmin = -2,
        vmax = 2,
        xticklabels = False,
        yticklabels = False
    )
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path_of_plot)
    plt.close()


SIGNATURES_FOR_SUMMARY = [f"CD8_{i}" for i in range(1,7)] + ["CD8_B", "CD8_G"]
ALPHA_MAIN = 0.05
ALPHA_SUGG = 0.10

def _load_fgsea_csv(path: Path) -> pd.DataFrame:
    """Load an fgsea result and normalize expected columns (pathway, NES, padj)."""
    df = pd.read_csv(path)
    # Make a lowercase->original map to find the columns regardless of exact case
    lower_map = {c.lower(): c for c in df.columns}
    required = ["pathway", "nes", "padj"]
    missing = [r for r in required if r not in lower_map]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    df = df.rename(columns={
        lower_map["pathway"]: "pathway",
        lower_map["nes"]: "NES",
        lower_map["padj"]: "padj"
    })
    return df[["pathway", "NES", "padj"]]

def _summarize_stratum(df_fgsea: pd.DataFrame, stratum_label: str) -> pd.DataFrame:
    """Subset to our signatures and compute direction/significance."""
    sub = df_fgsea[df_fgsea["pathway"].isin(SIGNATURES_FOR_SUMMARY)].copy()
    # Direction convention based on your pipeline:
    # Sex coded Female=0, Male=1, correlations ranked descending → positive NES = higher in males.
    sub["direction"] = np.where(
        sub["NES"] > 0, "higher in males",
        np.where(sub["NES"] < 0, "higher in females", "no difference")
    )
    sub["significance"] = pd.cut(
        sub["padj"],
        bins=[-1, ALPHA_MAIN, ALPHA_SUGG, float("inf")],
        labels=[f"significant (padj<{ALPHA_MAIN})", f"suggestive (padj<{ALPHA_SUGG})", "ns"],
        right=False
    )
    sub["stratum"] = stratum_label
    return sub.sort_values(["padj", "pathway"])

def _format_lines(df_summary: pd.DataFrame, stratum_label: str) -> str:
    rows = df_summary[df_summary["stratum"] == stratum_label]
    lines = [f"{stratum_label}:"]
    for _, r in rows.iterrows():
        lines.append(
            f"  {r['pathway']}: {r['direction']} "
            f"(NES={r['NES']:.2f}, padj={r['padj']:.3g}; {r['significance']})"
        )
    return "\n".join(lines)

def create_sex_diff_summary_from_fgsea_files() -> tuple[pd.DataFrame, str]:
    """
    Load the three fgsea-by-sex CSVs produced earlier in main(),
    build a tidy summary across strata, and return (summary_df, pretty_text).
    """
    p_all = Path(paths.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_all_samples)
    p_naive = Path(paths.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_naive_samples)
    p_exper = Path(paths.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_experienced_samples)

    df_all   = _summarize_stratum(_load_fgsea_csv(p_all),   "all")
    df_naive = _summarize_stratum(_load_fgsea_csv(p_naive), "ICB-naive")
    df_exper = _summarize_stratum(_load_fgsea_csv(p_exper), "ICB-experienced")

    summary = pd.concat([df_all, df_naive, df_exper], ignore_index=True)
    summary = summary[["stratum", "pathway", "NES", "padj", "direction", "significance"]]

    pretty_text = "\n\n".join([
        _format_lines(summary, "all"),
        _format_lines(summary, "ICB-naive"),
        _format_lines(summary, "ICB-experienced"),
    ])
    return summary, pretty_text


def _summarize_icb(df_fgsea: pd.DataFrame) -> pd.DataFrame:
    """
    ICB indicator coded Naive=0, Experienced=1 in the ranks (stratum == 'all').
    Positive NES ⇒ enriched in ICB-experienced; Negative NES ⇒ enriched in ICB-naive.
    """
    sub = df_fgsea[df_fgsea["pathway"].isin(SIGNATURES_FOR_SUMMARY)].copy()
    sub["direction"] = np.where(
        sub["NES"] > 0, "higher in experienced",
        np.where(sub["NES"] < 0, "higher in naive", "no difference")
    )
    sub["significance"] = pd.cut(
        sub["padj"],
        bins=[-1, ALPHA_MAIN, ALPHA_SUGG, float("inf")],
        labels=[f"significant (padj<{ALPHA_MAIN})", f"suggestive (padj<{ALPHA_SUGG})", "ns"],
        right=False
    )
    sub["stratum"] = "ICB (all samples)"
    return sub.sort_values(["padj", "pathway"])[["stratum","pathway","NES","padj","direction","significance"]]

def create_icb_status_summary_from_fgsea_file() -> tuple[pd.DataFrame, str]:
    """
    Load the single fgsea-by-ICB CSV (built in main() for stratum=='all') and return
    a tidy summary plus a human-readable pretty text block.
    """
    p_icb = Path(paths.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_ICB_status)
    df = _load_fgsea_csv(p_icb)
    summary = _summarize_icb(df)
    # pretty text
    lines = ["ICB status (Naive vs Experienced):"]
    for _, r in summary.iterrows():
        lines.append(
            f"  {r['pathway']}: {r['direction']} "
            f"(NES={r['NES']:.2f}, padj={r['padj']:.3g}; {r['significance']})"
        )
    pretty_text = "\n".join(lines)
    return summary, pretty_text

def create_icb_module_score_summary(
    df_stats_icb: pd.DataFrame
) -> pd.DataFrame:
    """
    Condense the Mann–Whitney + BH-FDR table into a simple summary:
    means (naive/experienced), difference, direction, significance.
    Assumes df_stats_icb was created with name_of_group_0='naive', name_of_group_1='experienced'.
    """
    df = df_stats_icb.copy()
    # Ensure canonical column names are present
    required = ["mean_value_for_naive","mean_value_for_experienced","FDR"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' missing from ICB stats table.")
    out = df[["mean_value_for_naive","mean_value_for_experienced","FDR"]].copy()
    out["difference_between_mean_values_for_experienced_and_naive"] = (
        out["mean_value_for_experienced"] - out["mean_value_for_naive"]
    )
    out["direction"] = np.where(
        out["difference_between_mean_values_for_experienced_and_naive"] > 0,
        "u_exp > u_naive",
        "u_naive > u_exp"
    )
    out["difference_between_mean_values_is_significant"] = out["FDR"] < ALPHA_MAIN
    return out


def _summarize_icb_by_sex(df_fgsea: pd.DataFrame, sex_label: str) -> pd.DataFrame:
    """
    For a single sex (Female or Male): NES>0 ⇒ enriched in ICB-experienced within that sex.
    """
    sub = df_fgsea[df_fgsea["pathway"].isin(SIGNATURES_FOR_SUMMARY)].copy()
    sub["direction"] = np.where(
        sub["NES"] > 0, "higher in experienced",
        np.where(sub["NES"] < 0, "higher in naive", "no difference")
    )
    sub["significance"] = pd.cut(
        sub["padj"],
        bins=[-1, ALPHA_MAIN, ALPHA_SUGG, float("inf")],
        labels=[f"significant (padj<{ALPHA_MAIN})", f"suggestive (padj<{ALPHA_SUGG})", "ns"],
        right=False
    )
    sub["sex"] = sex_label
    sub["stratum"] = f"ICB ({sex_label})"
    return sub.sort_values(["padj", "pathway"])[["sex","stratum","pathway","NES","padj","direction","significance"]]

def create_icb_status_summary_by_sex_from_fgsea_files() -> tuple[pd.DataFrame, str]:
    """
    Read the two FGSEA-by-ICB CSVs generated within each sex and produce a tidy summary + pretty text.
    """
    base = paths.outputs_of_completing_Aim_1_2
    p_f = Path(base) / "fgsea_re_ICB_status_female.csv"
    p_m = Path(base) / "fgsea_re_ICB_status_male.csv"
    df_f = _summarize_icb_by_sex(_load_fgsea_csv(p_f), "Female")
    df_m = _summarize_icb_by_sex(_load_fgsea_csv(p_m), "Male")
    summary = pd.concat([df_f, df_m], ignore_index=True)
    # pretty text (grouped by sex)
    lines = []
    for sex_label in ["Female", "Male"]:
        lines.append(f"ICB status (Naive vs Experienced) within {sex_label}:")
        for _, r in summary[summary["sex"]==sex_label].iterrows():
            lines.append(
                f"  {r['pathway']}: {r['direction']} "
                f"(NES={r['NES']:.2f}, padj={r['padj']:.3g}; {r['significance']})"
            )
        lines.append("")  # blank line between sexes
    pretty_text = "\n".join(lines).strip()
    return summary, pretty_text

def create_icb_module_score_summary_by_sex(
    df_stats_icb_female: pd.DataFrame,
    df_stats_icb_male: pd.DataFrame
) -> pd.DataFrame:
    """
    Condense Mann–Whitney + BH-FDR ICB-status tables within each sex into a single tidy frame.
    Assumes columns mean_value_for_naive / mean_value_for_experienced / FDR are present.
    """
    def _summ(df, sex_label):
        out = df[["mean_value_for_naive","mean_value_for_experienced","FDR"]].copy()
        out["difference_between_mean_values_for_experienced_and_naive"] = (
            out["mean_value_for_experienced"] - out["mean_value_for_naive"]
        )
        out["direction"] = np.where(
            out["difference_between_mean_values_for_experienced_and_naive"] > 0,
            "u_exp > u_naive",
            "u_naive > u_exp"
        )
        out["difference_between_mean_values_is_significant"] = out["FDR"] < ALPHA_MAIN
        out["sex"] = sex_label
        return out
    fem = _summ(df_stats_icb_female, "Female")
    mal = _summ(df_stats_icb_male, "Male")
    return pd.concat([fem, mal], axis=0)


def create_volcano_plots_for_gene_sets(
    expression_submatrix: pd.DataFrame,
    series_of_indicators_of_sex: pd.Series,
    series_of_indicators_of_ICB_status: pd.Series,
    dictionary_of_names_of_sets_of_genes_and_lists_of_genes: dict[str, list[str]],
    output_dir: Path
):
    """
    For each gene set, make a volcano plot for Female vs Male and for ICB Naive vs Experienced.
    Skips sets with fewer than 3 genes present in the matrix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for set_name, gene_list in dictionary_of_names_of_sets_of_genes_and_lists_of_genes.items():
        # subset to genes present
        genes_present = [g for g in gene_list if g in expression_submatrix.index]
        if len(genes_present) < 3:
            # too few genes to make a meaningful volcano
            continue

        expr_subset = expression_submatrix.loc[genes_present]

        # Sex contrast
        df_sex = create_data_frame_of_genes_log_FCs_p_values_and_FDRs(expr_subset, series_of_indicators_of_sex)
        create_volcano_plot(
            data_frame_of_genes_log_FCs_p_values_and_FDRs=df_sex,
            title=f"Volcano ({set_name}) — Female vs Male",
            path_of_plot=str(output_dir / f"volcano_{set_name}_female_vs_male.png")
        )

        # ICB contrast
        df_icb = create_data_frame_of_genes_log_FCs_p_values_and_FDRs(expr_subset, series_of_indicators_of_ICB_status)
        create_volcano_plot(
            data_frame_of_genes_log_FCs_p_values_and_FDRs=df_icb,
            title=f"Volcano ({set_name}) — ICB Naive vs Experienced",
            path_of_plot=str(output_dir / f"volcano_{set_name}_naive_vs_experienced.png")
        )


def main():
    paths.ensure_dependencies_for_comparing_enrichment_scores_exist()
    clinical_molecular_linkage_data = pd.read_csv(paths.clinical_molecular_linkage_data)
    data_frame_of_sample_IDs_and_patient_IDs = clinical_molecular_linkage_data[["RNASeq", "ORIENAvatarKey"]]
    data_frame_of_sample_IDs_and_patient_IDs = (
        data_frame_of_sample_IDs_and_patient_IDs
        .sort_values(by = ["RNASeq", "ORIENAvatarKey"])
        .drop_duplicates(
            subset = ["RNASeq"],
            keep = "first"
        )
    )
    dictionary_of_names_of_sets_of_genes_and_lists_of_genes = load_dictionary_of_names_of_sets_of_genes_and_lists_of_genes()
    expression_matrix = pd.read_csv(
        paths.expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest,
        index_col = 0
    )
    list_of_names_of_sets_of_genes = [f"CD8_{i}" for i in range(1, 6 + 1)]
    list_of_names_of_sets_of_genes_in_set_CD8_B = [f"CD8_{i}" for i in (1, 2, 3)]
    list_of_names_of_sets_of_genes_in_set_CD8_G = [f"CD8_{i}" for i in (4, 5, 6)]
    patient_data = pd.read_csv(paths.patient_data)
    series_of_indicators_of_ICB_status = create_series_of_indicators_of_ICB_status(
        clinical_molecular_linkage_data,
        expression_matrix
    )
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
    dictionary_of_names_of_sets_of_genes_and_lists_of_genes["CD8_B"] = sorted_list_of_genes_in_set_CD8_B
    dictionary_of_names_of_sets_of_genes_and_lists_of_genes["CD8_G"] = sorted_list_of_genes_in_set_CD8_G
    for stratum in ["all", "naive", "experienced"]:
        if stratum == "all":
            series_of_indicators_of_ICB_status_for_stratum = series_of_indicators_of_ICB_status
        elif stratum == "naive":
            series_of_indicators_of_ICB_status_for_stratum = series_of_indicators_of_ICB_status[
                series_of_indicators_of_ICB_status == 0
            ]
        elif stratum == "experienced":
            series_of_indicators_of_ICB_status_for_stratum = series_of_indicators_of_ICB_status[
                series_of_indicators_of_ICB_status == 1
            ]
        else:
            raise Exception("Stratum is invalid.")
        expression_submatrix = expression_matrix[series_of_indicators_of_ICB_status_for_stratum.index]
        index_of_sample_IDs = series_of_indicators_of_ICB_status_for_stratum.index
        metadata_frame = pd.DataFrame(
            {"sample_id": index_of_sample_IDs}
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
        series_of_indicators_of_sex = metadata_frame["Sex"].map(
            {
                "Female": 0,
                "Male": 1
            }
        )
        if stratum == "all":
            data_frame_of_genes_log_FCs_p_values_and_FDRs_for_female_and_male_samples = create_data_frame_of_genes_log_FCs_p_values_and_FDRs(
                expression_submatrix,
                series_of_indicators_of_sex
            )
            data_frame_of_genes_log_FCs_p_values_and_FDRs_for_female_and_male_samples.to_csv(
                paths.data_frame_of_genes_log_FCs_p_values_and_FDRs_for_female_and_male_samples
            )
            create_volcano_plot(
                data_frame_of_genes_log_FCs_p_values_and_FDRs = (
                    data_frame_of_genes_log_FCs_p_values_and_FDRs_for_female_and_male_samples
                ),
                title = "Volcano Plot of Genes for Female and Male Samples",
                path_of_plot = paths.volcano_plot_for_female_and_male_samples
            )
            create_expression_heatmap(
                expression_submatrix = expression_submatrix,
                series_of_indicators = series_of_indicators_of_sex,
                name_of_indicator_0 = "Female",
                name_of_indicator_1 = "Male",
                title = "Heatmap of Normalized Expressions\nfor Top Differentially Expressed Genes\nfor Female and Male Samples",
                path_of_plot = paths.heatmap_of_normalized_expressions_for_top_differentially_expressed_genes_for_female_and_male_samples,
                data_frame_of_genes_log_FCs_p_values_and_FDRs = data_frame_of_genes_log_FCs_p_values_and_FDRs_for_female_and_male_samples
            )
            data_frame_of_genes_log_FCs_p_values_and_FDRs_for_naive_and_experienced_samples = create_data_frame_of_genes_log_FCs_p_values_and_FDRs(
                expression_submatrix,
                series_of_indicators_of_ICB_status_for_stratum
            )
            data_frame_of_genes_log_FCs_p_values_and_FDRs_for_naive_and_experienced_samples.to_csv(
                paths.data_frame_of_genes_log_FCs_p_values_and_FDRs_for_naive_and_experienced_samples
            )
            create_volcano_plot(
                data_frame_of_genes_log_FCs_p_values_and_FDRs = data_frame_of_genes_log_FCs_p_values_and_FDRs_for_naive_and_experienced_samples,
                title = "Volcano Plot of Genes for Naive and Experienced Samples",
                path_of_plot = paths.volcano_plot_for_naive_and_experienced_samples
            )
            gene_sets_for_volcanos = {
                **{name: dictionary_of_names_of_sets_of_genes_and_lists_of_genes[name] for name in [f"CD8_{i}" for i in range(1, 7)]},
                "CD8_B": dictionary_of_names_of_sets_of_genes_and_lists_of_genes["CD8_B"],
                "CD8_G": dictionary_of_names_of_sets_of_genes_and_lists_of_genes["CD8_G"]
            }
            create_volcano_plots_for_gene_sets(
                expression_submatrix = expression_submatrix,
                series_of_indicators_of_sex = series_of_indicators_of_sex,
                series_of_indicators_of_ICB_status = series_of_indicators_of_ICB_status_for_stratum,
                dictionary_of_names_of_sets_of_genes_and_lists_of_genes = gene_sets_for_volcanos,
                output_dir = paths.outputs_of_completing_Aim_1_2 / "volcano_by_gene_set"
            )
            create_expression_heatmap(
                expression_submatrix = expression_submatrix,
                series_of_indicators = series_of_indicators_of_ICB_status_for_stratum,
                name_of_indicator_0 = "Naive",
                name_of_indicator_1 = "Experienced",
                title = "Heatmap of Normalized Expressions\nfor Top Differentially Expressed Genes\nfor Naive and Experienced Samples",
                path_of_plot = paths.heatmap_of_normalized_expressions_for_top_differentially_expressed_genes_for_naive_and_experienced_samples,
                data_frame_of_genes_log_FCs_p_values_and_FDRs = data_frame_of_genes_log_FCs_p_values_and_FDRs_for_naive_and_experienced_samples
            )
        data_frame_of_genes_and_statistics_re_sex = compute_point_biserial_correlation_for_each_gene(
            expression_submatrix,
            series_of_indicators_of_sex
        )
        if stratum == "all":
            data_frame_of_genes_and_statistics_re_sex.to_csv(paths.data_frame_of_genes_and_statistics_re_sex_for_all_samples)
        elif stratum == "naive":
            data_frame_of_genes_and_statistics_re_sex.to_csv(paths.data_frame_of_genes_and_statistics_re_sex_for_naive_samples)
        elif stratum == "experienced":
            data_frame_of_genes_and_statistics_re_sex.to_csv(paths.data_frame_of_genes_and_statistics_re_sex_for_experienced_samples)
        else:
            raise Exception("Stratum is invalid.")
        series_of_point_biserial_correlations_re_sex = data_frame_of_genes_and_statistics_re_sex["point_biserial_correlation"].copy()
        series_of_ranks_re_sex = series_of_point_biserial_correlations_re_sex.rank(method = "first")
        # Point biserial correlations are ranked in ascending order.
        # When 2 point biserial correlations are equal, the first receives a lower rank and the second receives a higher rank.
        series_of_point_biserial_correlations_re_sex = series_of_point_biserial_correlations_re_sex + (series_of_ranks_re_sex * 1e-12)
        series_of_ranked_point_biserial_correlations_re_sex = series_of_point_biserial_correlations_re_sex.sort_values(ascending = False)
        if stratum == "all":
            data_frame_of_genes_and_statistics_re_ICB_status = compute_point_biserial_correlation_for_each_gene(
                expression_submatrix,
                series_of_indicators_of_ICB_status_for_stratum
            )
            data_frame_of_genes_and_statistics_re_ICB_status.to_csv(paths.data_frame_of_genes_and_statistics_re_ICB_status)
            series_of_point_biserial_correlations_re_ICB_status = data_frame_of_genes_and_statistics_re_ICB_status["point_biserial_correlation"].copy()
            series_of_ranks_re_ICB_status = series_of_point_biserial_correlations_re_ICB_status.rank(method = "first")
            series_of_point_biserial_correlations_re_ICB_status = series_of_point_biserial_correlations_re_ICB_status + (series_of_ranks_re_ICB_status * 1e-12)
            series_of_ranked_point_biserial_correlations_re_ICB_status = series_of_point_biserial_correlations_re_ICB_status.sort_values(ascending = False)
        data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex = run_fgsea(
            series_of_ranked_point_biserial_correlations_re_sex,
            dictionary_of_names_of_sets_of_genes_and_lists_of_genes
        )
        if stratum == "all":
            data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex.to_csv(
                paths.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_all_samples,
                index = False
            )
        elif stratum == "naive":
            data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex.to_csv(
                paths.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_naive_samples,
                index = False
            )
        elif stratum == "experienced":
            data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex.to_csv(
                paths.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_experienced_samples,
                index = False
            )
        else:
            raise Exception("Stratum is invalid.")
        if stratum == "all":
            plot_FDR_vs_Normalized_Enrichment_Score(
                data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex,
                paths.plot_of_FDR_vs_Normalized_Enrichment_Score_re_sex_for_all_samples
            )
            for sex_label, sex_code in [("Female", 0), ("Male", 1)]:
                idx = series_of_indicators_of_sex[series_of_indicators_of_sex == sex_code].index
                if len(idx) >= 3:  # guard for tiny subsets
                    expr_sex = expression_submatrix[idx]
                    icb_sex = series_of_indicators_of_ICB_status_for_stratum.loc[idx]
                    df_genes_icb_sex = compute_point_biserial_correlation_for_each_gene(expr_sex, icb_sex)
                    ser_icb = df_genes_icb_sex["point_biserial_correlation"].copy()
                    ranks_icb = ser_icb.rank(method="first")
                    ser_icb = ser_icb + (ranks_icb * 1e-12)
                    ser_ranked_icb = ser_icb.sort_values(ascending=False)
                    df_fgsea_icb_sex = run_fgsea(ser_ranked_icb, dictionary_of_names_of_sets_of_genes_and_lists_of_genes)
                    # Save per-sex FGSEA CSVs (used later for summaries)
                    out_csv = paths.outputs_of_completing_Aim_1_2 / f"fgsea_re_ICB_status_{sex_label.lower()}.csv"
                    df_fgsea_icb_sex.to_csv(out_csv, index=False)
        elif stratum == "naive":
            plot_FDR_vs_Normalized_Enrichment_Score(
                data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex,
                paths.plot_of_FDR_vs_Normalized_Enrichment_Score_re_sex_for_naive_samples
            )
        elif stratum == "experienced":
            plot_FDR_vs_Normalized_Enrichment_Score(
                data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex,
                paths.plot_of_FDR_vs_Normalized_Enrichment_Score_re_sex_for_experienced_samples
            )
        else:
            raise Exception("Stratum is invalid.")
        if stratum == "all":
            data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_ICB_status = run_fgsea(
                series_of_ranked_point_biserial_correlations_re_ICB_status,
                dictionary_of_names_of_sets_of_genes_and_lists_of_genes
            )
            data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_ICB_status.to_csv(
                paths.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_ICB_status,
                index = False
            )
            plot_FDR_vs_Normalized_Enrichment_Score(
                data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_ICB_status,
                paths.plot_of_FDR_vs_Normalized_Enrichment_Score_re_ICB_status
            )    
        dictionary_of_names_of_sets_of_genes_and_series_of_module_scores = {}
        for name_of_set_of_genes in list_of_names_of_sets_of_genes:
            list_of_genes = dictionary_of_names_of_sets_of_genes_and_lists_of_genes.get(name_of_set_of_genes)
            dictionary_of_names_of_sets_of_genes_and_series_of_module_scores[name_of_set_of_genes] = create_series_of_module_scores(
                expression_submatrix,
                list_of_genes
            )
        data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes = pd.DataFrame(
            dictionary_of_names_of_sets_of_genes_and_series_of_module_scores
        )
        data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes.index.name = "sample_id"
        data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes["CD8_B"] = create_series_of_module_scores(
            expression_submatrix,
            sorted_list_of_genes_in_set_CD8_B
        )
        data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes["CD8_G"] = create_series_of_module_scores(
            expression_submatrix,
            sorted_list_of_genes_in_set_CD8_G
        )
        if stratum == "all":
            data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes.to_csv(
                paths.data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes_for_all_samples
            )
        elif stratum == "naive":
            data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes.to_csv(
                paths.data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes_for_naive_samples
            )
        elif stratum == "experienced":
            data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes.to_csv(
                paths.data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes_for_experienced_samples
            )
        else:
            raise Exception("Stratum is invalid.")
        if stratum == "all":
            for name_of_set_of_genes in list_of_names_of_sets_of_genes:
                series_of_module_scores = (
                    data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes[name_of_set_of_genes]
                )
                create_scatterplot_of_module_score_vs_ICB_status_by_sex(
                    series_of_module_scores = series_of_module_scores,
                    series_of_indicators_of_ICB_status = series_of_indicators_of_ICB_status_for_stratum,
                    series_of_indicators_of_sex = series_of_indicators_of_sex,
                    title = f"Module Score vs. ICB Status for Set of Genes {name_of_set_of_genes} by Sex",
                    path_to_plot = (
                        paths.outputs_of_completing_Aim_1_2 /
                        f"plot_of_module_score_vs_ICB_status_by_sex_for_set_of_genes_{name_of_set_of_genes}.png"
                    )
                )
                create_plot_of_mean_module_score_vs_ICB_status_by_sex(
                    series_of_module_scores = series_of_module_scores,
                    series_of_indicators_of_ICB_status = series_of_indicators_of_ICB_status_for_stratum,
                    series_of_indicators_of_sex = series_of_indicators_of_sex,
                    title = f"Mean Module Score vs. ICB Status for Set of Genes {name_of_set_of_genes} by Sex",
                    path_to_plot = (
                        paths.outputs_of_completing_Aim_1_2 /
                        f"plot_of_mean_module_score_vs_ICB_status_by_sex_for_set_of_genes_{name_of_set_of_genes}.png"
                    )
                )
        series_of_CD8_B_module_scores_for_samples = create_series_of_module_scores(
            expression_submatrix,
            sorted_list_of_genes_in_set_CD8_B
        ).rename("CD8_B_module_score_for_sample")
        series_of_CD8_G_module_scores_for_samples = create_series_of_module_scores(
            expression_submatrix,
            sorted_list_of_genes_in_set_CD8_G
        ).rename("CD8_G_module_score_for_sample")
        series_of_differences = (
            series_of_CD8_G_module_scores_for_samples - series_of_CD8_B_module_scores_for_samples
        ).rename("difference_between_CD8_G_and_CD8_B_module_scores")
        dictionary_of_names_of_modules_and_series_of_values = {
            "CD8_G_minus_CD8_B": series_of_differences,
            "CD8_B": series_of_CD8_B_module_scores_for_samples,
            "CD8_G": series_of_CD8_G_module_scores_for_samples
        }
        for name_of_set_of_genes in list_of_names_of_sets_of_genes:
            dictionary_of_names_of_modules_and_series_of_values[name_of_set_of_genes] = (
                data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes[name_of_set_of_genes]
            )
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
        if stratum == "all":
            data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences.to_csv(
                paths.data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences_for_all_samples
            )
        elif stratum == "naive":
            data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences.to_csv(
                paths.data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences_for_naive_samples
            )
        elif stratum == "experienced":
            data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences.to_csv(
                paths.data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences_for_experienced_samples
            )
        else:
            raise Exception("Stratum is invalid.")
        list_of_data_frames_of_categories_of_module_scores_and_statistics_re_sex = []
        for name_of_set_of_genes, series_of_values in dictionary_of_names_of_modules_and_series_of_values.items():
            list_of_data_frames_of_categories_of_module_scores_and_statistics_re_sex.append(
                create_data_frame_of_category_of_module_score_and_statistics(
                    series_of_values,
                    series_of_indicators_of_sex,
                    "females",
                    "males",
                    name_of_set_of_genes
                )
            )
        data_frame_of_categories_of_module_scores_and_statistics_re_sex = pd.concat(
            list_of_data_frames_of_categories_of_module_scores_and_statistics_re_sex,
            axis = 0
        )
        data_frame_of_categories_of_module_scores_and_statistics_re_sex["FDR"] = multipletests(
            data_frame_of_categories_of_module_scores_and_statistics_re_sex["p_value"],
            method = "fdr_bh"
        )[1]
        summary = data_frame_of_categories_of_module_scores_and_statistics_re_sex[
            ["mean_value_for_males", "mean_value_for_females", "FDR"]
        ].copy()
        summary["difference_between_mean_values_for_males_and_females"] = (
            summary["mean_value_for_males"] - summary["mean_value_for_females"]
        )
        summary["direction"] = np.where(
            summary["difference_between_mean_values_for_males_and_females"] > 0,
            "u_M > u_F",
            "u_F > u_M"
        )
        summary["difference_between_mean_values_is_significant"] = summary["FDR"] < 0.05
        if stratum == "all":
            data_frame_of_categories_of_module_scores_and_statistics_re_sex.to_csv(
                paths.data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_all_samples
            )
            summary.to_csv(paths.summary_for_all_samples)
        elif stratum == "naive":
            data_frame_of_categories_of_module_scores_and_statistics_re_sex.to_csv(
                paths.data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_naive_samples
            )
            summary.to_csv(paths.summary_for_naive_samples)
        elif stratum == "experienced":
            data_frame_of_categories_of_module_scores_and_statistics_re_sex.to_csv(
                paths.data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_experienced_samples
            )
            summary.to_csv(paths.summary_for_experienced_samples)
        else:
            raise Exception("Stratum is invalid.")
        if stratum == "all":
            plot_value_vs_indicator(
                series_of_differences,
                series_of_indicators_of_sex,
                "Female",
                "Male",
                "Difference between CD8 G Module Score and CD8 B Module Score\nvs. Sex",
                paths.plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_all_samples
            )
        elif stratum == "naive":
            plot_value_vs_indicator(
                series_of_differences,
                series_of_indicators_of_sex,
                "Female",
                "Male",
                "Difference between CD8 G Module Score and CD8 B Module Score\nvs. Sex",
                paths.plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_naive_samples
            )
        elif stratum == "experienced":
            plot_value_vs_indicator(
                series_of_differences,
                series_of_indicators_of_sex,
                "Female",
                "Male",
                "Difference between CD8 G Module Score and CD8 B Module Score\nvs. Sex",
                paths.plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_experienced_samples
            )
        else:
            raise Exception("Stratum is invalid.")
        first_percentile, ninety_ninth_percentile = np.nanpercentile(
            series_of_logs_of_ratios_of_CD8_G_module_scores_to_CD8_B_module_scores,
            [1, 99]
        )
        series_of_clipped_logs_of_ratios = series_of_logs_of_ratios_of_CD8_G_module_scores_to_CD8_B_module_scores.clip(
            lower = first_percentile,
            upper = ninety_ninth_percentile
        )
        if stratum == "all":
            plot_value_vs_indicator(
                series_of_clipped_logs_of_ratios,
                series_of_indicators_of_sex,
                "Female",
                "Male",
                "Log of Ratio of CD8 G Module Score and CD8 B Module Score\nvs. Sex",
                paths.plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_all_samples
            )
        elif stratum == "naive":
            plot_value_vs_indicator(
                series_of_clipped_logs_of_ratios,
                series_of_indicators_of_sex,
                "Female",
                "Male",
                "Log of Ratio of CD8 G Module Score and CD8 B Module Score\nvs. Sex",
                paths.plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_naive_samples
            )
        elif stratum == "experienced":
            plot_value_vs_indicator(
                series_of_clipped_logs_of_ratios,
                series_of_indicators_of_sex,
                "Female",
                "Male",
                "Log of Ratio of CD8 G Module Score and CD8 B Module Score\nvs. Sex",
                paths.plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_experienced_samples
            )
        else:
            raise Exception("Stratum is invalid.")
        if stratum == "all":
            list_of_data_frames_of_categories_of_module_scores_and_statistics_re_ICB_status = []
            data_frame_of_category_of_module_score_CD8_G_minus_CD8_B_and_statistics_re_ICB_status = create_data_frame_of_category_of_module_score_and_statistics(
                series_of_differences,
                series_of_indicators_of_ICB_status_for_stratum,
                "naive",
                "experienced",
                "CD8_G_minus_CD8_B"
            )
            list_of_data_frames_of_categories_of_module_scores_and_statistics_re_ICB_status.append(
                data_frame_of_category_of_module_score_CD8_G_minus_CD8_B_and_statistics_re_ICB_status
            )
            for name_of_set_of_genes, series_of_module_scores in {
                "CD8_B": series_of_CD8_B_module_scores_for_samples,
                "CD8_G": series_of_CD8_G_module_scores_for_samples
            }.items():
                list_of_data_frames_of_categories_of_module_scores_and_statistics_re_ICB_status.append(
                    create_data_frame_of_category_of_module_score_and_statistics(
                        series_of_module_scores,
                        series_of_indicators_of_ICB_status_for_stratum,
                        "naive",
                        "experienced",
                        name_of_set_of_genes
                    )
                )
            for name_of_set_of_genes in list_of_names_of_sets_of_genes:
                series_of_module_scores = data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes[name_of_set_of_genes]
                list_of_data_frames_of_categories_of_module_scores_and_statistics_re_ICB_status.append(
                    create_data_frame_of_category_of_module_score_and_statistics(
                        series_of_module_scores,
                        series_of_indicators_of_ICB_status_for_stratum,
                        "naive",
                        "experienced",
                        name_of_set_of_genes
                    )
                )
            data_frame_of_categories_of_module_scores_and_statistics_re_ICB_status = pd.concat(
                list_of_data_frames_of_categories_of_module_scores_and_statistics_re_ICB_status,
                axis = 0
            )
            data_frame_of_categories_of_module_scores_and_statistics_re_ICB_status["FDR"] = multipletests(
                data_frame_of_categories_of_module_scores_and_statistics_re_ICB_status["p_value"],
                method = "fdr_bh"
            )[1]
            data_frame_of_categories_of_module_scores_and_statistics_re_ICB_status.to_csv(
                paths.data_frame_of_categories_of_module_scores_and_statistics_re_ICB_status
            )
            icb_mod_summary = create_icb_module_score_summary(data_frame_of_categories_of_module_scores_and_statistics_re_ICB_status)
            icb_mod_summary_path = paths.outputs_of_completing_Aim_1_2 / "summary_by_ICB_status.csv"
            icb_mod_summary.to_csv(icb_mod_summary_path)
            sex_icb_tables = {}
            for sex_label, sex_code in [("Female", 0), ("Male", 1)]:
                idx = series_of_indicators_of_sex[series_of_indicators_of_sex == sex_code].index
                if len(idx) >= 3:
                    icb_ind = series_of_indicators_of_ICB_status_for_stratum.loc[idx]
                    per_sig = []
                    for module_name, series_vals in dictionary_of_names_of_modules_and_series_of_values.items():
                        per_sig.append(
                            create_data_frame_of_category_of_module_score_and_statistics(
                                series_vals.loc[idx], icb_ind, "naive", "experienced", module_name
                            )
                        )
                    df_sex = pd.concat(per_sig, axis=0)
                    df_sex["FDR"] = multipletests(df_sex["p_value"], method="fdr_bh")[1]
                    sex_icb_tables[sex_label] = df_sex
                    # write the per-sex detailed table
                    (paths.outputs_of_completing_Aim_1_2 / f"data_frame_of_categories_of_module_scores_and_statistics_re_ICB_status_{sex_label.lower()}.csv"
                    ).write_text(df_sex.to_csv())
            # concise module-score summaries by sex (if both exist)
            if "Female" in sex_icb_tables or "Male" in sex_icb_tables:
                fem_df = sex_icb_tables.get("Female", pd.DataFrame(columns=["mean_value_for_naive","mean_value_for_experienced","FDR"]))
                mal_df = sex_icb_tables.get("Male",   pd.DataFrame(columns=["mean_value_for_naive","mean_value_for_experienced","FDR"]))
                icb_mod_by_sex = create_icb_module_score_summary_by_sex(fem_df, mal_df)
                (paths.outputs_of_completing_Aim_1_2 / "summary_by_ICB_status_by_sex.csv").write_text(
                    icb_mod_by_sex.to_csv()
                )
            plot_value_vs_indicator(
                series_of_differences,
                series_of_indicators_of_ICB_status_for_stratum,
                "naive",
                "experienced",
                "Difference between CD8 G Module Score and CD8 B Module Score\nvs. ICB Status",
                paths.plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_ICB_status
            )
            plot_value_vs_indicator(
                series_of_clipped_logs_of_ratios,
                series_of_indicators_of_ICB_status_for_stratum,
                "naive",
                "experienced",
                "Log of Ratio of CD8 G Module Score and CD8 B Module Score\nvs. ICB Status",
                paths.plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_ICB_status
            )

    summary_df, pretty_text = create_sex_diff_summary_from_fgsea_files()
    summary_csv_path = paths.outputs_of_completing_Aim_1_2 / "summary_of_sex_differences_for_CD8_signatures_by_stratum.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    icb_summary_df, icb_pretty = create_icb_status_summary_from_fgsea_file()
    icb_summary_csv_path = paths.outputs_of_completing_Aim_1_2 / "summary_of_ICB_status_differences_for_CD8_signatures.csv"
    icb_summary_df.to_csv(icb_summary_csv_path, index=False)
    (paths.outputs_of_completing_Aim_1_2 / "summary_of_ICB_status_differences_pretty.txt").write_text(
        icb_pretty, encoding="utf-8"
    )
    icb_by_sex_df, icb_by_sex_pretty = create_icb_status_summary_by_sex_from_fgsea_files()
    icb_by_sex_csv = paths.outputs_of_completing_Aim_1_2 / "summary_of_ICB_status_differences_for_CD8_signatures_by_sex.csv"
    icb_by_sex_df.to_csv(icb_by_sex_csv, index=False)
    (paths.outputs_of_completing_Aim_1_2 / "summary_of_ICB_status_differences_by_sex_pretty.txt").write_text(
        icb_by_sex_pretty, encoding="utf-8"
    )


if __name__ == "__main__":
    main()