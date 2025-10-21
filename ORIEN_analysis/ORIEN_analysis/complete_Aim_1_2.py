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
    output_of_pairing_clinical_data_and_stages_of_tumors = pd.read_csv(paths.output_of_pairing_clinical_data_and_stages_of_tumors)
    medications_data = pd.read_csv(paths.medications_data)
    set_of_medications_for_ICB_therapy = {
        "Pembrolizumab", "Nivolumab", # PD‑1
        "Atezolizumab", # PD‑L1
        "Ipilimumab" # CTLA‑4
    }
    medications_data["medication_is_for_ICB_therapy"] = medications_data["Medication"].isin(
        set_of_medications_for_ICB_therapy
    )
    medications_data_for_ICB_therapy = medications_data[medications_data["medication_is_for_ICB_therapy"]].copy()
    medications_data_for_ICB_therapy["AgeAtMedStart"] = medications_data_for_ICB_therapy["AgeAtMedStart"].apply(numericize_age)
    data_frame_of_output_and_clinical_molecular_linkage_data = (
        output_of_pairing_clinical_data_and_stages_of_tumors[["ORIENSpecimenID", "EKN Assigned Stage", "AvatarKey"]]
        .merge(
            clinical_molecular_linkage_data[["DeidSpecimenID", "Age At Specimen Collection", "RNASeq"]],
            how = "left",
            left_on = "ORIENSpecimenID",
            right_on = "DeidSpecimenID",
            validate = "one_to_one"
        )
        .drop(columns = "DeidSpecimenID")
        .rename(
            columns = {
                "Age At Specimen Collection": "Age_At_Specimen_Collection",
                "EKN Assigned Stage": "EKN_Assigned_Stage"
            }
        )
    )
    data_frame_of_output_and_clinical_molecular_linkage_data["Age_At_Specimen_Collection"] = (
        data_frame_of_output_and_clinical_molecular_linkage_data["Age_At_Specimen_Collection"].apply(numericize_age)
    )
    data_frame_of_output_clinical_molecular_linkage_and_medications_data = (
        data_frame_of_output_and_clinical_molecular_linkage_data[["ORIENSpecimenID", "AvatarKey", "Age_At_Specimen_Collection"]]
        .merge(
            medications_data_for_ICB_therapy[["AvatarKey", "AgeAtMedStart"]],
            on = "AvatarKey",
            how = "left"
        )
    )
    data_frame_of_output_clinical_molecular_linkage_and_medications_data["patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"] = (
        data_frame_of_output_clinical_molecular_linkage_and_medications_data["AgeAtMedStart"] <= data_frame_of_output_clinical_molecular_linkage_and_medications_data["Age_At_Specimen_Collection"]
    )
    data_frame_of_specimen_IDs_and_indicators_that_patient_received_ICB_therapy = (
        data_frame_of_output_clinical_molecular_linkage_and_medications_data
        .groupby("ORIENSpecimenID", as_index = False)
        ["patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"]
        .agg(lambda series: series.any())
    )
    data_frame_of_enrichment_scores_and_clinical_and_QC_data = (
        data_frame_of_output_and_clinical_molecular_linkage_data[
            data_frame_of_output_and_clinical_molecular_linkage_data["RNASeq"].notna()
        ]
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
    data_frame_of_enrichment_scores_and_clinical_and_QC_data["patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"] = (
        data_frame_of_enrichment_scores_and_clinical_and_QC_data["patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"]
        .astype("boolean")
        .fillna(False)
    )
    series_of_indicators_of_ICB_status = (
        data_frame_of_enrichment_scores_and_clinical_and_QC_data[
            ["RNASeq", "patient_received_ICB_therapy_at_or_before_age_of_specimen_collection"]
        ]
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


def _per_gene_de_stats(expression_matrix: pd.DataFrame,
                       indicators: pd.Series,
                       name0: str,
                       name1: str,
                       pseudocount: float = 1e-6) -> pd.DataFrame:
    """
    Compute per-gene stats between group0 (indicator==0) and group1 (indicator==1).
    Returns a DataFrame with columns: log2_fc (group1/group0), p_value, FDR.
    """
    expr = expression_matrix.loc[:, indicators.index]  # align columns
    group0 = expr.loc[:, indicators == 0]
    group1 = expr.loc[:, indicators == 1]

    # log2 fold change of means with tiny pseudocount (robust for TPM)
    mean0 = (group0.mean(axis=1) + pseudocount)
    mean1 = (group1.mean(axis=1) + pseudocount)
    log2_fc = np.log2(mean1 / mean0)

    # Mann–Whitney U, two-sided (works for non-normal TPM)
    pvals = []
    for g in expr.index:
        a = group0.loc[g].to_numpy()
        b = group1.loc[g].to_numpy()
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and (a.size > 0) and (b.size > 0):
            # If all values identical across both groups, p=1.0 quickly
            if (np.nanstd(a) == 0) and (np.nanstd(b) == 0) and (np.nanmean(a) == np.nanmean(b)):
                pvals.append(1.0)
            else:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
                pvals.append(p)
        else:
            pvals.append(np.nan)
    pvals = pd.Series(pvals, index=expr.index, name="p_value")
    FDR = pd.Series(multipletests(pvals.fillna(1.0), method="fdr_bh")[1],
                    index=expr.index, name="FDR")

    out = pd.DataFrame({"log2_fc": log2_fc, "p_value": pvals, "FDR": FDR})
    out.index.name = "gene"
    return out


def create_volcano_plot(de_df: pd.DataFrame,
                        title: str,
                        path_to_plot: str,
                        fdr_thresh: float = 0.05,
                        lfc_thresh: float = 1.0,
                        top_labels: int = 15):
    """
    de_df has columns: log2_fc, p_value, FDR and index=gene.
    """
    df = de_df.copy()
    df["neglog10p"] = -np.log10(df["p_value"].clip(lower=1e-300))
    df["sig"] = (df["FDR"] < fdr_thresh)

    plt.figure(figsize=(7, 6))
    ax = sns.scatterplot(data=df, x="log2_fc", y="neglog10p", hue="sig", edgecolor=None, s=10)
    ax.axvline(-lfc_thresh, ls="--", lw=1, c="gray")
    ax.axvline( lfc_thresh, ls="--", lw=1, c="gray")
    ax.axhline(-np.log10(fdr_thresh), ls="--", lw=1, c="gray")
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)
    ax.legend(title=f"FDR < {fdr_thresh:g}", loc="upper right")

    # annotate a few top hits by |lfc| * -log10(p)
    score = df["neglog10p"] * df["log2_fc"].abs()
    for gene in score.nlargest(top_labels).index:
        x = df.at[gene, "log2_fc"]
        y = df.at[gene, "neglog10p"]
        ax.text(x, y, gene, fontsize=7)

    plt.tight_layout()
    plt.savefig(path_to_plot, dpi=300)
    plt.close()


def create_expression_heatmap(expression_matrix: pd.DataFrame,
                              indicators: pd.Series,
                              name0: str,
                              name1: str,
                              title: str,
                              path_to_plot: str,
                              de_df: pd.DataFrame | None = None,
                              max_genes: int = 100,
                              cluster_rows: bool = False):
    """
    Heatmap across a *subset* of genes to avoid OOM:
      - If de_df is provided, select up to max_genes (half up, half down) among FDR<0.05 by |log2_fc|.
      - Otherwise, randomly sample up to max_genes genes.
    Rows are z-scored per gene. Columns annotated by group.
    """
    # align
    expr = expression_matrix.loc[:, indicators.index]

    # choose genes
    if de_df is not None and ("log2_fc" in de_df.columns) and ("FDR" in de_df.columns):
        sig = de_df[de_df["FDR"] < 0.05].copy()
        if sig.empty:
            chosen = de_df.reindex(expr.index).nlargest(max_genes, columns="log2_fc", keep="all").index
        else:
            k_each = max_genes // 2
            up = sig.sort_values("log2_fc", ascending=False).head(k_each).index
            down = sig.sort_values("log2_fc", ascending=True).head(k_each).index
            chosen = pd.Index(up).union(pd.Index(down))
    else:
        chosen = pd.Index(expr.index).to_series().sample(min(max_genes, expr.shape[0]), random_state=0).index

    expr = expr.loc[chosen]

    # z-score per gene (float32 to reduce memory)
    z_expr = z_score_expressions_for_gene(expr).astype(np.float32)

    # column order: group0 then group1
    ordered_cols = list(indicators[indicators == 0].index) + list(indicators[indicators == 1].index)
    z_expr = z_expr.loc[:, ordered_cols]

    # column colors
    col_colors = indicators.loc[ordered_cols].map({0: name0, 1: name1})
    palette = {name0: "#1f77b4", name1: "#d62728"}  # blue / red
    col_colors = col_colors.map(palette)
    col_colors.name = "Sex" if name0 == "Female" else "ICB status"

    # Use heatmap (no row clustering) to avoid O(n^2) memory
    g = sns.clustermap(z_expr,
                       col_cluster=False,
                       row_cluster=cluster_rows,   # default False
                       cmap="RdBu_r",
                       xticklabels=False,
                       yticklabels=False,
                       col_colors=col_colors,
                       figsize=(8, 10))
    plt.suptitle(title, y=1.02)
    plt.savefig(path_to_plot, dpi=300, bbox_inches="tight")
    plt.close()


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
            de_sex = _per_gene_de_stats(
                expression_submatrix,
                series_of_indicators_of_sex,
                name0="Female", name1="Male"
            )
            de_sex.to_csv(paths.outputs_of_completing_Aim_1_2 / "DE_sex_male_vs_female_all_samples.csv")
            create_volcano_plot(
                de_df=de_sex,
                title="Volcano: Male vs Female (all samples)",
                path_to_plot=paths.outputs_of_completing_Aim_1_2 / "volcano_male_vs_female_all_samples.png"
            )
            # Sex heatmap
            create_expression_heatmap(
                expression_matrix=expression_submatrix,
                indicators=series_of_indicators_of_sex,
                name0="Female", name1="Male",
                title="Expression heatmap (top DE genes): Female vs Male",
                path_to_plot=paths.outputs_of_completing_Aim_1_2 / "heatmap_topDE_female_vs_male.png",
                de_df=de_sex, max_genes=100, cluster_rows=False
            )
        # ---------- NEW: Volcano + Heatmap for ICB (Experienced vs Naive) ----------
        if stratum == "all":
            de_icb = _per_gene_de_stats(
                expression_submatrix,
                series_of_indicators_of_ICB_status_for_stratum,
                name0="Naive", name1="Experienced"
            )
            de_icb.to_csv(paths.outputs_of_completing_Aim_1_2 / "DE_ICB_experienced_vs_naive_all_samples.csv")
            create_volcano_plot(
                de_df=de_icb,
                title="Volcano: ICB Experienced vs Naive (all samples)",
                path_to_plot=paths.outputs_of_completing_Aim_1_2 / "volcano_ICB_experienced_vs_naive_all_samples.png"
            )
            # ICB heatmap
            create_expression_heatmap(
                expression_matrix=expression_submatrix,
                indicators=series_of_indicators_of_ICB_status_for_stratum,
                name0="Naive", name1="Experienced",
                title="Expression heatmap (top DE genes): ICB Naive vs Experienced",
                path_to_plot=paths.outputs_of_completing_Aim_1_2 / "heatmap_topDE_ICB_naive_vs_experienced.png",
                de_df=de_icb, max_genes=100, cluster_rows=False
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
        data_frame_of_category_of_module_score_CD8_G_minus_CD8_B_and_statistics_re_sex = create_data_frame_of_category_of_module_score_and_statistics(
            series_of_differences,
            series_of_indicators_of_sex,
            "females",
            "males",
            "CD8_G_minus_CD8_B"
        )
        list_of_data_frames_of_categories_of_module_scores_and_statistics_re_sex.append(
            data_frame_of_category_of_module_score_CD8_G_minus_CD8_B_and_statistics_re_sex
        )
        for name_of_set_of_genes in list_of_names_of_sets_of_genes:
            series_of_module_scores = data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes[name_of_set_of_genes]
            list_of_data_frames_of_categories_of_module_scores_and_statistics_re_sex.append(
                create_data_frame_of_category_of_module_score_and_statistics(
                    series_of_module_scores,
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
        if stratum == "all":
            data_frame_of_categories_of_module_scores_and_statistics_re_sex.to_csv(
                paths.data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_all_samples
            )
        elif stratum == "naive":
            data_frame_of_categories_of_module_scores_and_statistics_re_sex.to_csv(
                paths.data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_naive_samples
            )
        elif stratum == "experienced":
            data_frame_of_categories_of_module_scores_and_statistics_re_sex.to_csv(
                paths.data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_experienced_samples
            )
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


if __name__ == "__main__":
    main()