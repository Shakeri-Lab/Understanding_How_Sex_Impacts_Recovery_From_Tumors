'''
create_expression_matrices.py

We build an expression matrix with 60,609 Ensembl IDs and 333 sample IDs.
We filter the expression matrix to columns with sample IDs
in a data frame of filtered QC data and IDs of cutaneous specimens
output by our pipeline for pairing clinical data and stages of tumors.
'''


from ORIEN_analysis.config import paths
from datetime import datetime, timezone
import numpy as np
import math
import os
import pandas as pd
from functools import reduce
import operator


def _build_ensembl_to_hgnc_mapping(expr_files: list) -> pd.Series:
    """
    Build a Series mapping Ensembl ID (no version) -> HGNC symbol,
    using the user's provided approach (first non-empty per gene_id).
    """
    if not expr_files:
        raise Exception(f"No .genes.results files were found in {paths.gene_and_transcript_expression_results}.")
    ser = (
        pd.concat(
            [pd.read_csv(f, sep="\t", usecols=["gene_id", "gene_symbol"]) for f in expr_files],
            ignore_index=True
        )
        .assign(gene_id=lambda df: df["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True))
        .query("gene_id != ''")
        .query("gene_symbol != ''")
        .drop_duplicates("gene_id")
        .set_index("gene_id")["gene_symbol"]
    )
    return ser


def _collapse_to_hgnc_median(tpm_ensembl: pd.DataFrame, ens_to_hgnc: pd.Series) -> pd.DataFrame:
    """
    Align Ensembl->HGNC mapping to TPM rows (Ensembl IDs without version),
    keep only mapped rows, attach HGNC, and collapse duplicates by per-sample median.
    """
    # Align mapping to the TPM row index (labels = Ensembl IDs)
    hgnc_aligned = ens_to_hgnc.reindex(tpm_ensembl.index)

    # Keep rows with a non-null HGNC symbol
    keep_mask = hgnc_aligned.notna()
    dropped = int((~keep_mask).sum())
    if dropped > 0:
        print(f"Dropping {dropped} Ensembl IDs without HGNC mapping.")

    # Subset TPM by the aligned boolean mask (indexes now match)
    tpm_mapped = tpm_ensembl.loc[keep_mask].copy()

    # Attach HGNC as a column (same order as tpm_mapped rows)
    tpm_mapped["HGNC"] = hgnc_aligned.loc[keep_mask].astype(str).values

    # Collapse Ensembl duplicates to HGNC by per-sample median
    tpm_hgnc = tpm_mapped.groupby("HGNC", sort=False).median(numeric_only=True)
    return tpm_hgnc


def _low_expression_filter(tpm_df: pd.DataFrame, threshold: float = 1.0, min_prop: float = 0.20) -> tuple[pd.DataFrame, pd.Series]:
    """
    Keep genes with TPM > threshold in at least ceil(min_prop * n_samples) samples.
    Returns (filtered_df, mask_used).
    """
    n_samples = tpm_df.shape[1]
    if n_samples == 0:
        return tpm_df.copy(), pd.Series(False, index=tpm_df.index)
    min_count = max(1, math.ceil(min_prop * n_samples))
    mask = (tpm_df > threshold).sum(axis=1) >= min_count
    return tpm_df.loc[mask], mask


def _log2_tpm_plus1(df: pd.DataFrame) -> pd.DataFrame:
    return np.log2(df.astype(float) + 1.0)


def _rowwise_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gene-wise z-score across samples (rows standardized).
    Uses population SD (ddof=0). Constant rows become 0.
    """
    means = df.mean(axis=1)
    stds = df.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = df.sub(means, axis=0).div(stds, axis=0).fillna(0.0)
    return z


def create_full_expression_matrix(list_of_paths: str) -> pd.DataFrame:
    dictionary_of_sample_IDs_and_series_of_expressions: dict[str, pd.Series] = {}
    for path in list_of_paths:
        sample_ID = os.path.basename(path).removesuffix(".genes.results")
        series_of_expressions = (
            pd.read_csv(
                path,
                sep = '\t',
                usecols = ["gene_id", "TPM"],
                dtype = {
                    "gene_id": str,
                    "TPM": float
                }
            )
            .assign(
                gene_id = lambda df: df["gene_id"].str.replace(r"\.\d+$", "", regex = True)
            )
            .set_index("gene_id")
            ["TPM"]
        )
        dictionary_of_sample_IDs_and_series_of_expressions[sample_ID] = series_of_expressions
    full_expression_matrix = pd.DataFrame(dictionary_of_sample_IDs_and_series_of_expressions)
    print(f"Full expression matrix has shape {full_expression_matrix.shape}.")
    print(f"The number of unique genes is {full_expression_matrix.index.nunique()}.")
    print(f"The number of unique sample IDs is {full_expression_matrix.columns.nunique()}.")
    return full_expression_matrix


def load_QC_data():
    QC_data = pd.read_csv(
        paths.QC_data,
        dtype = {
            "ORIENAvatarKey": str,
            "QCCheck": str,
            "SLID": str
        }
    )
    print(f"QC data has shape {QC_data.shape}.")
    print(f"The number of unique patient IDs is {QC_data['ORIENAvatarKey'].nunique()}.")
    print(f"The number of unique sample IDs is {QC_data['SLID'].nunique()}.")
    return QC_data


def load_clinical_molecular_linkage_data():
    clinical_molecular_linkage_data = pd.read_csv(
        paths.clinical_molecular_linkage_data,
        dtype = {
            "DeidSpecimenID": str,
            "RNASeq": str
        }
    )
    print(f"Clinical molecular linkage data has shape {clinical_molecular_linkage_data.shape}.")
    print(f"The number of unique specimen IDs is {clinical_molecular_linkage_data['DeidSpecimenID'].nunique()}.")
    print(f"The number of unique sample IDs is {clinical_molecular_linkage_data['RNASeq'].nunique()}.")
    return clinical_molecular_linkage_data


def load_output_of_pipeline():
    output_of_pipeline = pd.read_csv(
        paths.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors,
        dtype = {
            "AvatarKey": str,
            "AssignedPrimarySite": str
        }
    )
    print(f"Output of pipeline for pairing clinical data and stages of tumors has shape {output_of_pipeline.shape}.")
    print(f"The number of unique specimen IDs is {output_of_pipeline['ORIENSpecimenID'].nunique()}.")
    return output_of_pipeline


def load_diagnosis_data():
    diagnosis_data = pd.read_csv(paths.diagnosis_data).reset_index()
    print(f"Diagnosis data has shape {diagnosis_data.shape}")
    return diagnosis_data


def merge_data(
    QC_data,
    clinical_molecular_linkage_data,
    output_of_pipeline,
    diagnosis_data
):
    manifest = (
        QC_data[
            [
                "ORIENAvatarKey",
                "SLID",
                "NexusBatch"
            ]
        ]
        .rename(
            columns = {"ORIENAvatarKey": "PATIENT_ID"}
        )
        .merge(
            clinical_molecular_linkage_data[
                [
                    "Age At Specimen Collection", # Column "Age At Specimen Collection" values in the spirit of collection dates.
                    "DeidSpecimenID",
                    "RNASeq",
                    "SpecimenSiteOfCollection"
                ]
            ],
            how = "left",
            left_on = "SLID",
            right_on = "RNASeq"
        )
        .drop(columns = "RNASeq")
        .rename(columns = {"SpecimenSiteOfCollection": "SpecimenSite"})
        .merge(
            output_of_pipeline[
                [
                    "AssignedPrimarySite",
                    "ORIENSpecimenID",
                    "index_of_row_of_diagnosis_data_paired_with_specimen"
                ]
            ],
            how = "left",
            left_on = "DeidSpecimenID",
            right_on = "ORIENSpecimenID"
        )
        .merge(
            diagnosis_data[
                [
                    "index",
                    "HistologyCode"
                ]
            ],
            how = "left",
            left_on = "index_of_row_of_diagnosis_data_paired_with_specimen",
            right_on = "index"
        )
        .drop(columns = "index_of_row_of_diagnosis_data_paired_with_specimen")
        .rename(columns = {"index": "DiagnosisID"})
    )
    return manifest


def provide_name_of_first_column_whose_name_matches_a_candidate(
    QC_data: pd.DataFrame,
    list_of_candidates: list[str]
) -> str | None:
    for candidate in list_of_candidates:
        if candidate in QC_data.columns:
            return candidate
    return None


def create_dictionary_of_names_of_standard_columns_and_sources(QC_data):
    dictionary_of_names_of_standard_columns_and_possible_sources = {
        "AlignmentRate": ["AlignmentRate", "PercentAligned", "pct_aligned"],
        "rRNARate": ["rRNA Rate", "rRNARate", "Pct_rRNA"],
        "Duplication": ["Duplication", "DupRate", "PCT_DUPLICATION"],
        "MappedReads": ["MappedReads", "Total Mapped Reads"],
        "ExonicRate": ["ExonicRate", "Pct_Exonic"]
    }
    dictionary_of_names_of_standard_columns_and_sources = {
        name_of_standard_column: provide_name_of_first_column_whose_name_matches_a_candidate(QC_data, list_of_candidates)
        for name_of_standard_column, list_of_candidates in dictionary_of_names_of_standard_columns_and_possible_sources.items()
    }
    return dictionary_of_names_of_standard_columns_and_sources


def standardize_columns(
    QC_data: pd.DataFrame,
    dictionary_of_names_of_standard_columns_and_sources: dict[str, str],
    manifest: pd.DataFrame
) -> dict[str, str]:
    for name_of_standard_column, name_of_source in dictionary_of_names_of_standard_columns_and_sources.items():
        if name_of_source is not None:
            manifest[name_of_standard_column] = pd.to_numeric(QC_data[name_of_source], errors = "raise")
        else:
            manifest[name_of_standard_column] = pd.NA
    print(f"Dictionary of names of standard columns and sources is\n{dictionary_of_names_of_standard_columns_and_sources}.")


def create_series_of_indicators_that_comparisons_are_true(series: pd.Series, direction: str, threshold: float) -> pd.Series:
    if direction == "ge":
        return series >= threshold
    elif direction == "le":
        return series <= threshold
    else:
        raise Exception(f"Direction {direction} is invalid.")


def create_series_of_indicators_that_values_are_outliers(series: pd.Series) -> pd.Series:
    mean = series.mean()
    standard_deviation = series.std()
    return (series.sub(mean).abs().div(standard_deviation)).gt(3)


dictionary_of_names_of_standard_columns_and_dictionaries_of_directions_and_thresholds = {
    "AlignmentRate": {"direction": "ge", "threshold": 0.7},
    "rRNARate": {"direction": "le", "threshold": 0.2},
    "Duplication": {"direction": "le", "threshold": 0.6},
    "MappedReads": {"direction": "ge", "threshold": 10E6},
    "ExonicRate": {"direction": "ge", "threshold": 0.5}
}


def add_to_manifest_columns_of_indicators_that_comparisons_are_true(
    manifest: pd.DataFrame,
    dictionary_of_names_of_standard_columns_and_sources: dict[str, str]
):
    list_of_names_of_columns_of_indicators_that_comparisons_are_true = []
    for name_of_standard_column, dictionary_of_directions_and_thresholds in dictionary_of_names_of_standard_columns_and_dictionaries_of_directions_and_thresholds.items():
        if dictionary_of_names_of_standard_columns_and_sources.get(name_of_standard_column) is None:
            manifest[f"comparison_is_true_for_{name_of_standard_column}"] = pd.NA
        else:
            manifest[f"comparison_is_true_for_{name_of_standard_column}"] = create_series_of_indicators_that_comparisons_are_true(
                manifest[name_of_standard_column],
                dictionary_of_directions_and_thresholds["direction"],
                dictionary_of_directions_and_thresholds["threshold"]
            )
        list_of_names_of_columns_of_indicators_that_comparisons_are_true.append(f"comparison_is_true_for_{name_of_standard_column}")
    return list_of_names_of_columns_of_indicators_that_comparisons_are_true


def add_to_manifest_columns_of_indicators_that_values_are_outliers(
    manifest: pd.DataFrame,
    dictionary_of_names_of_standard_columns_and_sources: dict[str, str]
):
    list_of_names_of_columns_of_indicators_that_values_are_outliers = []
    for name_of_standard_column, dictionary_of_directions_and_thresholds in dictionary_of_names_of_standard_columns_and_dictionaries_of_directions_and_thresholds.items():
        if dictionary_of_names_of_standard_columns_and_sources.get(name_of_standard_column) is None:
            manifest[f"{name_of_standard_column}_is_outlier"] = False
        else:
            manifest[f"{name_of_standard_column}_is_outlier"] = create_series_of_indicators_that_values_are_outliers(
                manifest[name_of_standard_column]
            )
        list_of_names_of_columns_of_indicators_that_values_are_outliers.append(f"{name_of_standard_column}_is_outlier")
    return list_of_names_of_columns_of_indicators_that_values_are_outliers


def provide_reason_to_exclude_sample(row, dictionary_of_names_of_standard_columns_and_sources) -> str:
    list_of_reasons = []
    if not row.get("QC_Pass"):
        list_of_reasons.append("Value of QCCheck is not \"Pass\".")
    number_of_indicators_that_comparisons_are_false = row.get("number_of_indicators_that_comparisons_are_false")
    if number_of_indicators_that_comparisons_are_false >= 2:
        list_of_reasons.append(f"Number of indicators that comparisons are false is {number_of_indicators_that_comparisons_are_false}.")
    for name_of_standard_column in ["AlignmentRate", "Duplication", "ExonicRate", "MappedReads", "rRNARate"]:
        if row.get(f"{name_of_standard_column}_is_outlier"):
            list_of_reasons.append(f"{name_of_standard_column} is outlier.")
    if not row.get("sample_has_assigned_primary_site"):
        list_of_reasons.append("Sample does not have assigned primary site.")
    elif not row.get("sample_is_cutaneous"):
        assigned_primary_site = row.get("AssignedPrimarySite")
        list_of_reasons.append(f"Sample is not cutaneous and is {assigned_primary_site}.")
    return " ".join(list_of_reasons)


def create_manifest(QC_data, dictionary_of_names_of_standard_columns_and_sources) -> pd.DataFrame:
    clinical_molecular_linkage_data = load_clinical_molecular_linkage_data()
    output_of_pipeline = load_output_of_pipeline()
    diagnosis_data = load_diagnosis_data()
    manifest = merge_data(
        QC_data,
        clinical_molecular_linkage_data,
        output_of_pipeline,
        diagnosis_data
    )
    manifest["QC_Pass"] = QC_data["QCCheck"].eq("Pass")
    standardize_columns(QC_data, dictionary_of_names_of_standard_columns_and_sources, manifest)
    list_of_names_of_columns_of_indicators_that_comparisons_are_true = add_to_manifest_columns_of_indicators_that_comparisons_are_true(
        manifest,
        dictionary_of_names_of_standard_columns_and_sources
    )
    manifest["number_of_indicators_that_comparisons_are_false"] = (
        (manifest[list_of_names_of_columns_of_indicators_that_comparisons_are_true] == False).sum(axis = 1)
    )
    list_of_names_of_columns_of_indicators_that_values_are_outliers = add_to_manifest_columns_of_indicators_that_values_are_outliers(
        manifest,
        dictionary_of_names_of_standard_columns_and_sources
    )
    manifest["value_is_outlier"] = manifest[list_of_names_of_columns_of_indicators_that_values_are_outliers].any(axis = 1)
    manifest["sample_has_assigned_primary_site"] = manifest["AssignedPrimarySite"].notna()
    manifest["sample_is_cutaneous"] = manifest["AssignedPrimarySite"].eq("cutaneous")
    manifest["Included"] = (
        manifest["QC_Pass"] &
        manifest["number_of_indicators_that_comparisons_are_false"].lt(2) &
        manifest["sample_is_cutaneous"] &
        ~manifest["value_is_outlier"]
    )
    manifest["ExclusionReason"] = manifest.apply(
        lambda row: "" if row.get("Included") else provide_reason_to_exclude_sample(row, dictionary_of_names_of_standard_columns_and_sources),
        axis = 1
    )
    manifest_to_save = manifest[
        [
            "PATIENT_ID",
            "SLID",
            "DeidSpecimenID",
            "ORIENSpecimenID",
            "SpecimenSite",
            "AssignedPrimarySite",
            "HistologyCode",
            "DiagnosisID",
            "Age At Specimen Collection",
            "NexusBatch",
            "MappedReads",
            "ExonicRate",
            "AlignmentRate",
            "rRNARate",
            "Duplication",
            "comparison_is_true_for_MappedReads",
            "MappedReads_is_outlier",
            "comparison_is_true_for_ExonicRate",
            "ExonicRate_is_outlier",
            "comparison_is_true_for_AlignmentRate",
            "AlignmentRate_is_outlier",
            "comparison_is_true_for_rRNARate",
            "rRNARate_is_outlier",
            "comparison_is_true_for_Duplication",
            "Duplication_is_outlier",
            "QC_Pass",
            "Included",
            "ExclusionReason"            
        ]
    ]
    manifest_to_save.to_csv(paths.manifest, index = False)
    print("Manifest was saved.")
    print(f"Manifest has shape {manifest_to_save.shape}.")
    return manifest


def create_QC_summary_of_CSVs(
    QC_data: pd.DataFrame,
    dictionary_of_names_of_standard_columns_and_sources: dict[str, str],
    manifest: pd.DataFrame
):
    list_of_dictionaries_of_information_for_first_table_of_QC_summary: list[dict] = []
    for name_of_standard_column, dictionary_of_directions_and_thresholds in dictionary_of_names_of_standard_columns_and_dictionaries_of_directions_and_thresholds.items():
        comparison = dictionary_of_directions_and_thresholds["direction"]
        threshold = dictionary_of_directions_and_thresholds["threshold"]
        name_of_source = dictionary_of_names_of_standard_columns_and_sources.get(name_of_standard_column)
        standard_column_is_available = name_of_source is not None and QC_data[name_of_standard_column].notna().any()
        minimum_value = None
        median_value = None
        maximum_value = None
        number_of_values = 0
        number_of_samples_for_which_comparisons_are_true = 0
        number_of_samples_for_which_comparisons_are_false = 0
        number_of_values_that_are_outliers = 0
        if standard_column_is_available:
            series_of_values = manifest[name_of_standard_column]
            minimum_value = series_of_values.min()
            median_value = series_of_values.median()
            maximum_value = series_of_values.max()
            number_of_values = series_of_values.size
            series_of_indicators_that_comparisons_are_true = manifest[f"comparison_is_true_for_{name_of_standard_column}"]
            number_of_samples_for_which_comparisons_are_true = (series_of_indicators_that_comparisons_are_true == True).sum()
            number_of_samples_for_which_comparisons_are_false = (series_of_indicators_that_comparisons_are_true == False).sum()
            number_of_values_that_are_outliers = (manifest[f"{name_of_standard_column}_is_outlier"] == True).sum()
        list_of_dictionaries_of_information_for_first_table_of_QC_summary.append(
            {
                "name of standard column": name_of_standard_column,
                "comparison": comparison,
                "threshold": threshold,
                "name of source": name_of_source if name_of_source is not None else "not available",
                "availability": "available" if standard_column_is_available else "not available",
                "Min": minimum_value,
                "Median": median_value,
                "Max": maximum_value,
                "number of values": number_of_values,
                "number of samples for which comparisons are true": number_of_samples_for_which_comparisons_are_true,
                "number of samples_for_which_comparisons are false": number_of_samples_for_which_comparisons_are_false,
                "number of values that are outliers": number_of_values_that_are_outliers
            }
        )
    list_of_dictionaries_of_information_for_second_table_of_QC_summary: list[dict] = []
    number_of_samples = manifest.shape[0]
    number_of_samples_included_in_expression_matrix = manifest["Included"].sum()
    number_of_samples_excluded_from_expression_matrix = number_of_samples - number_of_samples_included_in_expression_matrix
    def add_row(description: str, number_of_samples: int):
        list_of_dictionaries_of_information_for_second_table_of_QC_summary.append(
            {
                "description": description,
                "number of samples": number_of_samples
            }
        )
    add_row("number of samples", number_of_samples)
    add_row("number of samples included in expression matrix", number_of_samples_included_in_expression_matrix)
    add_row("number of samples excluded from expression matrix", number_of_samples_excluded_from_expression_matrix)
    add_row("Value of QCCheck is not \"Pass\".", (~manifest["QC_Pass"]).sum())
    add_row(f"Number of indicators that comparisons are false is at least 2.", (manifest["number_of_indicators_that_comparisons_are_false"] >= 2).sum())
    for name_of_standard_column in ["MappedReads", "ExonicRate", "AlignmentRate", "rRNARate", "Duplication"]:
        name_of_source = dictionary_of_names_of_standard_columns_and_sources.get(name_of_standard_column)
        if name_of_source is None:
            add_row(f"Standard column {name_of_standard_column} is not available.", None)
        else:
            add_row(f"{name_of_standard_column} is outlier.", (manifest[f"{name_of_standard_column}_is_outlier"] == True).sum())
    add_row("Sample does not have assigned primary site.", (~manifest["sample_has_assigned_primary_site"]).sum())
    add_row("Sample is not cutaneous.", (manifest["sample_has_assigned_primary_site"] & ~manifest["sample_is_cutaneous"]).sum())
    first_data_frame_of_information_of_QC_summary = pd.DataFrame(list_of_dictionaries_of_information_for_first_table_of_QC_summary)
    second_data_frame_of_information_of_QC_summary = pd.DataFrame(list_of_dictionaries_of_information_for_second_table_of_QC_summary)
    with open(paths.QC_summary_of_CSVs, 'w') as QC_summary_of_CSVs:
        first_data_frame_of_information_of_QC_summary.to_csv(QC_summary_of_CSVs, index = False)
        QC_summary_of_CSVs.write('\n')
        second_data_frame_of_information_of_QC_summary.to_csv(QC_summary_of_CSVs, index = False)
    print("QC summary of Comma Separated Values was saved.")
    return list_of_dictionaries_of_information_for_second_table_of_QC_summary


def create_QC_summary_in_Markdown(
    dictionary_of_names_of_standard_columns_and_sources: dict[str, str],
    list_of_dictionaries_of_information_for_second_table_of_QC_summary: list[dict],
    manifest: pd.DataFrame
):
    list_of_contents_of_QC_summary_in_Markdown = []
    list_of_contents_of_QC_summary_in_Markdown.append(
        f'''# QC Summary\n
This summary was generated at {format_timestamp(datetime.now().timestamp())}.
'''
    )
    list_of_contents_describing_availability_and_comparisons = []
    for name_of_standard_column, dictionary_of_directions_and_thresholds in dictionary_of_names_of_standard_columns_and_dictionaries_of_directions_and_thresholds.items():
        name_of_source = dictionary_of_names_of_standard_columns_and_sources.get(name_of_standard_column)
        standard_column_is_available = "available" if name_of_source else "not available"
        comparison = "at least" if dictionary_of_directions_and_thresholds["direction"] == "ge" else "at most"
        threshold = dictionary_of_directions_and_thresholds["threshold"]
        list_of_contents_describing_availability_and_comparisons.append(
            f'''Standard column `{name_of_standard_column}` is {standard_column_is_available}.\n
Standard column `{name_of_standard_column}` has {"source " + name_of_source if name_of_source else "no source"}.\n
Values in standard column {name_of_standard_column} are compared as {comparison} {threshold}.
'''
        )
    list_of_contents_of_QC_summary_in_Markdown += list_of_contents_describing_availability_and_comparisons
    number_of_samples = manifest.shape[0]
    number_of_samples_included_in_expression_matrix = manifest["Included"].sum()
    number_of_samples_excluded_from_expression_matrix = number_of_samples - number_of_samples_included_in_expression_matrix
    list_of_contents_of_QC_summary_in_Markdown.append(
        f'''The number of samples in the manifest is {number_of_samples}.\n
The number of samples included in the expression matrix is {number_of_samples_included_in_expression_matrix}.\n
The number of samples excluded from the expression matrix is {number_of_samples_excluded_from_expression_matrix}.
'''
    )
    for row in list_of_dictionaries_of_information_for_second_table_of_QC_summary:
        if row["description"] not in (
            "number of samples",
            "number of samples included in expression matrix",
            "number of samples excluded from expression matrix",
            "Standard column AlignmentRate is not available.",
            "Standard column Duplication is not available."
        ):
            description = row["description"].strip(".")
            number_of_samples = row["number of samples"]
            list_of_contents_of_QC_summary_in_Markdown.append(
                f"{description} for {number_of_samples} samples.\n"
            )

    qc_mtime = format_timestamp(os.path.getmtime(paths.QC_data))
    cml_mtime = format_timestamp(os.path.getmtime(paths.clinical_molecular_linkage_data))
    pairing_mtime = format_timestamp(os.path.getmtime(paths.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors))
    dx_mtime = format_timestamp(os.path.getmtime(paths.diagnosis_data))
    expr_latest = format_timestamp(get_latest_timestamp_of_file(paths.gene_and_transcript_expression_results))
    list_of_contents_of_QC_summary_in_Markdown.append(
        f'''Gene and transcript expression results live at {paths.gene_and_transcript_expression_results} and were last modified at {expr_latest}.\n
QC data lives at {paths.QC_data} and was last modified at {qc_mtime}.\n
Clinical molecular linkage data lives at {paths.clinical_molecular_linkage_data} and was last modified at {cml_mtime}.\n
Diagnosis data lives at {paths.diagnosis_data} and was last modified at {dx_mtime}.\n
Output of pipeline for pairing clinical data and stages of tumors lives at {paths.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors} and was last modified at {pairing_mtime}.'''
    )
    with open(paths.QC_summary_in_Markdown, 'w') as f:
        f.write("\n".join(list_of_contents_of_QC_summary_in_Markdown))
    print("QC summary in Markdown was saved.")


def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz = timezone.utc).astimezone().isoformat(timespec = "seconds")


def get_latest_timestamp_of_file(path_to_directory):
    list_of_paths_of_files = list(path_to_directory.glob("*"))
    return max(path_of_file.stat().st_mtime for path_of_file in list_of_paths_of_files)


def main():
    paths.ensure_dependencies_for_creating_expression_matrices_exist()
    list_of_paths = list(paths.gene_and_transcript_expression_results.glob("*.genes.results"))
    full_expression_matrix = create_full_expression_matrix(list_of_paths)
    QC_data = load_QC_data()
    dictionary_of_names_of_standard_columns_and_sources = create_dictionary_of_names_of_standard_columns_and_sources(QC_data)
    manifest = create_manifest(QC_data, dictionary_of_names_of_standard_columns_and_sources)
    list_of_dictionaries_of_information_for_second_table_of_QC_summary = create_QC_summary_of_CSVs(
        QC_data,
        dictionary_of_names_of_standard_columns_and_sources,
        manifest
    )
    create_QC_summary_in_Markdown(
        dictionary_of_names_of_standard_columns_and_sources,
        list_of_dictionaries_of_information_for_second_table_of_QC_summary,
        manifest
    )




    # Restrict Ensembl TPM to included SLIDs (this is the *pre low-expression* matrix)
    list_of_included_SLIDs = manifest.loc[manifest["Included"], "SLID"].dropna().unique().tolist()
    tpm_ensembl_pre = full_expression_matrix.loc[:, [c for c in full_expression_matrix.columns if c in list_of_included_SLIDs]].copy()
    tpm_ensembl_pre.to_csv("tpm_ensembl_pre_filter.csv")
    print(f"TPM (Ensembl) pre-filter matrix has shape {tpm_ensembl_pre.shape}.")
    # Keep old name for backward-compatibility
    tpm_ensembl_pre.to_csv("filtered_expression_matrix.csv")

    # === Ensembl→HGNC mapping & HGNC collapse (median) ======================
    ens_to_hgnc = _build_ensembl_to_hgnc_mapping(list_of_paths)
    # Save mapping as both the configured series path and a two-column table
    try:
        ens_to_hgnc.to_csv("data_frame_of_Ensembl_IDs_and_HGNC_symbols.csv")
        print(f"Saved Ensembl→HGNC series to \"data_frame_of_Ensembl_IDs_and_HGNC_symbols.csv\".")
    except Exception as e:
        print(f"Warning: could not write mapping series to data_frame_of_Ensembl_IDs_and_HGNC_symbols.csv ({e}).")
    mapping_df = ens_to_hgnc.rename("HGNC").rename_axis("EnsemblID").reset_index()
    mapping_df.to_csv("ensembl_to_hgnc_mapping.csv", index=False)

    tpm_hgnc_pre = _collapse_to_hgnc_median(tpm_ensembl_pre, ens_to_hgnc)
    tpm_hgnc_pre.to_csv("tpm_hgnc_pre_filter.csv")
    print(f"TPM (HGNC-collapsed, median) pre-filter matrix has shape {tpm_hgnc_pre.shape}.")

    # === Low-expression filtering: TPM > 1 in ≥ 20% samples =================
    tpm_ensembl_post, mask_ens = _low_expression_filter(tpm_ensembl_pre, threshold=1.0, min_prop=0.20)
    tpm_ensembl_post.to_csv("tpm_ensembl_post_filter.csv")
    print(f"TPM (Ensembl) post-filter (TPM>1 in ≥20%) has shape {tpm_ensembl_post.shape} "
          f"(kept {mask_ens.sum()} / {mask_ens.size} genes).")

    tpm_hgnc_post, mask_hgnc = _low_expression_filter(tpm_hgnc_pre, threshold=1.0, min_prop=0.20)
    tpm_hgnc_post.to_csv("tpm_hgnc_post_filter.csv")
    print(f"TPM (HGNC) post-filter (TPM>1 in ≥20%) has shape {tpm_hgnc_post.shape} "
          f"(kept {mask_hgnc.sum()} / {mask_hgnc.size} genes).")

    # === Transforms: log2(TPM+1) and gene-wise z-scores =====================
    # (Save for both Ensembl and HGNC post-filter; HGNC versions are typically used for PCA/signatures)
    log2_ens = _log2_tpm_plus1(tpm_ensembl_post)
    log2_ens.to_csv("log2_tpm_plus1_ensembl_post_filter.csv")
    z_ens = _rowwise_zscore(log2_ens)  # z-score usually applied after log
    z_ens.to_csv("zscore_ensembl_post_filter.csv")

    log2_hgnc = _log2_tpm_plus1(tpm_hgnc_post)
    log2_hgnc.to_csv("log2_tpm_plus1_hgnc_post_filter.csv")
    z_hgnc = _rowwise_zscore(log2_hgnc)
    z_hgnc.to_csv("zscore_hgnc_post_filter.csv")

    print("CSV and Markdown QC summaries were written.")
    print("Saved:")
    print(" - tpm_ensembl_pre_filter.csv / tpm_ensembl_post_filter.csv")
    print(" - tpm_hgnc_pre_filter.csv / tpm_hgnc_post_filter.csv")
    print(" - log2_tpm_plus1_ensembl_post_filter.csv / zscore_ensembl_post_filter.csv")
    print(" - log2_tpm_plus1_hgnc_post_filter.csv / zscore_hgnc_post_filter.csv")
    print(" - ensembl_to_hgnc_mapping.csv (and mapping series to paths.series_of_Ensembl_IDs_and_HGNC_symbols)")


if __name__ == "__main__":
    main()