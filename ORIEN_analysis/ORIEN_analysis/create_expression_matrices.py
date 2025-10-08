'''
create_expression_matrices.py


Creating expression matrices earlier used data from all gene and transcript results.
Different genes and samples were included for creating an enrichment matrix, CD8 analysis, and CD8 groups analysis.

We created a full expression matrix (EM1).

We created a manifest using gene and transcript results, QC data, clinical molecular linkage data, diagnosis data, and the output of our pipeline for pairing clinical data and stages of tumors.
The manifest may be used to include in or exclude samples from expression matrices.
We created expression matrices approved by the manifest / restricted to cutaneous samples specified by the output of our pipeline that pass QC.

We created a QC summary of Comma Separated Values.

We created a QC summary in Markdown.

We created an expression matrix with SLIDs approved by the manifest and Ensembl IDs (EM2).

We created a data frame of Ensembl IDs and HGNC symbols.

We created an expression matrix with SLIDs approved by the manifest and HGNC symbols (EM3).

We created an expression matrix with SLIDs approved by the manifest, Ensembl IDs, and filtering to rows where at least 20 percent of samples had expressions greater than 1 (EM4).

We created an expression matrix with SLIDs approved by the manifest, HGNC symbols, and filtering to rows where at least 20 percent of samples had expressions greater than 1 (EM5).

We created an expression matrix with SLIDs approved by the manifest, Ensembl IDs, filtering, and applying a log (EM6).

We created an expression matrix with SLIDs approved by the manifest, HGNC symbols, filtering, and applying a log (EM7).

We created an expression matrix with SLIDs approved by the manifest, Ensembl IDs, filtering, and z scoring (EM8).

We created an expression matrix with SLIDs approved by the manifest, Ensembl IDs, filtering, and z scoring (EM9).

TODO: Implement batch handling.
TODO: Create a patient level manifest.
TODO: Use EM5 to create an enrichment matrix.
TODO: Use either EM6 or EM7 to conduct CD8 analysis and CD8 groups analysis.
'''


from ORIEN_analysis.config import paths
from datetime import datetime, timezone
import numpy as np
import math
import os
import pandas as pd
from functools import reduce
import operator


def create_full_expression_matrix(list_of_paths: list) -> pd.DataFrame:
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
        paths.output_of_pairing_clinical_data_and_stages_of_tumors,
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


def ensure_rates_are_proportions(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors = "raise")
    if series.max() > 1.0:
        series = series / 100.0
    return series


def standardize_columns(
    QC_data: pd.DataFrame,
    dictionary_of_names_of_standard_columns_and_sources: dict[str, str],
    manifest: pd.DataFrame
):
    for name_of_standard_column, name_of_source in dictionary_of_names_of_standard_columns_and_sources.items():
        if name_of_source is None:
            manifest[name_of_standard_column] = pd.NA
        else:
            source = QC_data[name_of_source]
            if name_of_standard_column in {"MappedReads"}:
                manifest[name_of_standard_column] = pd.to_numeric(source, errors = "raise")
            else:
                manifest[name_of_standard_column] = ensure_rates_are_proportions(source)
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
    for name_of_standard_column in dictionary_of_names_of_standard_columns_and_dictionaries_of_directions_and_thresholds.keys():
        if dictionary_of_names_of_standard_columns_and_sources.get(name_of_standard_column) is None:
            manifest[f"{name_of_standard_column}_is_outlier"] = pd.NA
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
        value_is_outlier = False if (row.get(f"{name_of_standard_column}_is_outlier") is pd.NA) else row.get(f"{name_of_standard_column}_is_outlier")
        if value_is_outlier:
            list_of_reasons.append(f"{name_of_standard_column} is outlier.")
    if not row.get("sample_has_assigned_primary_site"):
        list_of_reasons.append("Sample does not have assigned primary site.")
    elif not row.get("sample_is_cutaneous"):
        assigned_primary_site = row.get("AssignedPrimarySite")
        list_of_reasons.append(f"Sample is not cutaneous and is {assigned_primary_site}.")
    return " ".join(list_of_reasons)


def create_relaxed_manifest(QC_data: pd.DataFrame) -> pd.DataFrame:
    clinical_molecular_linkage_data = load_clinical_molecular_linkage_data()
    output_of_pipeline = load_output_of_pipeline()
    manifest = (
        QC_data[
            [
                #"ORIENAvatarKey",
                "SLID",
                #"NexusBatch"
            ]
        ]
        #.rename(
        #    columns = {"ORIENAvatarKey": "PATIENT_ID"}
        #)
        .merge(
            clinical_molecular_linkage_data[
                [
                    #"Age At Specimen Collection", # Column "Age At Specimen Collection" values in the spirit of collection dates.
                    "DeidSpecimenID",
                    "RNASeq",
                    #"SpecimenSiteOfCollection"
                ]
            ],
            how = "left",
            left_on = "SLID",
            right_on = "RNASeq"
        )
        .drop(columns = "RNASeq")
        #.rename(columns = {"SpecimenSiteOfCollection": "SpecimenSite"})
        .merge(
            output_of_pipeline[
                [
                    "AssignedPrimarySite",
                    "ORIENSpecimenID",
                    #"index_of_row_of_diagnosis_data_paired_with_specimen"
                ]
            ],
            how = "left",
            left_on = "DeidSpecimenID",
            right_on = "ORIENSpecimenID"
        )
    )
    manifest["Included"] = manifest["AssignedPrimarySite"].eq("cutaneous")
    return manifest


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
        standard_column_is_available = name_of_source is not None and manifest[name_of_standard_column].notna().any()
        minimum_value = None
        median_value = None
        maximum_value = None
        number_of_values = None
        number_of_non_NA_values = None
        number_of_samples_for_which_comparisons_are_true = None
        number_of_samples_for_which_comparisons_are_false = None
        number_of_values_that_are_outliers = None
        if standard_column_is_available:
            series_of_values = manifest[name_of_standard_column]
            minimum_value = series_of_values.min()
            median_value = series_of_values.median()
            maximum_value = series_of_values.max()
            number_of_values = series_of_values.size
            number_of_non_NA_values = series_of_values.notna().sum()
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
                "minimum": minimum_value,
                "median": median_value,
                "maximum": maximum_value,
                "number of values": number_of_values,
                "number of non NA values": number_of_non_NA_values,
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
    pairing_mtime = format_timestamp(os.path.getmtime(paths.output_of_pairing_clinical_data_and_stages_of_tumors))
    dx_mtime = format_timestamp(os.path.getmtime(paths.diagnosis_data))
    expr_latest = format_timestamp(get_latest_timestamp_of_file(paths.gene_and_transcript_expression_results))
    list_of_contents_of_QC_summary_in_Markdown.append(
        f'''Gene and transcript expression results live at {paths.gene_and_transcript_expression_results} and were last modified at {expr_latest}.\n
QC data lives at {paths.QC_data} and was last modified at {qc_mtime}.\n
Clinical molecular linkage data lives at {paths.clinical_molecular_linkage_data} and was last modified at {cml_mtime}.\n
Diagnosis data lives at {paths.diagnosis_data} and was last modified at {dx_mtime}.\n
Output of pipeline for pairing clinical data and stages of tumors lives at {paths.output_of_pairing_clinical_data_and_stages_of_tumors} and was last modified at {pairing_mtime}.'''
    )
    with open(paths.QC_summary_in_Markdown, 'w') as f:
        f.write("\n".join(list_of_contents_of_QC_summary_in_Markdown))
    print("QC summary in Markdown was saved.")


def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz = timezone.utc).astimezone().isoformat(timespec = "seconds")


def get_latest_timestamp_of_file(path_to_directory):
    list_of_paths_of_files = list(path_to_directory.glob("*"))
    return max(path_of_file.stat().st_mtime for path_of_file in list_of_paths_of_files)


def create_series_of_Ensembl_IDs_and_HGNC_symbols(list_of_paths: list) -> pd.Series:
    series_of_Ensembl_IDs_and_HGNC_symbols = (
        pd.concat(
            [
                pd.read_csv(
                    file_of_gene_results,
                    sep = '\t',
                    usecols = ["gene_id", "gene_symbol"],
                    dtype = {
                        "gene_id": str,
                        "gene_symbol": str
                    }
                )
                for file_of_gene_results in list_of_paths
            ]
        )
        .assign(
            gene_id = lambda df: df["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex = True)
        )
        .drop_duplicates("gene_id")
        .set_index("gene_id")
        ["gene_symbol"]
    )
    return series_of_Ensembl_IDs_and_HGNC_symbols


def create_expression_matrix_with_HGNC_symbols(
    expression_matrix: pd.DataFrame,
    series_of_Ensembl_IDs_and_HGNC_symbols: pd.Series
) -> pd.DataFrame:
    expression_matrix_with_HGNC_symbols = expression_matrix.copy()
    expression_matrix_with_HGNC_symbols["HGNC_symbol"] = series_of_Ensembl_IDs_and_HGNC_symbols
    expression_matrix_with_HGNC_symbols = (
        expression_matrix_with_HGNC_symbols
        .groupby(
            "HGNC_symbol",
            sort = False
        )
        .median()
    )
    return expression_matrix_with_HGNC_symbols


def apply_filter(expression_matrix: pd.DataFrame) -> pd.DataFrame:
    proportion_of_samples = 0.2
    threshold = 1.0
    number_of_samples = expression_matrix.shape[1]
    number_of_samples_that_must_have_expressions_greater_than_threshold = math.ceil(proportion_of_samples * number_of_samples)
    series_of_indicators_that_more_than_the_required_number_of_samples_have_expressions_greater_than_threshold = (
        (expression_matrix > threshold).sum(axis = 1) >= number_of_samples_that_must_have_expressions_greater_than_threshold
    )
    filtered_expression_matrix = expression_matrix.loc[
        series_of_indicators_that_more_than_the_required_number_of_samples_have_expressions_greater_than_threshold
    ]
    print(
        f"Expression matrix after applying filter has shape {filtered_expression_matrix.shape}.\n"
        f"{series_of_indicators_that_more_than_the_required_number_of_samples_have_expressions_greater_than_threshold.sum()} out of " +
        f"{len(series_of_indicators_that_more_than_the_required_number_of_samples_have_expressions_greater_than_threshold)} rows were kept."
    )
    return filtered_expression_matrix


def apply_log(expression_matrix: pd.DataFrame) -> pd.DataFrame:
    return np.log2(expression_matrix + 1.0)


def z_score(expression_matrix: pd.DataFrame) -> pd.DataFrame:
    series_of_means = expression_matrix.mean(axis = 1)
    series_of_standard_deviations = expression_matrix.std(axis = 1)
    z_score = expression_matrix.sub(series_of_means, axis = 0).div(series_of_standard_deviations, axis = 0)
    return z_score


def main():
    paths.ensure_dependencies_for_creating_expression_matrices_exist()
    list_of_paths = list(paths.gene_and_transcript_expression_results.glob("*.genes.results"))
    full_expression_matrix = create_full_expression_matrix(list_of_paths)
    full_expression_matrix.to_csv(paths.full_expression_matrix)

    QC_data = load_QC_data()
    dictionary_of_names_of_standard_columns_and_sources = create_dictionary_of_names_of_standard_columns_and_sources(QC_data)
    manifest = create_manifest(QC_data, dictionary_of_names_of_standard_columns_and_sources)
    relaxed_manifest = create_relaxed_manifest(QC_data)

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

    list_of_included_SLIDs = manifest.loc[manifest["Included"], "SLID"].tolist()
    expression_matrix_with_SLIDs_approved_by_manifest = full_expression_matrix.loc[
        :,
        [name_of_column for name_of_column in full_expression_matrix.columns if name_of_column in list_of_included_SLIDs]
    ]
    expression_matrix_with_SLIDs_approved_by_manifest.to_csv(paths.expression_matrix_with_SLIDs_approved_by_manifest)
    print(
        f'''Expression matrix with SLIDs approved by manifest was saved.
Expression matrix with SLIDs approved by manifest has shape {expression_matrix_with_SLIDs_approved_by_manifest.shape}.'''
    )

    list_of_included_SLIDs = relaxed_manifest.loc[relaxed_manifest["Included"], "SLID"].tolist()
    expression_matrix_with_SLIDs_approved_by_relaxed_manifest = full_expression_matrix.loc[
        :,
        [name_of_column for name_of_column in full_expression_matrix.columns if name_of_column in list_of_included_SLIDs]
    ]
    expression_matrix_with_SLIDs_approved_by_relaxed_manifest.to_csv(paths.expression_matrix_with_SLIDs_approved_by_relaxed_manifest)
    print(
        f'''Expression matrix with SLIDs approved by relaxed manifest was saved.
Expression matrix with SLIDs approved by relaxed manifest has shape {expression_matrix_with_SLIDs_approved_by_relaxed_manifest.shape}.'''
    )

    series_of_Ensembl_IDs_and_HGNC_symbols = create_series_of_Ensembl_IDs_and_HGNC_symbols(list_of_paths)
    series_of_Ensembl_IDs_and_HGNC_symbols.to_csv(paths.data_frame_of_Ensembl_IDs_and_HGNC_symbols)
    print("Data frame of Ensembl IDs and HGNC symbols was saved.")

    full_expression_matrix_with_HGNC_symbols = create_expression_matrix_with_HGNC_symbols(
        full_expression_matrix,
        series_of_Ensembl_IDs_and_HGNC_symbols
    )
    full_expression_matrix_with_HGNC_symbols.to_csv(paths.full_expression_matrix_with_HGNC_symbols)
    print(
        f'''Full expression matrix with HGNC symbols was saved.
Full expression matrix with HGNC symbols has shape {full_expression_matrix_with_HGNC_symbols.shape}.'''
    )

    expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = create_expression_matrix_with_HGNC_symbols(
        expression_matrix_with_SLIDs_approved_by_manifest,
        series_of_Ensembl_IDs_and_HGNC_symbols
    )
    expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.to_csv(
        paths.expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest
    )
    print(
        f'''Expression matrix with HGNC symbols and SLIDs approved by manifest was saved.
Expression matrix with HGNC symbols and SLIDs approved by manifest has shape {expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.shape}.'''
    )

    expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_relaxed_manifest = create_expression_matrix_with_HGNC_symbols(
        expression_matrix_with_SLIDs_approved_by_relaxed_manifest,
        series_of_Ensembl_IDs_and_HGNC_symbols
    )
    expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_relaxed_manifest.to_csv(
        paths.expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_relaxed_manifest
    )
    print(
        f'''Expression matrix with HGNC symbols and SLIDs approved by relaxed manifest was saved.
Expression matrix with HGNC symbols and SLIDs approved by relaxed manifest has shape {expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_relaxed_manifest.shape}.'''
    )

    filtered_expression_matrix_with_SLIDs_approved_by_manifest = apply_filter(expression_matrix_with_SLIDs_approved_by_manifest)
    filtered_expression_matrix_with_SLIDs_approved_by_manifest.to_csv(
        paths.filtered_expression_matrix_with_SLIDs_approved_by_manifest
    )
    print("Filtered expression matrix with SLIDs approved by manifest was saved.")

    filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = apply_filter(
        expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest
    )
    filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.to_csv(
        paths.filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest
    )
    print("Filtered expression matrix with HGNC symbols and SLIDs approved by manifest was saved.")

    logged_filtered_expression_matrix_with_SLIDs_approved_by_manifest = apply_log(
        filtered_expression_matrix_with_SLIDs_approved_by_manifest
    )
    logged_filtered_expression_matrix_with_SLIDs_approved_by_manifest.to_csv(
        paths.logged_filtered_expression_matrix_with_SLIDs_approved_by_manifest
    )
    print("Logged filtered expression matrix with SLIDs approved by manifest was saved.")

    logged_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = apply_log(
        filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest
    )
    logged_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.to_csv(
        paths.logged_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest
    )
    print("Logged filtered expression matrix with HGNC symbols and SLIDs approved by manifest was saved.")

    z_scored_filtered_expression_matrix_with_SLIDs_approved_by_manifest = z_score(
        filtered_expression_matrix_with_SLIDs_approved_by_manifest
    )
    z_scored_filtered_expression_matrix_with_SLIDs_approved_by_manifest.to_csv(
        paths.z_scored_filtered_expression_matrix_with_SLIDs_approved_by_manifest
    )
    print("z scored filtered expression matrix with SLIDs approved by manifest was saved.")

    z_scored_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = z_score(
        filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest
    )
    z_scored_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.to_csv(
        paths.z_scored_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest
    )
    print("z scored filtered expression matrix with HGNC symbols and SLIDs approved by manifest was saved.")


if __name__ == "__main__":
    main()