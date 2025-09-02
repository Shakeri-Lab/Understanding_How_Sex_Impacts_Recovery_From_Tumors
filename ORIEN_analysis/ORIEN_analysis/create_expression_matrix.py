'''
create_expression_matrix.py

We build an expression matrix with 60,609 Ensembl IDs and 333 sample IDs.
We filter the expression matrix to columns with sample IDs
in a data frame of filtered QC data and IDs of cutaneous specimens
output by our pipeline for pairing clinical data and stages of tumors.
'''


from ORIEN_analysis.config import paths
from datetime import datetime, timezone
import os
import pandas as pd
from functools import reduce
import operator


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "NA"
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _latest_mtime_in_dir(dir_path) -> float | None:
    try:
        files = list(dir_path.glob("*"))
    except Exception:
        return None
    if not files:
        return None
    try:
        return max(f.stat().st_mtime for f in files)
    except Exception:
        return None


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


def provide_name_of_first_column_whose_name_matches_a_candidate(QC_data: pd.DataFrame, list_of_candidates: list[str]) -> str | None:
    for candidate in list_of_candidates:
        if candidate in QC_data.columns:
            return candidate
    return None


def safe_bool(x, default: bool = False) -> bool:
    try:
        if pd.isna(x):
            return default
    except TypeError:
        return default
    return bool(x)


def standardize_columns(QC_data: pd.DataFrame) -> dict[str, str]:
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
    for name_of_standard_column, name_of_source in dictionary_of_names_of_standard_columns_and_sources.items():
        if name_of_source is not None:
            QC_data[name_of_standard_column] = pd.to_numeric(QC_data[name_of_source], errors = "raise")
        else:
            QC_data[name_of_standard_column] = pd.NA
    return dictionary_of_names_of_standard_columns_and_sources


def create_expression_matrix():
    # Create full expression matrix.
    list_of_paths = list(paths.gene_and_transcript_expression_results.glob("*.genes.results"))
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
    expression_matrix = pd.DataFrame(dictionary_of_sample_IDs_and_series_of_expressions)
    print(f"Expression matrix has shape {expression_matrix.shape}.")
    print(f"The number of unique genes is {expression_matrix.index.nunique()}.")
    print(f"The number of unique sample IDs is {expression_matrix.columns.nunique()}.")

    # Load QC data.
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

    # Standardize QC data.
    dictionary_of_names_of_standard_columns_and_sources = standardize_columns(QC_data)
    print(f"Dictionary of names of standard columns and sources is\n{dictionary_of_names_of_standard_columns_and_sources}.")

    # Add to QC data columns of indicators that comparisons are true and that values are outliers.
    dictionary_of_names_of_standard_columns_and_dictionaries_of_directions_and_thresholds = {
        "AlignmentRate": {"direction": "ge", "threshold": 0.7},
        "rRNARate": {"direction": "le", "threshold": 0.2},
        "Duplication": {"direction": "le", "threshold": 0.6},
        "MappedReads": {"direction": "ge", "threshold": 10E6},
        "ExonicRate": {"direction": "ge", "threshold": 0.5}
    }
    list_of_names_of_columns_of_indicators_that_comparisons_are_true = []
    list_of_names_of_columns_of_indicators_that_values_are_outliers = []
    for name_of_standard_column, dictionary_of_directions_and_thresholds in dictionary_of_names_of_standard_columns_and_dictionaries_of_directions_and_thresholds.items():
        standard_column_has_source = dictionary_of_names_of_standard_columns_and_sources.get(name_of_standard_column) is not None
        if not standard_column_has_source:
            QC_data[f"comparison_is_true_for_{name_of_standard_column}"] = pd.NA
            QC_data[f"{name_of_standard_column}_is_outlier"] = pd.NA
        else:
            if QC_data[name_of_standard_column].isna().all():
                QC_data[f"comparison_is_true_for_{name_of_standard_column}"] = pd.NA
                QC_data[f"{name_of_standard_column}_is_outlier"] = pd.NA
            else:
                QC_data[f"comparison_is_true_for_{name_of_standard_column}"] = create_series_of_indicators_that_comparisons_are_true(
                    QC_data[name_of_standard_column],
                    dictionary_of_directions_and_thresholds["direction"],
                    dictionary_of_directions_and_thresholds["threshold"]
                )
                out = create_series_of_indicators_that_values_are_outliers(
                    QC_data[name_of_standard_column]
                )
                out = out.mask(QC_data[name_of_standard_column].isna(), other = pd.NA)
                QC_data[f"{name_of_standard_column}_is_outlier"] = out
        list_of_names_of_columns_of_indicators_that_comparisons_are_true.append(f"comparison_is_true_for_{name_of_standard_column}")
        list_of_names_of_columns_of_indicators_that_values_are_outliers.append(f"{name_of_standard_column}_is_outlier")

    # Add to QC data column `QC_Pass`.
    QC_data["QC_Pass"] = QC_data["QCCheck"].eq("Pass")

    # Add to QC data columns of numbers of indicators that comparisons are false and indicators that values are outliers.
    available_pass_fail_cols = [
        c for c in list_of_names_of_columns_of_indicators_that_comparisons_are_true
        if QC_data[c].notna().any()
    ]
    QC_data["number_of_indicators_that_comparisons_are_false"] = (
        (QC_data[available_pass_fail_cols] == False).sum(axis = 1)
        if available_pass_fail_cols else 0
    )
    available_outlier_cols = [
        c for c in list_of_names_of_columns_of_indicators_that_values_are_outliers
        if QC_data[c].notna().any()
    ]
    if available_outlier_cols:
        QC_data["value_is_outlier"] = QC_data[available_outlier_cols].fillna(False).any(axis = 1)
    else:
        QC_data["value_is_outlier"] = False

    # Load clinical molecular linkage data.
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

    output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors = pd.read_csv(
        paths.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors,
        dtype = {
            "AvatarKey": str,
            "AssignedPrimarySite": str
        }
    )
    print(f"Output of pipeline for pairing clinical data and stages of tumors has shape {output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.shape}.")
    print(f"The number of unique specimen IDs is {output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors['ORIENSpecimenID'].nunique()}.")

    diagnosis_data = pd.read_csv(paths.diagnosis_data).reset_index()

    manifest = (
        QC_data[
            [
                "ORIENAvatarKey",
                "SLID",
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
                "number_of_indicators_that_comparisons_are_false",
                "value_is_outlier"
            ]
        ]
        .rename(columns = {"ORIENAvatarKey": "PATIENT_ID"})
        .merge(
            clinical_molecular_linkage_data[
                [
                    "RNASeq",
                    "DeidSpecimenID",
                    "SpecimenSiteOfCollection",
                    "Age At Specimen Collection" # Column "Age At Specimen Collection" values in the spirit of collection dates.
                ]
            ],
            how = "left",
            left_on = "SLID",
            right_on = "RNASeq"
        )
        .drop(columns = "RNASeq")
        .rename(columns = {"SpecimenSiteOfCollection": "SpecimenSite"})
        .merge(
            output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors[
                [
                    "index_of_row_of_diagnosis_data_paired_with_specimen",
                    "ORIENSpecimenID",
                    "AssignedPrimarySite"
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
    manifest["has_cml_link"] = manifest["DeidSpecimenID"].notna()
    manifest["has_site"] = manifest["AssignedPrimarySite"].notna()
    manifest["is_cutaneous"] = manifest["AssignedPrimarySite"].astype(str).str.strip().str.lower().eq("cutaneous")
    manifest["Included"] = (
        manifest["QC_Pass"] &
        manifest["has_cml_link"] &
        manifest["is_cutaneous"] &
        manifest["number_of_indicators_that_comparisons_are_false"].lt(2) &
        ~manifest["value_is_outlier"]
    )

    def provide_exclusion_reason(row) -> str:
        list_of_reasons = []
        if not safe_bool(row.get("QC_Pass", False)):
            list_of_reasons.append("Value of QCCheck is not \"Pass\".")
        number_of_indicators_that_comparisons_are_false = row.get("number_of_indicators_that_comparisons_are_false", 0)
        number_of_indicators_that_comparisons_are_false = (
            0
            if pd.isna(number_of_indicators_that_comparisons_are_false)
            else int(number_of_indicators_that_comparisons_are_false)
        )
        if number_of_indicators_that_comparisons_are_false >= 2:
            list_of_reasons.append(f"{number_of_indicators_that_comparisons_are_false} comparisons are false.")
        for name_of_standard_column in ["MappedReads", "ExonicRate", "AlignmentRate", "rRNARate", "Duplication"]:
            has_source = dictionary_of_names_of_standard_columns_and_sources.get(name_of_standard_column) is not None
            if not has_source:
                continue
            val = row.get(f"{name_of_standard_column}_is_outlier", False)
            if safe_bool(val, False):
                list_of_reasons.append(f"{name_of_standard_column} is outlier.")
        if not safe_bool(row.get("has_cml_link", False)):
            list_of_reasons.append("There is no link between SLID and DeidSpecimenID.")
        if not safe_bool(row.get("has_site", False)):
            list_of_reasons.append("AssignedPrimarySite is missing.")
        elif not safe_bool(row.get("is_cutaneous", False)):
            site = row.get("AssignedPrimarySite", pd.NA)
            list_of_reasons.append(f"Sample is not cutaneous and is {site}.")
        return " ".join(list_of_reasons)
    
    manifest["ExclusionReason"] = manifest.apply(
        lambda row: "" if safe_bool(row.get("Included"), False) else provide_exclusion_reason(row),
        axis = 1
    )

    rows_metrics: list[dict] = []
    for metric, th in dictionary_of_names_of_standard_columns_and_dictionaries_of_directions_and_thresholds.items():
        source = dictionary_of_names_of_standard_columns_and_sources.get(metric)
        available = source is not None and QC_data[metric].notna().any()
        avail_label = "available" if available else "not available"

        # Pass/fail/outlier counts
        if available:
            comp = QC_data[f"comparison_is_true_for_{metric}"]
            n_pass = int((comp == True).sum())
            n_fail = int((comp == False).sum())
            n_out = int((QC_data[f"{metric}_is_outlier"] == True).sum())
            series = QC_data[metric].dropna()
            v_min = float(series.min()) if not series.empty else None
            v_med = float(series.median()) if not series.empty else None
            v_max = float(series.max()) if not series.empty else None
            n_nonnull = int(series.size)
        else:
            n_pass = n_fail = n_out = n_nonnull = 0
            v_min = v_med = v_max = None

        rows_metrics.append({
            "Type": "metric",
            "Metric": metric,
            "SourceColumn": source if source is not None else "not available",
            "Availability": avail_label,
            "ThresholdDirection": th["direction"],
            "ThresholdValue": th["threshold"],
            "N_nonnull": n_nonnull,
            "N_pass_threshold": n_pass,
            "N_fail_threshold": n_fail,
            "N_outliers_>3SD": n_out,
            "Min": v_min,
            "Median": v_med,
            "Max": v_max
        })

    # Counts included/excluded by reason (high-level & breakdown)
    total_n = int(manifest.shape[0])
    n_included = int(manifest["Included"].sum())
    n_excluded = int(total_n - n_included)

    rows_reasons: list[dict] = []
    def add_reason(name: str, count: int):
        rows_reasons.append({"Type": "reason", "Reason": name, "Count": int(count)})

    add_reason("TOTAL", total_n)
    add_reason("Included", n_included)
    add_reason("Excluded", n_excluded)

    add_reason('QCCheck != "Pass"', int((~manifest["QC_Pass"].fillna(False)).sum()))
    add_reason(">=2 thresholds failed", int((manifest["number_of_indicators_that_comparisons_are_false"] >= 2).fillna(False).sum()))
    for metric in ["MappedReads", "ExonicRate", "AlignmentRate", "rRNARate", "Duplication"]:
        source = dictionary_of_names_of_standard_columns_and_sources.get(metric)
        if source is None:
            add_reason(f"{metric} outlier (not available)", 0)
        else:
            add_reason(f"{metric} outlier", int((manifest[f"{metric}_is_outlier"] == True).sum()))
    add_reason("No SLID↔DeidSpecimenID link", int((~manifest["has_cml_link"]).sum()))
    add_reason("AssignedPrimarySite missing", int((~manifest["has_site"]).sum()))
    add_reason("Not cutaneous", int((manifest["has_site"] & ~manifest["is_cutaneous"]).sum()))

    qc_summary_df = pd.concat([
        pd.DataFrame(rows_metrics),
        pd.DataFrame(rows_reasons)
    ], ignore_index=True)
    qc_summary_df.to_csv("qc_summary.csv", index=False)

    # Markdown summary ("brief")
    thresholds_lines = []
    for metric, th in dictionary_of_names_of_standard_columns_and_dictionaries_of_directions_and_thresholds.items():
        src = dictionary_of_names_of_standard_columns_and_sources.get(metric)
        thresholds_lines.append(f"- **{metric}** ({'available' if src else 'not available'}; source: `{src if src else 'NA'}`): "
                                f"{'≥' if th['direction']=='ge' else '≤'} {th['threshold']}")

    # Data/vintage paths (with mod-times)
    try:
        qc_mtime = os.path.getmtime(paths.QC_data)
    except Exception:
        qc_mtime = None
    try:
        cml_mtime = os.path.getmtime(paths.clinical_molecular_linkage_data)
    except Exception:
        cml_mtime = None
    try:
        pairing_mtime = os.path.getmtime(paths.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors)
    except Exception:
        pairing_mtime = None
    try:
        dx_mtime = os.path.getmtime(paths.diagnosis_data)
    except Exception:
        dx_mtime = None
    expr_latest = _latest_mtime_in_dir(paths.gene_and_transcript_expression_results)

    md = []
    md.append("# QC Summary")
    md.append("")
    md.append(f"- Generated: {_fmt_ts(datetime.now().timestamp())}")
    md.append(f"- Total SLIDs: **{total_n}** | Included: **{n_included}** | Excluded: **{n_excluded}**")
    md.append("")
    md.append("## Thresholds & Availability")
    md += thresholds_lines
    md.append("")
    md.append("## Data sources & vintage")
    md.append(f"- QC data: `{paths.QC_data}` (mtime: {_fmt_ts(qc_mtime)})")
    md.append(f"- Clinical–molecular linkage: `{paths.clinical_molecular_linkage_data}` (mtime: {_fmt_ts(cml_mtime)})")
    md.append(f"- Pairing output: `{paths.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors}` (mtime: {_fmt_ts(pairing_mtime)})")
    md.append(f"- Diagnosis data: `{paths.diagnosis_data}` (mtime: {_fmt_ts(dx_mtime)})")
    md.append(f"- Expression dir: `{paths.gene_and_transcript_expression_results}` (latest file mtime: {_fmt_ts(expr_latest)})")
    md.append("")
    md.append("## Exclusion breakdown (counts)")
    for row in rows_reasons:
        if row["Type"] == "reason" and row["Reason"] not in ("TOTAL", "Included", "Excluded"):
            md.append(f"- {row['Reason']}: **{row['Count']}**")

    with open("qc_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    columns_to_exclude_from_output = [
        "number_of_indicators_that_comparisons_are_false",
        "value_is_outlier",
        "has_cml_link",
        "has_site",
        "is_cutaneous",
    ]
    manifest_to_save = manifest.drop(columns = columns_to_exclude_from_output)
    manifest_to_save.to_csv("manifest.csv", index = False)
    print(f"Cohort manifest has shape {manifest_to_save.shape}.")

    list_of_included_SLIDs = manifest.loc[manifest["Included"], "SLID"].dropna().unique().tolist()
    filtered_expression_matrix = expression_matrix.loc[
        :,
        [column for column in expression_matrix.columns if column in list_of_included_SLIDs]
    ]
    filtered_expression_matrix.to_csv("filtered_expression_matrix.csv")
    print(f"Filtered expression matrix has shape {filtered_expression_matrix.shape}.")
    print(f"CSV and Markdown QC summaries were written.")


if __name__ == "__main__":
    create_expression_matrix()