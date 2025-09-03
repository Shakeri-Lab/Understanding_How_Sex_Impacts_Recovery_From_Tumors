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


def safe_bool(x, default: bool = False) -> bool:
    try:
        if pd.isna(x):
            return default
    except TypeError:
        return default
    return bool(x)


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


def standardize_columns(QC_data: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, str]:
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
            manifest[name_of_standard_column] = pd.to_numeric(QC_data[name_of_source], errors = "raise")
        else:
            manifest[name_of_standard_column] = pd.NA
    print(f"Dictionary of names of standard columns and sources is\n{dictionary_of_names_of_standard_columns_and_sources}.")
    return dictionary_of_names_of_standard_columns_and_sources


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


def create_expression_matrices():
    list_of_paths = list(paths.gene_and_transcript_expression_results.glob("*.genes.results"))
    full_expression_matrix = create_full_expression_matrix(list_of_paths)
    QC_data = load_QC_data()
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
    dictionary_of_names_of_standard_columns_and_sources = standardize_columns(QC_data, manifest)
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


    rows_metrics: list[dict] = []
    for metric, th in dictionary_of_names_of_standard_columns_and_dictionaries_of_directions_and_thresholds.items():
        source = dictionary_of_names_of_standard_columns_and_sources.get(metric)
        available = source is not None and QC_data[metric].notna().any()
        avail_label = "available" if available else "not available"

        # Pass/fail/outlier counts
        if available:
            comp = manifest[f"comparison_is_true_for_{metric}"]
            n_pass = int((comp == True).sum())
            n_fail = int((comp == False).sum())
            n_out = int((manifest[f"{metric}_is_outlier"] == True).sum())
            series = manifest[metric].dropna()
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
    add_reason("AssignedPrimarySite missing", int((~manifest["sample_has_assigned_primary_site"]).sum()))
    add_reason("Not cutaneous", int((manifest["sample_has_assigned_primary_site"] & ~manifest["sample_is_cutaneous"]).sum()))

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
    paths.ensure_dependencies_for_creating_expression_matrices_exist()
    create_expression_matrices()