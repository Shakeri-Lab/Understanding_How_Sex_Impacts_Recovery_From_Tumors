'''
create_expression_matrix.py

We build an expression matrix with 60,609 Ensembl IDs and 333 sample IDs.
We filter the expression matrix to columns with sample IDs
in a data frame of filtered QC data and IDs of cutaneous specimens
output by our pipeline for pairing clinical data and stages of tumors.
'''


from ORIEN_analysis.config import paths
import os
import pandas as pd
from functools import reduce
import operator


def create_series_of_indicators_that_values_are_outliers(series: pd.Series) -> pd.Series:
    mean = series.mean()
    standard_deviation = series.std()
    return (series.sub(mean).abs().div(standard_deviation)).gt(3)


def provide_name_of_first_column_whose_name_matches_a_candidate(filtered_QC_data: pd.DataFrame, list_of_candidates: list[str]) -> str | None:
    for candidate in list_of_candidates:
        if candidate in filtered_QC_data.columns:
            return candidate
    return None


def create_expression_matrix():
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
    expression_matrix.to_csv("expression_matrix.csv")
    print(f"Expression matrix has shape {expression_matrix.shape}.")
    print(f"The number of unique genes is {expression_matrix.index.nunique()}.")
    print(f"The number of unique sample IDs is {expression_matrix.columns.nunique()}.")
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
    list_of_dictionaries_of_information_re_columns = [
        {
            "name": "Alignment",
            "list_of_candidates": ["AlignmentRate", "PercentAligned", "pct_aligned"],
            "direction": "ge",
            "threshold": 0.7
        },
        {
            "name": "rRNA",
            "list_of_candidates": ["rRNA Rate", "rRNARate", "Pct_rRNA"],
            "direction": "le",
            "threshold": 0.2
        },
        {
            "name": "Duplication",
            "list_of_candidates": ["Duplication", "DupRate", "PCT_DUPLICATION"],
            "direction": "le",
            "threshold": 0.6
        },
        {
            "name": "Mapped reads",
            "list_of_candidates": ["MappedReads", "Total Mapped Reads"],
            "direction": "ge",
            "threshold": 10E6
        },
        {
            "name": "Exonic rate",
            "list_of_candidates": ["ExonicRate", "Pct_Exonic"],
            "direction": "ge",
            "threshold": 0.5
        }
    ]
    list_of_series_of_indicators_that_values_did_not_meet_condition = []
    list_of_series_of_indicators_that_values_are_outliers = []
    for dictionary_of_information_re_column in list_of_dictionaries_of_information_re_columns:
        name_of_dictionary = dictionary_of_information_re_column["name"]
        name_of_column = provide_name_of_first_column_whose_name_matches_a_candidate(
            QC_data,
            dictionary_of_information_re_column["list_of_candidates"]
        )
        direction = dictionary_of_information_re_column["direction"]
        threshold = dictionary_of_information_re_column["threshold"]
        if name_of_column is None:
            print(f"Column was not found for dictionary {name_of_dictionary}.")
            continue
        series_of_values = QC_data[name_of_column]
        if direction == "ge":
            series_of_indicators_that_values_met_condition = series_of_values >= threshold
        elif direction == "le":
            series_of_indicators_that_values_met_condition = series_of_values <= threshold
        else:
            raise Exception(f"Direction {direction} is invalid.")
        series_of_indicators_that_values_did_not_meet_condition = ~series_of_indicators_that_values_met_condition
        list_of_series_of_indicators_that_values_did_not_meet_condition.append(
            series_of_indicators_that_values_did_not_meet_condition
        )
        series_of_indicators_that_values_are_outliers = create_series_of_indicators_that_values_are_outliers(series_of_values)
        list_of_series_of_indicators_that_values_are_outliers.append(series_of_indicators_that_values_are_outliers)
    series_of_numbers_of_failures = sum(list_of_series_of_indicators_that_values_did_not_meet_condition)
    series_of_indicators_that_some_values_are_outliers = reduce(operator.or_, list_of_series_of_indicators_that_values_are_outliers)
    series_of_indicators_that_row_should_be_excluded = (series_of_numbers_of_failures >= 2) | series_of_indicators_that_some_values_are_outliers
    filtered_QC_data = QC_data.loc[~series_of_indicators_that_row_should_be_excluded]
    filtered_QC_data = filtered_QC_data[filtered_QC_data["QCCheck"] == "Pass"]
    print(f"Filtered QC data has shape {filtered_QC_data.shape}.")
    print(f"The number of unique patient IDs is {filtered_QC_data['ORIENAvatarKey'].nunique()}.")
    print(f"The number of unique sample IDs is {filtered_QC_data['SLID'].nunique()}.")
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
    set_of_SLIDs_in_QC_data = set(QC_data["SLID"])
    set_of_SLIDs_in_clinical_molecular_linkage_data = set(clinical_molecular_linkage_data["RNASeq"])
    list_of_SLIDs_in_QC_data_and_not_in_clinical_molecular_linkage_data = sorted(set_of_SLIDs_in_QC_data - set_of_SLIDs_in_clinical_molecular_linkage_data)
    list_of_SLIDs_in_clinical_molecular_linkage_data_and_not_in_QC_data = sorted(set_of_SLIDs_in_clinical_molecular_linkage_data - set_of_SLIDs_in_QC_data)
    print(f"List of SLIDS in QC data and not in clinical molecular linkage data is {list_of_SLIDs_in_QC_data_and_not_in_clinical_molecular_linkage_data}.")
    print(f"List of SLIDS in clinical molecular linkage data and not in QC data is {list_of_SLIDs_in_clinical_molecular_linkage_data_and_not_in_QC_data}.")
    # There is 1 to 1 overlap between SLIDs in clinical molecular linkage data and QC data before string normalization.
    data_frame_of_filtered_QC_data_and_specimen_IDs = filtered_QC_data.merge(
        clinical_molecular_linkage_data[["RNASeq", "DeidSpecimenID"]],
        how = "left",
        left_on = "SLID",
        right_on = "RNASeq"
    )
    print(f"Data frame of filtered QC data and specimen IDs has shape {data_frame_of_filtered_QC_data_and_specimen_IDs.shape}.")
    print(f"The number of unique specimen IDs is {data_frame_of_filtered_QC_data_and_specimen_IDs['DeidSpecimenID'].nunique()}.")
    print(f"The number of unique sample IDs is {data_frame_of_filtered_QC_data_and_specimen_IDs['RNASeq'].nunique()}.")
    output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors = pd.read_csv(
        paths.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors,
        dtype = {
            "AvatarKey": str,
            "AssignedPrimarySite": str
        }
    )
    print(f"Output of pipeline for pairing clinical data and stages of tumors has shape {output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.shape}.")
    print(f"The number of unique specimen IDs is {output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors['ORIENSpecimenID'].nunique()}.")
    set_of_ORIEN_specimen_IDs_in_output_of_pipeline = set(output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors["ORIENSpecimenID"])
    set_of_DEID_specimen_IDs_in_clinical_molecular_linkage_data = set(clinical_molecular_linkage_data["DeidSpecimenID"])
    list_of_ORIEN_specimen_IDs_that_are_not_DEID_specimen_IDs = sorted(set_of_ORIEN_specimen_IDs_in_output_of_pipeline - set_of_DEID_specimen_IDs_in_clinical_molecular_linkage_data)
    list_of_DEID_specimen_IDs_that_are_not_ORIEN_specimen_IDs = sorted(set_of_DEID_specimen_IDs_in_clinical_molecular_linkage_data - set_of_ORIEN_specimen_IDs_in_output_of_pipeline)
    print(f"List of ORIEN specimen IDs that are not DEID specimen IDs is {list_of_ORIEN_specimen_IDs_that_are_not_DEID_specimen_IDs}.")
    #print(f"List of DEID specimen IDs that are not ORIEN specimen IDs is {list_of_DEID_specimen_IDs_that_are_not_ORIEN_specimen_IDs}.")
    print(f"List of DEID specimen IDs that are not ORIEN specimen IDs is large.")
    # All ORIEN specimen IDs in output of pipeline are DEID specimen IDs in clinical molecular linkage data. 
    output_of_pipeline_for_cutaneous_specimens = output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors[
        output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors["AssignedPrimarySite"] == "cutaneous"
    ]
    print(f"Output of pipeline for cutaneous specimens has shape {output_of_pipeline_for_cutaneous_specimens.shape}.")
    print(f"The number of unique specimen IDs is {output_of_pipeline_for_cutaneous_specimens['ORIENSpecimenID'].nunique()}.")
    series_of_specimen_IDs_in_filtered_QC_data = data_frame_of_filtered_QC_data_and_specimen_IDs["DeidSpecimenID"]
    series_of_IDs_of_cutaneous_specimens = output_of_pipeline_for_cutaneous_specimens["ORIENSpecimenID"]
    set_of_IDs_of_cutaneous_specimens = set(series_of_IDs_of_cutaneous_specimens)
    data_frame_of_filtered_QC_data_and_IDs_of_cutaneous_specimens = data_frame_of_filtered_QC_data_and_specimen_IDs[
        series_of_specimen_IDs_in_filtered_QC_data.isin(set_of_IDs_of_cutaneous_specimens)
    ]
    print(f"Data frame of filtered QC data and IDs of cutaneous specimens has shape {data_frame_of_filtered_QC_data_and_IDs_of_cutaneous_specimens.shape}.")
    print(f"The number of unique specimen IDs is {data_frame_of_filtered_QC_data_and_IDs_of_cutaneous_specimens['DeidSpecimenID'].nunique()}.")
    print(f"The number of unique sample IDs is {data_frame_of_filtered_QC_data_and_IDs_of_cutaneous_specimens['SLID'].nunique()}.")
    series_of_sample_IDs = data_frame_of_filtered_QC_data_and_IDs_of_cutaneous_specimens["SLID"]
    filtered_expression_matrix = expression_matrix.loc[:, series_of_sample_IDs]
    print(f"Filtered expression matrix has shape {filtered_expression_matrix.shape}.")
    print(f"The number of unique genes is {filtered_expression_matrix.index.nunique()}.")
    print(f"The number of unique sample IDs is {filtered_expression_matrix.columns.nunique()}.")
    filtered_expression_matrix.to_csv("filtered_expression_matrix.csv")


if __name__ == "__main__":
    create_expression_matrix()