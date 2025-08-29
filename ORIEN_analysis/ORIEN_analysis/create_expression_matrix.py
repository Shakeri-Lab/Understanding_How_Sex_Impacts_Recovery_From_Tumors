'''
create_expression_matrix.py

We build an expression matrix with 60,609 Ensembl IDs and 333 sample IDs.
We filter the expression matrix to columns with sample IDs from QC data
for patients with specimens whose assigned primary site is "cutaneous".

QC data has 333 rows and 333 unique sample IDs.
Each row has a unique sample ID.
Filtering QC data to rows with mapped reads at least 10E6 and exonic rates at least 0.5 and
to rows with sample IDs for patients with cutaneous specimens
yields 318 rows and 318 unique sample IDs.
The filtered expression matrix has 60,609 Ensembl IDs and 318 sample IDs.
No KeyError occurred when selecting columns of the filtered expression matrix.
Every sample ID in the filtered QC data is the name of a column in the expression matrix.

We include all SLIDs for any patient who has a cutaneous specimen.
If a sample ID corresponds to a non-cutaneous specimen,
that sample ID will be included in the filtered expression matrix.
To include only sample IDs of cutaneous specimens in the filtered expression matrix,
we would need site information for each sample.
'''


from ORIEN_analysis.config import paths
import os
import pandas as pd


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
            "SLID": str
        }
    )
    print(f"QC data has shape {QC_data.shape}.")
    print(f"The number of unique patient IDs is {QC_data['ORIENAvatarKey'].nunique()}.")
    print(f"The number of unique sample IDs is {QC_data['SLID'].nunique()}.")
    filtered_QC_data = QC_data[
        (QC_data["MappedReads"] >= 10E6) &
        (QC_data["ExonicRate"] >= 0.5)
    ]
    print(f"Filtered QC data has shape {filtered_QC_data.shape}.")
    print(f"The number of unique patient IDs is {filtered_QC_data['ORIENAvatarKey'].nunique()}.")
    print(f"The number of unique sample IDs is {filtered_QC_data['SLID'].nunique()}.")
    output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors = pd.read_csv(
        paths.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors,
        dtype = {
            "AvatarKey": str,
            "AssignedPrimarySite": str
        }
    )
    print(f"Output of pipeline for pairing clinical data and stages of tumors has shape {output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.shape}.")
    print(f"The number of unique patient IDs is {output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors['AvatarKey'].nunique()}.")
    output_of_pipeline_for_cutaneous_specimens = output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors[
        output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors["AssignedPrimarySite"] == "cutaneous"
    ]
    print(f"Output of pipeline for cutaneous specimens has shape {output_of_pipeline_for_cutaneous_specimens.shape}.")
    print(f"The number of unique patient IDs is {output_of_pipeline_for_cutaneous_specimens['AvatarKey'].nunique()}.")
    series_of_patient_IDs_in_filtered_QC_Data = filtered_QC_data["ORIENAvatarKey"]
    series_of_patient_IDs_in_output_of_pipeline_for_cutaneous_specimens = output_of_pipeline_for_cutaneous_specimens["AvatarKey"]
    set_of_patient_IDs_in_output_of_pipeline_for_cutaneous_specimens = set(series_of_patient_IDs_in_output_of_pipeline_for_cutaneous_specimens)
    filtered_QC_data_for_patients_with_cutaneous_specimens = filtered_QC_data[series_of_patient_IDs_in_filtered_QC_Data.isin(set_of_patient_IDs_in_output_of_pipeline_for_cutaneous_specimens)]
    print(f"Filtered QC data for patients with cutaneous specimens has shape {filtered_QC_data_for_patients_with_cutaneous_specimens.shape}.")
    print(f"The number of unique patient IDs is {filtered_QC_data_for_patients_with_cutaneous_specimens['ORIENAvatarKey'].nunique()}.")
    print(f"The number of unique sample IDs is {filtered_QC_data_for_patients_with_cutaneous_specimens['SLID'].nunique()}.")
    series_of_sample_IDs = filtered_QC_data_for_patients_with_cutaneous_specimens["SLID"]
    filtered_expression_matrix = expression_matrix.loc[:, series_of_sample_IDs]
    print(f"Filtered expression matrix has shape {filtered_expression_matrix.shape}.")
    print(f"The number of unique genes is {filtered_expression_matrix.index.nunique()}.")
    print(f"The number of unique sample IDs is {filtered_expression_matrix.columns.nunique()}.")
    filtered_expression_matrix.to_csv("filtered_expression_matrix.csv")


if __name__ == "__main__":
    create_expression_matrix()