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
                usecols = ["gene_id", "TPM"]
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
    print(f"The number of unique genes is {len(expression_matrix.index.unique())}.")
    print(f"The number of unique sample IDs is {len(expression_matrix.columns.unique())}.")
    QC_data = pd.read_csv(paths.QC_data)
    print(f"QC data has shape {QC_data.shape}.")
    print(f"The number of unique patient IDs is {len(QC_data["ORIENAvatarKey"].unique())}.")
    print(f"The number of unique sample IDs is {len(QC_data["SLID"].unique())}.")
    output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors = pd.read_csv(
        "../pair_clinical_data_and_stages_of_tumors/output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv"
    )
    # We restrict strictly to the prespecified cutaneous melanoma cohort with paired clinical-tumor data.
    output_of_pipeline_for_cutaneous_specimens = output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors[
        output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors["AssignedPrimarySite"] == "cutaneous"
    ]
    series_of_patient_IDs_in_QC_Data = QC_data["ORIENAvatarKey"]
    series_of_patient_IDs_in_output_of_pipeline_for_cutaneous_specimens = output_of_pipeline_for_cutaneous_specimens["AvatarKey"]
    set_of_patient_IDs_in_output_of_pipeline_for_cutaneous_specimens = set(series_of_patient_IDs_in_output_of_pipeline_for_cutaneous_specimens)
    QC_data_for_patients_with_cutaneous_specimens = QC_data[series_of_patient_IDs_in_QC_Data.isin(set_of_patient_IDs_in_output_of_pipeline_for_cutaneous_specimens)]
    QC_data_for_patients_with_cutaneous_specimens.to_csv("QC_data_for_patients_with_cutaneous_specimens.csv")
    series_of_sample_IDs = QC_data_for_patients_with_cutaneous_specimens["SLID"]
    filtered_expression_matrix = expression_matrix.loc[:, series_of_sample_IDs]
    '''
    Filtering will raise if any sample ID is not the name of a column in expression matrix.
    This possibility serves as a check that all sample IDs are in expression matrix.
    '''
    filtered_expression_matrix.to_csv("filtered_expression_matrix.csv")


if __name__ == "__main__":
    create_expression_matrix()