from pathlib import Path
import os


class Paths():
    
    def __init__(self):

        # ORIEN_analysis
        # dependencies
        self.root = Path("/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors/ORIEN_analysis")
        self.output = self.root / "output"
        # -----
        # <no files>
        # outputs
        # <no files>

        # ORIEN_analysis/run_xCell_analysis.py
        # dependencies
        self.outputs_of_running_xCell_analysis = self.output / "running_xCell_analysis"
        # -----
        # <no files>
        # outputs
        # <no files>

        # ORIEN_analysis/create_expression_matrices.py
        # dependencies
        self.gene_and_transcript_expression_results = self.root / "../../RNAseq/gene_and_transcript_expression_results"
        self.manifest_and_QC_files = self.root / "../../Manifest_and_QC_Files"
        self.outputs_of_creating_expression_matrices = self.output / "creating_expression_matrices"
        # -----
        self.QC_data = self.manifest_and_QC_files / "24PRJ217UVA_20250130_RNASeq_QCMetrics.csv"
        self.clinical_molecular_linkage_data = self.root / "../../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv"
        self.diagnosis_data = self.root / "../../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_Diagnosis_V4.csv"
        self.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors = self.root / "../pair_clinical_data_and_stages_of_tumors/output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv"
        # outputs
        self.full_expression_matrix = self.outputs_of_creating_expression_matrices / "full_expression_matrix.csv" # EM1
        self.manifest = self.outputs_of_creating_expression_matrices / "manifest.csv"
        self.QC_summary_in_Markdown = self.outputs_of_creating_expression_matrices / "QC_summary.md"
        self.QC_summary_of_CSVs = self.outputs_of_creating_expression_matrices / "QC_summary.csv"
        self.expression_matrix_with_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "expression_matrix_with_SLIDs_approved_by_manifest.csv" # EM2
        self.data_frame_of_Ensembl_IDs_and_HGNC_symbols = self.outputs_of_creating_expression_matrices / "data_frame_of_Ensembl_IDs_and_HGNC_symbols.csv"
        self.expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.csv" # EM3
        self.filtered_expression_matrix_with_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "filtered_expression_matrix_with_SLIDs_approved_by_manifest.csv" # EM4
        self.filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.csv" # EM5
        self.logged_filtered_expression_matrix_with_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "logged_filtered_expression_matrix_with_SLIDs_approved_by_manifest.csv" # EM6
        self.logged_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "logged_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.csv" # EM7
        self.z_scored_filtered_expression_matrix_with_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "z_scored_filtered_expression_matrix_with_SLIDs_approved_by_manifest.csv" # EM8
        self.z_scored_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "z_scored_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.csv" # EM9


    def ensure_dependencies_for_creating_expression_matrices_exist(self):
        for path in [
            self.gene_and_transcript_expression_results,
            self.outputs_of_creating_expression_matrices
        ]:
            os.makedirs(path, exist_ok = True)
        for path in [
            self.QC_data,
            self.clinical_molecular_linkage_data,
            self.diagnosis_data,
            self.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors
        ]:
            assert os.path.exists(path), f"The dependency of creating expression matrices `{path}` does not exist."


    def ensure_dependencies_for_running_xCell_analysis_exist(self):
        for path in [
            self.outputs_of_running_xCell_analysis
        ]:
            os.makedirs(path, exist_ok = True)
        for path in [
            self.filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest
        ]:
            assert os.path.exists(path), f"The dependency of creating expression matrices `{path}` does not exist."


paths = Paths()