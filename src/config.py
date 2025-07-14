import os
from pathlib import Path


class Paths():
    
    def __init__(self):
        
        # src
        # dependencies
        # -----
        self.root = Path("/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors")
        self.manifest_and_QC_files = self.root / "../Manifest_and_QC_Files"
        self.QC_data = self.manifest_and_QC_files / "24PRJ217UVA_20250130_RNASeq_QCMetrics.csv"
        self.normalized_clinical_data = self.root / "../Clinical_Data/24PRJ217UVA_NormalizedFiles"
        
        # src/data_processing/eda.py
        # dependencies
        self.outputs_of_eda = self.root / "output/eda"
        self.eda_plots = self.outputs_of_eda / "plots"
        self.eda_reports = self.outputs_of_eda / "reports"
        # outputs
        self.map_from_sample_to_patient = self.root / "output/eda/sample_to_patient_map.csv"
        self.data_frame_of_melanoma_patient_and_sequencing_data = self.outputs_of_eda / "melanoma_patients_with_sequencing.csv"
        self.eda_summary_statistics = self.eda_reports / "summary_statistics.csv"
        self.eda_icb_distribution = self.eda_reports / "icb_distribution.csv"
        self.eda_sex_distribution = self.eda_plots / "sex_distribution.png"
        
        # src/immune_analysis/data_loading.py
        # dependencies
        self.gene_and_transcript_expression_results = self.root / "../RNAseq/gene_and_transcript_expression_results"
        self.outputs_of_data_loading = self.root / "output/data_loading"
        # -----
        self.diagnosis_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_Diagnosis_V4.csv"
        self.medications_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_Medications_V4.csv"
        self.patient_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_PatientMaster_V4.csv"
        self.surgery_biopsy_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_SurgeryBiopsy_V4.csv"
        # outputs
        self.melanoma_clinical_data = self.outputs_of_data_loading / "melanoma_clinical_data.csv"
        self.melanoma_expression_matrix = self.outputs_of_data_loading / "melanoma_expression_matrix.csv"
        
        # src/immune_analysis/microenv.py
        # dependencies
        self.outputs_of_microenv = self.root / "output/microenv"
        # -----
        # outputs
        self.melanoma_sample_immune_clinical_data = self.outputs_of_microenv / "melanoma_sample_immune_clinical.csv" # TODO: Rename.
        self.map_of_biopsy_locations_to_counts = self.outputs_of_microenv / "specimen_site_summary.csv"
        self.map_of_procedure_types_to_counts = self.outputs_of_microenv / "procedure_type_summary.csv"
        self.map_of_indicators_that_specimens_are_part_of_metastatic_disease_to_counts = self.outputs_of_microenv / "metastatic_status_summary.csv"
        self.data_frame_of_scores_by_sample_and_cell_type = self.outputs_of_microenv / "xcell_scores_raw.csv"
        self.focused_data_frame_of_scores_by_sample_and_cell_type = self.outputs_of_microenv / "xcell_scores_focused_panel.csv"
    
    
    def ensure_dependencies_for_src_exist(self):
        for path in [self.root, self.normalized_clinical_data, self.manifest_and_QC_files]:
            os.makedirs(path, exist_ok = True)
        assert os.path.exists(self.QC_data), f"The dependency of `src` `{path}` does not exist."
    
    
    def ensure_dependencies_for_eda_exist(self):
        self.ensure_dependencies_for_src_exist()
        for path in [self.outputs_of_eda, self.eda_plots, self.eda_reports]:
            os.makedirs(path, exist_ok = True)

        
    def ensure_dependencies_for_data_loading_exist(self):
        self.ensure_dependencies_for_eda_exist()
        for path in [self.gene_and_transcript_expression_results, self.outputs_of_data_loading]:
            os.makedirs(path, exist_ok = True)
        for path in [self.diagnosis_data, self.medications_data, self.patient_data, self.surgery_biopsy_data]:
            assert os.path.exists(path), f"The dependency of `src/immune_analysis/data_loading.py` `{path}` does not exist."
        

    def ensure_dependencies_for_microenv_exist(self):
        self.ensure_dependencies_for_src_exist()
        os.makedirs(self.outputs_of_microenv, exist_ok = True)
        assert os.path.exists(self.data_frame_of_melanoma_patient_and_sequencing_data), f"The dependency of `src/immune_analysis/microenv.py` `{self.data_frame_of_melanoma_patient_and_sequencing_data}` does not exist."


paths = Paths()