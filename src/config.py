import os
from pathlib import Path


class Paths():
    
    def __init__(self):
        
        # src/immune_analysis
        # dependencies
        self.root = Path("/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors")
        # -----
        self.map_from_sample_to_patient = self.root / "output/eda/sample_to_patient_map.csv"
        
        # src/immune_analysis/data_loading.py
        # dependencies
        self.gene_and_transcript_expression_results = self.root / "../RNAseq/gene_and_transcript_expression_results"
        self.manifest_and_QC_files = self.root / "../Manifest_and_QC_Files"
        self.normalized_clinical_data = self.root / "../Clinical_Data/24PRJ217UVA_NormalizedFiles"
        self.outputs_of_data_loading = self.root / "output/data_loading"
        # -----
        self.QC_data = self.manifest_and_QC_files / "24PRJ217UVA_20250130_RNASeq_QCMetrics.csv"
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
        self.data_frame_of_melanoma_patient_and_sequencing_data = self.root / "output/eda/melanoma_patients_with_sequencing.csv"
        # outputs
        self.melanoma_sample_immune_clinical_data = self.outputs_of_microenv / "melanoma_sample_immune_clinical.csv" # TODO: Rename.
        self.map_of_biopsy_locations_to_counts = self.outputs_of_microenv / "specimen_site_summary.csv"
        self.map_of_procedure_types_to_counts = self.outputs_of_microenv / "procedure_type_summary.csv"
        self.map_of_indicators_that_specimens_are_part_of_metastatic_disease_to_counts = self.outputs_of_microenv / "metastatic_status_summary.csv"
        self.data_frame_of_scores_by_sample_and_cell_type = self.outputs_of_microenv / "xcell_scores_raw.csv"
        self.focused_data_frame_of_scores_by_sample_and_cell_type = self.outputs_of_microenv / "xcell_scores_focused_panel.csv"
    
    
    def ensure_dependencies_for_immune_analysis_exist(self):
        os.makedirs(self.root, exist_ok = True)
        assert os.path.exists(self.map_from_sample_to_patient), f"The dependency of `src/immune_analysis` `{self.map_from_sample_to_patient}` does not exist."

        
    def ensure_dependencies_for_data_loading_exist(self):
        self.ensure_dependencies_for_immune_analysis_exist()
        for path in [self.gene_and_transcript_expression_results, self.manifest_and_QC_files, self.normalized_clinical_data, self.outputs_of_data_loading]:
            os.makedirs(path, exist_ok = True)
        for path in [self.QC_data, self.diagnosis_data, self.medications_data, self.patient_data, self.surgery_biopsy_data]:
            assert os.path.exists(path), f"The dependency of `src/immune_analysis/data_loading.py` `{path}` does not exist."
        

    def ensure_dependencies_for_microenv_exist(self):
        self.ensure_dependencies_for_immune_analysis_exist()
        os.makedirs(self.outputs_of_microenv, exist_ok = True)
        assert os.path.exists(self.data_frame_of_melanoma_patient_and_sequencing_data), f"The dependency of `src/immune_analysis/microenv.py` `{self.data_frame_of_melanoma_patient_and_sequencing_data}` does not exist."


paths = Paths()