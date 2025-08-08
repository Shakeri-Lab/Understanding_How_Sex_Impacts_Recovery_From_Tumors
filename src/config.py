import os
from pathlib import Path


# Define a list of columns for a focused panel for ICB response analysis.
FOCUSED_XCELL_PANEL = [
    'CD8+ T-cells',
    'CD4+ memory T-cells',
    'Tgd cells',
    'Macrophages M2',
    'Tregs',
    'cDC',
    'pDC',
    'Memory B-cells',
    'Plasma cells',
    'Endothelial cells',
    'Fibroblasts',
    'ImmuneScore',
    'StromaScore',
    'MicroenvironmentScore'
]


class Paths():
    '''
    Class Paths is a template for a singleton that records dependencies and outputs of and ensures dependencies exist for
    `src/data_processing/eda.py`,
    `src/immune_analysis/data_loading.py`,
    `src/immune_analysis/microenv.py`,
    `src/immune_analysis/immune_analysis.py`, and
    `src/immune_analysis/linear_mixed_models.py`.
    '''
    
    def __init__(self):
        
        # src
        # dependencies
        self.root = Path("/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors")
        self.manifest_and_QC_files = self.root / "../Manifest_and_QC_Files"
        self.normalized_clinical_data = self.root / "../Clinical_Data/24PRJ217UVA_NormalizedFiles"
        # -----
        self.QC_data = self.manifest_and_QC_files / "24PRJ217UVA_20250130_RNASeq_QCMetrics.csv"
        # outputs
        # <no files>
        
        # src/data_processing/eda.py
        # dependencies
        self.outputs_of_eda = self.root / "output/eda"
        self.eda_plots = self.outputs_of_eda / "plots"
        self.eda_reports = self.outputs_of_eda / "reports"
        # -----
        # files in self.normalized_clinical_data. Filenames are generated dynamically.
        # outputs
        self.map_from_sample_to_patient = self.outputs_of_eda / "sample_to_patient_map.csv"
        self.data_frame_of_melanoma_patient_and_sequencing_data = self.outputs_of_eda / "melanoma_patients_with_sequencing.csv"
        self.eda_summary_statistics = self.eda_reports / "summary_statistics.csv"
        self.eda_icb_distribution = self.eda_reports / "icb_distribution.csv"
        self.eda_sex_distribution = self.eda_plots / "sex_distribution.png"
        
        # src/immune_analysis/data_loading.py
        # dependencies
        self.outputs_of_data_loading = self.root / "output/data_loading"
        self.gene_and_transcript_expression_results = self.root / "../RNAseq/gene_and_transcript_expression_results"
        # -----
        # files in self.gene_and_transcript_expression_results. Filenames are generated dynamically.
        # self.map_from_sample_to_patient, which is defined above
        # self.QC_data, which is defined above
        self.clinical_molecular_linkage_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv"
        self.diagnosis_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_Diagnosis_V4.csv"
        self.medications_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_Medications_V4.csv"
        self.patient_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_PatientMaster_V4.csv"
        self.surgery_biopsy_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_SurgeryBiopsy_V4.csv"
        # outputs
        self.data_frame_of_Ensembl_IDs_and_HGNC_symbols = self.outputs_of_data_loading / "data_frame_of_Ensembl_IDs_and_HGNC_symbols.csv"
        self.melanoma_clinical_data = self.outputs_of_data_loading / "melanoma_clinical_data.csv"
        self.melanoma_expression_matrix = self.outputs_of_data_loading / "melanoma_expression_matrix.csv"
        
        # src/immune_analysis/microenv.py
        # dependencies
        self.outputs_of_microenv = self.root / "output/microenv"
        # -----
        # self.data_frame_of_melanoma_patient_and_sequencing_data, which is defined above
        # outputs
        self.melanoma_sample_immune_clinical_data = self.outputs_of_microenv / "melanoma_sample_immune_clinical.csv" # TODO: Rename.
        self.map_of_biopsy_locations_to_counts = self.outputs_of_microenv / "specimen_site_summary.csv"
        self.map_of_procedure_types_to_counts = self.outputs_of_microenv / "procedure_type_summary.csv"
        self.map_of_indicators_that_specimens_are_part_of_metastatic_disease_to_counts = self.outputs_of_microenv / "metastatic_status_summary.csv"
        self.enrichment_data_frame_per_xCell = self.outputs_of_microenv / "xcell_scores_raw_per_xCell.csv"
        self.focused_enrichment_data_frame = self.outputs_of_microenv / "xcell_scores_focused_panel.csv"
        self.enrichment_data_frame_per_xCell2_and_Pan_Cancer = self.outputs_of_microenv / "xcell_scores_raw_per_xCell2_and_Pan_Cancer.csv"
        self.enrichment_data_frame_per_xCell2_and_TME_Compendium = self.outputs_of_microenv / "xcell_scores_raw_per_xCell2_and_TME_Compendium.csv"
        
        # src/immune_analysis/immune_analysis.py
        # dependencies
        self.outputs_of_immune_analysis = self.root / "output/immune_analysis"
        self.distributions_of_abundance_of_cells_of_type_by_group = self.outputs_of_immune_analysis / "distributions_of_abundance_of_cells_of_type_by_group"
        # -----
        # self.melanoma_sample_immune_clinical_data, which is defined above
        # self.focused_data_frame_of_scores_by_sample_and_cell_type, which is defined above
        # outputs
        self.matrix_of_correlations_of_abundances_of_cell_types = self.outputs_of_immune_analysis / "matrix_of_correlations_of_abundances_of_cell_types.png"
        self.data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens = self.outputs_of_immune_analysis / "data_frame_of_indicators_of_metastasis_types_of_specimens_and_numbers_of_specimens.csv"
        self.data_frame_of_statistics_for_cell_types_and_sexes = self.outputs_of_immune_analysis / "data_frame_of_statistics_for_cell_types_and_sexes.csv"
        self.plot_of_numbers_of_samples_by_specimen_site_and_metastatic_status = self.outputs_of_immune_analysis / "plot_of_numbers_of_samples_by_specimen_site_and_metastatic_status.png"
        self.data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis = self.outputs_of_immune_analysis / "data_frame_of_statistics_for_cell_types_and_indicators_of_metastasis.csv"
        self.metastatic_vs_primary_heatmap = self.outputs_of_immune_analysis / "metastatic_vs_primary_heatmap.png"
        
        # src/immune_analysis/linear_mixed_models.py
        # dependencies
        self.outputs_of_linear_mixed_models = self.root / "output/linear_mixed_models"
        # -----
        # self.enrichment_data_frame_per_xCell, which is defined above
        # self.enrichment_data_frame_per_xCell2_and_Pan_Cancer, which is defined above
        # self.enrichment_data_frame_per_xCell2_and_TME_Compendium, which is defined above
        # self.melanoma_sample_immune_clinical_data, which is defined above
        # self.QC_data, which is defined above
        # outputs
        self.mixed_model_results_per_xCell = self.outputs_of_linear_mixed_models / "mixed_model_results_per_xCell.csv"
        self.mixed_model_results_significant_per_xCell = self.outputs_of_linear_mixed_models / "mixed_model_results_significant_per_xCell.csv"
        self.mixed_model_results_per_xCell2_and_Pan_Cancer = self.outputs_of_linear_mixed_models / "mixed_model_results_per_xCell2_and_Pan_Cancer.csv"
        self.mixed_model_results_significant_per_xCell2_and_Pan_Cancer = self.outputs_of_linear_mixed_models / "mixed_model_results_significant_per_xCell2_and_Pan_Cancer.csv"
        self.mixed_model_results_per_xCell2_and_TME_Compendium = self.outputs_of_linear_mixed_models / "mixed_model_results_per_xCell2_and_TME_Compendium.csv"
        self.mixed_model_results_significant_per_xCell2_and_TME_Compendium = self.outputs_of_linear_mixed_models / "mixed_model_results_significant_per_xCell2_and_TME_Compendium.csv"
        
        # src/immune_analysis/compare_enrichment_scores.py
        # dependencies
        self.outputs_of_compare_enrichment_scores = self.root / "output/compare_enrichment_scores"
        # -----
        # self.enrichment_data_frame_per_xCell, which is defined above
        # self.enrichment_data_frame_per_xCell2_and_Pan_Cancer, which is defined above
        # self.enrichment_data_frame_per_xCell2_and_TME_Compendium, which is defined above
        # outputs
        self.comparisons_for_females_and_males_and_xCell = self.outputs_of_compare_enrichment_scores / "comparisons_for_females_and_males_and_xCell.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell = self.outputs_of_compare_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell = self.outputs_of_compare_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell.csv"
        self.comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer = self.outputs_of_compare_enrichment_scores / "comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer = self.outputs_of_compare_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_Pan_Cancer = self.outputs_of_compare_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_Pan_Cancer.csv"
        self.comparisons_for_females_and_males_and_xCell2_and_TME_Compendium = self.outputs_of_compare_enrichment_scores / "comparisons_for_females_and_males_and_xCell2_and_TME_Compendium.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_TME_Compendium = self.outputs_of_compare_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_TME_Compendium.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_TME_Compendium = self.outputs_of_compare_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_TME_Compendium.csv"       
        
        # src/immune_analysis/treatment_analysis.py
        # dependencies
        self.outputs_of_treatment_analysis = self.root / "output/treatment_analysis"
        self.treatment_analysis_plots = self.outputs_of_treatment_analysis / "plots"
        # -----
        self.outcomes_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_Outcomes_V4.csv"
        self.vital_status_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_VitalStatus_V4.csv"
        # outputs
        self.mediation_results = self.outputs_of_treatment_analysis / "mediation_results.csv"
        # Files with names of the form {cell_type}_survival.png are created dynamically.
        self.immune_cell_differences = self.treatment_analysis_plots / "immune_cell_differences.png"
        self.plot_of_t_cell_phenotypes = self.treatment_analysis_plots / "t_cell_phenotypes.png"
        self.mediation_analysis = self.treatment_analysis_plots / "mediation_analysis.png"
        self.sex_differences_immune = self.outputs_of_treatment_analysis / "sex_differences_immune.csv"
        self.immune_cell_differences_2 = self.treatment_analysis_plots / "immune_cell_differences_2.png"
        self.response_patterns = self.outputs_of_treatment_analysis / "response_patterns.csv"
        self.sex_differences = self.outputs_of_treatment_analysis / "sex_differences.csv"
        self.data_frame_of_T_cell_phenotypes = self.outputs_of_treatment_analysis / "data_frame_of_T_cell_phenotypes.csv"
        self.mediation_results_2 = self.outputs_of_treatment_analysis / "mediation_results_2.csv"
    
    
    def ensure_dependencies_for_src_exist(self):
        for path in [self.root, self.manifest_and_QC_files, self.normalized_clinical_data]:
            os.makedirs(path, exist_ok = True)
        assert os.path.exists(self.QC_data), f"The dependency of `src` `{path}` does not exist."
    
    
    def ensure_dependencies_for_eda_exist(self):
        for path in [self.outputs_of_eda, self.eda_plots, self.eda_reports]:
            os.makedirs(path, exist_ok = True)

        
    def ensure_dependencies_for_data_loading_exist(self):
        for path in [self.gene_and_transcript_expression_results, self.outputs_of_data_loading]:
            os.makedirs(path, exist_ok = True)
        for path in [self.diagnosis_data, self.medications_data, self.patient_data, self.surgery_biopsy_data]:
            assert os.path.exists(path), f"The dependency of `src/immune_analysis/data_loading.py` `{path}` does not exist."
        

    def ensure_dependencies_for_microenv_exist(self):
        os.makedirs(self.outputs_of_microenv, exist_ok = True)
        assert os.path.exists(self.data_frame_of_melanoma_patient_and_sequencing_data), f"The dependency of `src/immune_analysis/microenv.py` `{self.data_frame_of_melanoma_patient_and_sequencing_data}` does not exist."
        
        
    def ensure_dependencies_for_immune_analysis_exist(self):
        for path in [
            self.outputs_of_immune_analysis,
            self.distributions_of_abundance_of_cells_of_type_by_group
        ]:
            os.makedirs(path, exist_ok = True)
    
    
    def ensure_dependencies_for_linear_mixed_models_exist(self):
        os.makedirs(self.outputs_of_linear_mixed_models, exist_ok = True)
        for path in [
            self.enrichment_data_frame_per_xCell,
            self.enrichment_data_frame_per_xCell2_and_Pan_Cancer,
            self.enrichment_data_frame_per_xCell2_and_TME_Compendium,
            self.melanoma_sample_immune_clinical_data,
            self.QC_data
        ]:
            assert os.path.exists(path), f"The dependency of `src/immune_analysis/linear_mixed_models.py` `{path}` does not exist."


    def ensure_dependencies_for_compare_enrichment_scores_exist(self):
        os.makedirs(self.outputs_of_compare_enrichment_scores, exist_ok = True)
        for path in [
            self.enrichment_data_frame_per_xCell,
            self.enrichment_data_frame_per_xCell2_and_Pan_Cancer,
            self.enrichment_data_frame_per_xCell2_and_TME_Compendium
        ]:
            assert os.path.exists(path), f"The dependency of `src/immune_analysis/compare_enrichment_scores.py` `{path}` does not exist."


paths = Paths()