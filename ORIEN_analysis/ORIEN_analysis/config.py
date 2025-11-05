from pathlib import Path
import os


class Paths():
    
    def __init__(self):

        # ORIEN_analysis
        # dependencies
        self.root = Path("/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors/ORIEN_analysis")
        self.normalized_clinical_data = self.root / "../../Clinical_Data/24PRJ217UVA_NormalizedFiles"
        self.output = self.root / "output"
        # -----
        # <no files>
        # outputs
        # <no files>

        # ORIEN_analysis/ORIEN_analysis/pair_clinical_data_and_stages_of_tumors.py
        # dependencies
        self.outputs_of_pairing_clinical_data_and_stages_of_tumors = self.output / "pairing_clinical_data_and_stages_of_tumors"
        # -----
        self.clinical_molecular_linkage_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv"
        self.diagnosis_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_Diagnosis_V4.csv"
        self.metastatic_disease_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_MetastaticDisease_V4.csv"
        # outputs
        self.output_of_pairing_clinical_data_and_stages_of_tumors = self.outputs_of_pairing_clinical_data_and_stages_of_tumors / "output_of_pairing_clinical_data_and_stages_of_tumors.csv"

        # ORIEN_analysis/ORIEN_analysis/create_expression_matrices.py
        # dependencies
        self.gene_and_transcript_expression_results = self.root / "../../RNAseq/gene_and_transcript_expression_results"
        self.manifest_and_QC_files = self.root / "../../Manifest_and_QC_Files"
        self.outputs_of_creating_expression_matrices = self.output / "creating_expression_matrices"
        # -----
        self.QC_data = self.manifest_and_QC_files / "24PRJ217UVA_20250130_RNASeq_QCMetrics.csv"
        # self.clinical_molecular_linkage_data, which is defined above
        # self.diagnosis_data, which is defined above
        # self.output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors, which is defined above
        # outputs
        self.full_expression_matrix = self.outputs_of_creating_expression_matrices / "full_expression_matrix.csv" # EM1
        self.full_expression_matrix_with_HGNC_symbols = self.outputs_of_creating_expression_matrices / "full_expression_matrix_with_HGNC_symbols.csv"
        self.manifest = self.outputs_of_creating_expression_matrices / "manifest.csv"
        self.QC_summary_in_Markdown = self.outputs_of_creating_expression_matrices / "QC_summary.md"
        self.QC_summary_of_CSVs = self.outputs_of_creating_expression_matrices / "QC_summary.csv"
        self.expression_matrix_with_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "expression_matrix_with_SLIDs_approved_by_manifest.csv" # EM2
        self.expression_matrix_with_SLIDs_approved_by_relaxed_manifest = self.outputs_of_creating_expression_matrices / "expression_matrix_with_SLIDs_approved_by_relaxed_manifest.csv"
        self.data_frame_of_Ensembl_IDs_and_HGNC_symbols = self.outputs_of_creating_expression_matrices / "data_frame_of_Ensembl_IDs_and_HGNC_symbols.csv"
        self.expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.csv" # EM3
        self.expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_relaxed_manifest = self.outputs_of_creating_expression_matrices / "expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_relaxed_manifest.csv"
        self.filtered_expression_matrix_with_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "filtered_expression_matrix_with_SLIDs_approved_by_manifest.csv" # EM4
        self.filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.csv" # EM5
        self.logged_filtered_expression_matrix_with_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "logged_filtered_expression_matrix_with_SLIDs_approved_by_manifest.csv" # EM6
        self.logged_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "logged_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.csv" # EM7
        self.z_scored_filtered_expression_matrix_with_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "z_scored_filtered_expression_matrix_with_SLIDs_approved_by_manifest.csv" # EM8
        self.z_scored_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest = self.outputs_of_creating_expression_matrices / "z_scored_filtered_expression_matrix_with_HGNC_symbols_and_SLIDs_approved_by_manifest.csv" # EM9

        # ORIEN_analysis/ORIEN_analysis/run_xCell_analysis.py
        # dependencies
        self.outputs_of_running_xCell_analysis = self.output / "running_xCell_analysis"
        # -----
        # <no files>
        # outputs
        self.enrichment_data_frame_per_xCell = self.outputs_of_running_xCell_analysis / "enrichment_data_frame_per_xCell.csv"
        self.enrichment_data_frame_per_xCell2_and_Pan_Cancer = self.outputs_of_running_xCell_analysis / "enrichment_data_frame_per_xCell2_and_Pan_Cancer.csv"
        self.focused_enrichment_data_frame_per_xCell = self.outputs_of_running_xCell_analysis / "focused_enrichment_data_frame_per_xCell.csv"

        # ORIEN_analysis/ORIEN_analysis/fit_linear_models.py
        # dependencies
        self.outputs_of_fitting_LMs = self.output / "fitting_linear_models"
        # -----
        # self.enrichment_data_frame_per_xCell, which is defined above
        # self.enrichment_data_frame_per_xCell2_and_Pan_Cancer, which is defined above
        self.diagnosis_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_Diagnosis_V4.csv"
        self.medications_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_Medications_V4.csv"
        self.patient_data = self.normalized_clinical_data / "24PRJ217UVA_20241112_PatientMaster_V4.csv"
        # outputs
        self.figures_of_box_plots_of_residuals_by_batch = self.outputs_of_fitting_LMs / "figures_of_box_plots_of_residuals_by_batch"
        # -----
        self.results_of_fitting_LMs_per_xCell = self.outputs_of_fitting_LMs / "results_of_fitting_LMs_per_xCell.csv"
        self.results_of_fitting_LMs_per_xCell2_and_Pan_Cancer = self.outputs_of_fitting_LMs / "results_of_fitting_LMs_per_xCell2_and_Pan_Cancer.csv"
        self.significant_results_of_fitting_LMs_per_xCell = self.outputs_of_fitting_LMs / "significant_results_of_fitting_LMs_per_xCell.csv"
        self.significant_results_of_fitting_LMs_per_xCell2_and_Pan_Cancer = self.outputs_of_fitting_LMs / "significant_results_of_fitting_LMs_per_xCell2_and_Pan_Cancer.csv"
        self.data_frame_for_reviewing_ICB_status = self.outputs_of_fitting_LMs / "data_frame_for_reviewing_ICB_status.csv"


        # ORIEN_analysis/ORIEN_analysis/compare_enrichment_scores.py
        # dependencies
        self.outputs_of_comparing_enrichment_scores = self.output / "comparing_enrichment_scores"
        # -----
        # self.enrichment_data_frame_per_xCell, which is defined above
        # self.enrichment_data_frame_per_xCell2_and_Pan_Cancer, which is defined above
        # outputs
        self.plots_for_comparing_enrichment_scores = self.outputs_of_comparing_enrichment_scores / "plots_for_comparing_enrichment_scores"
        # -----
        self.comparisons_for_females_and_males_and_xCell = self.outputs_of_comparing_enrichment_scores / "comparisons_for_female_and_males_and_xCell.csv"
        self.comparisons_for_female_and_male_naive_samples_and_xCell = self.outputs_of_comparing_enrichment_scores / "comparisons_for_female_and_male_naive_samples_and_xCell.csv"
        self.comparisons_for_female_and_male_experienced_samples_and_xCell = self.outputs_of_comparing_enrichment_scores / "comparisons_for_female_and_male_experienced_samples_and_xCell.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell = self.outputs_of_comparing_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell = self.outputs_of_comparing_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_and_xCell = self.outputs_of_comparing_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_and_xCell.csv"
        self.comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer = self.outputs_of_comparing_enrichment_scores / "comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer.csv"
        self.comparisons_for_female_and_male_naive_samples_and_xCell2_and_Pan_Cancer = self.outputs_of_comparing_enrichment_scores / "comparisons_for_female_and_male_naive_samples_and_xCell2_and_Pan_Cancer.csv"
        self.comparisons_for_female_and_male_experienced_samples_and_xCell2_and_Pan_Cancer = self.outputs_of_comparing_enrichment_scores / "comparisons_for_female_and_male_experienced_samples_and_xCell2_and_Pan_Cancer.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer = self.outputs_of_comparing_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_Pan_Cancer = self.outputs_of_comparing_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_Pan_Cancer.csv"
        self.comparisons_for_ICB_naive_and_experienced_samples_and_xCell2_and_Pan_Cancer = self.outputs_of_comparing_enrichment_scores / "comparisons_for_ICB_naive_and_experienced_samples_and_xCell2_and_Pan_Cancer.csv"


        # ORIEN_analysis/ORIEN_analysis/complete_Aim_1_2.py
        # dependencies
        self.outputs_of_completing_Aim_1_2 = self.output / "completing_Aim_1_2"
        # -----
        self.dictionary_of_names_of_sets_of_genes_and_lists_of_genes = self.root / "dictionary_of_names_of_sets_of_genes_and_lists_of_genes.json"
        # outputs
        self.data_frame_of_genes_and_statistics_re_sex_for_all_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_genes_and_statistics_re_sex_for_all_samples.csv"
        self.data_frame_of_genes_and_statistics_re_sex_for_naive_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_genes_and_statistics_re_sex_for_naive_samples.csv"
        self.data_frame_of_genes_and_statistics_re_sex_for_experienced_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_genes_and_statistics_re_sex_for_experienced_samples.csv"
        self.data_frame_of_genes_and_statistics_re_ICB_status = self.outputs_of_completing_Aim_1_2 / "data_frame_of_genes_and_statistics_re_ICB_status.csv"
        self.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_all_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_all_samples.csv"
        self.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_naive_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_naive_samples.csv"
        self.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_experienced_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_sex_for_experienced_samples.csv"
        self.data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_ICB_status = self.outputs_of_completing_Aim_1_2 / "data_frame_of_names_of_sets_of_genes_statistics_and_lists_of_genes_re_ICB_status.csv"
        self.plot_of_FDR_vs_Normalized_Enrichment_Score_re_sex_for_all_samples = self.outputs_of_completing_Aim_1_2 / "plot_of_FDR_vs_Normalized_Enrichment_Score_re_sex_for_all_samples.png"
        self.plot_of_FDR_vs_Normalized_Enrichment_Score_re_sex_for_naive_samples = self.outputs_of_completing_Aim_1_2 / "plot_of_FDR_vs_Normalized_Enrichment_Score_re_sex_for_naive_samples.png"
        self.plot_of_FDR_vs_Normalized_Enrichment_Score_re_sex_for_experienced_samples = self.outputs_of_completing_Aim_1_2 / "plot_of_FDR_vs_Normalized_Enrichment_Score_re_sex_for_experienced_samples.png"
        self.plot_of_FDR_vs_Normalized_Enrichment_Score_re_ICB_status = self.outputs_of_completing_Aim_1_2 / "plot_of_FDR_vs_Normalized_Enrichment_Score_re_ICB_status.png"
        self.data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes_for_all_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes_for_all_samples.csv"
        self.data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes_for_naive_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes_for_naive_samples.csv"
        self.data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes_for_experienced_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_sample_IDs_and_module_scores_for_6_sets_of_genes_for_experienced_samples.csv"
        self.data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences_for_all_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences_for_all_samples.csv"
        self.data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences_for_naive_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences_for_naive_samples.csv"
        self.data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences_for_experienced_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_sample_IDs_CD8_B_and_G_module_scores_and_differences_for_experienced_samples.csv"
        self.data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_all_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_all_samples.csv"
        self.data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_naive_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_naive_samples.csv"
        self.data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_experienced_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_categories_of_module_scores_and_statistics_re_sex_for_experienced_samples.csv"
        self.data_frame_of_categories_of_module_scores_and_statistics_re_ICB_status = self.outputs_of_completing_Aim_1_2 / "data_frame_of_categories_of_module_scores_and_statistics_re_ICB_status.csv"
        self.plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_all_samples = self.outputs_of_completing_Aim_1_2 / "plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_all_samples.png"
        self.plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_naive_samples = self.outputs_of_completing_Aim_1_2 / "plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_naive_samples.png"
        self.plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_experienced_samples = self.outputs_of_completing_Aim_1_2 / "plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_experienced_samples.png"
        self.plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_ICB_status = self.outputs_of_completing_Aim_1_2 / "plot_of_difference_between_CD8_G_module_score_and_CD8_B_module_score_vs_ICB_status.png"
        self.plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_all_samples = self.outputs_of_completing_Aim_1_2 / "plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_all_samples.png"
        self.plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_naive_samples = self.outputs_of_completing_Aim_1_2 / "plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_naive_samples.png"
        self.plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_experienced_samples = self.outputs_of_completing_Aim_1_2 / "plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_sex_for_experienced_samples.png"
        self.plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_ICB_status = self.outputs_of_completing_Aim_1_2 / "plot_of_log_of_ratio_of_CD8_G_module_score_and_CD8_B_module_score_vs_ICB_status.png"
        self.data_frame_of_genes_log_FCs_p_values_and_FDRs_for_naive_and_experienced_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_genes_log_FCs_p_values_and_FDRs_for_naive_and_experienced_samples.csv"
        self.data_frame_of_genes_log_FCs_p_values_and_FDRs_for_female_and_male_samples = self.outputs_of_completing_Aim_1_2 / "data_frame_of_genes_log_FCs_p_values_and_FDRs_for_female_and_male_samples.csv"
        self.volcano_plot_for_female_and_male_samples = self.outputs_of_completing_Aim_1_2 / "volcano_plot_for_female_and_male_samples.png"
        self.volcano_plot_for_naive_and_experienced_samples = self.outputs_of_completing_Aim_1_2 / "volcano_plot_for_naive_and_experienced_samples.png"
        self.heatmap_of_normalized_expressions_for_top_differentially_expressed_genes_for_female_and_male_samples = self.outputs_of_completing_Aim_1_2 / "heatmap_of_normalized_expressions_for_top_differentially_expressed_genes_for_female_and_male_samples.png"
        self.heatmap_of_normalized_expressions_for_top_differentially_expressed_genes_for_naive_and_experienced_samples = self.outputs_of_completing_Aim_1_2 / "heatmap_of_normalized_expressions_for_top_differentially_expressed_genes_for_naive_and_experienced_samples.png"
        self.summary_for_all_samples = self.outputs_of_completing_Aim_1_2 / "summary_for_all_samples.csv"
        self.summary_for_naive_samples = self.outputs_of_completing_Aim_1_2 / "summary_for_naive_samples.csv"
        self.summary_for_experienced_samples = self.outputs_of_completing_Aim_1_2 / "summary_for_experienced_samples.csv"

        # ORIEN_analysis/ORIEN_analysis/complete_Aim_2_1.py
        # dependencies
        self.outputs_of_completing_Aim_2_1 = self.output / "completing_Aim_2_1"
        # -----
        # <no files>
        # outputs
        # <no files>

        # ORIEN_analysis/ORIEN_analysis/build_tables.py
        # dependencies
        self.outputs_of_building_tables = self.output / "building_tables"
        # -----
        # <no files>
        # outputs
        # <no files>

        # ORIEN_analysis/ORIEN_analysis/add_paths_to_data_frame_of_IDs_of_patients_specimens_and_WES.py
        # dependencies
        # <no folders>
        # -----
        # <no files>
        # outputs
        # <no files>

        # ORIEN_analysis/ORIEN_analysis/create_data_frame_of_patient_IDs_specimen_IDs_and_indicators_of_mutations.py
        # dependencies
        # <no folders>
        # -----
        # <no files>
        # outputs
        # <no files>

        # ORIEN_analysis/ORIEN_analysis/create_summary_of_driver_mutations.py
        # dependencies
        # <no folders>
        # -----
        # <no files>
        # outputs
        # <no files>


    def ensure_dependencies_for_pairing_clinical_data_and_stages_of_tumors_exist(self):
        for path in [
            self.outputs_of_pairing_clinical_data_and_stages_of_tumors
        ]:
            os.makedirs(path, exist_ok = True)
        for path in [
            self.clinical_molecular_linkage_data,
            self.diagnosis_data,
            self.metastatic_disease_data
        ]:
            assert os.path.exists(path), f"The dependency of creating expression matrices `{path}` does not exist."


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
            self.output_of_pairing_clinical_data_and_stages_of_tumors
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
            assert os.path.exists(path), f"The dependency of running xCell analysis `{path}` does not exist."


    def ensure_dependencies_for_fitting_LMs_exist(self):
        for path in [
            self.outputs_of_fitting_LMs,
            self.figures_of_box_plots_of_residuals_by_batch
        ]:
            os.makedirs(path, exist_ok = True)
        for path in [
            self.enrichment_data_frame_per_xCell,
            self.enrichment_data_frame_per_xCell2_and_Pan_Cancer
        ]:
            assert os.path.exists(path), f"The dependency of fitting LMs `{path}` does not exist."


    def ensure_dependencies_for_comparing_enrichment_scores_exist(self):
        for path in [
            self.outputs_of_comparing_enrichment_scores,
            self.plots_for_comparing_enrichment_scores
        ]:
            os.makedirs(path, exist_ok = True)
        for path in [
            self.enrichment_data_frame_per_xCell,
            self.enrichment_data_frame_per_xCell2_and_Pan_Cancer
        ]:
            assert os.path.exists(path), f"The dependency of comparing enrichment scores `{path}` does not exist."


    def ensure_dependencies_for_completing_Aim_1_2_exist(self):
        for path in [

        ]:
            os.makedirs(path, exist_ok = True)
        for path in [

        ]:
            assert os.path.exists(path), f"The dependency of comparing enrichment scores `{path}` does not exist."


    def ensure_dependencies_for_completing_Aim_2_1_exist(self):
        for path in [

        ]:
            os.makedirs(path, exist_ok = True)
        for path in [

        ]:
            assert os.path.exists(path), f"The dependency of comparing enrichment scores `{path}` does not exist."


    def ensure_dependencies_for_building_tables_exist(self):
        for path in [
            self.outputs_of_building_tables
        ]:
            os.makedirs(path, exist_ok = True)
        for path in [

        ]:
            assert os.path.exists(path), f"The dependency of comparing enrichment scores `{path}` does not exist."


    def ensure_dependencies_for_adding_paths_to_data_frame_of_IDs_of_patients_specimens_and_WES_tables_exist(self):
        for path in [

        ]:
            os.makedirs(path, exist_ok = True)
        for path in [

        ]:
            assert os.path.exists(path), f"The dependency of comparing enrichment scores `{path}` does not exist."


    def ensure_dependencies_for_creating_data_frame_of_patient_IDs_specimen_IDs_and_indicators_of_mutations_exist(self):
        for path in [

        ]:
            os.makedirs(path, exist_ok = True)
        for path in [

        ]:
            assert os.path.exists(path), f"The dependency of comparing enrichment scores `{path}` does not exist."


    def ensure_dependencies_for_creating_summary_of_driver_mutations_exist(self):
        for path in [

        ]:
            os.makedirs(path, exist_ok = True)
        for path in [

        ]:
            assert os.path.exists(path), f"The dependency of comparing enrichment scores `{path}` does not exist."


paths = Paths()