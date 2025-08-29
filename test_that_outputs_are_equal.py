#!/usr/bin/env python3
'''
Test that various scripts produce the exact same outputs as sibling reference files.

Usage:
./miniconda3/envs/ici_sex/bin/pytest test_that_outputs_are_equal.py
'''

from pathlib import Path
from pandas.testing import assert_frame_equal
from PIL import Image, ImageChops
import pandas as pd
import pytest
import subprocess


def assert_that_are_equal_images(path_1: str, path_2: str) -> None:
    with Image.open(path_1) as image_1, Image.open(path_2) as image_2:
        assert image_1.size == image_2.size, "Image sizes differ."
        assert image_1.mode == image_2.mode, "Image color modes differ."
        difference = ImageChops.difference(image_1, image_2)
        assert difference.getbbox() is None, "Images differ."

        
def remove_outputs_except_references(directory: Path) -> None:
    if not directory.exists():
        return
    for path in directory.rglob("*"):
        if path.is_file() and not path.stem.endswith("_for_comparison"):
            path.unlink(missing_ok = True)
        
        
'''
Test script `src/immune_analysis/utils.py`, which has no outputs.

Test that outputs of the following scripts are equal to references.
`src/data_processing/eda.py`,
`src/immune_analysis/data_loading.py`,
`src/immune_analysis/microenv.py`,
`src/immune_analysis/linear_mixed_models.py`, and
TODO: `src/immune_analysis/immune_analysis.py`
'''
        

# src/data_processing/utils.py doesn't have any outputs.
def test_utils():
    # ./miniconda3/envs/ici_sex/bin/python src/data_processing/utils.py ../Manifest_and_QC_Files/24PRJ217UVA_20250130_RNASeq_QCMetrics.csv
    pass


def test_that_outputs_of_EDA_are_equal():
    # ./miniconda3/envs/ici_sex/bin/python -m src.data_processing.eda
    
    data_frame_of_melanoma_patient_and_sequencing_data = pd.read_csv("output/eda/melanoma_patients_with_sequencing.csv")
    data_frame_of_melanoma_patient_and_sequencing_data_for_comparison = pd.read_csv("output/eda/melanoma_patients_with_sequencing_for_comparison.csv")
    assert_frame_equal(data_frame_of_melanoma_patient_and_sequencing_data, data_frame_of_melanoma_patient_and_sequencing_data)
    
    map_from_sample_ID_to_patient_ID = pd.read_csv("output/eda/sample_to_patient_map.csv")
    map_from_sample_ID_to_patient_ID_for_comparison = pd.read_csv("output/eda/sample_to_patient_map_for_comparison.csv")
    assert_frame_equal(map_from_sample_ID_to_patient_ID, map_from_sample_ID_to_patient_ID_for_comparison)
    
    # bar chart of numbers of patients by sex
    assert_that_are_equal_images("output/eda/plots/sex_distribution.png", "output/eda/plots/sex_distribution_for_comparison.png")
    
    map_of_indicators_that_patients_have_ICB_therapy_to_counts_of_patients = pd.read_csv("output/eda/reports/icb_distribution.csv")
    map_of_indicators_that_patients_have_ICB_therapy_to_counts_of_patients_for_comparison = pd.read_csv("output/eda/reports/icb_distribution_for_comparison.csv")
    assert_frame_equal(map_of_indicators_that_patients_have_ICB_therapy_to_counts_of_patients, map_of_indicators_that_patients_have_ICB_therapy_to_counts_of_patients_for_comparison)
    
    data_frame_of_statistics_summarizing_data_frame_of_melanoma_patient_and_sequencing_data = pd.read_csv("output/eda/reports/summary_statistics.csv")
    data_frame_of_statistics_summarizing_data_frame_of_melanoma_patient_and_sequencing_data_for_comparison = pd.read_csv("output/eda/reports/summary_statistics_for_comparison.csv")
    assert_frame_equal(data_frame_of_statistics_summarizing_data_frame_of_melanoma_patient_and_sequencing_data, data_frame_of_statistics_summarizing_data_frame_of_melanoma_patient_and_sequencing_data_for_comparison)


def test_that_outputs_of_data_loading_are_equal():
    # ./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.data_loading
    
    series_of_Ensembl_IDs_and_HGNC_symbols = pd.read_csv("output/data_loading/series_of_Ensembl_IDs_and_HGNC_symbols.csv")
    series_of_Ensembl_IDs_and_HGNC_symbols_for_comparison = pd.read_csv("output/data_loading/series_of_Ensembl_IDs_and_HGNC_symbols_for_comparison.csv")
    assert_frame_equal(series_of_Ensembl_IDs_and_HGNC_symbols, series_of_Ensembl_IDs_and_HGNC_symbols_for_comparison)
    
    data_frame_of_melanoma_clinical_data = pd.read_csv("output/data_loading/melanoma_clinical_data.csv")
    data_frame_of_melanoma_clinical_data_for_comparison = pd.read_csv("output/data_loading/melanoma_clinical_data_for_comparison.csv")
    assert_frame_equal(data_frame_of_melanoma_clinical_data, data_frame_of_melanoma_clinical_data_for_comparison)

    melanoma_expression_matrix = pd.read_csv("output/data_loading/melanoma_expression_matrix.csv")
    melanoma_expression_matrix_for_comparison = pd.read_csv("output/data_loading/melanoma_expression_matrix_for_comparison.csv")
    assert_frame_equal(melanoma_expression_matrix, melanoma_expression_matrix_for_comparison)


def test_that_outputs_of_microenv_are_equal():
    # ./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.microenv
    
    data_frame_melanoma_sample_immune_clinical = pd.read_csv("output/microenv/melanoma_sample_immune_clinical.csv")
    data_frame_melanoma_sample_immune_clinical_for_comparison = pd.read_csv("output/microenv/melanoma_sample_immune_clinical_for_comparison.csv")
    assert_frame_equal(data_frame_melanoma_sample_immune_clinical, data_frame_melanoma_sample_immune_clinical_for_comparison)
    
    metastatic_status_summary = pd.read_csv("output/microenv/metastatic_status_summary.csv")
    metastatic_status_summary_for_comparison = pd.read_csv("output/microenv/metastatic_status_summary_for_comparison.csv")
    assert_frame_equal(metastatic_status_summary, metastatic_status_summary_for_comparison)
    
    procedure_type_summary = pd.read_csv("output/microenv/procedure_type_summary.csv")
    procedure_type_summary_for_comparison = pd.read_csv("output/microenv/procedure_type_summary_for_comparison.csv")
    assert_frame_equal(procedure_type_summary, procedure_type_summary_for_comparison)
    
    specimen_site_summary = pd.read_csv("output/microenv/specimen_site_summary.csv")
    specimen_site_summary_for_comparison = pd.read_csv("output/microenv/specimen_site_summary_for_comparison.csv")
    assert_frame_equal(specimen_site_summary, specimen_site_summary_for_comparison)
    
    xcell_scores_focused_panel = pd.read_csv("output/microenv/xcell_scores_focused_panel.csv")
    xcell_scores_focused_panel_for_comparison = pd.read_csv("output/microenv/xcell_scores_focused_panel_for_comparison.csv")
    assert_frame_equal(xcell_scores_focused_panel, xcell_scores_focused_panel_for_comparison)
    
    data_frame_of_xcell_scores = pd.read_csv("output/microenv/xcell_scores_raw_per_xCell.csv")
    data_frame_of_xcell_scores_for_comparison = pd.read_csv("output/microenv/xcell_scores_raw_per_xCell_for_comparison.csv")
    assert_frame_equal(data_frame_of_xcell_scores, data_frame_of_xcell_scores_for_comparison)

    data_frame_of_xcell_scores = pd.read_csv("output/microenv/xcell_scores_raw_per_xCell2_and_Pan_Cancer.csv")
    data_frame_of_xcell_scores_for_comparison = pd.read_csv("output/microenv/xcell_scores_raw_per_xCell2_and_Pan_Cancer_for_comparison.csv")
    assert_frame_equal(data_frame_of_xcell_scores, data_frame_of_xcell_scores_for_comparison)
    
    data_frame_of_xcell_scores = pd.read_csv("output/microenv/xcell_scores_raw_per_xCell2_and_TME_Compendium.csv")
    data_frame_of_xcell_scores_for_comparison = pd.read_csv("output/microenv/xcell_scores_raw_per_xCell2_and_TME_Compendium_for_comparison.csv")
    assert_frame_equal(data_frame_of_xcell_scores, data_frame_of_xcell_scores_for_comparison)

    
# TODO: Test that outputs of immune analysis are equal.


def test_that_outputs_of_linear_mixed_models_are_equal():
    # ./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.linear_mixed_models
    
    data_frame_of_mixed_model_results_per_xCell = pd.read_csv("output/linear_mixed_models/mixed_model_results_per_xCell.csv")
    data_frame_of_mixed_model_results_per_xCell_for_comparison = pd.read_csv("output/linear_mixed_models/mixed_model_results_per_xCell_for_comparison.csv")
    assert_frame_equal(data_frame_of_mixed_model_results_per_xCell, data_frame_of_mixed_model_results_per_xCell_for_comparison)
    
    data_frame_of_mixed_model_results_per_xCell_and_Pan_Cancer = pd.read_csv("output/linear_mixed_models/mixed_model_results_per_xCell2_and_Pan_Cancer.csv")
    data_frame_of_mixed_model_results_per_xCell_and_Pan_Cancer_for_comparison = pd.read_csv("output/linear_mixed_models/mixed_model_results_per_xCell2_and_Pan_Cancer_for_comparison.csv")
    assert_frame_equal(data_frame_of_mixed_model_results_per_xCell_and_Pan_Cancer, data_frame_of_mixed_model_results_per_xCell_and_Pan_Cancer_for_comparison)
    
    data_frame_of_mixed_model_results_per_xCell_and_TME_Compendium = pd.read_csv("output/linear_mixed_models/mixed_model_results_per_xCell2_and_TME_Compendium.csv")
    data_frame_of_mixed_model_results_per_xCell_and_TME_Compendium_for_comparison = pd.read_csv("output/linear_mixed_models/mixed_model_results_per_xCell2_and_TME_Compendium_for_comparison.csv")
    assert_frame_equal(data_frame_of_mixed_model_results_per_xCell_and_TME_Compendium, data_frame_of_mixed_model_results_per_xCell_and_TME_Compendium_for_comparison)
    
    data_frame_of_mixed_model_results_significant_per_xCell = pd.read_csv("output/linear_mixed_models/mixed_model_results_significant_per_xCell.csv")
    data_frame_of_mixed_model_results_significant_per_xCell_for_comparison = pd.read_csv("output/linear_mixed_models/mixed_model_results_significant_per_xCell_for_comparison.csv")
    assert_frame_equal(data_frame_of_mixed_model_results_significant_per_xCell, data_frame_of_mixed_model_results_significant_per_xCell_for_comparison)
    
    data_frame_of_mixed_model_results_significant_per_xCell_and_Pan_Cancer = pd.read_csv("output/linear_mixed_models/mixed_model_results_significant_per_xCell2_and_Pan_Cancer.csv")
    data_frame_of_mixed_model_results_significant_per_xCell_and_Pan_Cancer_for_comparison = pd.read_csv("output/linear_mixed_models/mixed_model_results_significant_per_xCell2_and_Pan_Cancer_for_comparison.csv")
    assert_frame_equal(data_frame_of_mixed_model_results_significant_per_xCell_and_Pan_Cancer, data_frame_of_mixed_model_results_significant_per_xCell_and_Pan_Cancer_for_comparison)
    
    data_frame_of_mixed_model_results_significant_per_xCell_and_TME_Compendium = pd.read_csv("output/linear_mixed_models/mixed_model_results_significant_per_xCell2_and_TME_Compendium.csv")
    data_frame_of_mixed_model_results_significant_per_xCell_and_TME_Compendium_for_comparison = pd.read_csv("output/linear_mixed_models/mixed_model_results_significant_per_xCell2_and_TME_Compendium_for_comparison.csv")
    assert_frame_equal(data_frame_of_mixed_model_results_significant_per_xCell_and_TME_Compendium, data_frame_of_mixed_model_results_significant_per_xCell_and_TME_Compendium_for_comparison)


def test_that_outputs_of_compare_enrichment_scores_are_equal():
    # ./miniconda3/envs/ici_sex/bin/python -m src.immune_analysis.compare_enrichment_scores
    
    comparisons_for_females_and_males_and_xCell = pd.read_csv("output/compare_enrichment_scores/comparisons_for_females_and_males_and_xCell.csv")
    comparisons_for_females_and_males_and_xCell_for_comparison = pd.read_csv("output/compare_enrichment_scores/comparisons_for_females_and_males_and_xCell_for_comparison.csv")
    assert_frame_equal(comparisons_for_females_and_males_and_xCell, comparisons_for_females_and_males_and_xCell_for_comparison)
    
    comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer = pd.read_csv("output/compare_enrichment_scores/comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer.csv")
    comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer_for_comparison = pd.read_csv("output/compare_enrichment_scores/comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer_for_comparison.csv")
    assert_frame_equal(comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer, comparisons_for_females_and_males_and_xCell2_and_Pan_Cancer_for_comparison)
    
    comparisons_for_females_and_males_and_xCell2_and_TME_Compendium = pd.read_csv("output/compare_enrichment_scores/comparisons_for_females_and_males_and_xCell2_and_TME_Compendium.csv")
    comparisons_for_females_and_males_and_xCell2_and_TME_Compendium_for_comparison = pd.read_csv("output/compare_enrichment_scores/comparisons_for_females_and_males_and_xCell2_and_TME_Compendium_for_comparison.csv")
    assert_frame_equal(comparisons_for_females_and_males_and_xCell2_and_TME_Compendium, comparisons_for_females_and_males_and_xCell2_and_TME_Compendium_for_comparison)
    
    comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell.csv")
    comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell_for_comparison = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell_for_comparison.csv")
    assert_frame_equal(comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell, comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell_for_comparison)
    
    comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer.csv")
    comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer_for_comparison = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer_for_comparison.csv")
    assert_frame_equal(comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer, comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer_for_comparison)
    
    comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_TME_Compendium = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_TME_Compendium.csv")
    comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_TME_Compendium_for_comparison = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_TME_Compendium_for_comparison.csv")
    assert_frame_equal(comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_TME_Compendium, comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_TME_Compendium_for_comparison)
    
    comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell.csv")
    comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell_for_comparison = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell_for_comparison.csv")
    assert_frame_equal(comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell, comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell_for_comparison)
    
    comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_Pan_Cancer = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer.csv")
    comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_Pan_Cancer_for_comparison = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_Pan_Cancer_for_comparison.csv")
    assert_frame_equal(comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_Pan_Cancer, comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_Pan_Cancer_for_comparison)
    
    comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_TME_Compendium = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_TME_Compendium.csv")
    comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_TME_Compendium_for_comparison = pd.read_csv("output/compare_enrichment_scores/comparisons_for_ICB_naive_and_experienced_samples_of_females_and_xCell2_and_TME_Compendium_for_comparison.csv")
    assert_frame_equal(comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_TME_Compendium, comparisons_for_ICB_naive_and_experienced_samples_of_males_and_xCell2_and_TME_Compendium_for_comparison)