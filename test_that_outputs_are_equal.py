#!/usr/bin/env python3
'''
Test that various scripts produce the exact same outputs as sibling reference files.
'''


from pandas.testing import assert_frame_equal
from PIL import Image, ImageChops
import pandas as pd


def assert_that_are_equal_images(path_1: str, path_2: str) -> None:
    with Image.open(path_1) as image_1, Image.open(path_2) as image_2:
        assert image_1.size == image_2.size, "Image sizes differ."
        assert image_1.mode == image_2.mode, "Image color modes differ."
        difference = ImageChops.difference(image_1, image_2)
        assert difference.getbbox() is None, "Images differ."


def test_that_outputs_of_EDA_are_equal() -> bool:
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


def test_that_outputs_of_data_loading_are_equal() -> bool:
    data_frame_of_melanoma_clinical_data = pd.read_csv("output/data_loading/melanoma_clinical_data.csv")
    data_frame_of_melanoma_clinical_data_for_comparison = pd.read_csv("output/data_loading/melanoma_clinical_data_for_comparison.csv")
    assert_frame_equal(data_frame_of_melanoma_clinical_data, data_frame_of_melanoma_clinical_data_for_comparison)

    melanoma_expression_matrix = pd.read_csv("output/data_loading/melanoma_expression_matrix.csv")
    melanoma_expression_matrix_for_comparison = pd.read_csv("output/data_loading/melanoma_expression_matrix_for_comparison.csv")
    assert_frame_equal(melanoma_expression_matrix, melanoma_expression_matrix_for_comparison)
