#!/usr/bin/env python3
'''
Test that ./src/immune_analysis/data_loading.py produces the exact same outputs as sibling reference files.
'''


import pandas as pd
from pandas.testing import assert_frame_equal


def test_that_outputs_are_equal() -> bool:
    data_frame_of_melanoma_clinical_data = pd.read_csv("output/data_loading/melanoma_clinical_data.csv")
    data_frame_of_melanoma_clinical_data_for_comparison = pd.read_csv("output/data_loading/melanoma_clinical_data_for_comparison.csv")
    assert_frame_equal(data_frame_of_melanoma_clinical_data, data_frame_of_melanoma_clinical_data_for_comparison)

    melanoma_expression_matrix = pd.read_csv("output/data_loading/melanoma_expression_matrix.csv")
    melanoma_expression_matrix_for_comparison = pd.read_csv("output/data_loading/melanoma_expression_matrix_for_comparison.csv")
    assert_frame_equal(melanoma_expression_matrix, melanoma_expression_matrix_for_comparison)
