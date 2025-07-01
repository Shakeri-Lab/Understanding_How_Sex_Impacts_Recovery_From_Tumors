#!/usr/bin/env python3
'''
Test that ./src/immune_analysis/data_loading.py produces the exact same outputs as sibling reference files.
'''

from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest


PATH_TO_OUTPUTS_OF_DATA_LOADING = Path("output/data_loading")


@pytest.fixture(scope = "session")
def pairs() -> list[tuple[str, str]]:
    pairs = [
        ("melanoma_clinical_data.csv", "melanoma_clinical_data_for_comparison.csv"),
        ("melanoma_expression_matrix.csv", "melanoma_expression_matrix_for_comparison.csv")
    ]
    return pairs


def load(path: Path) -> pd.DataFrame:
    '''
    Return a data frame based on a CSV file.
    '''
    df = pd.read_csv(path)
    return df


def test_that_outputs_are_equal(pairs) -> bool:
    for pair in pairs:
        f1, f2 = (PATH_TO_OUTPUTS_OF_DATA_LOADING / p for p in pair)
        df1, df2 = map(load, (f1, f2))
        assert_frame_equal(df1, df2)
