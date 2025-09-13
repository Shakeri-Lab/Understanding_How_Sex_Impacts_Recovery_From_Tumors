'''
Usage
pytest -q test_pipeline_for_pairing_clinical_data_and_stages_of_tumors.py > output_of_test_pipeline_for_pairing_clinical_data_and_stages_of_tumors.txt 2>&1

Verify that the output of the pipeline for pairing clinical data and stages of tumors exactly matches ORIEN Tumor Staging Key.
'''

from pathlib import Path
from pandas.testing import assert_frame_equal
from ORIEN_analysis.config import paths
import pandas as pd
import pytest
from ORIEN_analysis.pair_clinical_data_and_stages_of_tumors import run_pipeline


PATH_TO_CLINICAL_DATA = Path("../Clinical_Data/24PRJ217UVA_NormalizedFiles")
PATH_TO_CLINICAL_MOLECULAR_LINKAGE_DATA = PATH_TO_CLINICAL_DATA / "24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv"
PATH_TO_DIAGNOSIS_DATA = PATH_TO_CLINICAL_DATA / "24PRJ217UVA_20241112_Diagnosis_V4.csv"
PATH_TO_METASTATIC_DISEASE_DATA = PATH_TO_CLINICAL_DATA / "24PRJ217UVA_20241112_MetastaticDisease_V4.csv"
PATH_TO_KEY = "ORIEN_analysis/ORIEN_Tumor_Staging_Key.csv"

@pytest.fixture(scope = "session")
def output_of_pipeline() -> pd.DataFrame:
    return run_pipeline(
        path_to_clinical_molecular_linkage_data = PATH_TO_CLINICAL_MOLECULAR_LINKAGE_DATA,
        path_to_diagnosis_data = PATH_TO_DIAGNOSIS_DATA,
        path_to_metastatic_disease_data = PATH_TO_METASTATIC_DISEASE_DATA
    )

@pytest.fixture(scope = "session")
def orien_tumor_staging_key() -> pd.DataFrame:
    return pd.read_csv(PATH_TO_KEY)

def test_that_output_of_pipeline_equals_key(output_of_pipeline: pd.DataFrame, orien_tumor_staging_key: pd.DataFrame) -> None:
    assert_frame_equal(output_of_pipeline, orien_tumor_staging_key)