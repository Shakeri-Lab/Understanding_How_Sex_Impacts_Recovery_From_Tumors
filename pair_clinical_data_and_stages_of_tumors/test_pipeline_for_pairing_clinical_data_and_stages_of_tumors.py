'''
Usage
pytest -q test_pipeline_for_pairing_clinical_data_and_stages_of_tumors.py > output_of_test_pipeline_for_pairing_clinical_data_and_stages_of_tumors.txt 2>&1

Verify that the output of the pipeline for pairing clinical data and stages of tumors exactly matches ORIEN Tumor Staging Key.
'''

import pandas as pd
from pathlib import Path
from pipeline_for_pairing_clinical_data_and_stages_of_tumors import run_pipeline
import pytest
import sys


DATA_ROOT = Path("/sfs/gpfs/tardis/project/orien/data/aws/24PRJ217UVA_IORIG/Clinical_Data/24PRJ217UVA_NormalizedFiles")
CSV_CM  = DATA_ROOT / "24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv"
CSV_DX  = DATA_ROOT / "24PRJ217UVA_20241112_Diagnosis_V4.csv"
CSV_MD  = DATA_ROOT / "24PRJ217UVA_20241112_MetastaticDisease_V4.csv"
CSV_TH  = DATA_ROOT / "24PRJ217UVA_20241112_Medications_V4.csv"
KEY_CSV = "ORIEN_Tumor_Staging_Key.csv"

@pytest.fixture(scope="session")
def output_of_pipeline() -> pd.DataFrame:
    return run_pipeline(
        clinmol = CSV_CM,
        diagnosis = CSV_DX,
        metadisease = CSV_MD,
        therapy = CSV_TH,
        strict = True
    )

def test_that_output_of_pipeline_equals_key(output_of_pipeline: pd.DataFrame) -> None:
    orien_tumor_staging_key = pd.read_csv(KEY_CSV).sort_values(by = ["AvatarKey", "ORIENSpecimenID"]).reset_index(drop = True)
    if output_of_pipeline.shape != orien_tumor_staging_key.shape:
        print(f"Shapes of output of pipeline {output_of_pipeline.shape} and ORIEN Tumor Staging Key {orien_tumor_staging_key.shape} are mismatched.")
        sys.exit(1)
    if list(output_of_pipeline.columns) != list(orien_tumor_staging_key.columns):
        print(
            "Output of pipeline and ORIEN Tumor Staging key have different columns.\n" +
            "Output of pipeline has columns " + ', '.join([name_of_column for name_of_column in output_of_pipeline.columns]) + ".\n" +
            "ORIEN tumor staging key has columns " + ', '.join([name_of_column for name_of_column in orien_tumor_staging_key.columns]) + "."
        )
        sys.exit(1)

    mask_of_equality = output_of_pipeline.values == orien_tumor_staging_key.values
    if mask_of_equality.all():
        print("All corresponding cells in output of pipeline and ORIEN Tumor Staging Key match.")
        return

    tuple_of_arrays_of_row_and_colummn_indices = (~mask_of_equality).nonzero()
    for index_of_row, index_of_column in zip(*tuple_of_arrays_of_row_and_colummn_indices):
        name_of_column = orien_tumor_staging_key.columns[index_of_column]
        value_of_orien_tumor_staging_key = orien_tumor_staging_key.iat[index_of_row, index_of_column]
        value_of_output_of_pipeline = output_of_pipeline.iat[index_of_row, index_of_column]
        print(f"Row {index_of_row} | Column '{name_of_column}': {value_of_output_of_pipeline!r}  !=  {value_of_orien_tumor_staging_key!r}")

    print(f"{len(tuple_of_arrays_of_row_and_colummn_indices[0])} cell(s) in output of pipeline differ from corresponding cells in ORIEN Tumor Staging Key match.")
    sys.exit(1)