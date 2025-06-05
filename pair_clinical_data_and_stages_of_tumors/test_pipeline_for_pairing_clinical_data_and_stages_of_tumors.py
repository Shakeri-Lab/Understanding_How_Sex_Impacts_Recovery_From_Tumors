'''
Usage
pytest -q test_pipeline_for_pairing_clinical_data_and_stages_of_tumors.py > output_of_test_pipeline_for_pairing_clinical_data_and_stages_of_tumors.txt 2>&1
'''

from pathlib import Path

import pandas as pd
import pytest

# The module we just refactored
from pipeline_for_pairing_clinical_data_and_stages_of_tumors import run_pipeline

###############################################################################
# CONFIG – adjust to your file layout once per project
###############################################################################
DATA_ROOT = Path("/sfs/gpfs/tardis/project/orien/data/aws/24PRJ217UVA_IORIG/Clinical_Data/24PRJ217UVA_NormalizedFiles")

CSV_CM  = DATA_ROOT / "24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv"
CSV_DX  = DATA_ROOT / "24PRJ217UVA_20241112_Diagnosis_V4.csv"
CSV_MD  = DATA_ROOT / "24PRJ217UVA_20241112_MetastaticDisease_V4.csv"
CSV_TH  = DATA_ROOT / "24PRJ217UVA_20241112_Medications_V4.csv"


###############################################################################
# PIPELINE FIXTURE – run once per test session
###############################################################################
@pytest.fixture(scope="session")
def paired_df() -> pd.DataFrame:
    """Run the full pairing/staging pipeline once and cache the result."""
    return run_pipeline(
        clinmol     = CSV_CM,
        diagnosis   = CSV_DX,
        metadisease = CSV_MD,
        therapy     = CSV_TH,
        strict      = True
    )


###############################################################################
# EXAMPLE-DRIVEN EXPECTATION TABLE
###############################################################################
# Only include examples where the rules document states a *definite* outcome.
# (Many ‘selection’ examples in Group-C/D are about tie-breaks and do not give
# a final stage; they are omitted here.)

EXAMPLES = [
    # ── Cutaneous rules ──────────────────────────────────────────────────────
    ("2AP9EDU231" , "III", "CUT4"),
    ("300QLLH1JW" , "III", "CUT4"),
    ("0VZ1LP53XJ" , "III", "CUT5"),
    ("9TUP5F9XV7" , "IV" , "CUT6"),
    ("EQ8CK21XQT" , "IV" , "CUT7"),
    ("GONDO7FLPP" , "III", "CUT8"),
    ("7EE85P5EGN" , "IV" , "CUT9"),
    ("9CJCPWLERP" , "IV" , "CUT9"),
    ("ABE47J13C5" , "IV" , "CUT9"),
    ("CG6JRI0XIX" , "IV" , "CUT9"),
    ("0BUVFK9AQ1" , "III", "CUT10"),
    ("6DL054517A" , "III", "CUT10"),
    ("A594OFU98I" , "III", "CUT10"),
    ("2T0D2KKDXB" , "IV" , "CUT11"),
    # ── Ocular rules ─────────────────────────────────────────────────────────
    ("2DC14Y2AQS" , "IV" , "OC3"),
    ("8OR7RX5NO5" , "III", "OC4"),
    # ── Mucosal rules ────────────────────────────────────────────────────────
    ("Z7CEUA8SAJ" , "IV" , "MU4"),
]


###############################################################################
# PARAMETERISED TEST
###############################################################################
@pytest.mark.parametrize("avatar, expected_stage, expected_rule", EXAMPLES)
def test_example_case(paired_df: pd.DataFrame,
                      avatar: str,
                      expected_stage: str,
                      expected_rule: str) -> None:
    """Every worked example from the spec should hit the documented rule & stage."""
    row = paired_df.loc[paired_df["AvatarKey"] == avatar]
    assert not row.empty, f"{avatar} did not appear in pipeline output"

    got_stage = row["AssignedStage"].iloc[0]
    got_rule  = row["StageRuleHit"].iloc[0]

    assert got_stage == expected_stage, \
        f"{avatar}: stage {got_stage!r} ≠ expected {expected_stage!r}"
    assert got_rule  == expected_rule,  \
        f"{avatar}: rule  {got_rule!r}  ≠ expected {expected_rule!r}"