#!/usr/bin/env python3

'''
`pipeline_for_pairing_clinical_data_and_stages_of_tumors.py`

This module is a pipeline implementing "ORIEN Specimen Staging Revised Rules" for pairing patients' melanoma tumor specimens with the appropriate primary diagnosis site, patient grouping, AJCC stage, and rule.

2. From "ORIEN Specimen Staging Revised Rules":
2. Definitions
    - Melanoma diagnosis (Diagnosis file): HistologyCode = {list of melanoma codes previously sent}
    - Tumor sequenced (Molecular Linkage file): Tumor/Germline variable = Tumor

A melanoma diagnosis is a row in `24PRJ217UVA_20241112_Diagnosis_V4.csv` with a histology code of the form 87<digit><digit>/<digit>.

A sequenced tumor is a row in `24PRJ217UVA_20241112_MetastaticDisease_V4.csv` with a value of "Tumor" in column with label "Tumor/Germline".

From "ORIEN Specimen Staging Revised Rules":
1. Using data  from:
    - Molecular Linkage file (main file)
    - Diagnosis file (for primary site and stage info)
    - Metastatic Disease (to help assign stage)

Usage:
python pipeline_for_pairing_clinical_data_and_stages_of_tumors.py --clinmol ../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv --diagnosis ../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_Diagnosis_V4.csv --metadisease ../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_MetastaticDisease_V4.csv --therapy ../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_Medications_V4.csv --out output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv > output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.txt 2>&1
'''

from __future__ import annotations
import argparse
import logging
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple


#######################
# CONSTANTS AND REGEXES
#######################

'''
From "ORIEN Specimen Staging Revised Rules":
4. AssignedPrimarySite (SAME DEFINITIONS AS BEFORE)
    - IF PrimaryDiagnosisSite contains "skin" OR "EAR" OR "eyelid" OR "vulva", THEN AssignedPrimarySite = cutaneous
        - Vulvar melanoma is included here given that all appear to have been staged as cutaneous (not mucosal) melanoma.
    - If PrimaryDiagnosisSite contains "choroid" OR "ciliary body" OR "conjunctiva", then AssignedPrimarySite = ocular.
    - If PrimaryDiagnosisSite contains "sinus" OR "gum" OR "nasal" OR "urethra" then AssignedPrimarySite = mucosal
        - The list only includes the primary mucosal sites present in this data set and not all possible mucosal sites for melanoma in general.
    - If PrimaryDiagnosisSite contains 'unknown', then AssignedPrimarySite = unknown
'''

SITE_KEYWORDS = {
    "cutaneous": "skin|ear|eyelid|vulva",
    "ocular": "choroid|ciliary body|conjunctiva",
    "mucosal": "sinus|gum|nasal|urethra",
    "unknown": "unknown"
}

AGE_FUDGE = 0.005 # years, or approximately 1.8 days.
_ROMAN_RE = re.compile(r"\b(?:Stage\s*)?([IV]{1,3})(?:[ABCD])?\b", re.I)
_MELANOMA_RE = re.compile(r"^87\d\d/\d$")

ICB_PATTERN = re.compile(r"immune checkpoint|pembrolizumab|nivolumab|ipilimumab|atezolizumab|durvalumab|avelumab|cemiplimab|relatlimab", re.I)

CUTANEOUS_RE = re.compile(SITE_KEYWORDS["cutaneous"], re.I)
NODE_RE = re.compile(r"lymph node", re.I)
PAROTID_RE = re.compile(r"parotid", re.I)
_SITE_LOCAL_RE = re.compile(r"skin|ear|eyelid|vulva|head|scalp|soft tissue[s]?|breast|lymph node|parotid", re.I)


#################
# SMALL UTILITIES
#################

def _float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

    
def _first_roman(stage_txt: str) -> Optional[str]:
    '''
    Return the core Roman numeral (I, II, III, or IV) in a stage string or None.
    '''
    m = _ROMAN_RE.search(str(stage_txt))
    return None if m is None else m.group(1).upper()
    
    
#############
# CSV LOADING
#############

def _strip_cols(df: pd.DataFrame) -> None:
    df.columns = df.columns.str.strip()


def load_inputs(
    clinmol_csv: Path,
    dx_csv: Path,
    meta_csv: Path,
    therapy_csv: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    logging.info("Loading Clinical‑Molecular Linkage...")
    cm = pd.read_csv(clinmol_csv, dtype = str)
    _strip_cols(cm)
    cm = cm[cm["Tumor/Germline"].str.lower() == "tumor"].copy()

    logging.info("Loading Diagnosis...")
    dx = pd.read_csv(dx_csv, dtype = str)
    _strip_cols(dx)
    dx = dx[dx["HistologyCode"].str.match(_MELANOMA_RE, na=False)].copy()

    logging.info("Loading Metastatic Disease...")
    md = pd.read_csv(meta_csv, dtype = str)
    _strip_cols(md)

    th = None
    if therapy_csv is not None:
        logging.info("Loading Therapy...")
        th = pd.read_csv(therapy_csv, dtype = str)
        _strip_cols(th)

    return cm, dx, md, th


#######################
# COUNTING AND GROUPING
#######################

def add_counts(cm: pd.DataFrame, dx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    From "ORIEN Specimen Staging Revised Rules":
    3. New variables to create:
        3.a. MelanomaDiagnosisCount: count for number of melanoma clinical diagnoses for a patient
            - Create using the number of unique [AgeAtDiagnosis and PrimaryDiagnosisSite] combinations for each patient
            - This approach may work best since a few patients have multiple melanomas diagnosed at the same age, so it [age at diagnosis] cannot be used on its own.
                - Example with multiple diagnoses at same age with same stage: ILE2DL0KMW
        3.b. SequencedTumorCount: count for number of sequenced tumor samples for a patient
            - Create using the number of unique [DeidSpecimenID and AvatarKey] combinations] for each patient
            - This approach may work best since a few patients have multiple sequenced tumors at the same age (or stage, etc), so these variables cannot be used on their own.
                - Example with multiple tumors sequenced at the same age/stage: 59OP5X1AZL
    '''
    
    '''
    Create a new column `_age` in data frame of diagnoses that is column `AgeAtDiagnosis` with NA values equal to "NA".
    Drop duplicate rows by patient ID, `_age`, and primary diagnosis site.
    Determine how many distinct melanoma diagnoses each patient has.
    '''
    diag_unique = dx.assign(_age = dx["AgeAtDiagnosis"].fillna("NA")).drop_duplicates(subset = ["AvatarKey", "_age", "PrimaryDiagnosisSite"])
    diag_counts  = diag_unique.groupby("AvatarKey").size().rename("MelanomaDiagnosisCount")
    
    '''
    Drop duplicate rows in data frame of tumors by patient and specimen IDs.
    Determine how many distinct tumors each patient has.
    '''
    tumor_counts = cm.drop_duplicates(["ORIENAvatarKey", "DeidSpecimenID"]).groupby("ORIENAvatarKey").size().rename("SequencedTumorCount")
    
    # Logic that uses both diagnosis count and tumor count is executed while iterating over rows in cm.
    cm = cm.join(tumor_counts, on = "ORIENAvatarKey")
    cm = cm.join(diag_counts, on = "ORIENAvatarKey")
    
    return cm, dx


def _patient_group(row) -> str:
    '''
    From "ORIEN Specimen Staging Revised Rules":
    5. AssignedGroup (SAME DEFINITIONS AS BEFORE)
        - Group A: MelanomaDiagnosisCount = 1 AND SequencedTumorCount = 1 -> n = 327
        - Group B: MelanomaDiagnosisCount = 1 AND SequencedTumorCount > 1 -> n = 19
        - Group C: MelanomaDiagnosisCount > 1 AND SequencedTumorCount = 1 -> n = 30
        - Group D: MelanomaDiagnosisCount > 1 AND SequencedTumorCount > 1 -> n=3
    '''
    mdx = row.get("MelanomaDiagnosisCount", 0)
    mts = row.get("SequencedTumorCount", 0)
    if mdx == 1 and mts == 1:
        return "A"
    if mdx == 1 and mts > 1:
        return "B"
    if mdx > 1 and mts == 1:
        return "C"
    if mdx > 1 and mts > 1:
        return "D"
    else:
        raise Exception("Group could not be assigned.")

    
def _assign_icb_status(
    spec_row: pd.Series,
    therapy_patient: Optional[pd.DataFrame]
) -> str:
    age_spec = _float(spec_row.get("Age At Specimen Collection"))
    if therapy_patient is None:
        return "Unknown"
    icb_rows = therapy_patient[
        therapy_patient["Medication"].str.contains(ICB_PATTERN, na = False)
    ].copy()
    if icb_rows.empty:
        return "No-ICB"
    icb_rows["_age"] = icb_rows["AgeAtMedStart"].apply(_float)
    icb_rows = icb_rows[pd.notna(icb_rows["_age"])]
    if icb_rows.empty or age_spec is None:
        return "Unknown"
    return "Post-ICB" if age_spec >= icb_rows["_age"].min() else "Pre-ICB"


def _filter_meta_before(meta_patient: pd.DataFrame, age_spec: float | None, allow_unknown_age: bool = True) -> pd.DataFrame:
    if meta_patient.empty:
        return meta_patient.iloc[0:0]
    m = meta_patient.copy()
    m["_age"] = m["AgeAtMetastaticSite"].apply(_float)
    if age_spec is None:
        return m if allow_unknown_age else m[pd.notna(m["_age"])]
    keep = m[pd.notna(m["_age"]) & (m["_age"] <= age_spec + AGE_FUDGE)]
    if allow_unknown_age:
        keep = pd.concat([keep, m[pd.isna(m["_age"])]], ignore_index=True).drop_duplicates()
    return keep


#########################
# PRIMARY‑SITE ASSIGNMENT
#########################

def assign_primary_site(primary_diagnosis_site: str) -> str:
    txt = str(primary_diagnosis_site).lower()
    for site, pat in SITE_KEYWORDS.items():
        if re.search(pat, txt): # i.e., if the text is in the pattern
            return site
    raise ValueError(f"Unrecognized primary diagnosis site: '{primary_diagnosis_site}'")


##############################################################################################
# STAGING RULES
# The first rule that matches wins.
# A specimen is not evaluated against any rule later than a rule that applies to the specimen.
##############################################################################################

def stage_by_ordered_rules(
    spec: pd.Series,
    dx: pd.Series,
    meta_patient: pd.DataFrame
) -> Tuple[str, str]:
    '''
    Return (EKN Assigned Stage, NEW RULE) following the 10 ordered rules of "ORIEN Specimen Staging Revised Rules".
    '''

    # Short aliases ----------------------------------------------------------
    age_diag_txt = str(dx.get("AgeAtDiagnosis", "")).strip()
    age_diag_f = _float(age_diag_txt)
    path_stg = str(dx.get("PathGroupStage", "")).strip()
    clin_stg = str(dx.get("ClinGroupStage", "")).strip()
    site_coll = str(spec.get("SpecimenSiteOfCollection", "")).lower()
    age_spec = _float(spec.get("Age At Specimen Collection"))

    def _meta_prior_distant() -> bool:
        if meta_patient.empty:
            return False
        m = _filter_meta_before(meta_patient, age_spec, allow_unknown_age = False)
        if m.empty:
            return False
        distant = m["MetastaticDiseaseInd"].str.contains("Yes - Distant", na = False)
        site_ok = m["MetsDzPrimaryDiagnosisSite"].str.contains(
            r"skin|ear|eyelid|vulva|eye|choroid|ciliary body|conjunctiva|sinus|gum|nasal|urethra",
            na = False,
            case = False
        )
        return (distant & site_ok).any()

    def _prior_skin_regional() -> bool:
        """
        Any *regional or NOS* cutaneous mets dated **on/-before** the specimen?
        """
        if meta_patient.empty:
            return False
        m = _filter_meta_before(meta_patient, age_spec, allow_unknown_age = False)
        if m.empty:
            return False
        site_ok = m["MetsDzPrimaryDiagnosisSite"].str.contains(
            r"skin|ear|eyelid|vulva", case=False, na=False
        )
        reg_ok = m["MetastaticDiseaseInd"].str.contains(
            r"Yes - Regional|Yes - NOS", case=False, na=False
        )
        return (site_ok & reg_ok).any()

    def _skin_distant_unknown() -> bool:
        if meta_patient.empty:
            return False
        site_ok = meta_patient["MetsDzPrimaryDiagnosisSite"].str.contains(CUTANEOUS_RE, na = False)
        distant = meta_patient["MetastaticDiseaseInd"].str.contains(r"Yes - Distant|Yes - NOS", na = False)
        unk_age = meta_patient["AgeAtMetastaticSite"].str.strip().str.lower().eq("age unknown/not recorded")
        return (site_ok & distant & unk_age).any()
    
    # RULE 1 - AGE
    if age_diag_txt.lower() == "age 90 or older":
        return _first_roman(path_stg or clin_stg) or "Unknown", "AGE"

    # RULE 2 - PATHIV
    if "IV" in path_stg.upper():
        return "IV", "PATHIV"

    # RULE 3 - CLINIV
    if "IV" in clin_stg.upper():
        return "IV", "CLINIV"

    # RULE 4 - METSITE
    if not _SITE_LOCAL_RE.search(site_coll):
        return "IV", "METSITE"

    # RULE 5 - PRIORDISTANT
    if _meta_prior_distant():
        return "IV", "PRIORDISTANT"

    # RULE 6 - NOMETS
    if meta_patient.empty or meta_patient["MetastaticDiseaseInd"].str.lower().eq("no").all():
        return _first_roman(path_stg or clin_stg) or "Unknown", "NOMETS"

    # RULE 7 - NODE
    if NODE_RE.search(site_coll) or PAROTID_RE.search(site_coll):
        return "III", "NODE"

    # RULE 8 - SKINLESS90D
    within90 = _within_90_days_after(
        age_spec + AGE_FUDGE if age_spec is not None else None,
        age_diag_f
    )

    skin_site = re.search(
        r"skin|ear|eyelid|vulva|head|soft tissue[s]?|breast",
        site_coll
    )

    if within90 and skin_site and not _prior_skin_regional():
        if path_stg in [
            "Unknown/Not Reported",
            "No TNM applicable for this site/histology combination",
            "Unknown/Not Applicable",
        ]:
            return _first_roman(clin_stg) or "Unknown", "SKINLESS90D"
        return _first_roman(path_stg) or "Unknown", "SKINLESS90D"

    # RULE 9 - SKINREG
    if re.search(r"skin|ear|eyelid|vulva|head|soft tissue[s]?|breast|lymph node", site_coll) and not _skin_distant_unknown():
        return "III", "SKINREG"

    # RULE 10 - SKINUNK
    if _skin_distant_unknown():
        return "IV", "SKINUNK"

    # Fallback (should never be reached according to ORIEN Specimen Staging Revised Rules)
    return "Unknown", "UNMATCHED"


def _within_90_days_after(age_spec: float | None, age_diag: float | None) -> bool:
    if age_spec is None or age_diag is None:
        return False
    diff = age_spec - age_diag
    return 0 <= abs(diff) <= 90 / 365.25


def _select_specimen_B(patient_cm: pd.DataFrame) -> pd.Series:
    '''
    From "ORIEN Specimen Staging Revised Rules":
    6. AssignedGroup
    Group B = 1 melanoma diagnosis and >1 tumor sequenced -> n=19
    CHANGE FROM PRIOR: Do not exclude any, can still use those with WES only for TMB analysis.
    A few clarifications on the rules also in red text below.
        - If RNAseq is available for just one tumor, select the tumor with RNAseq data (even if no WES)
        - If RNAseq data is available for > 1 tumors OR if only WES is available for all tumors:
    '''

    has_rna = patient_cm["RNASeq"].notna()
    n_rna = has_rna.sum()

    if n_rna == 1:
        return patient_cm.loc[has_rna].iloc[0]

    if n_rna > 1 or n_rna == 0:
        site = patient_cm["SpecimenSiteOfCollection"].str.lower().fillna("")
        skin  = site.str.contains("skin")
        soft  = site.str.contains("soft tissue")
        lnode = site.str.contains("lymph node")

        def earliest(df):
            return df.sort_values("Age At Specimen Collection", ascending=True).iloc[0]

        if skin.any() and not (lnode.any() or soft.any()):
            return earliest(patient_cm[skin])

        if not (skin.any() or soft.any()) and lnode.any():
            return earliest(patient_cm[lnode])

        if (skin | soft).any() and lnode.any():
            candidates = patient_cm[skin | soft | lnode]
            earliest_age = candidates["Age At Specimen Collection"].min()
            same_age = candidates[candidates["Age At Specimen Collection"] == earliest_age]
            if same_age.shape[0] == 1:
                return same_age.iloc[0]
            
            if "Primary/Met" in same_age.columns and (same_age["Primary/Met"].str.lower() == "primary").any():
                return same_age[same_age["Primary/Met"].str.lower() == "primary"].iloc[0]
            
            if lnode.any():
                return earliest(patient_cm[lnode])
            
            return earliest(candidates)

        return earliest(patient_cm)

    raise RuntimeError("Unexpected branch fall-through")

        
def _hist_clean(txt: str) -> str:
    return re.sub(r"[^A-Za-z]", "", str(txt)).lower()


def _select_diagnosis_C(dx_patient: pd.DataFrame, spec_row: pd.Series, meta_patient: pd.DataFrame) -> pd.Series:
    if dx_patient.empty:
        raise ValueError("No diagnosis rows supplied to _select_diagnosis_C")
    
    dxp = dx_patient.copy()
    age_spec = _float(spec_row["Age At Specimen Collection"])

    primary_met = spec_row["Primary/Met"].strip().lower()

    if primary_met == "primary":
        age_diag = dxp["AgeAtDiagnosis"].apply(_float)
        prox = age_diag.apply(lambda x: _within_90_days_after(age_spec, x))
        if prox.sum() == 1:
            return dxp[prox].iloc[0]

        if prox.any():
            site_match = dxp["PrimaryDiagnosisSite"].str.lower() == spec_row["SpecimenSiteOfCollection"].lower()
            match_rows = dxp[prox & site_match].copy()
            if not match_rows.empty:
                match_rows["_age"] = match_rows["AgeAtDiagnosis"].apply(_float)
                return match_rows.sort_values("_age", na_position = "last").iloc[0]

        histologies_differ = dxp["Histology"].apply(_hist_clean).nunique() > 1
        if (prox.sum() == 0) or (age_diag.isna().any() and histologies_differ):
            hist_spec = _hist_clean(spec_row["Histology/Behavior"])
            hist_match = dxp["Histology"].apply(_hist_clean) == hist_spec
            if hist_match.sum() == 1:
                return dxp[hist_match].iloc[0]

    else:
        site_coll = spec_row["SpecimenSiteOfCollection"].lower()
        if not re.search(r"soft tissue|lymph node", site_coll):
            return dxp.sort_values("AgeAtDiagnosis").iloc[0]

        prox = dxp["AgeAtDiagnosis"].apply(_float).apply(lambda x: _within_90_days_after(age_spec, x))
        
        if "soft tissue" in site_coll and prox.sum() == 1:
            return dxp[prox].iloc[0]

        if "lymph node" in site_coll:
            if prox.any():
                pos_node = ~dxp["PathNStage"].str.contains(
                    r"\bN0\b|Nx|unknown/not applicable|no tnm",
                    case = False,
                    na = False
                )
                pos_node &= prox
                if pos_node.sum() == 1:
                    return dxp[pos_node].iloc[0]
            if prox.sum() == 0 and not meta_patient.empty:
                match_site = meta_patient["MetsDzPrimaryDiagnosisSite"].str.lower().unique()
                site_match = dxp["PrimaryDiagnosisSite"].str.lower().isin(match_site)
                if site_match.sum() == 1:
                    return dxp[site_match].iloc[0]

    dxp["_age"] = dxp["AgeAtDiagnosis"].astype(float)
    return dxp.sort_values("_age").iloc[0]


def _select_specimen_D(patient_cm: pd.DataFrame) -> pd.Series:
    cm = patient_cm.copy()
    cm_rna = cm[cm["RNASeq"].notna()]
    if len(cm_rna) == 1:
        return cm_rna.iloc[0]
    if len(cm_rna) > 1:
        cm_rna["_age"] = cm_rna["Age At Specimen Collection"].apply(_float)
        return cm_rna.sort_values("_age", na_position = "last").iloc[0]
    cm["_age"] = cm["Age At Specimen Collection"].apply(_float)
    return cm.sort_values("_age", na_position = "last").iloc[0]


def _select_diagnosis_D(dx_patient: pd.DataFrame, spec_row: pd.Series) -> pd.Series:
    site = spec_row["SpecimenSiteOfCollection"].lower()
    dxp = dx_patient.copy()
    dxp["_age"] = dxp["AgeAtDiagnosis"].apply(_float)
    if re.search(r"lymph node", site):
        return dxp.sort_values("_age", ascending = False).iloc[0]
    if re.search(r"skin|soft tissue", site):
        return dxp.sort_values("_age").iloc[0]
    return dxp.sort_values("_age").iloc[0]


def run_pipeline(
    clinmol: Path,
    diagnosis: Path,
    metadisease: Path,
    therapy: Path | None = None
) -> pd.DataFrame:
    cm, dx, md, th = load_inputs(clinmol, diagnosis, metadisease, therapy)
    cm, dx = add_counts(cm, dx)

    cm["Group"] = cm.apply(_patient_group, axis = 1)
    
    patient_groups = cm.groupby("ORIENAvatarKey").first()
    count_A = (patient_groups['Group'] == 'A').sum()
    count_B = (patient_groups['Group'] == 'B').sum()
    count_C = (patient_groups['Group'] == 'C').sum()
    count_D = (patient_groups['Group'] == 'D').sum()
    assert count_A == 327, f"Number of patients in A was {count_A} and should be 327."
    assert count_B == 19, f"Number of patients in B was {count_B} and should be 19."
    assert count_C == 30, f"Number of patients in C was {count_C} and should be 30."
    assert count_D == 3, f"Number of patients in D was {count_D} and should be 3."
    
    cm[["MelanomaDiagnosisCount", "SequencedTumorCount"]] = cm[["MelanomaDiagnosisCount", "SequencedTumorCount"]].fillna(0)

    output_rows: List[Dict[str, str]] = []

    for avatar, specs in cm.groupby("ORIENAvatarKey", sort = False):
        dx_patient = dx[dx["AvatarKey"] == avatar]
        meta_patient = md[md["AvatarKey"] == avatar]
        therapy_patient = th[th["AvatarKey"] == avatar] if th is not None else None
        group = specs["Group"].iloc[0]

        if group == "A":
            spec_row, diag_row = specs.iloc[0], dx_patient.iloc[0]
        elif group == "B":
            spec_row = _select_specimen_B(specs)
            diag_row = dx_patient.iloc[0]
        elif group == "C":
            spec_row = specs.iloc[0]
            diag_row = _select_diagnosis_C(dx_patient, spec_row, meta_patient)
        elif group == "D":
            spec_row = _select_specimen_D(specs)
            diag_row = _select_diagnosis_D(dx_patient, spec_row)
        else:
            raise RuntimeError(f"Unknown group: {group}")

        primary_site = assign_primary_site(diag_row["PrimaryDiagnosisSite"])
        stage, rule = stage_by_ordered_rules(spec_row, diag_row, meta_patient)
        
        '''
        From "ORIEN Specimen Staging Revised Rules":
        3.c. AssignedPrimarySite: {cutaneous, ocular, mucosal, unknown}
            - Based on the parameters outlined below; primary site variable to be used for the analysis
        3.d. AssignedStage [EKN Assigned Stage in "ORIEN_Tumor_Staging_Key.csv"]: {I, II, III, IV}
            - Based on the parameters outlined below; stage variable to be used for the analysis
        3.e. AssignedGroup [Group in "ORIEN_Tumor_Staging_Key.csv"]: {A, B, C, D} (NEW - just to keep track of these better)
        '''
        
        output_rows.append(
            {
                "AvatarKey": avatar,
                "ORIENSpecimenID": spec_row["DeidSpecimenID"],
                "AssignedPrimarySite": primary_site,
                "Group": group,
                "EKN Assigned Stage": stage,
                "NEW RULE": rule
            }
        )

    return pd.DataFrame(output_rows).sort_values(by = ["AvatarKey", "ORIENSpecimenID"]).reset_index(drop = True)


def main():
    parser = argparse.ArgumentParser(description = "Pair clinical data and stages of tumors.")
    parser.add_argument("--clinmol", required = True, type = Path)
    parser.add_argument("--diagnosis", required = True, type = Path)
    parser.add_argument("--metadisease", required = True, type = Path)
    parser.add_argument("--therapy", type = Path)
    parser.add_argument("--out", required = True, type = Path)
    args = parser.parse_args()

    logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")

    df = run_pipeline(
        clinmol = args.clinmol,
        diagnosis = args.diagnosis,
        metadisease = args.metadisease,
        therapy = args.therapy
    )

    logging.info("Writing %d rows to %s", len(df), args.out)
    df.to_csv(args.out, index = False)


if __name__ == "__main__":
    main()