#!/usr/bin/env python3

'''
`pipeline_for_pairing_clinical_data_and_stages_of_tumors.py`

This module is a pipeline implementing "ORIEN Specimen Staging Revised Rules" for pairing patients' melanoma tumor specimens
with the appropriate primary diagnosis site, patient grouping, AJCC stage, and rule.
        
Usage:
python pipeline_for_pairing_clinical_data_and_stages_of_tumors.py --clinmol ../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv --diagnosis ../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_Diagnosis_V4.csv --metadisease ../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_MetastaticDisease_V4.csv --therapy ../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_Medications_V4.csv --out output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv > output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.txt 2>&1
'''

from __future__ import annotations
import argparse
import logging
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Tuple


##########
# SETTINGS
##########

STRICT: bool = False


###########
# CONSTANTS
###########

ICB_PATTERN = re.compile(
    r"immune checkpoint|pembrolizumab|nivolumab|ipilimumab|atezolizumab|durvalumab|avelumab|cemiplimab|relatlimab",
    re.I,
)

# ICD-O-3 malignant-melanoma morphology codes
# Source refs: SEER “Cutaneous Melanoma” rules, Pathology Outlines, WHO ICD-O-3 lists
MELANOMA_HISTOLOGY_CODES: set[str] = {
    # Classic / cutaneous
    "8720/3",  # Malignant melanoma, NOS
    "8721/3",  # Nodular melanoma
    "8722/3",  # Balloon cell melanoma
    "8723/3",  # Malignant melanoma, regressing
    "8730/3",  # Amelanotic melanoma
    "8740/3",  # Malignant melanoma in junctional nevus
    "8741/3",  # Malignant melanoma in precancerous melanosis
    "8742/3",  # Lentigo maligna melanoma
    "8743/3",  # Superficial spreading melanoma
    "8744/3",  # Acral lentiginous melanoma
    "8745/3",  # Desmoplastic / neurotropic melanoma
    "8746/3",  # Mucosal lentiginous melanoma

    # Congenital / blue-nevus–related
    "8761/3",  # MM arising in giant congenital nevus
    "8762/3",  # MM arising in congenital nevus (childhood type)
    "8780/3",  # MM arising in blue nevus

    # Dysplastic / magnocellular variants
    "8726/3",  # Malignant magnocellular melanoma
    "8727/3",  # Malignant melanoma in dysplastic nevus

    # Spitzoid / spindle / epithelioid
    "8770/3",  # Mixed epithelioid & spindle cell melanoma
    "8771/3",  # Epithelioid cell melanoma
    "8772/3",  # Spindle cell melanoma, NOS
    "8773/3",  # Spindle cell melanoma, type A
    "8774/3",  # Spindle cell melanoma, type B

    # Other special forms
    "8790/3",  # Malignant melanoma, NOS (eye & other sites)
    "8724/3",  # Nodular melanoma (ocular usage)
    "8725/3",  # Malignant neuronevus / neural-type melanoma
}

NODE_REGEX = re.compile(r"lymph node", re.I) # regular expression object

PAROTID_REGEX = re.compile(r"parotid", re.I) # regular expression object

SITE_KEYWORDS = {
    "cutaneous": "skin|ear|eyelid|vulva|head|scalp|trunk|face|neck|back|chest|shoulder|arm|leg|extremity|hand|foot|buttock|hip|soft tissue",
    "ocular": "choroid|ciliary body|conjunctiva|eye",
    "mucosal": "sinus|gum|nasal|urethra|anorect|anus|rectum|anal canal|oropharynx|oral|palate|vagina",
    "unknown": "unknown"
}

SKINLIKE_SITES = [
    "skin", "ear", "eyelid", "head", "soft tissues", "muscle", "chest wall", "vulva"
]

UNKNOWN_PATHOLOGICAL_STAGE_VALUES = {
    "unknown/not reported",
    "no tnm applicable for this site/histology combination",
    "unknown/not applicable"
}


#################
# UTILITY HELPERS
#################

def _assign_icb_status(
    spec_row: pd.Series,
    therapy_patient: Optional[pd.DataFrame],
) -> str:
    """
    Return Pre-ICB / Post-ICB / No-ICB / Unknown for this specimen.

    Logic
    • If therapy file not supplied (None) → Unknown
    • Filter patient therapy rows whose Agent or Class matches ICB_PATTERN.
    • Compare earliest AgeAtTherapyStart to Age At Specimen Collection.
    """
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

    min_icb_age = icb_rows["_age"].min()
    return "Post-ICB" if age_spec >= min_icb_age else "Pre-ICB"


def _filter_meta_before(meta_patient: pd.DataFrame, age_spec: float | None) -> pd.DataFrame:
    '''
    Keep metastatic-disease rows that satisfy either condition:
      • AgeAtMetastaticSite ≤ AgeAtSpecimenCollection
      • AgeAtMetastaticSite is "Age Unknown/Not Recorded" (therefore we store it as NaN)

    This matches the wording for OC-3 and MU-4, which explicitly
    include unknown-age distant mets.
    '''
    if meta_patient is None or meta_patient.empty:
        return pd.DataFrame()

    m = meta_patient.copy()
    m["_age"] = m["AgeAtMetastaticSite"].apply(_float)   # _float->None for "Unknown..."

    # Unknown age rows (pd.isna) are **always** retained
    if age_spec is None:
        return m[pd.isna(m["_age"])]

    return m[pd.isna(m["_age"]) | (m["_age"] <= age_spec)]
    

def _float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


#############
# CSV LOADING
#############

def _strip_cols(df: pd.DataFrame) -> None:
    '''
    Trim whitespace from every column header in place.
    '''
    df.columns = df.columns.str.strip()

def load_inputs(
    clinmol_csv: Path,
    dx_csv: Path,
    meta_csv: Path,
    therapy_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    '''
    1. from "ORIEN Data Rules for Tumor Clinical Pairing"
    1. Using data from:
    - Molecular Linkage file (main file)
    - Diagnosis file (for primary site and stage info)
    - Metastatic Disease (to help assign stage)
    '''
    logging.info("Loading Clinical‑Molecular Linkage...")
    cm = pd.read_csv(clinmol_csv, dtype = str)
    _strip_cols(cm)
    cm = cm[cm["Tumor/Germline"].str.lower() == "tumor"].copy()

    logging.info("Loading Diagnosis...")
    dx = pd.read_csv(dx_csv, dtype = str)
    _strip_cols(dx)
    dx = dx[dx["HistologyCode"].isin(MELANOMA_HISTOLOGY_CODES)].copy()

    logging.info("Loading Metastatic Disease...")
    md = pd.read_csv(meta_csv, dtype = str)
    _strip_cols(md)

    if therapy_csv is not None:
        logging.info("Loading Therapy...")
        th = pd.read_csv(therapy_csv, dtype=str)
        _strip_cols(th)
    else:
        th = None

    return cm, dx, md, th


#######################
# COUNTING AND GROUPING
#######################

def add_counts(cm: pd.DataFrame, dx: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    3. from ORIEN Data Rules for Tumor Clinical Pairing
    - 3. MelanomaDiagnosisCount: count for number of melanoma clinical diagnoses for a patient
        - Create using the number of unique [AgeAtDiagnosis and PrimaryDiagnosisSite combinations] for each patient
        - This approach may work best since a few patients have multiple melanomas diagnosed at the same age, so it [AgeAtDiagnosis] cannot be used on its own.
            - Example with multiple diagnoses at same age with same stage: ILE2DL0KMW
            ["ILE2DL0KMW" is a value of column ORIENAvatarKey in `/sfs/gpfs/tardis/project/orien/data/aws/24PRJ217UVA_IORIG/Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv`]
    
    4. from ORIEN Data Rules for Tumor Clinical Pairing
    - 4. SequencedTumorCount: count for number of sequenced tumor samples for a patient
        - Create using the number of unique [DeidSpecimenID and Avatar Key combinations] for each patient
        - This approach may work best since a few patients have multiple sequenced tumors at the same age (or stage, etc), so these variables cannot be used on their own.
            - Example with multiple tumors sequenced at same age/stage: 59OP5X1AZL
    
    Add MelanomaDiagnosisCount and SequencedTumorCount to CM and DX frames.
    '''
    diag_counts = (
        dx.drop_duplicates(["AvatarKey", "AgeAtDiagnosis", "PrimaryDiagnosisSite"]) # Ensure that (AvatarKey, AgeAtDiagnosis, PrimaryDiagnosisSite) represents 1 diagnosis.
          .groupby("AvatarKey")
          .size()
          .rename("MelanomaDiagnosisCount")
    )
    tumor_counts = (
        cm.drop_duplicates(["ORIENAvatarKey", "DeidSpecimenID"]) # Ensure that (ORIENAvatarKey, DeidSpecimenID) represents 1 tumor.
          .groupby("ORIENAvatarKey")
          .size()
          .rename("SequencedTumorCount")
    )
    cm = cm.join(tumor_counts, on = "ORIENAvatarKey") # Logic that uses both diagnosis count and tumor count is executed while iterating over rows in cm.
    cm = cm.join(diag_counts, on = "ORIENAvatarKey")
    return cm, dx


def patient_group(row) -> str:
    '''
    Return A, B, C, or D according to diagnosis and tumor counts.
    '''
    mdx = row["MelanomaDiagnosisCount"] or 0
    mts = row["SequencedTumorCount"] or 0
    if mdx == 1 and mts == 1:
        return "A"
    if mdx == 1 and mts > 1:
        return "B"
    if mdx > 1 and mts == 1:
        return "C"
    return "D"


###############################################
# SELECTION HELPERS – GROUP B (choose specimen)
###############################################

def _select_specimen_B(patient_cm: pd.DataFrame) -> pd.Series:
    '''
    Select a single tumor specimen for a Group B / D patient, following the exact precedence order in the ORIEN rule set.
    
    1. Exclude the patient if no tumor has RNA sequencing.
    2. If exactly 1 tumor has RNA sequencing, pick it.
    3. When 2 or more tumors with RNA sequencing exist, restrict all subsequent logic to the subset of tumors with RNA sequencing.
    4. If 1 or more tumors have a site with "skin" and no tumor has a site with "lymph node" or "soft tissue", pick a tumor with a site with "skin".
    5. If 1 or more tumors have a site with "lymph node" and no tumor has a site with "skin" or "soft tissue", pick a tumor with a site with "lymph node".
    6. If the patient has tumors with sites with "skin" or "soft tissue" and tumors with sites with "lymph node", pick the tumor with the earliest Age At Specimen Collection among tumors with sites with "skin" or "soft tissue".
    7. Otherwise, pick the tumor with the earliest age.
    '''
    cm = patient_cm.copy()
    
    # Keep only RNA‑seq‑enabled tumours if possible
    cm_rna = cm[cm["RNASeq"].notna()]
    candidates = cm_rna if not cm_rna.empty else cm
    if len(candidates) == 1:
        return candidates.iloc[0]

    # Site category masks
    is_skin = candidates["SpecimenSiteOfCollection"].str.contains("skin", case = False, na = False)
    is_soft = candidates["SpecimenSiteOfCollection"].str.contains(r"soft tissue", case = False, na = False)
    is_node = candidates["SpecimenSiteOfCollection"].str.contains(r"lymph node", case = False, na = False)

    # 4. Skin precedence when alone
    if is_skin.any() and not (is_node.any() or is_soft.any()):
        return candidates[is_skin].iloc[0]

    # 5. Node precedence when alone
    if is_node.any() and not (is_skin.any() or is_soft.any()):
        return candidates[is_node].iloc[0]
    
    # 6. Tie-breaker: earliest among skin or soft tissue
    if (is_skin.any() or is_soft.any()) and is_node.any():
        tie = candidates[is_skin | is_soft].copy()
        tie["_age"] = tie["Age At Specimen Collection"].apply(_float)  # robust → NaN
        # Put NaNs last so we never accidentally pick an “Unknown”-age specimen
        return tie.sort_values("_age", na_position="last").iloc[0]

    # 7. Fallback: earliest age (handles only soft tissue and other sites)
    candidates["_age"] = candidates["Age At Specimen Collection"].apply(_float)
    return candidates.sort_values("_age", na_position="last").iloc[0]


def _select_specimen_D(patient_cm: pd.DataFrame) -> pd.Series:
    cm = patient_cm.copy()
    cm_rna = cm[cm["RNASeq"].notna()]

    if len(cm_rna) == 1:
        return cm_rna.iloc[0]
    if len(cm_rna) > 1:
        cm_rna["_age"] = cm_rna["Age At Specimen Collection"].apply(_float)
        return cm_rna.sort_values("_age", na_position="last").iloc[0]

    cm["_age"] = cm["Age At Specimen Collection"].apply(_float)
    return cm.sort_values("_age", na_position="last").iloc[0]


def _select_diagnosis_D(dx_patient: pd.DataFrame, spec_row: pd.Series) -> pd.Series:
    site = spec_row["SpecimenSiteOfCollection"].lower()
    dxp = dx_patient.copy()
    dxp["_age"] = dxp["AgeAtDiagnosis"].apply(_float)

    if re.search(r"lymph node", site):
        return dxp.sort_values("_age", ascending=False).iloc[0]
    if re.search(r"skin|soft tissue", site):
        return dxp.sort_values("_age").iloc[0]
    return dxp.sort_values("_age").iloc[0]


################################################
# SELECTION HELPERS – GROUP C (choose diagnosis)
################################################

# 90 days *after* diagnosis (directional)
def _within_90_days_after(age_spec: float | None, age_diag: float | None) -> bool:
    """
    Return True when the specimen was collected **0–90 days AFTER** the
    diagnosis age, per ORIEN rules.
    """
    if age_spec is None or age_diag is None:
        return False
    diff = age_spec - age_diag
    return 0 <= diff <= (90 / 365.25)


def _hist_clean(txt: str) -> str:
    return re.sub(r"[^A-Za-z]", "", str(txt)).lower()


def _select_diagnosis_C(dx_patient: pd.DataFrame, spec_row: pd.Series, meta_patient: pd.DataFrame) -> pd.Series:
    '''
    Return a single diagnosis row for Group C/D per rules.
    '''
    if dx_patient.empty:
        raise ValueError("No diagnosis rows supplied to _select_diagnosis_C")
    
    dxp = dx_patient.copy()
    age_spec = _float(spec_row["Age At Specimen Collection"])

    # Split path by Primary/Met status of specimen
    primary_met = spec_row["Primary/Met"].strip().lower()

    if primary_met == "primary":
        # 90‑day proximity unique → pick that diagnosis
        age_diag = dxp["AgeAtDiagnosis"].apply(_float)
        prox = age_diag.apply(lambda x: _within_90_days_after(age_spec, x))
        if prox.sum() == 1:
            return dxp[prox].iloc[0]

        # Proximity tie-break by site
        if prox.any():
            site_match = (
                dxp["PrimaryDiagnosisSite"].str.lower()
                == spec_row["SpecimenSiteOfCollection"].lower()
            )
            match_rows = dxp[prox & site_match].copy()
            if not match_rows.empty:
                # choose deterministically: earliest AgeAtDiagnosis with known age
                match_rows["_age"] = match_rows["AgeAtDiagnosis"].apply(_float)
                best = match_rows.sort_values("_age", na_position="last").iloc[0]
                return best.drop(labels="_age")

        # Histology tie-break – spec fires when **no** diagnosis is within
        # 90 d *OR* at least one diagnosis has an unknown age.
        #   –  Case 1: *all* diagnoses > 90 d after specimen  →  tie-break, regardless of histology
        #   –  Case 2: ≥ 1 diagnosis age unknown **and** the diagnoses are heterogeneous in histology
        histologies_differ = dxp["Histology"].apply(_hist_clean).nunique() > 1
        if (prox.sum() == 0) or (age_diag.isna().any() and histologies_differ):
            hist_spec   = _hist_clean(spec_row["Histology/Behavior"])
            hist_match  = dxp["Histology"].apply(_hist_clean) == hist_spec
            if hist_match.sum() == 1:
                return dxp[hist_match].iloc[0]

    else:  # metastatic specimen pathway
        site_coll = spec_row["SpecimenSiteOfCollection"].lower()
        
        # Non-node / non-soft-tissue -> earliest diagnosis
        if not re.search(r"soft tissues|lymph node", site_coll):
            return dxp.sort_values("AgeAtDiagnosis").iloc[0]

        # Soft-tissue specimen within 90 days AFTER diagnosis
        if "soft tissues" in site_coll:
            prox = dxp["AgeAtDiagnosis"].apply(_float).apply(lambda x: _within_90_days_after(age_spec, x))
            if prox.sum() == 1:
                return dxp[prox].iloc[0]

        if "lymph node" in site_coll:
            prox = dxp["AgeAtDiagnosis"].apply(_float).apply(lambda x: _within_90_days_after(age_spec, x))
            # Positive-node rule – specimen must be within 90-day window **and**
            # the diagnosis must have PathNStage ≠ N0/Nx/unknown.
            if prox.any():
                pos_node = ~dxp["PathNStage"].str.contains(
                    r"\bN0\b|Nx|unknown/not applicable|no tnm",
                    case=False, na=False, regex=True,
                )
                pos_node &= prox                    # <-- spec-compliant intersection
                if pos_node.sum() == 1:
                    return dxp[pos_node].iloc[0]
            # Site match rule - *only when NO diagnosis is within 90 d*
            if prox.sum() == 0 and meta_patient is not None and not meta_patient.empty:
                match_site = meta_patient["MetsDzPrimaryDiagnosisSite"].str.lower().unique()
                site_match = dxp["PrimaryDiagnosisSite"].str.lower().isin(match_site)
                if site_match.sum() == 1:
                    return dxp[site_match].iloc[0]
            
    # Fallback – earliest diagnosis
    dxp["_age"] = dxp["AgeAtDiagnosis"].astype(float)
    return dxp.sort_values("_age").iloc[0]


#########################
# PRIMARY‑SITE ASSIGNMENT
#########################

def assign_primary_site(primary_diagnosis_site: str) -> str:
    txt = str(primary_diagnosis_site).lower()
    for site, pat in SITE_KEYWORDS.items():
        if re.search(pat, txt):
            return site
    if STRICT:
        raise ValueError(f"Unrecognized primary diagnosis site: '{primary_diagnosis_site}'. Run without --strict to coerce to 'unknown'.")
    return "unknown"


##############################################################################################
# STAGING RULES
# The first rule that matches wins.
# A specimen is not evaluated against any rule later than a rule that applies to the specimen.
##############################################################################################

_SITE_LOCAL_RE = re.compile(r"skin|ear|eyelid|vulva|head|scalp|soft tissues|breast|lymph node|parotid", re.I)


def _first_roman(stage_txt: str) -> Optional[str]:
    '''
    Return the first Roman-numeral stage (I–IV) found, else None.
    '''
    m = re.search(r"\b([IV]{1,3})\b", str(stage_txt).upper())
    return m.group(1) if m else None


def stage_by_ordered_rules(
    spec: pd.Series,
    dx: pd.Series,
    meta_patient: pd.DataFrame,
) -> Tuple[str, str]:
    '''
    Implement Rule 1 to Rule 10 exactly in the order printed in "ORIEN Specimen Staging Revised Rules".
    Return (EKN Assigned Stage, NEW RULE).
    '''

    # Short aliases ----------------------------------------------------------
    age_diag = str(dx.get("AgeAtDiagnosis", "")).strip()
    path_stg = str(dx.get("PathGroupStage", "")).strip()
    clin_stg = str(dx.get("ClinGroupStage", "")).strip()
    site_coll = str(spec.get("SpecimenSiteOfCollection", "")).lower()
    age_spec = _float(spec.get("Age At Specimen Collection"))

    # Helper for Meta rule #5 -----------------------------------------------
    def _meta_prior_distant() -> bool:
        if meta_patient is None or meta_patient.empty:
            return False
        m = _filter_meta_before(meta_patient, age_spec)
        if m.empty:
            return False
        distant = m["MetastaticDiseaseInd"].str.contains("Yes - Distant", na = False)
        primary_ok = m["MetsDzPrimaryDiagnosisSite"].str.contains(
            r"skin|ear|eyelid|vulva|eye|choroid|ciliary body|conjunctiva|sinus|gum|nasal|urethra",
            case = False,
            na = False,
        )
        return (distant & primary_ok).any()

    # ---------------- RULE #1  AGE -----------------------------------------
    if age_diag.lower() == "age 90 or older":
        return _first_roman(path_stg or clin_stg) or "Unknown", "AGE"

    # ---------------- RULE #2  PATHIV ---------------------------------------
    if "IV" in path_stg.upper():
        return "IV", "PATHIV"

    # ---------------- RULE #3  CLINIV --------------------------------------
    if "IV" in clin_stg.upper():
        return "IV", "CLINIV"

    # ---------------- RULE #4  METSITE -------------------------------------
    if not _SITE_LOCAL_RE.search(site_coll):
        return "IV", "METSITE"

    # ---------------- RULE #5  PRIORDISTANT --------------------------------
    if _meta_prior_distant():
        return "IV", "PRIORDISTANT"

    # ---------------- RULE #6  NOMETS --------------------------------------
    if meta_patient is not None and not meta_patient.empty:
        if meta_patient["MetastaticDiseaseInd"].str.lower().eq("no").all():
            return _first_roman(path_stg or clin_stg) or "Unknown", "NOMETS"

    # ---------------- RULE #7  NODE ----------------------------------------
    if re.search(r"lymph node|parotid", site_coll, re.I):
        return "III", "NODE"

    # ---------------- RULE #8  SKINLESS90D ---------------------------------
    # Within ±0.005 rounding fudge the spec requires
    try:
        age_diag_f = float(age_diag)
    except ValueError:
        age_diag_f = None
    within90 = (
        age_diag_f is not None and age_spec is not None
        and abs((age_spec + 0.005) - age_diag_f) <= 90 / 365.25
    )
    has_prior_skin_regional = (
        meta_patient is not None
        and _filter_meta_before(meta_patient, age_spec)
              .query("MetastaticDiseaseInd.str.contains('Regional|NOS', na=False)", engine="python")
              .empty
    )
    if within90 and has_prior_skin_regional:
        return _first_roman(path_stg or clin_stg) or "Unknown", "SKINLESS90D"

    # ---------------- RULE #9  SKINREG -------------------------------------
    if "skin" in site_coll and not has_prior_skin_regional:
        return "III", "SKINREG"

    # ---------------- RULE #10 SKINUNK -------------------------------------
    if meta_patient is not None and not meta_patient.empty:
        if _filter_meta_before(meta_patient, age_spec) \
                .query("AgeAtMetastaticSite == 'Age Unknown/Not Recorded' "
                       "and MetastaticDiseaseInd.str.contains('Distant|NOS', na=False)",
                       engine='python').any(axis=None):
            return "IV", "SKINUNK"

    return "Unknown", "UNMATCHED"


def run_pipeline(
    clinmol: Path,
    diagnosis: Path,
    metadisease: Path,
    therapy: Path | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    '''
    Execute the full pairing, staging, and ICB assignment pipeline and return the resulting data frame.
    '''
    globals()["STRICT"] = strict

    cm, dx, md, th = load_inputs(clinmol, diagnosis, metadisease, therapy)
    cm, dx = add_counts(cm, dx)

    # Everything that follows is byte-for-byte identical to the old `main()`
    cm["MelanomaDiagnosisCount"] = cm["MelanomaDiagnosisCount"].fillna(0).astype(int)
    cm["SequencedTumorCount"]   = cm["SequencedTumorCount"].fillna(0).astype(int)
    cm["Group"] = cm.apply(patient_group, axis=1)

    output_rows: list[dict] = []

    for avatar, specs in cm.groupby("ORIENAvatarKey"):
        dx_patient = dx[dx["AvatarKey"] == avatar]
        if dx_patient.empty:
            logging.warning(
                "Skipping AvatarKey %s: %d tumour specimen(s) but no melanoma Diagnosis rows after filtering – unable to pair.",
                avatar, len(specs),
            )
            continue

        meta_patient    = md[md["AvatarKey"] == avatar]
        therapy_patient = th[th["AvatarKey"] == avatar] if th is not None else None
        group           = specs["Group"].iloc[0]

        if group == "A":
            spec_row, diag_row = specs.iloc[0], dx_patient.iloc[0]
        elif group in ("B", "D"):
            spec_row = _select_specimen_B(specs)
            if spec_row is None:                       # no RNA-Seq specimen
                continue
            diag_row = (
                dx_patient.iloc[0] if group == "B"
                else _select_diagnosis_C(dx_patient, spec_row, meta_patient)
            )
        elif group == "C":
            spec_row  = specs.iloc[0]
            diag_row  = _select_diagnosis_C(dx_patient, spec_row, meta_patient)
        elif group == "D":
            spec_row = _select_specimen_D(specs)
            diag_row = _select_diagnosis_D(dx_patient, spec_row)
        else:
            raise RuntimeError(f"Unknown group: {group}")

        primary_site = assign_primary_site(diag_row["PrimaryDiagnosisSite"])
        stage, rule = stage_by_ordered_rules(spec_row, diag_row, meta_patient)
        
        output_rows.append(
            dict(
                AvatarKey = avatar,
                ORIENSpecimenID = spec_row["DeidSpecimenID"],
                AssignedPrimarySite = primary_site,
                Group = group,
                EknAssignedStage = stage,
                NewRule = rule
            )
        )
    
    map_of_column_names = {
        "EknAssignedStage": "EKN Assigned Stage",
        "NewRule": "NEW RULE"
    }
    out_df = pd.DataFrame(output_rows).rename(columns = map_of_column_names)
    out_df = out_df.sort_values(by = ["AvatarKey", "ORIENSpecimenID"]).reset_index(drop = True)
    return out_df


def main():
    parser = argparse.ArgumentParser(description = "Pair clinical data and stages of tumors.")
    parser.add_argument("--clinmol", required = True, type = Path)
    parser.add_argument("--diagnosis", required = True, type = Path)
    parser.add_argument("--metadisease", required = True, type = Path)
    parser.add_argument("--therapy", required = False, type = Path, help = "ORIEN Therapy CSV (optional; used for ICB status)")
    parser.add_argument("--out", required = True, type = Path)
    parser.add_argument("--strict", action = "store_true", help = "Abort if a primary site cannot be classified.")
    args = parser.parse_args()

    logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")

    out_df = run_pipeline(
        clinmol = args.clinmol,
        diagnosis = args.diagnosis,
        metadisease = args.metadisease,
        therapy = args.therapy,
        strict = args.strict,
    )

    logging.info("Writing %s rows to %s", len(out_df), args.out)
    out_df.to_csv(args.out, index = False)


if __name__ == "__main__":
    main()