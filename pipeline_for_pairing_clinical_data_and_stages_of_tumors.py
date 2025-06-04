#!/usr/bin/env python3

'''
`pipeline_for_pairing_clinical_data_and_stages_of_tumors.py`

This module is based on "ORIEN Data Rules for Tumor Clinical Pairing". The purpose of this module is to produce, for every patient, rows of melanoma diagnosis information and melanoma tumor information, each enriched. A patient is identified by values in fields 'ORIENAvatarKey` in a Clinical Molecular Linkage table and/or `AvatarKey` in Diagnosis, Metastatic Disease, and Medications tables. A row is enriched with a primary site, an AJCC stage at the time of specimen collection, an identifier of the rule used to determine the stage, and an ICB status. A primary site may be "cutaneous", "ocular", "mucosal", or "unknown". An AJCC stage may be "I", "II", "III", or "IV". An ICB status may be "Pre-ICB", "Post-ICB", "No-ICB", or "Unknown".

Melanoma diagnosis information is a row in the Diagnosis CSV file whose value of field `HistologyCode` is a ICD-O-3 code for melanoma. ICD-O-3 stands for International Classification of Disease for Oncology 3. Melanoma diagnosis information describes a case of skin cancer for a patient. Distinct diagnoses information for a patient are identified by pairs of values of fields `AgeAtDiagnosis` and `PrimaryDiagnosisSite`.

Melanoma tumor information is a row in the Clinical Molecular Linkage CSV file with value "Tumor" for field "Tumor/Germline". Melanoma tumor information represents a melanoma tumor for which at least Whole Exome Sequencing (WES) or RNA sequencing data were generated. Distinct tumor information for a patient are identified by pairs of values of fields `ORIENSpecimenID` and `ORIENAvatarKey` in table Clinical Molecular Linkage.

An AJCC stage is a grouping defined by the American Joint Committee on Cancer. An AJCC stage corresponds to fields `ClinGroupStage` and `PathGroupStage` in table Diagnosis. `ClinGroupStage` represents AJCC stages determined during clinical assessments. `PathGroupStage` represents AJCC stages determined during pathological assessments. AJCC stages indicate how much melanoma is in the body and where it is located.

A patient is in Group A if the patient has 1 diagnosis and 1 tumor. A patient is in Group B if the patient has 1 diagnosis and more than 1 tumor. A patient is in Group C if the patient has more than 1 diagnosis and 1 tumor. A patient is in Group D if a patienr has more than 1 diagnosis and more than 1 tumor.

This module
    1. loads Clinical Molecular Linkage, Diagnosis, and Metastatic Disease, and Medication CSV files;
    2. filters table Diagnosis to melanoma diagnosis information and table Clinical Molecular Linkage to melanoma tumor information;
    3. derives a melanoma diagnosis count equal to the number of unique pairs of values of fields `AgeAtDiagnosis` and `PrimaryDiagnosisSite` in table Diagnosis;
    4. derives a tumor count equal to the number of unique pairs of values of fields `ORIENSpecimenID` and `ORIENAvatarKey` in table Clinical Molecular Linkage;
    5. assigns patients in groups A, B, C, and D;
    6. reduces each patient to one row of a subset of diagnosis information and a subset of tumor information using the selection logic for that patient's group, dropping a patient if no valid pairing can be found (e.g., if all tumors lack RNA sequencing data);
    7.  enriches a patient's row with a primary site;
    8.  enriches a patient's row with an AJCC stage using a rule set that is specific to a primary site;
    9.  enriches a patient's row with an identifier of the rule used to determine the stage;
    10. enriches a patient's row with an ICB status by comparing age of patient when specimen was collected to earliest ICB therapy; and
    11. writes rows to an output CSV file.
        
Usage:

python pipeline_for_pairing_clinical_data_and_stages_of_tumors.py --clinmol ../../../Avatar_CLINICAL_Data/20250317_UVA_ClinicalMolLinkage_V4.csv --diagnosis ../../../Avatar_CLINICAL_Data/NormalizedFiles/20250317_UVA_Diagnosis_V4.csv --metadisease ../../../Avatar_CLINICAL_Data/NormalizedFiles/20250317_UVA_MetastaticDisease_V4.csv --therapy ../../../Avatar_CLINICAL_Data/NormalizedFiles/20250317_UVA_Medications_V4.csv --out output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv > output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.txt 2>&1
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
    "cutaneous": r"skin|ear|eyelid|vulva|head|scalp",
    "ocular": r"choroid|ciliary body|conjunctiva|eye",
    "mucosal": r"sinus|gum|nasal|urethra|anorect|anus|rectum|anal canal|oropharynx|oral|vagina|esophagus|palate",
    "unknown": r"unknown"
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


def _contains(value: str | float | int | None, pattern: str | re.Pattern) -> bool:
    if pd.isna(value):
        return False
    return bool(re.search(pattern, str(value).lower()))


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


def _filter_meta_after(meta_patient: pd.DataFrame, age_spec: float | None) -> pd.DataFrame:
    '''
    Rows with AgeAtMetastaticSite > specimen age (interval mets).
    '''
    if meta_patient is None or meta_patient.empty or age_spec is None:
        return pd.DataFrame()

    m = meta_patient.copy()
    m["_age"] = m["AgeAtMetastaticSite"].apply(_float)
    return m[pd.notna(m["_age"]) & (m["_age"] > age_spec)]


def _float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _strip_cols(df: pd.DataFrame) -> None:
    '''
    Trim whitespace from every column header in place.
    '''
    df.columns = df.columns.str.strip()
    
    
#############
# CSV LOADING
#############

def load_inputs(
    clinmol_csv: Path,
    dx_csv: Path,
    meta_csv: Path,
    therapy_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
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
        logging.info("Loading Therapy ...")
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
    Add MelanomaDiagnosisCount and SequencedTumorCount to CM and DX frames.
    '''
    diag_counts = (
        dx.drop_duplicates(["AvatarKey", "AgeAtDiagnosis", "PrimaryDiagnosisSite"])
          .groupby("AvatarKey")
          .size()
          .rename("MelanomaDiagnosisCount")
    )
    tumor_counts = (
        cm.drop_duplicates(["ORIENAvatarKey", "ORIENSpecimenID"])
          .groupby("ORIENAvatarKey")
          .size()
          .rename("SequencedTumorCount")
    )
    cm = cm.join(tumor_counts, on = "ORIENAvatarKey")
    cm = cm.join(diag_counts, on = "ORIENAvatarKey")
    dx = dx.join(diag_counts, on = "AvatarKey")
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
    
    # RNA sequencing gate
    if cm["RNASeq"].notna().sum() == 0:
        return None
    
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


###############
# STAGING RULES
###############

def stage_cutaneous(spec: pd.Series, dx: pd.Series, meta_patient: pd.DataFrame) -> Tuple[str, str]:
    '''
    Return (stage, rule_id) for cutaneous primary.
    '''
    path_stage = str(dx.get("PathGroupStage", "")).upper()
    clin_stage = str(dx.get("ClinGroupStage", "")).upper()
    site_coll = spec["SpecimenSiteOfCollection"].lower()
    primary_met = spec["Primary/Met"].lower()
    age_spec = _float(spec["Age At Specimen Collection"])

    meta_before = _filter_meta_before(meta_patient, age_spec)  # ≤ specimen age
    meta_after  = _filter_meta_after(meta_patient, age_spec)   #   > specimen age

    def _meta_yes(rows: pd.DataFrame,
                  kind_pat: str,
                  site_pat: str,
                  ind_cat: str | None = None) -> bool:
        """
        ind_cat: 'distant', 'regional', 'regional_nos', 'distant_nos', or None (no filter)
        """
        if rows.empty:
            return False
        ok = rows["MetsDzPrimaryDiagnosisSite"].str.contains(kind_pat, case=False, na=False)
        ok &= rows["MetastaticDiseaseSite"].str.contains(site_pat, case=False, na=False)
        if ind_cat is not None:
            ind_pat = {
                "distant":        r"Yes - Distant",
                "regional":       r"Yes - Regional",
                "regional_nos":   r"Yes - Regional|Yes - NOS",
                "distant_nos":    r"Yes - Distant|Yes - NOS",
            }[ind_cat]
            ok &= rows["MetastaticDiseaseInd"].str.contains(ind_pat, case=False, na=False)
        return ok.any()
    
    # --- RULE CUT‑1: PathGroupStage contains IV --------------------------------
    if "IV" in path_stage:
        return "IV", "CUT1"

    # --- RULE CUT‑2: Clin contains IV & Path unknown ---------------------------
    if (
        "IV" in clin_stage
        and path_stage.lower() in UNKNOWN_PATHOLOGICAL_STAGE_VALUES
    ):
        return "IV", "CUT2"

    # --- RULE CUT-3 -----------------------------------------------------------
    # metastatic specimen from a site that is NOT skin, lymph-node, soft-tissue,
    # muscle, *or* parotid (explicitly).
    if (primary_met == "metastatic"
        and not _contains(site_coll,
            r"skin|lymph node|soft tissue|muscle|parotid|chest wall|head|scalp")):
        return "IV", "CUT3"

    # Lymph node helper booleans
    is_node_spec = bool(NODE_REGEX.search(site_coll))

    # --- RULE CUT‑4: Node specimen, stage III at diagnosis ---------------------
    if (
        is_node_spec
        and _meta_yes(meta_before, r"skin|ear|eyelid|vulva", "lymph node", "regional_nos")
        and "III" in path_stage
    ):
        return "III", "CUT4"

    # ---------------- CUT‑5 ------------------------
    if is_node_spec and ("III" not in path_stage) and ("IV" not in path_stage) and ("IV" not in clin_stage):
        return "III", "CUT5"

    # --- RULE CUT‑6: Node specimen, distant nodal recurrence → stage IV --------
    if (
        is_node_spec
        and _meta_yes(meta_before, r"skin|ear|eyelid|vulva", "lymph node", "distant")
        and "IV" not in path_stage
    ):
        return "IV", "CUT6"

    # --- RULE CUT‑7: Parotid specimen distant recurrence → stage IV ------------
    # --- RULE CUT‑8: Parotid specimen regional → stage III ---------------------
    if PAROTID_REGEX.search(site_coll):
        if _meta_yes(meta_before, r"skin|ear|eyelid|vulva", "parotid", "distant"):
            return "IV", "CUT7"
        if _meta_yes(meta_before, r"skin|ear|eyelid|vulva", r"parotid|lymph node", "regional_nos"):
            return "III", "CUT8"

    cut_site_pat = r"skin|ear|eyelid|head|soft tissues|muscle|chest wall|vulva"
    if (_contains(site_coll, cut_site_pat)
        and primary_met == "metastatic"
        and "IV" not in path_stage):

        # CUT-9 – distant cutaneous recurrence (≤ specimen age, Distant or NOS)
        if (_meta_yes(meta_before, r"skin|ear|eyelid|vulva", cut_site_pat, "distant_nos")
            or _meta_yes(meta_before, r"skin|ear|eyelid|vulva", r".*", "distant_nos")):
            return "IV", "CUT9"

        # CUT-10 – regional cutaneous recurrence (≤ specimen age, Regional or NOS)
        if (_meta_yes(meta_before, r"skin|ear|eyelid|vulva", cut_site_pat, "regional_nos")
            or _meta_yes(meta_before, r"skin|ear|eyelid|vulva", r".*", "regional_nos")):
            return "III", "CUT10"
   
        
    # CUT-11 – primary specimen collected *after* interval distant mets
    if (primary_met == "primary"
        and _contains(site_coll, cut_site_pat)
        and _meta_yes(meta_before, r"skin|ear|eyelid|vulva", r".*", "distant_nos")):
        return "IV", "CUT11"

    # --- RULE CUT‑12: Fallback to numeric stage -------------------------------
    for src, rule in ((path_stage, "CUT12P"), (clin_stage, "CUT12C")):
        m = re.match(r"([IV]+)", src)
        if m:
            return m.group(1), rule

    # Default unknown
    return "Unknown", "CUT‑UNK"


def stage_ocular(spec: pd.Series, dx: pd.Series, meta_patient: pd.DataFrame) -> Tuple[str, str]:
    '''
    Ocular primary staging (rules OC-1, OC-2, OC-3, and OC-4)
    '''
    
    path_stage = str(dx.get("PathGroupStage", "")).upper()
    clin_stage = str(dx.get("ClinGroupStage", "")).upper()
    age_spec = _float(spec["Age At Specimen Collection"])
    site_coll = str(spec["SpecimenSiteOfCollection"]).lower()

    # OC-1 / OC-2: Stage IV at diagnosis
    if "IV" in path_stage:
        return "IV", "OC1"
    if "IV" in clin_stage and path_stage.lower() in UNKNOWN_PATHOLOGICAL_STAGE_VALUES:
        return "IV", "OC2"

    # Prepare metastatic disease rows (<= specimen age or unknown)
    meta_rows = _filter_meta_before(meta_patient, age_spec)
    if meta_rows.empty:
        return "Unknown", "OC-UNK"
    
    # OC-4 pre-condition: origin must be ocular AND specimen is a lymph-node.
    origin_ocular = _contains(spec.get("SpecimenSiteOfOrigin"), r"eye|choroid|ciliary body|conjunctiva")
    is_node_spec  = bool(NODE_REGEX.search(site_coll))
    has_distant   = meta_rows["MetastaticDiseaseInd"].str.contains("Yes - Distant", na=False).any()
    reg_nos       = meta_rows["MetastaticDiseaseInd"].str.contains(r"Yes - Regional|Yes - NOS", na=False).any()
    if origin_ocular and is_node_spec and reg_nos and not has_distant:
        return "III", "OC4"
    
    # OC-3: interval distant metastases (runs *after* OC-4 so OC-4 can win when appropriate)
    if has_distant:
        return "IV", "OC3"

    return "Unknown", "OC‑UNK"


def stage_mucosal(spec: pd.Series, dx: pd.Series, meta_patient: pd.DataFrame) -> Tuple[str, str]:
    path_stage = str(dx.get("PathGroupStage", "")).upper()
    clin_stage = str(dx.get("ClinGroupStage", "")).upper()
    age_spec = _float(spec["Age At Specimen Collection"])

    if "IV" in path_stage:
        return "IV", "MU1"
    if "IV" in clin_stage and path_stage.lower() in UNKNOWN_PATHOLOGICAL_STAGE_VALUES:
        return "IV", "MU2"

    # ─── MU-3P / MU-3C — Primary specimen, stage I-III numerical fallback ───
    if spec["Primary/Met"].lower() == "primary":
        for src, rule in ((path_stage, "MU3P"), (clin_stage, "MU3C")):
            m = re.match(r"([IV]+)", src)
            if m and m.group(1) != "IV":
                return m.group(1), rule

    # ─── Metastatic-specimen pathway ─────────────────────────────────────────
    meta_rows = _filter_meta_before(meta_patient, age_spec)

    # MU-4 — distant mets present  → stage IV
    if not meta_rows.empty and \
       meta_rows["MetastaticDiseaseInd"].str.contains("Yes - Distant", na=False).any():
        return "IV", "MU4"

    # MU-3M — metastatic specimen **without** distant mets (past *or* future) → III
    if spec["Primary/Met"].lower() == "metastatic" and "IV" not in path_stage:
        meta_after = _filter_meta_after(meta_patient, age_spec)
        no_future_distant = meta_after.empty or \
            not meta_after["MetastaticDiseaseInd"].str.contains("Yes - Distant", na=False).any()
        if no_future_distant:
            return "III", "MU3M"

    return "Unknown", "MU-UNK"


def stage_unknown_primary(spec, dx, meta) -> Tuple[str, str]:
    '''
    Rules UN-1 ... UN-4  (spec section 4.4)

    UN-1 : PathGroupStage contains IV
    UN-2 : ClinGroupStage contains IV  AND PathGroupStage unknown / not-applicable
    UN-3P: PathGroupStage contains I / II / III            ← new
    UN-3C: ClinGroupStage contains I / II / III (when Path unknown)  ← new
    UN-UNK: otherwise
    '''
    path_stage = str(dx.get("PathGroupStage", "")).upper()
    clin_stage = str(dx.get("ClinGroupStage", "")).upper()
    if "IV" in path_stage:
        return "IV", "UN1"
    if "IV" in clin_stage and path_stage.lower() in UNKNOWN_PATHOLOGICAL_STAGE_VALUES:
        return "IV", "UN2"
    
    # ---- UN-3  (fallback to the earliest numeric stage in Path → Clin) -------
    stage_re = re.compile(r"\b([I]{1,3})\b")          # captures I, II or III only
    m_path = stage_re.search(path_stage)
    if m_path:
        return m_path.group(1), "UN3P"

    if path_stage.lower() in UNKNOWN_PATHOLOGICAL_STAGE_VALUES:
        m_clin = stage_re.search(clin_stage)
        if m_clin:
            return m_clin.group(1), "UN3C"
    
    return "Unknown", "UN‑UNK"

################################################################################
# DRIVER
################################################################################

def main():
    parser = argparse.ArgumentParser(description = "Pair melanoma tumours with AJCC stages.")
    parser.add_argument("--clinmol", required = True, type = Path)
    parser.add_argument("--diagnosis", required = True, type = Path)
    parser.add_argument("--metadisease", required = True, type = Path)
    parser.add_argument("--therapy", required = False, type = Path, help = "ORIEN Therapy CSV (optional; used for ICB status)")
    parser.add_argument("--out", required = True, type = Path)
    parser.add_argument("--strict", action = "store_true", help = "Abort if a primary site cannot be classified.")
    args = parser.parse_args()
    
    globals()["STRICT"] = args.strict

    logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")

    cm, dx, md, th = load_inputs(args.clinmol, args.diagnosis, args.metadisease, args.therapy)
    cm, dx = add_counts(cm, dx)

    # Attach group labels to CM rows (one per specimen)
    cm["MelanomaDiagnosisCount"] = cm["MelanomaDiagnosisCount"].fillna(0).astype(int)
    cm["SequencedTumorCount"] = cm["SequencedTumorCount"].fillna(0).astype(int)
    cm["Group"] = cm.apply(patient_group, axis=1)

    output_rows: List[Dict] = []

    for avatar, specs in cm.groupby("ORIENAvatarKey"):
        dx_patient = dx[dx["AvatarKey"] == avatar]
        '''
        If a patient has no melanoma diagnoses that survived our histology filter, we cannot assign a stage unambiguously.
        Rather than crashing later, log it and move on to the next patient.
        '''
        if dx_patient.empty:
            logging.warning(
                "Skipping AvatarKey %s: %d tumour specimen(s) but no melanoma Diagnosis rows after filtering – unable to pair.",
                avatar,
                len(specs),
            )
            continue
        meta_patient = md[md["AvatarKey"] == avatar]
        therapy_patient = th[th["AvatarKey"] == avatar] if th is not None else None
        group = specs["Group"].iloc[0]

        if group == "A":
            spec_row = specs.iloc[0]
            diag_row = dx_patient.iloc[0]
        elif group in ("B", "D"):
            spec_row = _select_specimen_B(specs)
            if spec_row is None:
                logging.info("AvatarKey %s excluded: no specimen passes RNA-seq rule.", avatar)
                continue
            diag_row = (
                dx_patient.iloc[0]
                if group == "B" else _select_diagnosis_C(dx_patient, spec_row, meta_patient)
            )
        elif group == "C":
            spec_row = specs.iloc[0]
            diag_row = _select_diagnosis_C(dx_patient, spec_row, meta_patient)
        else:
            raise RuntimeError(f"Unknown group: {group}")

        primary_site = assign_primary_site(diag_row["PrimaryDiagnosisSite"])
        if primary_site == "cutaneous":
            stage, rule = stage_cutaneous(spec_row, diag_row, meta_patient)
        elif primary_site == "ocular":
            stage, rule = stage_ocular(spec_row, diag_row, meta_patient)
        elif primary_site == "mucosal":
            stage, rule = stage_mucosal(spec_row, diag_row, meta_patient)
        else:
            stage, rule = stage_unknown_primary(spec_row, diag_row, meta_patient)

        icb_status = _assign_icb_status(spec_row, therapy_patient)

        output_rows.append(
            {
                "AvatarKey": avatar,
                "ORIENSpecimenID": spec_row["ORIENSpecimenID"],
                "DiagnosisIndex": int(diag_row.name),
                "AgeAtSpecimenCollection": spec_row["Age At Specimen Collection"],
                "AssignedPrimarySite": primary_site,
                "AssignedStage": stage,
                "StageRuleHit": rule,
                "ICBStatus": icb_status,
            }
        )

    out_df = pd.DataFrame(output_rows)
    logging.info("Writing %s rows → %s", len(out_df), args.out)
    out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()