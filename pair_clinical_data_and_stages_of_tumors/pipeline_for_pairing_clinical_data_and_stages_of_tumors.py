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


###############################
# GLOBAL SETTINGS AND CONSTANTS
###############################

AGE_FUDGE = 0.005 # years, or approximately 1.8 days.
_ROMAN_RE = re.compile(r"\b([IV]{1,3})(?:[ABCD])?\b", re.I)
STRICT: bool = False


ICB_PATTERN = re.compile(r"immune checkpoint|pembrolizumab|nivolumab|ipilimumab|atezolizumab|durvalumab|avelumab|cemiplimab|relatlimab", re.I)

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

# All melanoma morphology codes begin with “87” in ICD-O-3.  The formal list
# above is still authoritative, but the revised rules state that *every*
# melanoma diagnosis must be retained.  Some partner centres occasionally
# write in additional site-specific “878x/3”, “879x/3”, etc. codes that do not
# appear in the historic list.  We therefore accept the full 87xx/3 range as a
# fall-back.

MEL_REGEX = re.compile(r"^87\d\d/3$")

SITE_KEYWORDS = {
    "cutaneous": "skin|ear|eyelid|vulva|head|scalp|trunk|face|neck|back|chest|shoulder|arm|leg|extremity|hand|foot|buttock|hip|soft tissue|breast",
    "ocular": "choroid|ciliary body|conjunctiva|eye",
    "mucosal": "sinus|gum|nasal|urethra|anorect|anus|rectum|anal canal|oropharynx|oral|palate|vagina",
    "unknown": "unknown"
}

CUTANEOUS_RE = re.compile(SITE_KEYWORDS["cutaneous"], re.I)
NODE_RE       = re.compile(r"lymph node", re.I)
PAROTID_RE    = re.compile(r"parotid", re.I)

'''
NODE_REGEX = re.compile(r"lymph node", re.I) # regular expression object
PAROTID_REGEX = re.compile(r"parotid", re.I) # regular expression object
REGEX_PRI_SITE_SKIN = r"skin|ear|eyelid|vulva"

SKINLIKE_SITES = [
    "skin", "ear", "eyelid", "head", "soft tissues", "muscle", "chest wall", "vulva"
]

UNKNOWN_PATHOLOGICAL_STAGE_VALUES = {
    "unknown/not reported",
    "no tnm applicable for this site/histology combination",
    "unknown/not applicable"
}
'''


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
    Return the first Roman-numeral stage (I–IV) found, else None.
    '''
    m = re.search(r"\b([IV]{1,3})\b", str(stage_txt).upper())
    return m.group(1) if m else None
    
    
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
    logging.info("Loading Clinical‑Molecular Linkage...")
    cm = pd.read_csv(clinmol_csv, dtype = str)
    _strip_cols(cm)
    cm = cm[cm["Tumor/Germline"].str.lower() == "tumor"].copy()

    logging.info("Loading Diagnosis...")
    dx = pd.read_csv(dx_csv, dtype = str)
    _strip_cols(dx)
    is_exact = dx["HistologyCode"].isin(MELANOMA_HISTOLOGY_CODES)
    is_any87 = dx["HistologyCode"].str.match(MEL_REGEX, na = False)
    dx = dx[is_exact | is_any87].copy()

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
    diag_counts = dx.groupby("AvatarKey").size().rename("MelanomaDiagnosisCount")
    tumor_counts = cm.drop_duplicates(["ORIENAvatarKey", "DeidSpecimenID"]).groupby("ORIENAvatarKey").size().rename("SequencedTumorCount")
    cm = cm.join(tumor_counts, on = "ORIENAvatarKey") # Logic that uses both diagnosis count and tumor count is executed while iterating over rows in cm.
    cm = cm.join(diag_counts, on = "ORIENAvatarKey")
    return cm, dx


def _patient_group(row) -> str:
    '''
    Return A, B, C, or D according to diagnosis and tumor counts.
    '''
    mdx = int(row.get("MelanomaDiagnosisCount", 0) or 0)
    mts = int(row.get("SequencedTumorCount",    0) or 0)
    if mdx == 1 and mts == 1:
        return "A"
    if mdx == 1 and mts > 1:
        return "B"
    if mdx > 1 and mts == 1:
        return "C"
    return "D"

    
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
        keep = pd.concat([keep, m[pd.isna(m["_age"])]]).drop_duplicates()
    return keep


#########################
# PRIMARY‑SITE ASSIGNMENT
#########################

def assign_primary_site(primary_diagnosis_site: str) -> str:
    txt = str(primary_diagnosis_site).lower()
    for site, pat in SITE_KEYWORDS.items():
        if re.search(pat, txt):
            return site
    if STRICT:
        raise ValueError(f"Unrecognized primary diagnosis site: '{primary_diagnosis_site}'")
    return "unknown"


##############################################################################################
# STAGING RULES
# The first rule that matches wins.
# A specimen is not evaluated against any rule later than a rule that applies to the specimen.
##############################################################################################

_SITE_LOCAL_RE = re.compile(r"skin|ear|eyelid|vulva|head|scalp|soft tissues|breast|lymph node|parotid", re.I)

def stage_by_ordered_rules(
    spec: pd.Series,
    dx: pd.Series,
    meta_patient: pd.DataFrame
) -> Tuple[str, str]:
    '''
    Return (EKN Assigned Stage, NEW RULE) following Rule 1 to Rule 10 exactly in the order printed in "ORIEN Specimen Staging Revised Rules"..
    '''

    # Short aliases ----------------------------------------------------------
    age_diag_txt = str(dx.get("AgeAtDiagnosis", "")).strip()
    age_diag_f = _float(age_diag_txt)
    path_stg = str(dx.get("PathGroupStage", "")).strip()
    clin_stg = str(dx.get("ClinGroupStage", "")).strip()
    site_coll = str(spec.get("SpecimenSiteOfCollection", "")).lower()
    age_spec = _float(spec.get("Age At Specimen Collection"))

    def _meta_prior_distant() -> bool:
        if meta_patient is None or meta_patient.empty:
            return False
        m = _filter_meta_before(meta_patient, age_spec, allow_unknown_age = False)
        if m.empty:
            return False
        distant = m["MetastaticDiseaseInd"].str.contains("Yes - Distant", na = False)
        site_ok = m["MetsDzPrimaryDiagnosisSite"].str.contains(CUTANEOUS_RE, na = False)
        return (distant & site_ok).any()

    def _prior_skin_regional() -> bool:
        if meta_patient is None or meta_patient.empty:
            return False
        m = _filter_meta_before(meta_patient, age_spec, allow_unknown_age = False)
        if m.empty:
            return False
        site_ok = m["MetsDzPrimaryDiagnosisSite"].str.contains(CUTANEOUS_RE, na = False)
        reg_ok  = m["MetastaticDiseaseInd"].str.contains(r"Yes - Regional|Yes - NOS", na = False)
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
    if not meta_patient.empty and meta_patient["MetastaticDiseaseInd"].str.lower().eq("no").all():
        if meta_patient["MetastaticDiseaseInd"].str.lower().eq("no").all():
            return _first_roman(path_stg or clin_stg) or "Unknown", "NOMETS"

    # RULE 7 - NODE
    if NODE_RE.search(site_coll) or PAROTID_RE.search(site_coll):
        return "III", "NODE"

    # RULE 8 - SKINLESS90D
    within90 = (
        age_spec is not None and age_diag_f is not None and
        0 <= (age_spec + AGE_FUDGE) - age_diag_f <= 90 / 365.25
    )
    if within90 and not _prior_skin_regional() and re.search(r"skin|ear|eyelid|vulva|head|soft tissues|breast", site_coll):
        return _first_roman(path_stg or clin_stg) or "Unknown", "SKINLESS90D"

    # RULE 9 - SKINREG
    if re.search(r"skin|ear|eyelid|vulva|head|soft tissues|breast|lymph node", site_coll) and not _skin_distant_unknown():
        return "III", "SKINREG"

    # RULE 10 - SKINUNK
    if _skin_distant_unknown():
        return "IV", "SKINUNK"

    # Fallback (should never be reached according to ORIEN Specimen Staging Revised Rules)
    return "Unknown", "UNMATCHED"


def _hist_clean(txt: str) -> str:
    return re.sub(r"[^A-Za-z]", "", str(txt)).lower()


def _within_90_days_after(age_spec: float | None, age_diag: float | None) -> bool:
    """
    Return True when the specimen was collected **0–90 days AFTER** the
    diagnosis age, per ORIEN rules.
    """
    if age_spec is None or age_diag is None:
        return False
    diff = age_spec - age_diag
    return 0 <= diff <= (90 / 365.25)


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

    cm["Group"] = cm.apply(_patient_group, axis = 1)
    cm["MelanomaDiagnosisCount"] = cm["MelanomaDiagnosisCount"].fillna(0)
    cm["SequencedTumorCount"] = cm["SequencedTumorCount"].fillna(0)

    output_rows: List[Dict[str, str]] = []

    for avatar, specs in cm.groupby("ORIENAvatarKey", sort = False):
        dx_patient = dx[dx["AvatarKey"] == avatar]
        meta_patient    = md[md["AvatarKey"] == avatar]
        therapy_patient = th[th["AvatarKey"] == avatar] if th is not None else None
        group           = specs["Group"].iloc[0]

        if group == "A":
            spec_row, diag_row = specs.iloc[0], dx_patient.iloc[0]
        elif group == "B":
            spec_row = _select_specimen_B(specs)
            diag_row = dx_patient.iloc[0]
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
            {
                "AvatarKey": avatar,
                "ORIENSpecimenID": spec_row["DeidSpecimenID"],
                "AssignedPrimarySite": primary_site,
                "Group": group,
                "EKN Assigned Stage": stage,
                "NEW RULE": rule
            }
        )

    out_df = pd.DataFrame(output_rows).sort_values(by = ["AvatarKey", "ORIENSpecimenID"]).reset_index(drop = True)
    return out_df


def main():
    parser = argparse.ArgumentParser(description = "Pair clinical data and stages of tumors.")
    parser.add_argument("--clinmol", required = True, type = Path)
    parser.add_argument("--diagnosis", required = True, type = Path)
    parser.add_argument("--metadisease", required = True, type = Path)
    parser.add_argument("--therapy", type = Path)
    parser.add_argument("--out", required = True, type = Path)
    parser.add_argument("--strict", action = "store_true")
    args = parser.parse_args()

    logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")

    df = run_pipeline(
        clinmol = args.clinmol,
        diagnosis = args.diagnosis,
        metadisease = args.metadisease,
        therapy = args.therapy,
        strict = args.strict,
    )

    logging.info("Writing %d rows to %s", len(df), args.out)
    df.to_csv(args.out, index = False)


if __name__ == "__main__":
    main()