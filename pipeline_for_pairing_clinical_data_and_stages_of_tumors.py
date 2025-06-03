#!/usr/bin/env python3

'''
Tumor‑Stage Pairing Pipeline
=================================
Pairs melanoma clinical diagnoses with sequenced tumor specimens and assigns an
AJCC stage (I, II, III, IV) at the time of specimen collection, exactly as
described in "ORIEN Data Rules for Tumor Clinical Pairing".

A melanoma clinical diagnosis is a row in the Diagnosis CSV file describes a case of skin cancer for a patient.
TODO: How can a medical clinical diagnosis be paired?
A sequenced tumor specimen is a row in the Clinical Molecular Linkage CSV file with value "Tumor" for field "Tumor/Germline" that represents a biological sample for which at least Whole Exome Sequencing (WES) or RNA sequencing data were generated.
TODO: How can a sequenced tumor specimen be paired?
An AJCC stage is a grouping defined by the American Joint Committee on Cancer. AJCC stage may be I, II, III, or IV. An AJCC stage corresponds to fields `ClinGroupStage` and `PathGroupStage` in table Diagnosis. `ClinGroupStage` represents AJCC stages determined during clinical assessments. `PathGroupStage` represents AJCC stages determined during pathological assessments.
TODO: What does an AJCC stage signify?

This single script
    1.  reads Clinical Molecular Linkage, Diagnosis, and Metastatic Disease CSV files;
    2.  filters to melanoma diagnoses (ICD‑O‑3 codes) and tumor specimens;
        TODO: How does this script identify melanoma diagnoses?
        TODO: How does this script identify tumor specimens?
        TODO: How does this script filter to melanoma diagnoses and tumor specimens?
    3.  derives per‑patient summary metrics (MelanomaDiagnosisCount /
        SequencedTumorCount) and assigns patients to Group A, B, C, or D;
        TODO: How are patients assigned to Group A, B, C, or D?
    4.  reduces each patient to one (specimen, diagnosis) pair using the selection logic for that patient's group, dropping a patient if no valid pairing can be found (e.g., if all tumors lack RNA sequencing data);
        TODO: What is a specimen?
        TODO: What is a diagnosis?
    5.  assigns a value to `AssignedPrimarySite` equal to "cutaneous", "ocular", "mucosal", or "unknown";
    6.  assigns an `AssignedStage` using a rule set that is specific to a primary site;
    7.  assigns a code (StageRuleHit) identifying which rule produced the stage; and
    8.  emits one row per paired specimen with traceability fields.
        TODO: What is a paired specimen?
        TODO: What is a traceability field?
        
Usage:

python pipeline_for_pairing_clinical_data_and_stages_of_tumors.py --clinmol ../../../Avatar_CLINICAL_Data/20250317_UVA_ClinicalMolLinkage_V4.csv --diagnosis ../../../Avatar_CLINICAL_Data/NormalizedFiles/20250317_UVA_Diagnosis_V4.csv --metadisease ../../../Avatar_CLINICAL_Data/NormalizedFiles/20250317_UVA_MetastaticDisease_V4.csv --out output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv > output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.txt 2>&1
'''

from __future__ import annotations
import argparse
import logging
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Tuple


###########
# CONSTANTS
###########

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
    "mucosal": r"sinus|gum|nasal|urethra",
    "unknown": r"unknown"
}

SKINLIKE_SITES = [
    "skin", "ear", "eyelid", "head", "soft tissues", "muscle", "chest wall", "vulva"
]

UNKNOWN_PATH_STAGE_VALUES = {
    # TODO: What does "PATH" mean?
    "unknown/not reported",
    "no tnm applicable for this site/histology combination",
    "unknown/not applicable"
}


#################
# UTILITY HELPERS
#################

def _contains(value: str | float | int | None, pattern: str | re.Pattern) -> bool:
    if pd.isna(value):
        return False
    return bool(re.search(pattern, str(value).lower()))


def _filter_meta(meta_patient: pd.DataFrame, age_spec: float | None) -> pd.DataFrame:
    '''
    Retain metastatic-disease rows that occurred on/before the specimen age
    OR whose age is "Age Unknown/Not Recorded".
    This guarantees correct behaviour for rules that say
        "... AgeAtMetastaticSite <= AgeAtSpecimenCollection OR Age Unknown/Not Recorded".
    '''
    if meta_patient is None or meta_patient.empty:
        return pd.DataFrame()

    m = meta_patient.copy()
    m["_age"] = m["AgeAtMetastaticSite"].apply(_float)   # _float->None for "Unknown..."

    if age_spec is None:
        # If specimen age is missing, keep only age-unknown rows
        return m[pd.isna(m["_age"])]

    return m[pd.isna(m["_age"]) | (m["_age"] <= age_spec)]


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

def load_inputs(clinmol_csv: Path, dx_csv: Path, meta_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    return cm, dx, md


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
    Return a single specimen row for a Group B/D patient or None if patient should be dropped.
    '''
    cm = patient_cm.copy()
    
    if cm["RNASeq"].notna().sum() == 0:
        return None
    
    # 1. Keep only RNA‑seq‑enabled tumours if possible
    if cm["RNASeq"].notna().any():
        cm_rna = cm[cm["RNASeq"].notna()]
        if len(cm_rna) == 1:
            return cm_rna.iloc[0]

    # 2. Skin taking precedence (no nodes/soft tissue present)
    has_skin = cm["SpecimenSiteOfCollection"].str.contains("skin", case = False, na = False)
    has_node_or_st = cm["SpecimenSiteOfCollection"].str.contains("lymph node|soft tissue", case = False, na = False)
    if has_skin.any() and not has_node_or_st.any():
        return cm[has_skin].iloc[0]

    # 3. Prefer lymph node when no skin / soft‑tissue
    has_node = cm["SpecimenSiteOfCollection"].str.contains("lymph node", case = False, na = False)
    if not has_skin.any() and has_node.any():
        return cm[has_node].iloc[0]

    # 4. Earliest AgeAtSpecimenCollection
    cm["_age"] = cm["Age At Specimen Collection"].astype(float)
    return cm.sort_values("_age").iloc[0]


################################################
# SELECTION HELPERS – GROUP C (choose diagnosis)
################################################

def _within_90_days(age_spec: float | None, age_diag: float | None) -> bool:
    if age_spec is None or age_diag is None:
        return False
    return abs(age_spec - age_diag) <= (90 / 365.25)


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
        prox = dxp["AgeAtDiagnosis"].apply(_float).apply(lambda x: _within_90_days(age_spec, x))
        if prox.sum() == 1:
            return dxp[prox].iloc[0]

        # Proximity tie‑break by site
        if prox.any():
            site_match = dxp["PrimaryDiagnosisSite"].str.lower() == spec_row["SpecimenSiteOfCollection"].lower()
            match_rows = dxp[prox & site_match]
            if len(match_rows) == 1:
                return match_rows.iloc[0]

        # Histology match
        hist_spec = _hist_clean(spec_row["Histology/Behavior"])
        dxp["_hist_clean"] = dxp["Histology"].apply(_hist_clean)
        hist_match = dxp["_hist_clean"] == hist_spec
        if hist_match.sum() == 1:
            return dxp[hist_match].iloc[0]

    else:  # metastatic specimen pathway
        site_coll = spec_row["SpecimenSiteOfCollection"].lower()
        
        # Non-node / non-soft-tissue -> earliest diagnosis
        if not re.search(r"soft tissues|lymph node", site_coll):
            return dxp.sort_values("AgeAtDiagnosis").iloc[0]

        # Soft tissue within 90 days
        if "soft tissues" in site_coll:
            prox = dxp["AgeAtDiagnosis"].apply(_float).apply(lambda x: _within_90_days(age_spec, x))
            if prox.sum() == 1:
                return dxp[prox].iloc[0]

        if "lymph node" in site_coll:
            # Positive nodes only one?
            pos_node = ~dxp["PathNStage"].str.contains(r"\bN0\b|Nx|unknown/not applicable|no tnm", case = False, na = False)
            if pos_node.sum() == 1:
                return dxp[pos_node].iloc[0]
            # PrimaryDiagnosisSite matched to MetsDz…
            if meta_patient is not None and not meta_patient.empty:
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

    # Meta rows with age <= specimen or "Unknown/Not Recorded"
    meta_rows = _filter_meta(meta_patient, age_spec)

    def _meta_yes(kind_pat: str, site_pat: str, distant: Optional[bool] = None) -> bool:
        if meta_rows.empty:
            return False
        ok = meta_rows["MetsDzPrimaryDiagnosisSite"].str.contains(kind_pat, case = False, na = False)
        ok &= meta_rows["MetastaticDiseaseSite"].str.contains(site_pat, case = False, na = False)
        if distant is not None:
            ind_pat = "Yes - Distant" if distant else r"Yes - Regional|Yes - NOS"
            ok &= meta_rows["MetastaticDiseaseInd"].str.contains(ind_pat, case = False, na = False)
        return ok.any()
    
    # --- RULE CUT‑1: PathGroupStage contains IV --------------------------------
    if "IV" in path_stage:
        return "IV", "CUT1"

    # --- RULE CUT‑2: Clin contains IV & Path unknown ---------------------------
    if (
        "IV" in clin_stage
        and path_stage.lower() in UNKNOWN_PATH_STAGE_VALUES
    ):
        return "IV", "CUT2"

    # --- RULE CUT‑3: Metastatic specimen from non‑skin/non‑node site -----------
    if primary_met == "metastatic" and not (
        _contains(site_coll, r"skin|lymph node|soft tissues|muscle|parotid|chest wall|head|scalp")
    ):
        return "IV", "CUT3"

    # Lymph node helper booleans
    is_node_spec = bool(NODE_REGEX.search(site_coll))

    # --- RULE CUT‑4: Node specimen, stage III at diagnosis ---------------------
    if (
        is_node_spec
        and _meta_yes(r"skin|ear|eyelid|vulva", "lymph node", distant = False)
        and "III" in path_stage
    ):
        return "III", "CUT4"

    # ---------------- CUT‑5 ------------------------
    if is_node_spec and ("III" not in path_stage) and ("IV" not in path_stage) and ("IV" not in clin_stage):
        return "III", "CUT5"

    # --- RULE CUT‑6: Node specimen, distant nodal recurrence → stage IV --------
    if (
        is_node_spec
        and _meta_yes(r"skin|ear|eyelid|vulva", "lymph node", distant = True)
        and "IV" not in path_stage
    ):
        return "IV", "CUT6"

    # --- RULE CUT‑7: Parotid specimen distant recurrence → stage IV ------------
    # --- RULE CUT‑8: Parotid specimen regional → stage III ---------------------
    if PAROTID_REGEX.search(site_coll):
        if _meta_yes(r"skin|ear|eyelid|vulva", "parotid", distant=True):
            return "IV", "CUT7"
        if _meta_yes(r"skin|ear|eyelid|vulva", r"parotid|lymph node", distant=False):
            return "III", "CUT8"

    # --- RULE CUT‑9: Distant cutaneous recurrence → stage IV -------------------
    # --- RULE CUT‑10: Regional cutaneous recurrence → stage III ---------------
    cut_site_pat = r"skin|ear|eyelid|head|soft tissues|muscle|chest wall|vulva"
    if _contains(site_coll, cut_site_pat) and primary_met == "metastatic" and "IV" not in path_stage:
        if _meta_yes(r"skin|ear|eyelid|vulva", cut_site_pat, distant=True) or _meta_yes(
            r"skin|ear|eyelid|vulva", r".*", distant=True
        ):
            return "IV", "CUT9"
        return "III", "CUT10"

    # --- RULE CUT‑11: Primary specimen after interval distant mets → stage IV --
    if primary_met == "primary" and _contains(site_coll, cut_site_pat) and _meta_yes(
        r"skin|ear|eyelid|vulva", r".*", distant=True
    ):
        return "IV", "CUT11"

    # --- RULE CUT‑12: Fallback to numeric stage -------------------------------
    for src, rule in ((path_stage, "CUT12P"), (clin_stage, "CUT12C")):
        m = re.match(r"([IV]+)", src)
        if m:
            return m.group(1), rule

    # Default unknown
    return "Unknown", "CUT‑UNK"


def stage_ocular(spec: pd.Series, dx: pd.Series, meta_patient: pd.DataFrame) -> Tuple[str, str]:
    path_stage = str(dx.get("PathGroupStage", "")).upper()
    clin_stage = str(dx.get("ClinGroupStage", "")).upper()
    age_spec = _float(spec["Age At Specimen Collection"])

    if "IV" in path_stage:
        return "IV", "OC1"
    if "IV" in clin_stage and path_stage.lower() in UNKNOWN_PATH_STAGE_VALUES:
        return "IV", "OC2"

    # Interval metastatic disease (≤ specimen age OR unknown)
    meta_rows = _filter_meta(meta_patient, age_spec)
    if not meta_rows.empty:
        if not meta_rows[meta_rows["MetastaticDiseaseInd"].str.contains("Yes - Distant", na=False)].empty:
            return "IV", "OC3"
        if not meta_rows[meta_rows["MetastaticDiseaseInd"].str.contains(r"Yes - Regional|Yes - NOS", na=False)].empty:
            return "III", "OC4"

    return "Unknown", "OC‑UNK"


def stage_mucosal(spec: pd.Series, dx: pd.Series, meta_patient: pd.DataFrame) -> Tuple[str, str]:
    path_stage = str(dx.get("PathGroupStage", "")).upper()
    clin_stage = str(dx.get("ClinGroupStage", "")).upper()
    age_spec = _float(spec["Age At Specimen Collection"])

    if "IV" in path_stage:
        return "IV", "MU1"
    if "IV" in clin_stage and path_stage.lower() in UNKNOWN_PATH_STAGE_VALUES:
        return "IV", "MU2"

    if spec["Primary/Met"].lower() == "primary":
        for src, rule in ((path_stage, "MU3P"), (clin_stage, "MU3C")):
            m = re.match(r"([IV]+)", src)
            if m and m.group(1) != "IV":
                return m.group(1), rule

    meta_rows = _filter_meta(meta_patient, age_spec)
    if not meta_rows.empty and not meta_rows[meta_rows["MetastaticDiseaseInd"].str.contains("Yes - Distant", na=False)].empty:
        return "IV", "MU4"

    return "Unknown", "MU‑UNK"


def stage_unknown_primary(spec, dx, meta) -> Tuple[str, str]:
    path_stage = str(dx.get("PathGroupStage", "")).upper()
    clin_stage = str(dx.get("ClinGroupStage", "")).upper()
    if "IV" in path_stage:
        return "IV", "UN1"
    if "IV" in clin_stage and path_stage.lower() in UNKNOWN_PATH_STAGE_VALUES:
        return "IV", "UN2"
    return "Unknown", "UN‑UNK"

################################################################################
# DRIVER
################################################################################

def main():
    parser = argparse.ArgumentParser(description = "Pair melanoma tumours with stages.")
    parser.add_argument("--clinmol", required = True, type = Path)
    parser.add_argument("--diagnosis", required = True, type = Path)
    parser.add_argument("--metadisease", required = True, type = Path)
    parser.add_argument("--out", required = True, type = Path)
    args = parser.parse_args()

    logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")

    cm, dx, md = load_inputs(args.clinmol, args.diagnosis, args.metadisease)
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

        output_rows.append(
            {
                "AvatarKey": avatar,
                "ORIENSpecimenID": spec_row["ORIENSpecimenID"],
                "DiagnosisIndex": int(diag_row.name), # index of row in table Diagnosis for traceability
                "AgeAtSpecimenCollection": spec_row["Age At Specimen Collection"],
                "AssignedPrimarySite": primary_site,
                "AssignedStage": stage,
                "StageRuleHit": rule,
            }
        )

    out_df = pd.DataFrame(output_rows)
    logging.info("Writing %s rows → %s", len(out_df), args.out)
    out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()