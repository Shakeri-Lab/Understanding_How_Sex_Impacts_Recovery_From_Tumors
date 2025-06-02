#!/usr/bin/env python3

'''
Tumor‑Stage Pairing Pipeline
=================================
Pairs melanoma clinical diagnoses with sequenced tumor specimens and assigns an
AJCC stage (I, II, III, IV) at the time of specimen collection, exactly as
described in "ORIEN Data Rules for Tumor Clinical Pairing".

TODO: What is a medical clinical diagnosis?
TODO: How can a medical clinical diagnosis be paired?
TODO: What is a sequenced tumor specimen?
TODO: How can a sequenced tumor specimen be paired?
TODO: What is an AJCC stage?

This single script
    1.  reads Clinical Molecular Linkage, Diagnosis, and Metastatic Disease CSV files;
    2.  identifies melanoma diagnoses (ICD‑O‑3 codes) and tumor specimens;
        TODO: How does this script identify melanoma diagnoses?
        TODO: How does this script identify tumor specimens?
    3.  derives per‑patient summary metrics (MelanomaDiagnosisCount /
        SequencedTumorCount) → Group A/B/C/D;
        TODO: Change "(MelanomaDiagnosisCount /
        SequencedTumorCount) → Group A/B/C/D." into a sentence or part of a sentence.
    4.  reduces each patient to one (specimen, diagnosis) pair using the
        selection logic for that patient's group;
        TODO: What is a specimen?
        TODO: What is a diagnosis?
    5.  assigns an `AssignedPrimarySite`;
    6.  assigns an `AssignedStage` using rule sets that are specific to cutaneous, ocular, mucosal, or unknown primary site; and
    7.  emits one row per paired specimen with traceability fields plus a code (StageRuleHit) identifying which rule produced the stage.
        TODO: What is a paired specimen?
        TODO: What is a traceability field?
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

MELANOMA_HISTOLOGY_CODES: set[str] = {
    # Cutaneous / NOS codes
    # TODO: Why are Not Otherwise Specified codes included?
    # TODO: Should unknown codes be included?
    "8720/3", "8721/3", "8722/3", "8723/3", "8728/3", "8730/3",
    "8740/3", "8741/3", "8742/3", "8743/3", "8744/3", "8745/3",
    # Acral / mucosal / ocular sub‑types
    # TODO: Are sub-types codes?
    # TODO: Why are acral codes included?
    "8761/3", "8770/3", "8771/3", "8772/3", "8773/3", "8774/3", "8780/3"
}

NODE_REGEX = re.compile(r"lymph node", re.I) # regular expression object

PAROTID_REGEX = re.compile(r"parotid", re.I) # regular expression object

SITE_KEYWORDS = {
    "cutaneous": r"skin|ear|eyelid|vulva",
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
    Return A/B/C/D according to counts.
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
    Return a single specimen row for a Group B/D patient per rules.
    '''
    cm = patient_cm.copy()
    # 1. Keep only RNA‑seq‑enabled tumours if possible
    if cm["RNASeq"].notna().any():
        cm = cm[cm["RNASeq"].notna()]
        if len(cm) == 1:
            return cm.iloc[0]

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


def _hist_clean(txt) -> str:
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
            if site_match.any():
                return dxp[prox & site_match].iloc[0]

        # Histology match
        hist_spec = _hist_clean(spec_row["Histology/Behavior"])
        dxp["_hist_clean"] = dxp["Histology"].apply(_hist_clean)
        hist_match = dxp["_hist_clean"] == hist_spec
        if hist_match.any():
            return dxp[hist_match].iloc[0]

    else:  # metastatic specimen pathway
        site_coll = spec_row["SpecimenSiteOfCollection"].lower()
        if not re.search(r"soft tissues|lymph node", site_coll):
            return dxp.sort_values("AgeAtDiagnosis").iloc[0]

        if "soft tissues" in site_coll and _within_90_days(
            age_spec, _float(dxp["AgeAtDiagnosis"].iloc[0])
        ):
            prox = dxp["AgeAtDiagnosis"].apply(_float).apply(lambda x: _within_90_days(age_spec, x))
            if prox.any():
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
        _contains(site_coll, r"skin|lymph node|soft tissues|muscle|parotid|chest wall|head")
    ):
        return "IV", "CUT3"

    # Lymph node helper booleans
    is_node_spec = bool(NODE_REGEX.search(site_coll))

    # Gather meta‑disease snippets for quick filters
    meta_rows = meta_patient if meta_patient is not None else pd.DataFrame()

    def _meta_yes(kind_pat: str, site_pat: str, distant: bool | None = None):
        if meta_rows.empty:
            return False
        ok = meta_rows["MetsDzPrimaryDiagnosisSite"].str.contains(kind_pat, case = False, na = False)
        ok &= meta_rows["MetastaticDiseaseSite"].str.contains(site_pat, case = False, na = False)
        if distant is not None:
            ind_pat = "Yes - Distant" if distant else r"Yes - Regional|Yes - NOS"
            ok &= meta_rows["MetastaticDiseaseInd"].str.contains(ind_pat, case = False, na = False)
        return ok.any()

    # --- RULE CUT‑4: Node specimen, stage III at diagnosis ---------------------
    if (
        is_node_spec
        and _meta_yes(r"skin|ear|eyelid|vulva", "lymph node", distant = False)
        and "III" in path_stage
    ):
        return "III", "CUT4"

    # --- RULE CUT‑5: Node specimen, early stage at diagnosis → now stage III ---
    if is_node_spec and not any(s in path_stage for s in ("III", "IV")) and "IV" not in clin_stage:
        return "III", "CUT5"

    # --- RULE CUT‑6: Node specimen, distant nodal recurrence → stage IV --------
    if (
        is_node_spec
        and _meta_yes(r"skin|ear|eyelid|vulva", "lymph node", distant = True)
        and "IV" not in path_stage
    ):
        return "IV", "CUT6"

    # --- RULE CUT‑7: Parotid specimen distant recurrence → stage IV ------------
    if PAROTID_REGEX.search(site_coll) and _meta_yes(r"skin|ear|eyelid|vulva", "parotid", distant = True):
        return "IV", "CUT7"

    # --- RULE CUT‑8: Parotid specimen regional → stage III ---------------------
    if PAROTID_REGEX.search(site_coll) and _meta_yes(r"skin|ear|eyelid|vulva", r"parotid|lymph node", distant = False):
        return "III", "CUT8"

    # --- RULE CUT‑9: Distant cutaneous recurrence → stage IV -------------------
    if (
        _contains(site_coll, r"skin|ear|eyelid|head|soft tissues|muscle|chest wall|vulva")
        and primary_met == "metastatic"
        and "IV" not in path_stage
        and (
            _meta_yes(r"skin|ear|eyelid|vulva", r"skin|ear|eyelid|head|soft tissues|muscle|chest wall", distant = True)
            or _meta_yes(r"skin|ear|eyelid|vulva", r".*", distant = True)
        )
    ):
        return "IV", "CUT9"

    # --- RULE CUT‑10: Regional cutaneous recurrence → stage III ---------------
    if (
        _contains(site_coll, r"skin|ear|eyelid|head|soft tissues|muscle|chest wall|vulva")
        and primary_met == "metastatic"
        and "IV" not in path_stage
    ):
        return "III", "CUT10"

    # --- RULE CUT‑11: Primary specimen after interval distant mets → stage IV --
    if (
        primary_met == "primary"
        and _contains(site_coll, r"skin|ear|eyelid|head|soft tissues|muscle|chest wall|vulva")
        and _meta_yes(r"skin|ear|eyelid|vulva", r".*", distant = True)
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
    site_coll = spec["SpecimenSiteOfOrigin"].lower() if pd.notna(spec["SpecimenSiteOfOrigin"]) else ""

    if "IV" in path_stage:
        return "IV", "OC1"
    if "IV" in clin_stage and path_stage.lower() in UNKNOWN_PATH_STAGE_VALUES:
        return "IV", "OC2"

    # Interval development of distant disease → stage IV
    if meta_patient is not None and not meta_patient.empty:
        distant = meta_patient[
            meta_patient["MetastaticDiseaseInd"].str.contains("Yes - Distant", na = False)
        ]
        for _, row in distant.iterrows():
            age_met = _float(row["AgeAtMetastaticSite"])
            age_spec = _float(spec["Age At Specimen Collection"])
            if age_met is None or age_spec is None:
                continue
            if age_met <= age_spec:
                return "IV", "OC3"

        regional = meta_patient[
            meta_patient["MetastaticDiseaseInd"].str.contains(r"Yes - Regional|Yes - NOS", na = False)
        ]
        for _, row in regional.iterrows():
            age_met = _float(row["AgeAtMetastaticSite"])
            age_spec = _float(spec["Age At Specimen Collection"])
            if age_met is None or age_spec is None:
                continue
            if age_met <= age_spec:
                return "III", "OC4"

    return "Unknown", "OC‑UNK"


def stage_mucosal(spec: pd.Series, dx: pd.Series, meta_patient: pd.DataFrame) -> Tuple[str, str]:
    path_stage = str(dx.get("PathGroupStage", "")).upper()
    clin_stage = str(dx.get("ClinGroupStage", "")).upper()

    if "IV" in path_stage:
        return "IV", "MU1"
    if "IV" in clin_stage and path_stage.lower() in UNKNOWN_PATH_STAGE_VALUES:
        return "IV", "MU2"

    if spec["Primary/Met"].lower() == "primary":
        for src, rule in ((path_stage, "MU3P"), (clin_stage, "MU3C")):
            m = re.match(r"([IV]+)", src)
            if m and m.group(1) != "IV":
                return m.group(1), rule

    if meta_patient is not None and not meta_patient.empty:
        distant = meta_patient[meta_patient["MetastaticDiseaseInd"].str.contains("Yes - Distant", na = False)]
        for _, row in distant.iterrows():
            age_met = _float(row["AgeAtMetastaticSite"])
            age_spec = _float(spec["Age At Specimen Collection"])
            if age_met is None or age_spec is None:
                continue
            if age_met <= age_spec:
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
                "Skipping AvatarKey %s: %d tumour specimen(s) but no "
                "melanoma Diagnosis rows after filtering – unable to pair.",
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
            if group == "B":
                diag_row = dx_patient.iloc[0]
            else:  # D
                diag_row = _select_diagnosis_C(dx_patient, spec_row, meta_patient)
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