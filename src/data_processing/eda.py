'''
`eda.py` is a module that generates various CSV files and a plot.

Usage:
./miniconda3/envs/ici_sex/bin/python -m src.data_processing.eda
'''

from __future__ import annotations

from pathlib import Path
import argparse
from collections import defaultdict
import logging
import numpy as np
import pandas as pd
import re
from typing import Dict, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-v0_8-whitegrid")

from src.data_processing.utils import create_map_from_qc
from src.config import paths


logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


ICB_LOOKUP = {
    **dict.fromkeys(["PEMBROLIZUMAB", "NIVOLUMAB", "CEMIPLIMAB", "SINTILIMAB"], "PD‑1"),
    **dict.fromkeys(["ATEZOLIZUMAB", "AVELUMAB", "DURVALUMAB"], "PD‑L1"),
    **dict.fromkeys(["IPILIMUMAB", "TREMELIMUMAB"], "CTLA‑4"),
    "RELATLIMAB": "LAG3",
}
ICB_DRUGS_FLAT = {k.lower() for k in ICB_LOOKUP}

MELANOMA_CODES = {
    "8720", "8721", "8722", "8723", "8728", "8730", "8740", "8741", "8742", "8743",
    "8744", "8745", "8746", "8761", "8770", "8771", "8772", "8773", "8774", "8780"
}

STAGE_MAP = {
    "0": "Stage 0", "I": "Stage I", "II": "Stage II", "III": "Stage III", "IV": "Stage IV",
    "IA": "Stage IA", "IB": "Stage IB",
    "IIA": "Stage IIA", "IIB": "Stage IIB", "IIC": "Stage IIC",
    "IIIA": "Stage IIIA", "IIIB": "Stage IIIB", "IIIC": "Stage IIIC", "IIID": "Stage IIID",
    "IVA": "Stage IVA", "IVB": "Stage IVB", "IVC": "Stage IVC"
}

STRICT_MEL_CODES = {"8720", "8721", "8730", "8740", "8742", "8761", "8771", "8772"}


def classify_icb(med):
    if isinstance(med, str):
        med_l = med.lower()
        return next((d for d in ICB_DRUGS_FLAT if d in med_l), None)
    return None


def clean_stage(raw):
    if pd.isna(raw):
        return "Unknown"
    s = re.sub(r"[^A-Za-z0-9]", "", str(raw).upper()).removeprefix("STAGE")
    return STAGE_MAP.get(s, "Unknown")


def convert_age(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    if "90" in s and "older" in s:
        return 90.0
    return float(s) if re.fullmatch(r"\d+(?:\.\d+)?", s) else np.nan


def get_stage_at_icb(patient_id: str, icb_start_age: float | np.nan, melanoma_diag_df: pd.DataFrame) -> str:
    """
    Pick the melanoma stage that best represents a patient at ICB start.
    """
    if pd.isna(icb_start_age):
        return "Unknown"

    patient = melanoma_diag_df[melanoma_diag_df['PATIENT_ID'] == patient_id].copy()
    if patient.empty:
        return "Unknown"

    patient['DiagAge'] = patient['AgeAtDiagnosis'].apply(convert_age)
    patient['CleanPathStage'] = patient['PathGroupStage'].apply(clean_stage)
    patient['CleanClinStage'] = patient['ClinGroupStage'].apply(clean_stage)

    # diagnoses on / before ICB
    relevant = (patient[patient['DiagAge'] <= icb_start_age].sort_values('DiagAge', ascending = False))

    # 1. most-recent diagnosis ≤ ICB
    scan = relevant if not relevant.empty else patient.sort_values('DiagAge')
    for _, row in scan.iterrows():
        if row['CleanPathStage'] != "Unknown":
            return row['CleanPathStage']
        if row['CleanClinStage'] != "Unknown":
            return row['CleanClinStage']

    return "Unknown"


def is_melanoma_base(code) -> bool:
    """Return True if the first four digits of the provided code match a melanoma code. Ignore behavior."""
    if pd.isna(code):
        return False
    base = str(code).split('/')[0][:4]
    return base in MELANOMA_CODES


def is_melanoma_hist(code) -> bool:
    """
    Return True for melanoma histology codes with malignant behavior; i.e., behavior 3.
    """
    if pd.isna(code):
        return False
    m = re.match(r"^(\d{5})", str(code).strip())
    if not m:
        return False
    prefix = m.group(1)
    base, beh = prefix[:4], prefix[4]
    return (base in STRICT_MEL_CODES) and (beh == "3")


def load_csvs():
    patterns = {
        "patients": ["PatientMaster"],
        "diagnoses": ["Diagnosis"],
        "treatments": ["Medications"],
        "clinical_mol_linkage": ["ClinicalMolLinkage"],
        "outcomes": ["Outcomes"],
        "vital_status": ["VitalStatus"],
    }
    out: Dict[str, pd.DataFrame] = {}
    for key, pats in patterns.items():
        file = next((f for f in paths.normalized_clinical_data.glob("*.csv") if any(p.lower() in f.name.lower() for p in pats)), None)
        if file is None:
            logger.warning("No file for %s", key)
            continue
        df = pd.read_csv(file).rename(columns = {"AvatarKey": "PATIENT_ID", "ORIENAvatarKey": "PATIENT_ID"})
        out[key] = df
        logger.info("%s → %d rows", file.name, len(df))
    return out


def process_clinical(dfs):
    req = {"patients", "diagnoses", "treatments", "clinical_mol_linkage"}
    if missing := req - dfs.keys():
        logger.error("Missing data: %s", ", ".join(missing))
        return pd.DataFrame()

    diag = dfs["diagnoses"].assign(IsMelanoma = lambda d: d["HistologyCode"].map(is_melanoma_base))
    mel_diag = diag.query("IsMelanoma")
    logger.info("Melanoma diagnoses: %d", len(mel_diag))

    linkage = dfs["clinical_mol_linkage"]
    mel_linkage = linkage[linkage["Histology/Behavior"].map(is_melanoma_hist, na_action = "ignore")]
    logger.info("Melanoma‐linked sequencing rows: %d", len(mel_linkage))
    pts = pd.Index(mel_diag["PATIENT_ID"]).intersection(mel_linkage["PATIENT_ID"])
    logger.info("Cohort size after intersection with melanoma linkage: %d", len(pts))

    patients = dfs["patients"].query("PATIENT_ID in @pts").assign(AgeAtClinicalRecordCreation = lambda d: d["AgeAtClinicalRecordCreation"].map(convert_age))

    diag_clean = mel_diag.assign(
        AgeAtDiagnosis = lambda d: d["AgeAtDiagnosis"].map(convert_age),
        ClinStage = lambda d: d["ClinGroupStage"].map(clean_stage),
        PathStage = lambda d: d["PathGroupStage"].map(clean_stage),
    )

    diag_agg = diag_clean.groupby("PATIENT_ID").agg(
        MelanomaHistologyCodes = ("HistologyCode", lambda x: list(x.unique())),
        EarliestMelanomaDiagnosisAge = ("AgeAtDiagnosis", "min"),
        MelanomaClinStages = ("ClinStage", lambda x: list(x.unique())),
        MelanomaPathStages = ("PathStage", lambda x: list(x.unique())),
    ).reset_index()
    patients = patients.merge(diag_agg, on = "PATIENT_ID", how = "left")

    treat = dfs["treatments"].query("PATIENT_ID in @pts")
    med_col = next((c for c in ["Medication", "MedicationName", "DrugName", "TreatmentName"] if c in treat), None)
    if med_col is None:
        patients["HAS_ICB"] = 0
        patients["ICB_START_AGE"] = np.nan
    else:
        icb = treat.assign(ICB = lambda d: d[med_col].map(classify_icb)).dropna(subset = ["ICB"]).assign(StartAge = lambda d: d["AgeAtMedStart"].map(convert_age))
        icb_agg = icb.sort_values("StartAge").groupby("PATIENT_ID").agg(
            ICB_Treatments = (med_col, list),
            ICB_START_AGE = ("StartAge", "first")
        )
        patients = patients.merge(icb_agg, on = "PATIENT_ID", how = "left")
        patients["HAS_ICB"] = patients["ICB_START_AGE"].notna().astype(int)

    patients["STAGE_AT_ICB"] = patients.apply(
        lambda r: get_stage_at_icb(r.PATIENT_ID, r.get("ICB_START_AGE"), diag_clean),
        axis = 1
    )

    patients["REFERENCE_AGE"] = patients["ICB_START_AGE"].fillna(patients["EarliestMelanomaDiagnosisAge"])

    if "outcomes" in dfs:
        outcomes = dfs["outcomes"].query("PATIENT_ID in @pts").copy()
        outcomes["AgeAtCurrentDiseaseStatus"] = outcomes["AgeAtCurrentDiseaseStatus"].map(convert_age)
        outcomes = outcomes.sort_values("AgeAtCurrentDiseaseStatus", ascending = False).groupby("PATIENT_ID").first().reset_index()
        patients = patients.merge(
            outcomes[["PATIENT_ID", "CurrentDiseaseStatus", "AgeAtCurrentDiseaseStatus"]],
            on = "PATIENT_ID",
            how = "left"
        )
    else:
        logger.warning("No outcomes data; skipping CurrentDiseaseStatus & AgeAtCurrentDiseaseStatus.")

    if "vital_status" in dfs:
        vital = dfs["vital_status"].query("PATIENT_ID in @pts").groupby("PATIENT_ID").first().reset_index()
        patients = patients.merge(
            vital[["PATIENT_ID", "VitalStatus", "AgeAtLastContact"]],
            on = "PATIENT_ID",
            how = "left"
        )
        patients["AgeAtLastContact"] = patients["AgeAtLastContact"].map(convert_age)
        vital_status = patients["VitalStatus"].str.upper().str.strip()
        status_map = {"DECEASED": 1, "LIVING": 0, "DEAD": 1, "ALIVE": 0}
        patients["OS_STATUS"] = vital_status.map(status_map).fillna(0).astype(int)
        patients["OS_TIME"] = (patients["AgeAtLastContact"] - patients["EarliestMelanomaDiagnosisAge"]) * 12
        patients.loc[patients["OS_TIME"] < 0, "OS_TIME"] = np.nan
    else:
        logger.warning("No vital status data; skipping OS_STATUS & OS_TIME.")

    mel_samples = mel_linkage.query("PATIENT_ID in @pts")
    patient_sequencing: dict[str, list[dict]] = defaultdict(list)
    for _, row in mel_samples.iterrows():
        pid = row["PATIENT_ID"]
        age = convert_age(row.get("Age At Specimen Collection", np.nan))
        site = row.get("SpecimenSiteOfOrigin", "Unknown")
        hist = row.get("Histology/Behavior", "Unknown")
        sample_type = "Tumor"
        if pd.notna(row.get("WES")) and str(row["WES"]).strip():
            patient_sequencing[pid].append({
                "type": "WES",
                "id": row["WES"].strip(),
                "age": age,
                "site": site,
                "histology": hist,
                "sample_type": sample_type
            })
        if pd.notna(row.get("RNASeq")) and str(row["RNASeq"]).strip():
            patient_sequencing[pid].append({
                "type": "RNASeq",
                "id": row["RNASeq"].strip(),
                "age": age,
                "site": site,
                "histology": hist,
                "sample_type": sample_type
            })

    earliest = {pid: min([s['age'] for s in seqs if pd.notna(s['age'])] or [np.nan]) for pid, seqs in patient_sequencing.items()}
    es = pd.Series(earliest, name = "EarliestSequencingAge").rename_axis("PATIENT_ID").reset_index()
    patients = patients.merge(es, on = "PATIENT_ID", how = "left")
            
    patients["MelanomaSequencingSamples"] = patients["PATIENT_ID"].map(
        lambda pid: str([f"{s['type']}:{s['id']}" for s in patient_sequencing.get(pid, [])])
    )
    patients["SequencingAges"] = patients["PATIENT_ID"].map(
        lambda pid: str([s["age"] for s in patient_sequencing.get(pid, [])])
    )
    
    def get_sequencing_before_icb(row):
        try:
            ages = eval(row["SequencingAges"])
        except:
            ages = []
        return str(["No ICB"] * len(ages))

    patients["SequencingBeforeICB"] = patients.apply(get_sequencing_before_icb, axis = 1)
    
    patients["SequencingSites"] = patients["PATIENT_ID"].map(
        lambda pid: str([s["site"] for s in patient_sequencing.get(pid, [])])
    )
    patients["SequencingHistologies"] = patients["PATIENT_ID"].map(
        lambda pid: str([s["histology"] for s in patient_sequencing.get(pid, [])])
    )
    patients["SequencingSampleTypes"] = patients["PATIENT_ID"].map(
        lambda pid: str([s["sample_type"] for s in patient_sequencing.get(pid, [])])
    )

    return patients


def main():
    
    paths.ensure_dependencies_for_eda_exist()
    
    dfs = load_csvs()
    print("The following CSVs were loaded.")
    print([type_of_data for type_of_data in dfs.keys()])
    
    df = process_clinical(dfs)

    if df.empty:
        logger.error("No data processed; aborting write.")
        return
    
    for col in ['MelanomaHistologyCodes', 'MelanomaClinStages', 'MelanomaPathStages', 'ICB_Treatments']:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, list) else str([] if pd.isna(x) else x)
            )

    id_map = create_map_from_qc(paths.QC_data, sample_col = None, patient_col = None)
    if id_map:
        map_df = pd.DataFrame(id_map.items(), columns = ["SampleID", "PatientID"])
        map_df.to_csv(paths.map_from_sample_to_patient, index = False)
        logger.info("Saved %d mappings → %s", len(map_df), paths.map_from_sample_to_patient)
    else:
        logger.warning("QC file parsed but returned an empty mapping – no sample_to_patient_map.csv written")
    
    print(f"Data frame will be saved to {paths.data_frame_of_melanoma_patient_and_sequencing_data}.")
    df.to_csv(paths.data_frame_of_melanoma_patient_and_sequencing_data, index = False)
    logger.info("Saved %d patients → %s", len(df), paths.data_frame_of_melanoma_patient_and_sequencing_data)
    
    summary_stats = df.describe(include = "all")
    summary_stats.to_csv(paths.eda_summary_statistics)
    logger.info("Saved summary statistics → %s", paths.eda_summary_statistics)

    if "HAS_ICB" in df.columns:
        icb_dist = df["HAS_ICB"].value_counts().rename_axis("HAS_ICB").reset_index(name = "count").sort_values("HAS_ICB", ascending = False)
        icb_dist.to_csv(paths.eda_icb_distribution, index = False)
        logger.info("Saved ICB distribution → %s", paths.eda_icb_distribution)
    else:
        logger.warning("'HAS_ICB' column missing – icb_distribution.csv not generated.")
    
    if "Sex" in df.columns and not df["Sex"].isna().all():
        plt.figure(figsize = (8, 6))
        sns.countplot(data = df, x = "Sex")
        plt.title("Sex Distribution")
        plt.xlabel("Sex")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(paths.eda_sex_distribution)
        plt.close()
        logger.info("Saved sex distribution plot → %s", paths.eda_sex_distribution)
    else:
        logger.warning("No 'Sex' column or all values are NaN - sex distribution.png not generated.")


if __name__ == "__main__":
    main()