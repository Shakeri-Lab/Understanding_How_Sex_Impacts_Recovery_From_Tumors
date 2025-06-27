#!/usr/bin/env python3

'''
`pipeline_for_pairing_clinical_data_and_stages_of_tumors.py`

This module is a pipeline implementing "ORIEN Specimen Staging Revised Rules" for pairing patients' melanoma tumor specimens with the appropriate primary diagnosis site, patient grouping, AJCC stage, and rule.

From "ORIEN Specimen Staging Revised Rules":
1. Using data  from:
    - Molecular Linkage file (main file)
    - Diagnosis file (for primary site and stage info)
    - Metastatic Disease (to help assign stage)

Usage:
python pipeline_for_pairing_clinical_data_and_stages_of_tumors.py --path_to_clinical_molecular_linkage_data ../../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv --path_to_diagnosis_data ../../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_Diagnosis_V4.csv --path_to_metastatic_disease_data ../../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_MetastaticDisease_V4.csv --path_to_output_data output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv > output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.txt 2>&1

From "ORIEN Specimen Staging Revised Rules":
2. Definitions
    - Melanoma diagnosis (Diagnosis file): HistologyCode = {list of melanoma codes previously sent}
    - Tumor sequenced (Molecular Linkage file): Tumor/Germline variable = Tumor

A melanoma diagnosis is a row in `24PRJ217UVA_20241112_Diagnosis_V4.csv` with a histology code of the form 87<digit><digit>/<digit>.
A sequenced tumor is a row in `24PRJ217UVA_20241112_MetastaticDisease_V4.csv` with a value of "Tumor" in column with label "Tumor/Germline".
'''

import argparse
import logging
import pandas as pd
from pathlib import Path
import re
from typing import Optional, Tuple


def add_counts(data_frame_of_clinical_molecular_linkage_data: pd.DataFrame, data_frame_of_diagnosis_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    
    series_of_numbers_of_sequenced_tumor_samples = data_frame_of_clinical_molecular_linkage_data.drop_duplicates(["ORIENAvatarKey", "DeidSpecimenID"]).groupby("ORIENAvatarKey").size().rename("SequencedTumorCount")
    
    series_of_numbers_of_melanoma_clinical_diagnoses = data_frame_of_diagnosis_data.assign(_age = data_frame_of_diagnosis_data["AgeAtDiagnosis"].fillna("NA")).drop_duplicates(subset = ["AvatarKey", "_age", "PrimaryDiagnosisSite"]).groupby("AvatarKey").size().rename("MelanomaDiagnosisCount")
    
    # Logic that uses both number of diagnoses and number of samples is executed while iterating over rows in data_frame_of_clinical_molecular_linkage_data.
    data_frame_of_clinical_molecular_linkage_data = data_frame_of_clinical_molecular_linkage_data.join(series_of_numbers_of_sequenced_tumor_samples, on = "ORIENAvatarKey")
    
    data_frame_of_clinical_molecular_linkage_data = data_frame_of_clinical_molecular_linkage_data.join(series_of_numbers_of_melanoma_clinical_diagnoses, on = "ORIENAvatarKey")
    
    data_frame_of_clinical_molecular_linkage_data[["MelanomaDiagnosisCount", "SequencedTumorCount"]] = data_frame_of_clinical_molecular_linkage_data[["MelanomaDiagnosisCount", "SequencedTumorCount"]].fillna(0)
    
    return data_frame_of_clinical_molecular_linkage_data


'''
From "ORIEN Specimen Staging Revised Rules":
4. AssignedPrimarySite (SAME DEFINITIONS AS BEFORE)
    - IF PrimaryDiagnosisSite contains "skin" OR "ear" OR "eyelid" OR "vulva", THEN AssignedPrimarySite = cutaneous
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


def assign_primary_site(primary_diagnosis_site: str) -> str:
    lowercase_primary_diagnosis_site = str(primary_diagnosis_site).lower()
    for primary_site, pattern in SITE_KEYWORDS.items():
        if re.search(pattern, lowercase_primary_diagnosis_site):
            return primary_site
    raise ValueError(f"A primary site could not be assigned for primary diagnosis site {primary_diagnosis_site}.")


def run_pipeline(
    path_to_clinical_molecular_linkage_data: Path,
    path_to_diagnosis_data: Path,
    path_to_metastatic_disease_data: Path
) -> pd.DataFrame:
    
    data_frame_of_clinical_molecular_linkage_data, data_frame_of_diagnosis_data, data_frame_of_metastatic_disease_data = load_data(path_to_clinical_molecular_linkage_data, path_to_diagnosis_data, path_to_metastatic_disease_data)
    
    data_frame_of_clinical_molecular_linkage_data = add_counts(data_frame_of_clinical_molecular_linkage_data, data_frame_of_diagnosis_data)

    '''
    From "ORIEN Specimen Staging Revised Rules":
    5. AssignedGroup (SAME DEFINITIONS AS BEFORE)
        - Group A: MelanomaDiagnosisCount = 1 AND SequencedTumorCount = 1 -> n = 327
        - Group B: MelanomaDiagnosisCount = 1 AND SequencedTumorCount > 1 -> n = 19
        - Group C: MelanomaDiagnosisCount > 1 AND SequencedTumorCount = 1 -> n = 30
        - Group D: MelanomaDiagnosisCount > 1 AND SequencedTumorCount > 1 -> n=3
    '''
    data_frame_of_clinical_molecular_linkage_data["Group"] = data_frame_of_clinical_molecular_linkage_data.apply(assign_patient_to_group, axis = 1)
    data_frame_of_first_rows_for_patients = data_frame_of_clinical_molecular_linkage_data.groupby("ORIENAvatarKey").first()
    number_of_patients_in_A = (data_frame_of_first_rows_for_patients["Group"] == 'A').sum()
    number_of_patients_in_B = (data_frame_of_first_rows_for_patients["Group"] == 'B').sum()
    number_of_patients_in_C = (data_frame_of_first_rows_for_patients["Group"] == 'C').sum()
    number_of_patients_in_D = (data_frame_of_first_rows_for_patients["Group"] == 'D').sum()
    assert number_of_patients_in_A == 327, f"Number of patients in A was {number_of_patients_in_A} and should be 327."
    assert number_of_patients_in_B == 19, f"Number of patients in B was {number_of_patients_in_B} and should be 19."
    assert number_of_patients_in_C == 30, f"Number of patients in C was {number_of_patients_in_C} and should be 30."
    assert number_of_patients_in_D == 3, f"Number of patients in D was {count_D} and should be 3."

    list_of_dictionaries_representing_rows_in_output_data: List[Dict[str, str]] = []

    # An ORIEN Avatar Key identifies a patient.
    for ORIENAvatarKey, data_frame_of_clinical_molecular_linkage_data_for_patient in data_frame_of_clinical_molecular_linkage_data.groupby("ORIENAvatarKey", sort = False):
        data_frame_of_diagnosis_data_for_patient = data_frame_of_diagnosis_data[data_frame_of_diagnosis_data["AvatarKey"] == ORIENAvatarKey]
        data_frame_of_metastatic_disease_data_for_patient = data_frame_of_metastatic_disease_data[data_frame_of_metastatic_disease_data["AvatarKey"] == ORIENAvatarKey]
        group = data_frame_of_clinical_molecular_linkage_data_for_patient["Group"].iloc[0]

        if group == "A":
            series_of_clinical_molecular_linkage_data_for_patient = data_frame_of_clinical_molecular_linkage_data_for_patient.iloc[0]
            series_of_diagnosis_data_for_patient = data_frame_of_diagnosis_data_for_patient.iloc[0]
        elif group == "B":
            series_of_clinical_molecular_linkage_data_for_patient = select_tumor_for_patient_in_B(data_frame_of_clinical_molecular_linkage_data_for_patient)
            series_of_diagnosis_data_for_patient = data_frame_of_diagnosis_data_for_patient.iloc[0]
        elif group == "C":
            series_of_clinical_molecular_linkage_data_for_patient = data_frame_of_clinical_molecular_linkage_data_for_patient.iloc[0]
            series_of_diagnosis_data_for_patient = select_diagnosis_for_patient_in_C(data_frame_of_diagnosis_data_for_patient, series_of_clinical_molecular_linkage_data_for_patient, data_frame_of_metastatic_disease_data_for_patient)
        elif group == "D":
            series_of_clinical_molecular_linkage_data_for_patient = select_tumor_for_patient_in_D(data_frame_of_clinical_molecular_linkage_data_for_patient)
            series_of_diagnosis_data_for_patient = select_diagnosis_for_patient_in_D(data_frame_of_diagnosis_data_for_patient, series_of_clinical_molecular_linkage_data_for_patient)
        else:
            raise RuntimeError(f"Group {group} is unknown.")

        ORIENSpecimenID = series_of_clinical_molecular_linkage_data_for_patient["DeidSpecimenID"]
        primary_site = assign_primary_site(series_of_diagnosis_data_for_patient["PrimaryDiagnosisSite"])
        
        '''
        From "ORIEN Specimen Staging Revised Rules":
        9. Additional Fields for Stage Assignments 
            a. Field Name: Discrepancy (0/1 or No/Yes) 
        This is to keep track of all the questionable ones that I reviewed with Dr. Slingluff regardless of whether we made an exception to the rule for staging or kept with the staging assigned by the rule. These will be identified under each rule, but are summarized here since I imagine this will be coded separately from the rules (this field doesn’t impact the rules/exceptions).
            - Discrepancy = 1 if the AvatarKey = ["087FO3NF65" or "227RDTKST8" or "2AP9EDU231" or "2X7USSLPJC" or "6DL054517A" or "6HWEJIP63S" or "7YX8AJLMWR" or "87AJ4KITK8" or "9DLKDVIQ2W" or "9EYYI5H9SU" or "9HA9MZCSU2" or "A594OFU98I" or "AC2EJBKWJO" or "APGLZDFLYJ" or "DTUPUJ06B5" or "EKZGU61JTP" or "FUAZTE7LVQ" or "GEM0S42KIH" or "GITAF8OSTV" or "HTKAEZOC7V" or "HZD0O858UJ" or "IALUL9JC9Y" or "ILE2DL0KMW" or "KEHK6YTVAK" or "L2R9RJJ88C" or "MD5OTA3E8A" or "MPHAPLR8K1" or "N5Q9122LTG" or "NXPOH3RBWY" or "QC7QX0VWAY" or "QE43I70DGC" or "QLWU5QNQIB" or "TIZXXXVCV9" or "X9AZUY3R1C" or "XHTXLE3MLC" or "XPZE95IE7I" or "YMC959TA29" or "YRGE6MVYNK" or "Z7CEUA8SAJ"]
            - For all other AvatarKey IDs not specified, Discrepancy = 0
        '''
        
        discrepancy = 0
        if ORIENAvatarKey in [
            "087FO3NF65",
            "227RDTKST8",
            "2AP9EDU231",
            "2X7USSLPJC",
            "6DL054517A",
            "6HWEJIP63S",
            "7YX8AJLMWR",
            "87AJ4KITK8",
            "9DLKDVIQ2W",
            "9EYYI5H9SU",
            "9HA9MZCSU2",
            "A594OFU98I",
            "AC2EJBKWJO",
            "APGLZDFLYJ",
            "DTUPUJ06B5",
            "EKZGU61JTP",
            "FUAZTE7LVQ",
            "GEM0S42KIH",
            "GITAF8OSTV",
            "HTKAEZOC7V",
            "HZD0O858UJ",
            "IALUL9JC9Y",
            "ILE2DL0KMW",
            "KEHK6YTVAK",
            "L2R9RJJ88C",
            "MD5OTA3E8A",
            "MPHAPLR8K1",
            "N5Q9122LTG",
            "NXPOH3RBWY",
            "QC7QX0VWAY",
            "QE43I70DGC",
            "QLWU5QNQIB",
            "TIZXXXVCV9",
            "X9AZUY3R1C",
            "XHTXLE3MLC",
            "XPZE95IE7I",
            "YMC959TA29",
            "YRGE6MVYNK",
            "Z7CEUA8SAJ"
        ]:
            discrepancy = 1
        
        '''
        9.b. Field Name: Possible New Primary (0/1 or No/Yes) 
        This is to keep track of the ones that may be new primary melanomas but we do not have an additional diagnosis or other information to definitively know this. The staging for these will be carried out like all the others. These will be identified under each rule, but are summarized here since I imagine this will be coded separately from the rules (this field doesn’t impact the rules/exceptions). 
        - Possible New Primary = 1 if the AvatarKey = ["87AJ4KITK8" or "9DLKDVIQ2W" or "9EYYI5H9SU" or "9HA9MZCSU2" or "A594OFU98I" or "AC2EJBKWJO" or "FUAZTE7LVQ" or "HZD0O858UJ" or "L2R9RJJ88C" or "MD5OTA3E8A" or "NXPOH3RBWY" or "QC7QX0VWAY" or "QE43I70DGC" or "XHTXLE3MLC"] 
        - For all other AvatarKey IDs not specified, Possible New Primary = 0
        '''
        possible_new_primary = 0
        if ORIENAvatarKey in ["87AJ4KITK8", "9DLKDVIQ2W", "9EYYI5H9SU", "9HA9MZCSU2", "A594OFU98I", "AC2EJBKWJO", "FUAZTE7LVQ", "HZD0O858UJ", "L2R9RJJ88C", "MD5OTA3E8A", "NXPOH3RBWY", "QC7QX0VWAY", "QE43I70DGC", "XHTXLE3MLC"]:
            possible_new_primary = 1
        
        stage, rule = assign_stage_and_rule(series_of_clinical_molecular_linkage_data_for_patient, series_of_diagnosis_data_for_patient, data_frame_of_metastatic_disease_data_for_patient)
        
        '''
        From "ORIEN Specimen Staging Revised Rules":
        3.c. AssignedPrimarySite: {cutaneous, ocular, mucosal, unknown}
            - Based on the parameters outlined below; primary site variable to be used for the analysis
        3.d. AssignedStage [EKN Assigned Stage in "ORIEN_Tumor_Staging_Key.csv"]: {I, II, III, IV}
            - Based on the parameters outlined below; stage variable to be used for the analysis
        3.e. AssignedGroup [Group in "ORIEN_Tumor_Staging_Key.csv"]: {A, B, C, D} (NEW - just to keep track of these better)
        '''
        
        list_of_dictionaries_representing_rows_in_output_data.append(
            {
                "AvatarKey": ORIENAvatarKey,
                "ORIENSpecimenID": ORIENSpecimenID,
                "AssignedPrimarySite": primary_site,
                "Group": group,
                "EKN Assigned Stage": stage,
                "NEW RULE": rule,
                "Discrepancy": discrepancy,
                "Possible New Primary": possible_new_primary
            }
        )

    return pd.DataFrame(list_of_dictionaries_representing_rows_in_output_data).sort_values(by = ["AvatarKey", "ORIENSpecimenID"]).reset_index(drop = True)


def assign_patient_to_group(row) -> str:
    
    melanoma_diagnosis_count = row.get("MelanomaDiagnosisCount", 0)
    sequenced_tumor_count = row.get("SequencedTumorCount", 0)
    
    if melanoma_diagnosis_count == 1 and sequenced_tumor_count == 1:
        return "A"
    if melanoma_diagnosis_count == 1 and sequenced_tumor_count > 1:
        return "B"
    if melanoma_diagnosis_count > 1 and sequenced_tumor_count == 1:
        return "C"
    if melanoma_diagnosis_count > 1 and sequenced_tumor_count > 1:
        return "D"
    else:
        raise Exception("Group could not be assigned.")

        
def select_tumor_for_patient_in_B(data_frame_of_clinical_molecular_linkage_data_for_patient: pd.DataFrame) -> pd.Series:
    '''
    From "ORIEN Specimen Staging Revised Rules":
    6. AssignedGroup
    Group B = 1 melanoma diagnosis and >1 tumor sequenced -> n=19
    CHANGE FROM PRIOR: Do not exclude any, can still use those with WES only for TMB analysis. A few clarifications on the rules also in red text below.
    SpecimenSiteofCollection is the correct field 
For the record: these two patients had the wrong SpecimenID in my file but the correct SpecimenID in by your code according to the rules below. Patient ID 317K6G9N41 should have SpecimenID = 53LFUMZSW8ACOX5IPUZ45ICJ0. Patient ID U2CUPQJ4T1 should have SpecimenID = HIWF190182C5JJE3FHYR5BIXR
    A few clarifications on the rules also in red text below.
        - If RNAseq is available for just one tumor, select the tumor with RNAseq data (even if no WES)
        - If RNAseq data is available for > 1 tumors OR if only WES is available for all tumors:
            - If the patient has a tumor with SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains "skin" and does not also have a tumor with SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains either "lymph node" or "soft tissue", then select the tumor[s] with SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains "skin"
                - If multiple skin specimens meet this rule, then use the one with earliest Age At Specimen Collection
            - If none of the patient's tumors have SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains "skin" or "soft tissue", then select the tumor with SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains "lymph node"
            - If a patient has a tumor with a SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains ["skin" OR "soft tissue"] AND a tumor with a SpecimenSiteofCollection that contains "lymph node", then select the one [the tumor with specimen site of collection containing "skin" or "soft tissue" or the tumor with a specimen site of collection containing "lymph node"] with the earliest Age At Specimen Collection.
                - If both skin/soft tissue and lymph node collected at same age (including if both are in the Age 90 or older category), then use the specimen with Primary/Met field = "Primary" (usually will be the skin specimen)
                - If none of the specimens have Primary/Met field = "Primary", then use the lymph node specimen.
            - If the patient does NOT have any tumor with a SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains ["skin" OR "soft tissue" OR "lymph node"], then select the tumor with the earliest Age At Specimen Collection
    '''
    
    mask_of_indicators_that_RNA_sequencing_data_is_available = data_frame_of_clinical_molecular_linkage_data_for_patient["RNASeq"].notna() & data_frame_of_clinical_molecular_linkage_data_for_patient["RNASeq"].str.strip().ne("")
    number_of_tumors_with_RNA_sequencing_data = mask_of_indicators_that_RNA_sequencing_data_is_available.sum()
    if number_of_tumors_with_RNA_sequencing_data == 1:
        return data_frame_of_clinical_molecular_linkage_data_for_patient.loc[mask_of_indicators_that_RNA_sequencing_data_is_available].iloc[0]

    mask_of_indicators_that_WES_data_is_available = data_frame_of_clinical_molecular_linkage_data_for_patient["WES"].notna() & data_frame_of_clinical_molecular_linkage_data_for_patient["WES"].str.strip().ne("")
    number_of_tumors_with_WES_sequencing_data = mask_of_indicators_that_WES_data_is_available.sum()
    if number_of_tumors_with_RNA_sequencing_data > 1 or (number_of_tumors_with_RNA_sequencing_data == 0 and mask_of_indicators_that_WES_data_is_available.all()):
        
        series_of_specimen_sites_of_collection_for_patient = data_frame_of_clinical_molecular_linkage_data_for_patient["SpecimenSiteOfCollection"].str.lower().fillna("")
        mask_of_indicators_that_specimen_sites_of_collection_contain_skin = series_of_specimen_sites_of_collection_for_patient.str.contains("skin")
        mask_of_indicators_that_specimen_sites_of_collection_contain_soft_tissue = series_of_specimen_sites_of_collection_for_patient.str.contains("soft tissue")
        mask_of_indicators_that_specimen_sites_of_collection_contain_lymph_node = series_of_specimen_sites_of_collection_for_patient.str.contains("lymph node")
        if mask_of_indicators_that_specimen_sites_of_collection_contain_skin.any() and not (mask_of_indicators_that_specimen_sites_of_collection_contain_lymph_node.any() or mask_of_indicators_that_specimen_sites_of_collection_contain_soft_tissue.any()):
            return data_frame_of_clinical_molecular_linkage_data_for_patient[mask_of_indicators_that_specimen_sites_of_collection_contain_skin].sort_values(by = "Age At Specimen Collection").iloc[0]
        
        if not (mask_of_indicators_that_specimen_sites_of_collection_contain_skin.any() or mask_of_indicators_that_specimen_sites_of_collection_contain_soft_tissue.any()):
            if not mask_of_indicators_that_specimen_sites_of_collection_contain_lymph_node.any():
                return data_frame_of_clinical_molecular_linkage_data_for_patient.sort_values(by = "Age At Specimen Collection").iloc[0]
            else:
                data_frame_of_clinical_molecular_linkage_data_with_specimen_sites_of_collection_containing_lymph_node = data_frame_of_clinical_molecular_linkage_data_for_patient[mask_of_indicators_that_specimen_sites_of_collection_contain_lymph_node]
                if data_frame_of_clinical_molecular_linkage_data_with_specimen_sites_of_collection_containing_lymph_node.shape[0] > 1:
                    logging.warn("6.2.2. A case is not specified.")
                    return data_frame_of_clinical_molecular_linkage_data_with_specimen_sites_of_collection_containing_lymph_node.sort_values(by = "Age At Specimen Collection").iloc[0]
                else:
                    return data_frame_of_clinical_molecular_linkage_data_with_specimen_sites_of_collection_containing_lymph_node.iloc[0]
        
        if (mask_of_indicators_that_specimen_sites_of_collection_contain_skin.any() or mask_of_indicators_that_specimen_sites_of_collection_contain_soft_tissue.any()) and mask_of_indicators_that_specimen_sites_of_collection_contain_lymph_node.any():
                        
            data_frame_of_candidates = data_frame_of_clinical_molecular_linkage_data_for_patient[(mask_of_indicators_that_specimen_sites_of_collection_contain_skin | mask_of_indicators_that_specimen_sites_of_collection_contain_soft_tissue) | mask_of_indicators_that_specimen_sites_of_collection_contain_lymph_node]
            
            earliest_age = data_frame_of_candidates["Age At Specimen Collection"].min()
            mask_of_indicators_that_ages_at_specimen_collection_are_earliest = data_frame_of_candidates["Age At Specimen Collection"] == earliest_age
            data_frame_of_candidates_with_earliest_age = data_frame_of_candidates[mask_of_indicators_that_ages_at_specimen_collection_are_earliest]
            if data_frame_of_candidates_with_earliest_age.shape[0] == 1:
                return data_frame_of_candidates_with_earliest_age.iloc[0]
            
            series_of_specimen_sites_of_collection_of_candidates_with_earliest_age = data_frame_of_candidates_with_earliest_age["SpecimenSiteOfCollection"].str.lower().fillna("")
            mask_of_indicators_that_specimen_sites_of_collection_contain_skin_or_soft_tissue = series_of_specimen_sites_of_collection_of_candidates_with_earliest_age.str.contains("skin") | series_of_specimen_sites_of_collection_of_candidates_with_earliest_age.str.contains("soft tissue")
            mask_of_indicators_that_specimen_sites_of_collection_contain_lymph_node = series_of_specimen_sites_of_collection_of_candidates_with_earliest_age.str.contains("lymph node")
            if mask_of_indicators_that_specimen_sites_of_collection_contain_skin_or_soft_tissue.any() and mask_of_indicators_that_specimen_sites_of_collection_contain_lymph_node.any():
                if "Primary/Met" in data_frame_of_candidates_with_earliest_age:
                    mask_of_indicators_that_value_of_primary_met_is_primary = data_frame_of_candidates_with_earliest_age["Primary/Met"].str.lower() == "primary"
                    if mask_of_indicators_that_value_of_primary_met_is_primary.any():
                        data_frame_of_candidates_with_earliest_age_and_value_of_primary_met_of_primary = data_frame_of_candidates_with_earliest_age[mask_of_indicators_that_value_of_primary_met_is_primary]
                        if data_frame_of_candidates_with_earliest_age_and_value_of_primary_met_of_primary.shape[0] > 1:
                            logging.warn("6.2.3. A case where multiple candidates with earliest age have a value of Primary/Met of primary is not specified.")
                            return data_frame_of_candidates_with_earliest_age_and_value_of_primary_met_of_primary.sort_values(by = "Age At Specimen Collection").iloc[0]
                        else:
                            return data_frame_of_candidates_with_earliest_age_and_value_of_primary_met_of_primary.iloc[0]
                else:
                    data_frame_of_candidates_with_earliest_age_and_specimen_sites_of_collection_containing_lymph_node = data_frame_of_candidates_with_earliest_age[mask_of_indicators_that_specimen_sites_of_collection_contain_lymph_node]
                    if data_frame_of_candidates_with_earliest_age_and_specimen_sites_of_collection_containing_lymph_node.shape[0] > 1:
                        logging.warn("6.2.3. A case where multiple candidates with earliest age have specimen sites of collection containing lymph node is not specified.")
                        return data_frame_of_candidates_with_earliest_age_and_specimen_sites_of_collection_containing_lymph_node.sort_values(by = "Age At Specimen Collection").iloc[0]
                    else:
                        return data_frame_of_candidates_with_earliest_age_and_specimen_sites_of_collection_containing_lymph_node.iloc[0]
    
    logging.warn("We reached the end of the process for selecting a tumor for a patient in B without selecting a patient.")
    return data_frame_of_clinical_molecular_linkage_data_for_patient.sort_values(by = "Age At Specimen Collection").iloc[0]


def select_diagnosis_for_patient_in_C(
    data_frame_of_diagnosis_data_for_patient: pd.DataFrame,
    series_of_clinical_molecular_linkage_data_for_patient: pd.Series,
    data_frame_of_metastatic_disease_data_for_patient: pd.DataFrame
) -> pd.Series:
    '''
    From "ORIEN Specimen Staging Revised Rules":
    7. Group C = > 1 melanoma diagnosis and 1 tumor sequenced -> n=30
    CHANGE FROM PRIOR: Simplified rules with changes in red text to ensure correct diagnosis is selected.
    SpecimenSiteofCollection [SpecimenSiteOfCollection] is the correct field.
    - For specimens with the field Primary/Met = "Primary"
        - IF AgeAtSpecimenCollection is WITHIN 90 days AFTER the AgeAtDiagnosis for only one of the diagnoses, then use that diagnosis.
            - 2FEX3JHKCW, 8BCBHGEO4N, FNP53S81KA, L5876HQBT9, MD578977I9, QC7QX0VWAY, VLY2FWYN29
        - If AgeAtSpecimenCollection is WITHIN 90 days AFTER the AgeAtDiagnosis for more than one of the diagnoses OR AgeAtDiagnosis is unknown for at least one of the diagnoses, but the PrimaryDiagnosisSite (from the Diagnosis file) is NOT the same for all diagnoses, then then use the diagnosis that has a PrimaryDiagnosisSite = SpecimenSiteOfOrigin
            - ILE2DL0KMW, L2R9RJJ88C
    - For specimens with the filed Primary/Met = "Metastatic"
        - IF "SpecimenSiteOfCollection does not contain "lymph node" AND only one diagnosis contains PathGroupStage EQUALS "IV" OR [ClinGroupStage EQUALS "IV" AND PathGroupStage is ["Unknown/Not Reported" OR "No TNM applicable for this site/histology combination" OR "Unknown/Not Applicable"]], then use the diagnosis with stage IV
            - 9NOLH4M870, HLIXS3VDZ6, SHTJKKY76C, DEB9M36STN
        - IF SpecimenSiteOfCollection does not contain "lymph node" AND NONE of the diagnoses have {PathGroupStage EQUALS "IV" OR [ClinGroupStage EQUALS "IV" AND PathGroupStage is ["Unknown/Not Reported" OR "No TNM applicable for this site/histology combination" OR "Unknown/Not Applicable"]]}, then use the diagnosis with the earliest AgeAtDiagnosis
            - 087FO3NF65, AC2EJBKWJO, BXVDLL792A, F3BE85LAWN, JUDAJ1LHL9, R06W2EUXCM, R2TNVTF684
        - IF SpecimenSiteOfCollection contains "lymph node" AND AgeAtSpecimenCollection is WITHIN 90 days AFTER the AgeAtDiagnosis for only one diagnosis, then use that diagnosis within 90 days of specimen collection.
            - 643X8OLYWR, ILKRH6I83A, RAB7UH51TS
        - IF SpecimenSiteOfCollection contains "lymph node" AND AgeAtSpecimenCollection is WITHIN 90 days AFTER the AgeAtDiagnosis for more than one diagnosis AND only one diagnosis has PathNStage that does NOT contain ["N0", "Nx", "Unknown/Not Applicable", "No TNM applicable for this site/histology combination"], then use that diagnosis (the one with known positive nodes of PathNStage).
            - 39TYSJBNKK, 5BS8L7PCCE, 6RX3G5GV02
        - IF SpecimenSiteOfCollection contains "lymph node" AND AgeAtSpecimenCollection is GREATER THAN 90 days AFTER the AgeAtDiagnosis for all of the diagnoses AND the PrimaryDiagnosisSite (from Diagnosis file) = SpecimenSiteofOrigin for only one of the diagnosis [diagnoses], then use the diagnosis associated with that PrimaryDiagnosisSite.
            - 7HU06PZK4Q, KWLPMWV0FM, XPZE95IE7I
        - IF SpecimenSiteOfCollection contains "lymph node" AND AgeAtSpecimenCollection is GREATER THAN 90 days AFTER the AgeAtDiagnosis for all of the diagnoses AND the PrimaryDiagnosisSite (from Diagnosis file) DOES NOT EQUAL SpecimenSiteOfOrigin for any diagnosis, then use the diagnosies [diagnosis] with earliest age of AgeAtDiagnosis [AgeAtDiagnosis].
            - 383CIRVHH2
    '''
    value_of_field_primary_met = series_of_clinical_molecular_linkage_data_for_patient["Primary/Met"].strip().lower()
    series_of_ages_at_diagnosis = data_frame_of_diagnosis_data_for_patient["AgeAtDiagnosis"].apply(lambda x: pd.NA if x == "Age Unknown/Not Recorded" else float(x))
    age_at_specimen_collection = pd.to_numeric(series_of_clinical_molecular_linkage_data_for_patient["Age At Specimen Collection"], errors = "raise")
    mask_of_indicators_that_age_at_specimen_collection_is_within_90_days_after_ages_at_diagnosis = series_of_ages_at_diagnosis.apply(lambda age_at_diagnosis: are_within_90_days_after(age_at_specimen_collection, age_at_diagnosis) if pd.notna(age_at_diagnosis) else False)
    mask_of_indicators_that_age_at_specimen_collection_is_greater_than_90_days_after_ages_at_diagnosis = series_of_ages_at_diagnosis.apply(lambda age_at_diagnosis: are_greater_than_90_days_after(age_at_specimen_collection, age_at_diagnosis) if pd.notna(age_at_diagnosis) else False)
    series_of_primary_diagnosis_sites = data_frame_of_diagnosis_data_for_patient["PrimaryDiagnosisSite"].str.lower()
    specimen_site_origin = str(series_of_clinical_molecular_linkage_data_for_patient["SpecimenSiteOfOrigin"]).lower()
    mask_of_indicators_that_primary_diagnosis_sites_are_specimen_site_of_origin = series_of_primary_diagnosis_sites == specimen_site_origin
    
    if value_of_field_primary_met == "primary":
        
        if mask_of_indicators_that_age_at_specimen_collection_is_within_90_days_after_ages_at_diagnosis.sum() == 1:
            return data_frame_of_diagnosis_data_for_patient[mask_of_indicators_that_age_at_specimen_collection_is_within_90_days_after_ages_at_diagnosis].iloc[0]

        unknown_age_exists = series_of_ages_at_diagnosis.isna().any()
        primary_diagnosis_sites_differ = series_of_primary_diagnosis_sites.nunique() > 1
        if (mask_of_indicators_that_age_at_specimen_collection_is_within_90_days_after_ages_at_diagnosis.sum() > 1 or unknown_age_exists) and primary_diagnosis_sites_differ:
            
            data_frame_of_diagnosis_data_for_patient_where_primary_diagnosis_sites_are_specimen_site_of_origin = data_frame_of_diagnosis_data_for_patient[mask_of_indicators_that_primary_diagnosis_sites_are_specimen_site_of_origin].copy()
            
            if not data_frame_of_diagnosis_data_for_patient_where_primary_diagnosis_sites_are_specimen_site_of_origin.empty:
                
                if data_frame_of_diagnosis_data_for_patient_where_primary_diagnosis_sites_are_specimen_site_of_origin.shape[0] > 1:
                    logging.warning("7.1.2. A case is not specified.")
                
                data_frame_of_diagnosis_data_for_patient_where_primary_diagnosis_sites_are_specimen_site_of_origin["_age"] = pd.to_numeric(data_frame_of_diagnosis_data_for_patient_where_primary_diagnosis_sites_are_specimen_site_of_origin["AgeAtDiagnosis"], errors = "raise")
                return data_frame_of_diagnosis_data_for_patient_where_primary_diagnosis_sites_are_specimen_site_of_origin.sort_values("_age", na_position = "last").iloc[0]

    elif value_of_field_primary_met == "metastatic":
        
        specimen_site_of_collection = series_of_clinical_molecular_linkage_data_for_patient["SpecimenSiteOfCollection"].lower()
        
        if "lymph node" not in specimen_site_of_collection:
            
            series_of_raw_pathological_group_stages = data_frame_of_diagnosis_data_for_patient["PathGroupStage"].fillna("").str.strip()
            series_of_uppercase_pathological_group_stages = series_of_raw_pathological_group_stages.str.upper()
            series_of_uppercase_clinical_group_stages = data_frame_of_diagnosis_data_for_patient["ClinGroupStage"].fillna("").str.strip().str.upper()

            mask_of_indicators_that_group_stage_is_IV = (
                (series_of_uppercase_pathological_group_stages == "IV") |
                (
                    (series_of_uppercase_clinical_group_stages == "IV") &
                    series_of_raw_pathological_group_stages.str.lower().isin(
                        {"unknown/not reported", "no tnm applicable for this site/histology combination", "unknown/not applicable"}
                    )
                )
            )

            if mask_of_indicators_that_group_stage_is_IV.sum() == 1:
                return data_frame_of_diagnosis_data_for_patient[mask_of_indicators_that_group_stage_is_IV].iloc[0]
            
            if mask_of_indicators_that_group_stage_is_IV.sum() == 0:
                copy_of_data_frame_of_diagnosis_data_for_patient = data_frame_of_diagnosis_data_for_patient.copy()
                copy_of_data_frame_of_diagnosis_data_for_patient["_age"] = series_of_ages_at_diagnosis
                return copy_of_data_frame_of_diagnosis_data_for_patient.sort_values("_age", na_position = "last").iloc[0]

        if "lymph node" in specimen_site_of_collection:
            
            if mask_of_indicators_that_age_at_specimen_collection_is_within_90_days_after_ages_at_diagnosis.sum() == 1:
                
                return data_frame_of_diagnosis_data_for_patient[mask_of_indicators_that_age_at_specimen_collection_is_within_90_days_after_ages_at_diagnosis].iloc[0]
            
            if mask_of_indicators_that_age_at_specimen_collection_is_within_90_days_after_ages_at_diagnosis.sum() > 1:
                
                mask_of_indicators_of_positives_nodes = ~data_frame_of_diagnosis_data_for_patient["PathNStage"].str.contains(
                    "N0|Nx|unknown/not applicable|no tnm applicable for this site/histology combination",
                    case = False,
                    na = False
                )
                
                if mask_of_indicators_of_positives_nodes.sum() == 1:
                    
                    return data_frame_of_diagnosis_data_for_patient[mask_of_indicators_of_positives_nodes].iloc[0]
               
            if mask_of_indicators_that_age_at_specimen_collection_is_greater_than_90_days_after_ages_at_diagnosis.all():
                
                if mask_of_indicators_that_primary_diagnosis_sites_are_specimen_site_of_origin.sum() == 1:
                    
                    data_frame_of_diagnosis_data_for_patient_where_primary_diagnosis_sites_are_specimen_site_of_origin = data_frame_of_diagnosis_data_for_patient[mask_of_indicators_that_primary_diagnosis_sites_are_specimen_site_of_origin].copy()
                    
                    if not data_frame_of_diagnosis_data_for_patient_where_primary_diagnosis_sites_are_specimen_site_of_origin.empty:
                        
                        data_frame_of_diagnosis_data_for_patient_where_primary_diagnosis_sites_are_specimen_site_of_origin["_age"] = pd.to_numeric(data_frame_of_diagnosis_data_for_patient_where_primary_diagnosis_sites_are_specimen_site_of_origin["AgeAtDiagnosis"], errors = "raise")
                        return data_frame_of_diagnosis_data_for_patient_where_primary_diagnosis_sites_are_specimen_site_of_origin.sort_values("_age", na_position = "last").iloc[0]
                    
                if mask_of_indicators_that_primary_diagnosis_sites_are_specimen_site_of_origin.sum() == 0:
                    
                    copy_of_data_frame_of_diagnosis_data_for_patient = data_frame_of_diagnosis_data_for_patient.copy()
                    copy_of_data_frame_of_diagnosis_data_for_patient["_age"] = pd.to_numeric(data_frame_of_diagnosis_data_for_patient["AgeAtDiagnosis"], errors = "raise")
                    return copy_of_data_frame_of_diagnosis_data_for_patient.sort_values("_age", na_position = "last").iloc[0]
                
    deid_specimen_id = series_of_clinical_molecular_linkage_data_for_patient["DeidSpecimenID"].strip()
    logging.warning(f"We reached the end of selecting diagnosis for patient in C for specimen with ID {deid_specimen_id} and value of field `Primary/Met` {value_of_field_primary_met}.")
    copy_of_data_frame_of_diagnosis_data_for_patient = data_frame_of_diagnosis_data_for_patient.copy()
    copy_of_data_frame_of_diagnosis_data_for_patient["_age"] = series_of_ages_at_diagnosis
    return copy_of_data_frame_of_diagnosis_data_for_patient.sort_values("_age").iloc[0]


def are_within_90_days_after(age_at_specimen_collection: float | None, age_diag: float | None) -> bool:
    if age_at_specimen_collection is None or age_diag is None:
        return False
    diff = age_at_specimen_collection - age_diag
    return 0 <= diff <= 90 / 365.25


def are_greater_than_90_days_after(age_at_specimen_collection: float | None, age_diag: float | None) -> bool:
    if age_at_specimen_collection is None or age_diag is None:
        return False
    diff = age_at_specimen_collection - age_diag
    return diff > 90 / 365.25


def select_tumor_for_patient_in_D(data_frame_of_clinical_molecular_linkage_data_for_patient: pd.DataFrame) -> pd.Series:
    '''
    From "ORIEN Specimen Staging Revised Rules":
    8. Group D: >1 melanoma diagnosis and >1 tumor sequenced -> n=3
    CHANGE FROM PRIOR: Rules below instead of using rules from Group B and C, to ensure correct tumors [tumor] and diagnosis pairs are selected.
    SpecimenSiteofCollection [SpecimenSiteOfCollection] is the correct field.
        8.a. First select tumor:
            i. If just one tumor with RNAseq data, then select that tumor.
                1. 59OP5X1AZL, FUAZTE7LVQ
            ii. If > 1 tumor with RNAseq data, then select tumor with earliest Age At Specimen Collection
                1. 7HOWLJKDEM
    '''
    
    data_frame_of_clinical_molecular_linkage_data_for_patient_with_RNA_sequencing_data = data_frame_of_clinical_molecular_linkage_data_for_patient[data_frame_of_clinical_molecular_linkage_data_for_patient["RNASeq"].notna()]
    
    if len(data_frame_of_clinical_molecular_linkage_data_for_patient_with_RNA_sequencing_data) == 1:
        
        return data_frame_of_clinical_molecular_linkage_data_for_patient_with_RNA_sequencing_data.iloc[0]
    
    if len(data_frame_of_clinical_molecular_linkage_data_for_patient_with_RNA_sequencing_data) > 1:
        
        data_frame_of_clinical_molecular_linkage_data_for_patient_with_RNA_sequencing_data["_age"] = pd.to_numeric(data_frame_of_clinical_molecular_linkage_data_for_patient_with_RNA_sequencing_data["Age At Specimen Collection"], errors = "raise")
        return data_frame_of_clinical_molecular_linkage_data_for_patient_with_RNA_sequencing_data.sort_values("_age", na_position = "last").iloc[0]

    
def select_diagnosis_for_patient_in_D(
    data_frame_of_diagnosis_data_for_patient: pd.DataFrame,
    series_of_clinical_molecular_linkage_data_for_patient: pd.Series
) -> pd.Series:
    '''
    From "ORIEN Specimen Staging Revised Rules":
    8.b. Then select diagnosis:
        i. If SpecimenSiteOfCollection contains "lymph node", then select diagnosis with latest Age At Diagnosis
            1. 59OP5X1AZL
        ii. If SpecimenSiteOfCollection contains either "skin" or "soft tissue", then select diagnosis with earliest Age At Diagnosis
            - 7HOWLJKDEM, FUAZTE7LVQ
    '''
    
    copy_of_data_frame_of_diagnosis_data_for_patient = data_frame_of_diagnosis_data_for_patient.copy()
    copy_of_data_frame_of_diagnosis_data_for_patient["_age"] = pd.to_numeric(copy_of_data_frame_of_diagnosis_data_for_patient["AgeAtDiagnosis"], errors = "raise")
    specimen_site_of_collection = series_of_clinical_molecular_linkage_data_for_patient["SpecimenSiteOfCollection"].lower()
    
    if "lymph node" in specimen_site_of_collection:
        
        return copy_of_data_frame_of_diagnosis_data_for_patient.sort_values("_age", ascending = False, na_position = "last").iloc[0]
    
    if re.search("skin|soft tissue", specimen_site_of_collection):
        
        return copy_of_data_frame_of_diagnosis_data_for_patient.sort_values("_age", na_position = "last").iloc[0]
    
    
def assign_stage_and_rule(
    series_of_clinical_molecular_linkage_data_for_patient: pd.Series,
    series_of_diagnosis_data_for_patient: pd.Series,
    data_frame_of_metastatic_disease_data_for_patient: pd.DataFrame
) -> Tuple[str, str]:

    age_coll = float(90.0 if series_of_clinical_molecular_linkage_data_for_patient.get("Age At Specimen Collection") == "Age 90 or older" else series_of_clinical_molecular_linkage_data_for_patient.get("Age At Specimen Collection"))

    MetsDzPrimaryDiagnosisSite = ""
    MetastaticDiseaseInd = ""
    metastatic_sites_text = ""
    ages = None
    AgeAtMetastaticSite = None
    
    if not data_frame_of_metastatic_disease_data_for_patient.empty:
        MetsDzPrimaryDiagnosisSite = "|".join(
            data_frame_of_metastatic_disease_data_for_patient["MetsDzPrimaryDiagnosisSite"]
            .dropna()
            .astype(str)
        ).lower()
        
        MetastaticDiseaseInd = "|".join(
            data_frame_of_metastatic_disease_data_for_patient["MetastaticDiseaseInd"]
            .dropna()
            .astype(str)
        ).lower()
        
        metastatic_sites_text = "|".join(data_frame_of_metastatic_disease_data_for_patient["MetastaticDiseaseSite"].dropna().astype(str)).lower()

        # ❷  Convert every age entry to float (treat “Age 90 or older” as 90.0),
        #     then take the earliest (minimum) non-null value.
        ages = (
            data_frame_of_metastatic_disease_data_for_patient["AgeAtMetastaticSite"]
            .apply(lambda x: 90.0 if str(x).strip() == "Age 90 or older" else _float(x))
        )
        AgeAtMetastaticSite = ages.min() if ages.notna().any() else None
    
    
    '''
    From "ORIEN Specimen Staging Revised Rules":
    10. These are the exceptions to the staging rules. I imagine this should be coded before the other staging rules in your script, so I’ve named it Rule#0 EXCEPTION, but you can put it in your code differently if there is a better way.
    
    10.a. RULE#0 EXCEPTION
    - If AvatarKey = ["227RDTKST8" or "ILE2DL0KMW" or "YRGE6MVYNK" or "EKZGU61JTP" or "3Z82S0R5IE" or "MXL3WK5YF0"], then AssignedStage = III
        - EKZGU61JTP: vulvar melanoma stage IIIB (T3bN1aM0, 7th) at age 67.0. Specimen is vagina (NOS), designated as “primary”, obtained at age 67.73. Reported to have regional nodes (inguinal) at age 67.07 and 67.20. Rule seems appropriate to consider this vaginal specimen as metastatic site of vulvar disease rather than new primary mucosal lesion; however, will be considered regional disease, not distant.
            - For this patient ID, set Discrepancy = 1.
    - If AvatarKey = ["GEM0S42KIH" or "CG6JRI0XIX" or "WDMTMU4SV2"], then AssignedStage = IV
    - This rule applies to 9 patients, all cutaneous; 8 patients in Group A and 1 patient in Group C (ILE2DL0KMW)
    '''
    if series_of_clinical_molecular_linkage_data_for_patient["ORIENAvatarKey"] in ["227RDTKST8", "ILE2DL0KMW", "YRGE6MVYNK", "EKZGU61JTP", "3Z82S0R5IE", "MXL3WK5YF0"]:
        return "III", "EXCEPTION"
    elif series_of_clinical_molecular_linkage_data_for_patient["ORIENAvatarKey"] in ["GEM0S42KIH", "CG6JRI0XIX", "WDMTMU4SV2"]:
        return "IV", "EXCEPTION"
    
    '''
    From "ORIEN Specimen Staging Revised Rules":
    10.b. Once AssignedStage field is populated by a rule, it should not be further evaluated by the remainder of the script. This simplifies the rules if the script can be written so that the assignment happens in this order and the patient is no longer evaluated once assigned.
        - For example, patient 50I7H1PID7 has AgeAtSpecimenCollection = "Age 90 or older" and a Path Stage of IV. It will be assigned stage IV by the AGE rule and then removed from further evaluation, so should not then be evaluated/assigned for the PATHIV rule.
        - Simplified rules now apply to all specimens regardless of their primary site (cutaneous, ocular, mucosal, or unknown)
        - I have provided the counts that meet each rule by primary site and group.
    
    Rule #1: AGE
        - If AgeAtDiagnosis = "Age 90 or older", then AssignedStage = numerical value of PathGroupStage OR ClinGroupStage (if PathGroupStage is [“Unknown/Not Reported” OR “No TNM applicable for this site/histology combination” OR  “Unknown/Not Applicable”])
            - Total: 5 patients, all cutaneous
                - 4 in group A
                - 1 in group B (D42OPM2PLC)
    '''
    string_representation_of_age_at_diagnosis = str(series_of_diagnosis_data_for_patient.get("AgeAtDiagnosis", "")).strip().lower()
    pathological_group_stage = str(series_of_diagnosis_data_for_patient.get("PathGroupStage", "")).strip().upper()
    clinical_group_stage = str(series_of_diagnosis_data_for_patient.get("ClinGroupStage", "")).strip().upper()
    
    if string_representation_of_age_at_diagnosis == "age 90 or older":
        if pathological_group_stage.lower() in ["unknown/not reported", "no tnm applicable for this site/histology combination", "unknown/not applicable"]:
            roman_numeral_in_clinical_group_stage = roman_numeral_in(clinical_group_stage)
            if roman_numeral_in_clinical_group_stage:
                return roman_numeral_in_clinical_group_stage, "AGE"
            else:
                return "Unknown", "AGE"
        roman_numeral_in_pathological_group_stage = roman_numeral_in(pathological_group_stage)
        if roman_numeral_in_pathological_group_stage:
            return roman_numeral_in_pathological_group_stage, "AGE"
        else:
            return "Unknown", "AGE"

    '''
    From "ORIEN Specimen Staging Revised Rules":
    10.c. Rule #2: PATHIV
        - If PathGroupStage contains "IV", then AssignedStage = "IV"
            - Total: 75 patients
                - Cutaneous: 70 patients
                    - 66 in group A; 1 in group B (317K6G9N41); 3 in group C (9NOLH4M870, DEB9M36STN, HLIXS3VDZ6)
                - Ocular: 1 patient, group A
                - Mucosal: 3 patients
                    - 2 patients in group A; 1 patient in group C (R06W2EUXCM)
                - Unknown: 1 patient, group B (MYCVCULC8L)
    '''
    if "IV" in pathological_group_stage:
        return "IV", "PATHIV"

    '''
    From "ORIEN Specimen Staging Revised Rules":
    10.d. Rule #3: CLINIV
        - If ClinGroupStage contains "IV", THEN AssignedStage = "IV"
            - This rule no longer state requires [requires] path staging to be unknown. In all but one case is the path stage also IV or unknown. In that one case, the path staging and metastatic disease file are unclear, and it appears appropriate to classify as stage IV disease.
                - Confirm with Slingluff: confirmed to follow rule, keep as stage IV
                    - QLWU5QNQIB: Trunk melanoma diagnosed at age 60.35 with path stage III (TxN3M0, 7th) and clinical stage IV (but recorded as TxN3M0 for that as well). Specimen is lymph node (NOS) obtained at age 60.64 (105 days after initial diagnosis). Reported to have "regional" colon and peritoneum mets at unknown age, as well as regional node (NOS) at unknown age, all associated with the trunk diagnosis. Also reported to have a distant skin met on lower extremity, also at unknown age and associated with trunk diagnosis. Rule seems appropriate given unknowns.
                        - For this patient ID, set Discrepancy = 1.
            - 27 patients
                - Cutaneous: 24 patients
                    - 22 patients in group A; 1 patient in group B (WAYUEWGM1O), 1 patient in group C (SHTJKKY76C)
                - Mucosal: 2 patients, group A
                - Ocular: 1 patient, group A
    '''
    if "IV" in clinical_group_stage:
        return "IV", "CLINIV"

    '''
    From "ORIEN Specimen Staging Revised Rules":
    10.e. Rule #4: METSITE
    SpecimenSiteofCollection [SpecimenSiteOfCollection] is the correct field.
        - IF SpecimenSiteofCollection [SpecimenSiteOfCollection] does not contain ["skin" OR "ear" OR "eyelid" OR "vulva", OR "head" OR "soft tissues" OR "breast" OR "lymph node" OR "parotid" OR "vagina"], THEN AssignedStage = "IV"
            - Specimen from a distant metastatic site 
                - Confirm with Slingluff: confirmed, rule captures any muscle or chest wall invasion as stage IV, but does not automatically assign breast or other soft tissue specimens as stage IV due to these not all appearing to be stage IV disease in this database
            - This rule applies to all patients regardless of primary site (cutaneous, ocular, mucosal, or unknown). This works for our database since it does not appear that we have any primary mucosal specimens. This rule would need to be edited if used in the future with mucosal melanomas. 
            - Confirm with Slingluff the appropriateness of stage IV assignment, confirmed; updated rule to exclude vaginal specimen and move EKZGU61JTP to rule #8 (SKINLESS90D)
                - KEHK6YTVAK: trunk melanoma stage IIIC (T4aN3M0, 8th) at age 63.28. Specimen is anus (NOS), designated as "primary", obtained at age 63.55. Reported to have distant pelvic nodes at 63.55 associated with trunk melanoma, and then two additional distant mets reported (inguinal nodes, soft tissue of pelvis) associated with trunk diagnosis. No mention of an anal melanoma. Rule seems appropriate to consider this as metastatic site instead of new primary mucosal (in addition to cutaneous primary of trunk); confirmed, follow rule to assign stage IV given previously reported distant mets
                    - For this patient ID, set Discrepancy = 1.
            - 37 patients 
                - Cutaneous: 35 patients 
                    - 30 patients in group A; 3 patients in group B (CILTNWAT6B, P5JJX6UI7G, 48J2GNDBEN); 2 patients in group C (F3BE85LAWN, BXVDLL792A)
                - Ocular: 2 patients, both group C (JUDAJ1LHL9, R2TNVTF684)
    '''
    specimen_site_of_collection = str(series_of_clinical_molecular_linkage_data_for_patient.get("SpecimenSiteOfCollection", "")).lower()
    pattern = re.compile("skin|ear|eyelid|vulva|head|soft tissues|breast|lymph node|parotid|vagina", re.I)
    if not pattern.search(specimen_site_of_collection):
        return "IV", "METSITE"

    '''
    From "ORIEN Specimen Staging Revised Rules":
    10.f. Rule #5: PRIORDISTANT
        - IF {combined entry on Metastatic file with: MetsDzPrimaryDiagnosisSite contains ["skin" OR "ear" OR "eyelid" OR "vulva" OR "eye" OR "choroid" OR "ciliary body" OR "conjunctiva" OR ["]sinus" OR "gum" OR "nasal" OR "urethra"] AND MetastaticDiseaseInd = "Yes – Distant", AND AgeAtMetastaticSite <= AgeAtSpecimenCollection+0.005}, THEN AssignedStage = "IV"
            - Patients with recorded distant metastatic disease prior to specimen collection 
            - Must add 0.005 to the AgeAtSpecimenCollection due to only 2 decimal points recorded for age in that file, but 3 decimal points recorded on the metastatic disease file.
            - This works for our database. This rule would need to be edited if used in the future to reflect the primary diagnosis site keywords applicable to that dataset (e.g., if there are other mucosal sites that should be included).
                - 37 patients 
                    - Cutaneous: 36 patients
                        - 31 patients in group A; 3 patients in group B (PP7QY6B66M, U2CUPQJ4T1, WX0NQIQIXM); 2 patients in group C (087FO3NF65, QC7QX0VWAY) 
                    - Ocular: 1 patient, group A 
                - Count update reflects exceptions listed below as well as the two patients (CG6JRI0XIX, WDMTMU4SV2) that I had erroneously counted by this rule but that weren’t actually captured. CG6JRI0XIX has prior metastatic disease (liver) reported with MetastaticDiseaseInd = "Yes – NOS". WDMTMU4SV2 has prior metastatic disease (bone) reported with MetastaticDiseaseInd = "Yes – Regional" due to upper extremity melanoma; consider metastasis to bone as stage IV disease.
            - This rule assumes that the designation of "Distant" (instead of Regional or NOS) in the metastatic disease file is accurate. In general, this designation seems appropriate for our database, but it is possible that some cases are misclassified. 
                - Confirm with Slingluff the appropriateness of stage IV assignment, confirmed details below 
                    - 087FO3NF65: Trunk IIIC (T3aN2aM0, 7th) at age 61.8. Regional axillary nodes at 62, then “Breast, NOS” met classified as “Distant” at age 63.59. Specimen is the Breast NOS met. Reported to have lung mets with “regional” intrathoracic nodes at age 64.32. Rule seems appropriate in this case; keep as stage IV for distant met.
                        - For this patient ID, set Discrepancy = 1 and Exception = 0. 
                    - ILE2DL0KMW: Two diagnoses at the same time: scalp IIC and trunk IIIC (T4bN2cM0, 8th). Specimen is skin of trunk, collected 57 days after diagnosis, designated as "primary". Reported to have "distant" axillary nodes associated with the trunk diagnosis (at the same time as the trunk skin specimen collection) and multiple "regional" skin mets associated with the scalp diagnosis. No additional entries to suggest later distant mets. No ICB treatment. Surgery file shows trunk diagnosis with first related biopsy taken at time of specimen collection. This may represent the primary lesion at IIIC diagnosis. Make exception to the rule to assign as stage III. 
                        - For this patient ID, set Discrepancy = 1. The PRIORDISTANT rule does not apply since this patient will be captured by EXCEPTION rule. This patient is not included in the patient counts for this rule anymore. 
                    - 227RDTKST8: Trunk IIIC (T3N3cM0, 8th) initial diagnosis. Specimen is lymph node (NOS), designated as "metastatic", collected 35 days after diagnosis. Reported to have "distant" axillary nodes at same age as diagnosis, then "regional" inguinal nodes at same age as specimen collection. Specimen likely represents an inguinal node. Specimen collected Pre-ICB treatment. No additional entries to suggest later distant mets. Surgery file shows three entries, one at diagnosis and two at age of specimen collection. Perhaps axillary node was found first, then later discovered skin lesion and inguinal nodal basin involvement, but this would still be IIIC disease.  Make exception to the rule to assign as stage III. 
                        - For this patient ID, set Discrepancy = 1. The PRIORDISTANT rule does not apply since this patient will be captured by EXCEPTION rule. This patient is not included in the patient counts for this rule anymore.   
                    - 9EYYI5H9SU: Lower extremity clinical IIB disease (no path stage) at initial diagnosis at age 43. Reported to have lung mets at age 59 associated with lower extremity diagnosis. Specimen is skin of trunk, designated as "primary", collected at age 63.46. Reported to later develop pancreas mets at age 63.64, associated with lower extremity diagnosis. Surgery file only contains entries for lower extremity diagnosis, including time points that correspond to age of specimen collection. Trunk lesion could represent new primary, but rule seems appropriate in this case.  Keep as stage IV with likely new primary in the setting of metastatic disease. 
                        - For this patient ID, set Discrepancy = 1. 
                        - For this patient ID, set Possible New Primary = 1.
                    - HTKAEZOC7V: Scalp/neck III (T4bN2bM0, 6th) disease at initial diagnosis at age 70.9. Specimen is axillary node, designated as “metastatic”, at age 82.55. Reported to have regional skin mets at age 71, then "distant" axillary nodes at age 79.7, followed by regional nodes of head/face/neck at age 81. Later lung mets at age 83.9. Specimen collected post-ICB. Rule seems appropriate in this case. Follow rule, keep as stage IV. 
                        - For this patient ID, set Discrepancy = 1.
                    - IALUL9JC9Y: Trunk IIIA (T3aN1aM0, 7th) at age 26.08. Reported to have "distant" nodes of head/face/neck at age 26.12 associated with trunk diagnosis. Then reported to have "regional" lymph node (NOS) associated with trunk diagnosis at age 27.22. Specimen is lymph node (NOS) at age 27.22. No ICB treatment. No additional entries to suggest later distant mets. Given prior "distant" nodes in head/face/neck, considered stage IV. Not sure about how different nodal basins are generally viewed for trunk melanomas. Follow rule, keep as stage IV. 
                        - For this patient ID, set Discrepancy = 1.
                    - QE43I70DGC: Lower extremity IIIB (T3bN1M0, 6th) disease at age 55.115. Reported to have “distant” inguinal nodes at age 55.115 (same time as initial diagnosis). Specimen is a skin of lower extremity obtained at age 63.6, designated as a "primary". Reported to later have brain mets at age 64. Skin specimen could represent new primary given timing, but could also represent regional recurrence of prior disease. Not sure why inguinal nodes were labelled as distant at initial diagnosis since not reported as stage IV at the time. Surgery file reports two specimens at age 55.115 (likely skin + node), but then another surgery at 55.293 before the specimen that we have at 63.4, and another later surgery at 63.79. Specimen obtained post-ICB treatment. Given brain mets within a year of specimen, rule likely appropriate given limited information. Keep as stage IV with likely new primary in the setting of metastatic disease. 
                        - For this patient ID, set Discrepancy = 1. 
                        - For this patient ID, set Possible New Primary = 1.
                    - XHTXLE3MLC: Lower extremity IIC disease at age 50.5. Reported distant upper extremity skin met at 53.44, associated with the lower extremity diagnosis. Specimen is skin from lower extremity, designated as "primary", obtained at age 54.56. No additional entries to suggest additional distant disease. Specimen obtained post-ICB treatment. While specimen could represent new primary, rule seems appropriate given limited information. Keep as stage IV with likely new primary in the setting of metastatic disease.
                        - For this patient ID, set Discrepancy = 1. 
                        - For this patient ID, set Possible New Primary = 1.
                    - YMC959TA29: Trunk IIIB (T4aN3cM0, 7th) disease at age 37.35. Specimen is skin (NOS), designated as "metastatic", obtained at age 37.43 (29 days after initial diagnosis). Reported to have "distant" skin met (NOS) at that time, associated with trunk diagnosis. Specimen collected pre-ICB treatment. Surgery file has first entry at 37.43, followed by two later (age 40) entries. This could represent the skin from the initial IIIB diagnosis, but since listed as "metastatic" and there is a reported "distant" skin (NOS) met, classified as stage IV. Rule seems appropriate given limited information. Follow rule, keep as stage IV. 
                        - For this patient ID, set Discrepancy = 1.
                    - YRGE6MVYNK: Lower extremity IIIA (T2aN1aM0, 8th) disease at age 29.97. Reported "distant" inguinal nodes at age 30.02, associated with lower extremity diagnosis. Specimen is skin from lower extremity, designated as "primary", obtained at age 30.02 (same age as "distant" inguinal nodes). No additional entries to suggest later distant mets. No ICB treatment. Surgery file has an entry at same age of diagnosis labeled as “unknown site” and another entry at same age of specimen collection, also labeled as "unknown site", followed by another surgery at age 30.56 that does have lower extremity diagnosis associated with it. This skin specimen likely represents the initial IIIA diagnosis. Make exception to the rule to assign as stage III. 
                        - For this patient ID, set Discrepancy = 1. The PRIORDISTANT rule does not apply since this patient will be captured by EXCEPTION rule. This patient is not included in the patient counts for this rule anymore.
                    - QC7QX0VWAY: Two melanoma diagnoses: trunk IIB at age 68.518 and lower extremity IIB at age 75.08. Lung met reported at age 71 associated with the trunk melanoma. No other mets reported. Specimen is skin from lower extremity, designated as "primary", obtained at age 75.08 (time of that diagnosis). Specimen obtained pre-ICB treatment. Since the rule does not distinguish by site (trunk, extremity, etc), it counts the lung met as distant disease for the lower extremity melanoma, and thus assigns stage IV. If this lesion represents a new primary, then this is incorrect. Unable to know from information available. Keep as stage IV with likely new primary in the setting of metastatic disease. 
                        - For this patient ID, set Discrepancy = 1. 
                        - For this patient ID, set Possible New Primary = 1.
                        - This is the patient that had the correct stage (IV) by your script according to the rules, but an error in my file (key) for this patient. The key has been corrected.
    '''
    age_at_specimen_collection = float(series_of_clinical_molecular_linkage_data_for_patient.get("Age At Specimen Collection"))
    AGE_FUDGE = 0.005 # years, or approximately 1.8 days.
    
    if not data_frame_of_metastatic_disease_data_for_patient.empty:
        
        copy_of_data_frame_of_metastatic_disease_data_for_patient = data_frame_of_metastatic_disease_data_for_patient.copy()
        copy_of_data_frame_of_metastatic_disease_data_for_patient["_age_at_metastatic_site"] = copy_of_data_frame_of_metastatic_disease_data_for_patient["AgeAtMetastaticSite"].apply(numericize_age_at_metastatic_site)
        mask_of_indicators_that_condition_for_rule_5_is_met = (
            copy_of_data_frame_of_metastatic_disease_data_for_patient["MetsDzPrimaryDiagnosisSite"].str.contains(
                "skin|ear|eyelid|vulva|eye|choroid|ciliary body|conjunctiva|sinus|gum|nasal|urethra",
                case = False,
                na = False
            )
            & copy_of_data_frame_of_metastatic_disease_data_for_patient["MetastaticDiseaseInd"].str.strip().str.lower().eq("yes - distant")
            & copy_of_data_frame_of_metastatic_disease_data_for_patient["_age_at_metastatic_site"].notna()
            & (copy_of_data_frame_of_metastatic_disease_data_for_patient["_age_at_metastatic_site"] <= (age_at_specimen_collection + AGE_FUDGE))
        )

        if mask_of_indicators_that_condition_for_rule_5_is_met.any():
            
            return "IV", "PRIORDISTANT"

    '''
    From "ORIEN Specimen Staging Revised Rules":
    10.g. Rule #6: NOMETS
        - If Metastatic Disease file field MetastaticDiseaseInd = "No", THEN AssignedStage = numerical value of PathGroupStage OR ClinGroupStage (if PathGroupStage is [“Unknown/Not Reported” OR “No TNM applicable for this site/histology combination” OR “Unknown/Not Applicable”])
            - No recorded regional or distant metastatic disease. Rule assigns stage at initial diagnosis. 
            - Must add 0.005 to the AgeAtSpecimenCollection due to only 2 decimal points recorded for age in that file, but 3 decimal points recorded on the diagnosis file.
            - 27 patients, all cutaneous 
                - 21 patients in group A; 1 patient in group B (92S5RV6HJS), 2 patients in group C (5BS8L7PCCE, L5876HQBT9) 
            - MetastaticDiseaseInd = No (meaning no regional or distant metastatic disease reported). 
                - Except for 1 lymph node specimen (5BS8L7PCCE), all that have MetastaticDiseaseInd = No are skin specimens listed as "Primary". However, the time between age at specimen collection and age at diagnosis ranges from 0 days to 721 days, including about half that were obtained within 30 days of initial diagnosis. About 70% have clinical or pathologic stage II disease at initial diagnosis. While some could represent new primaries, the skin site of specimen collection is the same as the primary skin site of diagnosis for all but one, and in that one case, the specimen site is "skin NOS". While some of these may represent new primaries, this rule treats all specimens as related to the initial diagnosis given the limited information to know if these are truly new primaries.  
            - Three patients with specimens obtained > 90 days: 
                - 9DLKDVIQ2W: initial IIB upper extremity diagnosis; specimen is skin of upper extremity obtained 721 days later; likely represents new primary; given limited information, keep as stage II 
                    - For this patient ID, set Discrepancy = 1
                    - For this patient ID, set Possible New Primary = 1 
                - MD5OTA3E8A: initial IIC lower extremity diagnosis; specimen is skin of lower extremity obtained 592 days later; likely represents new primary; given limited information, keep as stage II 
                    - For this patient ID, set Discrepancy = 1
                    - For this patient ID, set Possible New Primary = 1 
                - MPHAPLR8K1: initial IIIC face melanoma diagnosis; specimen is skin of face obtained 94 days later; keep as stage III assuming locoregional recurrence (not new primary) 
                    - For this patient ID, set Discrepancy = 1
    '''
    if data_frame_of_metastatic_disease_data_for_patient.empty or data_frame_of_metastatic_disease_data_for_patient["MetastaticDiseaseInd"].str.lower().eq("no").all():
        if pathological_group_stage.lower() in ["unknown/not reported", "no tnm applicable for this site/histology combination", "unknown/not applicable"]:
            roman_numeral_in_clinical_group_stage = roman_numeral_in(clinical_group_stage)
            if roman_numeral_in_clinical_group_stage:
                return roman_numeral_in_clinical_group_stage, "NOMETS"
            else:
                return "Unknown", "NOMETS"
        roman_numeral_in_pathological_group_stage = roman_numeral_in(pathological_group_stage)
        if roman_numeral_in_pathological_group_stage:
            return roman_numeral_in_pathological_group_stage, "NOMETS"
        else:
            return "Unknown", "NOMETS"

    '''
    From "ORIEN Specimen Staging Revised Rules":
    10.h. Rule #7: NODE
    SpecimenSiteofCollection [SpecimenSiteOfCollection] is the correct field
        - IF SpecimenSiteofCollection contains [“lymph node” OR “parotid”], THEN AssignedStage = III
            - Assigns stage III to node specimens that remain after the above rules have been evaluated.
            - 100 patients
                - 98 cutaneous
                    - 87 patients in Group A
                    - 1 patient in Group B: WOW011YH6I
                        - This patient was appropriately captured by this rule in your script, but I had an error in my file (key) that had the correct stage (III) but incorrect rule called. The key has been corrected.
                    - 9 patients in Group C: 383CIRVHH2, 39TYSJBNKK, 643X8OLYWR, 6RX3G5GV02, 7HU06PZK4Q, ILKRH6I83A, KWLPMWV0FM, RAB7UH51TS, XPZE95IE7I
                    - 1 patient in Group D: 59OP5X1AZL
                - 1 ocular patient: 8OR7RX5NO5 (group A) 
                - 1 mucosal patient: Z7CEUA8SAJ (group A)
        - Confirm with Slingluff; confirmed, all to follow rule (stage III), but all should be marked with Discrepancy = 1. 
            - N5Q9122LTG: Lower extremity stage III (clinical, no path information) diagnosed age 36. Specimen is lymph node (NOS) obtained at age 55 (~20 years later). Reported to only have "distant" inguinal nodes at unknown age. Surgery file has multiple entries ranging from age 43 – 57.9 associated with this diagnosis (no information on what the specimens were). No additional information to suggest distant mets. Specimen obtained Pre-ICB treatment. Rule seems appropriate given the limited information.
            - 2X7USSLPJC: Lower extremity stage III (clinical T3cN1bM0, no path information) at age 55.13. Specimen is a lymph node (NOS) obtained at age 55.54. Only one entry in metastatic disease file: distant pelvic nodes at age 55.578 (so about 2 weeks after specimen collection). Surgery file has 3 entries (no information on types of specimens): two done at 55.13 (so likely represents skin + node) and then one done at 55.578. Specimen obtained post-ICB treatment (looks like just one dose ipi/nivo since both meds have start and stop ages at 55.197). Rule seems appropriate given information available.
            - 7YX8AJLMWR: Lower extremity stage IIIB (T4bN1aM0, 7th) disease at age 47.29. Specimen is lymph node (NOS) obtained at age 49.19 (about 2 years after initial diagnosis). Metastatic disease file only has two entries for regional node (inguinal) and regional skin met, both at age 47.468. Multiple surgery file entries ranging from age 47.29 at diagnosis to age 50.23 (no specimen information available). Specimen obtained post-ICB treatment (end age 48.995). Rule seems appropriate given limited information.
            - DTUPUJ06B5: Trunk melanoma IIIA (T2N1aM0, 7th) at age 77. Reported regional inguinal nodes at 77.44 (160 days after initial diagnosis). Specimen is lymph node ("intra-abdominal") at 77.44 (same age as reported regional inguinal nodes). Two entries in surgery file: one at 77 and the other at 77.44. No ICB treatment. Despite being labeled as "intra-abdominal" nodes, no other information to suggest stage IV disease, so rule seems appropriate with assumption that these were the regional inguinal nodes.
            - X9AZUY3R1C: Lower extremity IIIC (T4bN2cM0, 7th) at age 48.67. Regional inguinal nodes reported at that initial diagnosis. Specimen is lymph node (NOS) obtained at 52.7. No ICB treatment. No additional information to suggest other mets. Rule seems appropriate.
            - 2AP9EDU231: Face melanoma III (T3N2cM0, 8th) at age 62.56. Reported regional nodes of face/head/neck at unknown age. Specimen is node (NOS, non-sentinel) at age 69.66. Surgery file with just one entry at 69.658. Specimen obtained pre-ICB treatment. No other information. Rule seems appropriate given limited information.
            - 6HWEJIP63S: Lower extremity III (T4aN2bM0, 7th) at age 52.9. Reported regional nodes (NOS) at unknown age. Also brain mets at unknown age. Specimen is node (NOS, regional) obtained at 55.14. Specimen is pre-ICB treatment. Rule seems appropriate given limited information.
            - GITAF8OSTV: Lower extremity III (T3bN3Mx, 7th) at age 60.29. Regional node (NOS) reported at unknown age. Specimen is node (NOS) at age 60.72. No ICB treatment. Rule seems appropriate given limited information.
            - XPZE95IE7I: Lower extremity III (T3bN2Mx, 7th) at age 62.87. Regional nodes (inguinal, pelvic) both at unknown age. Specimen is pelvic node obtained at 63.99, post-ICB treatment. Rule seems appropriate given limited information.
            - Z7CEUA8SAJ: Urethra melanoma stage III at initial diagnosis at age 77.78. Specimen is a lymph node (NOS, non-sentinel) obtained at age 77.82 (48 days after initial diagnosis). Reported to have distant inguinal nodes at unknown age. No additional entries in the metastatic disease file or surgery file to suggest additional later distant mets. Rule keeps this stage as III and ignores the "distant" classification for the inguinal nodes. Rule seems appropriate in this case.
    '''
    if ("lymph node" in specimen_site_of_collection) or ("parotid" in specimen_site_of_collection):
        return "III", "NODE"
    
    # ────────────────────────────────────────────────────────────────
    # RULE 8 – SKINLESS90D
    #
    #   Apply stage III if BOTH are true:
    #   1.  Age At Specimen Collection (fudged) is within ±90 days of Age At Diagnosis, and
    #   2.  There is no single metastatic-disease row that simultaneously has
    #       • MetsDzPrimaryDiagnosisSite matching "skin|ear|eyelid|vulva",
    #       • MetastaticDiseaseInd matching "yes - regional|yes - nos",
    #       • MetastaticDiseaseSite matching "skin|ear|eyelid|vulva|breast", and
    #       • AgeAtMetastaticSite ≤ AgeSpec + AGE_FUDGE
    # ────────────────────────────────────────────────────────────────
    age_diag_f = _float(string_representation_of_age_at_diagnosis)
    
    age_at_specimen_collection_fudged = None if age_at_specimen_collection is None else age_at_specimen_collection + AGE_FUDGE
    within_90d = are_within_90_days(age_at_specimen_collection_fudged, age_diag_f)

    no_row_hits_all_four = True
    if (
        within_90d
        and not data_frame_of_metastatic_disease_data_for_patient.empty
        and age_at_specimen_collection_fudged is not None
    ):
        meta_df = data_frame_of_metastatic_disease_data_for_patient.copy()

        # numeric metastatic age
        meta_df["_age_meta"] = meta_df["AgeAtMetastaticSite"].apply(
            lambda x: 90.0 if str(x).strip() == "Age 90 or older" else _float(x)
        )

        four_crit_mask = (
            meta_df["MetsDzPrimaryDiagnosisSite"].str.contains(
                "skin|ear|eyelid|vulva", case=False, na=False
            )
            & meta_df["MetastaticDiseaseInd"].str.contains(
                r"yes\s*-\s*(?:regional|nos)", case=False, na=False
            )
            & meta_df["MetastaticDiseaseSite"].str.contains(
                "skin|ear|eyelid|vulva|breast", case=False, na=False
            )
            & meta_df["_age_meta"].notna()
            & (meta_df["_age_meta"] <= age_at_specimen_collection_fudged)
        )

        no_row_hits_all_four = not four_crit_mask.any()

    if within_90d and no_row_hits_all_four:
        if pathological_group_stage.lower() in {
            "unknown/not reported",
            "no tnm applicable for this site/histology combination",
            "unknown/not applicable",
        }:
            return (roman_numeral_in(clinical_group_stage) or "Unknown"), "SKINLESS90D"
        return (roman_numeral_in(pathological_group_stage) or "Unknown"), "SKINLESS90D"


    # ────────────────────────────────────────────────────────────────
    # RULE 9 – SKINREG
    #
    # Stage III if EITHER of the following is true:
    #
    #   A)  There exists ≥ 1 metastatic-disease row that simultaneously has
    #        • MetsDzPrimaryDiagnosisSite ∼ "skin|ear|eyelid|vulva"
    #        • MetastaticDiseaseInd      ∼ "yes - regional|yes - nos"
    #        • MetastaticDiseaseSite     ∼ "skin|ear|eyelid|vulva|head|soft tissues|breast|lymph node"
    #        • AgeAtMetastaticSite       ≤ AgeSpec + AGE_FUDGE
    #
    #   B)  No metastatic-disease row simultaneously has
    #        • MetsDzPrimaryDiagnosisSite ∼ "skin|ear|eyelid|vulva"
    #        • MetastaticDiseaseInd       == "yes - distant"
    #        • AgeAtMetastaticSite        == "Age Unknown/Not Recorded"
    # ────────────────────────────────────────────────────────────────
    if not data_frame_of_metastatic_disease_data_for_patient.empty and age_at_specimen_collection is not None:
        meta_df = data_frame_of_metastatic_disease_data_for_patient.copy()

        # numeric version of AgeAtMetastaticSite
        meta_df["_age_meta"] = meta_df["AgeAtMetastaticSite"].apply(
            lambda x: 90.0 if str(x).strip() == "Age 90 or older" else _float(x)
        )

        # ----------  condition A  ----------
        cond_A_mask = (
            meta_df["MetsDzPrimaryDiagnosisSite"].str.contains(
                "skin|ear|eyelid|vulva", case=False, na=False
            )
            & meta_df["MetastaticDiseaseInd"].str.contains(
                r"yes\s*-\s*(?:regional|nos)", case=False, na=False
            )
            & meta_df["MetastaticDiseaseSite"].str.contains(
                "skin|ear|eyelid|vulva|head|soft tissues|breast|lymph node",
                case=False,
                na=False,
            )
            & meta_df["_age_meta"].notna()
            & (meta_df["_age_meta"] <= (age_at_specimen_collection + AGE_FUDGE))
        )

        # ----------  condition B  ----------
        cond_B_mask = (
            meta_df["MetsDzPrimaryDiagnosisSite"].str.contains(
                "skin|ear|eyelid|vulva", case=False, na=False
            )
            & meta_df["MetastaticDiseaseInd"].str.strip().str.lower().eq("yes - distant")
            & meta_df["AgeAtMetastaticSite"].str.strip().eq("Age Unknown/Not Recorded")
        )

        if cond_A_mask.any() or (not cond_B_mask.any()):
            return "III", "SKINREG"


    # ────────────────────────────────────────────────────────────────
    # RULE 10 – SKINUNK
    #
    # Stage IV if there exists at least one row in the metastatic disease
    # table that simultaneously has
    #   • MetsDzPrimaryDiagnosisSite matching "skin|ear|eyelid|vulva",
    #   • MetastaticDiseaseInd matching "yes - distant|yes - nos", and
    #   • AgeAtMetastaticSite of "Age Unknown/Not Recorded".
    # ────────────────────────────────────────────────────────────────
    if not data_frame_of_metastatic_disease_data_for_patient.empty:
        meta_df = data_frame_of_metastatic_disease_data_for_patient.copy()

        skinunk_mask = (
            meta_df["MetsDzPrimaryDiagnosisSite"].str.contains(
                "skin|ear|eyelid|vulva", case=False, na=False
            )
            & meta_df["MetastaticDiseaseInd"].str.contains(
                r"yes\s*-\s*(?:distant|nos)", case=False, na=False
            )
            & meta_df["AgeAtMetastaticSite"].str.strip().eq("Age Unknown/Not Recorded")
        )

        if skinunk_mask.any():
            return "IV", "SKINUNK"

    # Fallback (should never be reached according to ORIEN Specimen Staging Revised Rules)
    return "Unknown", "UNMATCHED"
    

def roman_numeral_in(stage: str) -> str | None:
    pattern = re.compile("(IV|III|II|I)(?![IV])", re.I)
    match = pattern.search(str(stage))
    return None if match is None else match.group(1).upper()
    
    
def numericize_age_at_metastatic_site(string_representation_of_age_at_metastatic_site: str):
    if str(string_representation_of_age_at_metastatic_site).strip() == "Age 90 or older":
        return 90.0
    elif str(string_representation_of_age_at_metastatic_site).strip() == "Age Unknown/Not Recorded":
        return pd.NA
    elif str(string_representation_of_age_at_metastatic_site).strip() == "Unknown/Not Applicable":
        return pd.NA
    else:
        return float(string_representation_of_age_at_metastatic_site)
    

ICB_PATTERN = re.compile("immune checkpoint|pembrolizumab|nivolumab|ipilimumab|atezolizumab|durvalumab|avelumab|cemiplimab|relatlimab", re.I)
CUTANEOUS_RE = re.compile(SITE_KEYWORDS["cutaneous"], re.I)



def _float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
    
    
def load_data(
    path_to_clinical_molecular_linkage_data: Path,
    path_to_diagnosis_data: Path,
    path_to_metastatic_disease_data: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    logging.info("Clinical molecular linkage data will be loaded.")
    data_frame_of_clinical_molecular_linkage_data = pd.read_csv(path_to_clinical_molecular_linkage_data, dtype = str)
    data_frame_of_clinical_molecular_linkage_data.columns = data_frame_of_clinical_molecular_linkage_data.columns.str.strip()
    mask_of_indicators_that_specimens_are_tumors = data_frame_of_clinical_molecular_linkage_data["Tumor/Germline"].str.lower() == "tumor"
    data_frame_of_clinical_molecular_linkage_data = data_frame_of_clinical_molecular_linkage_data[mask_of_indicators_that_specimens_are_tumors]

    logging.info("Diagnosis data will be loaded.")
    data_frame_of_diagnosis_data = pd.read_csv(path_to_diagnosis_data, dtype = str)
    data_frame_of_diagnosis_data.columns = data_frame_of_diagnosis_data.columns.str.strip()
    pattern_of_histology_codes_of_melanoma = re.compile(r"^87\d\d/\d$")
    mask_of_indicators_that_histology_codes_represent_melanoma = data_frame_of_diagnosis_data["HistologyCode"].str.match(pattern_of_histology_codes_of_melanoma, na = False)
    data_frame_of_diagnosis_data = data_frame_of_diagnosis_data[mask_of_indicators_that_histology_codes_represent_melanoma]

    logging.info("Metastatic disease data will be loaded.")
    data_frame_of_metastatic_disease_data = pd.read_csv(path_to_metastatic_disease_data, dtype = str)
    data_frame_of_metastatic_disease_data.columns = data_frame_of_metastatic_disease_data.columns.str.strip()

    return data_frame_of_clinical_molecular_linkage_data, data_frame_of_diagnosis_data, data_frame_of_metastatic_disease_data







def are_within_90_days(age_at_specimen_collection: float | None, age_diag: float | None) -> bool:
    if age_at_specimen_collection is None or age_diag is None:
        return False
    diff = age_at_specimen_collection - age_diag
    return 0 <= abs(diff) <= 90 / 365.25

        
def _hist_clean(txt: str) -> str:
    return re.sub("[^A-Za-z]", "", str(txt)).lower()


def main():
    parser = argparse.ArgumentParser(description = "Pair clinical data and stages of tumors.")
    parser.add_argument("--path_to_clinical_molecular_linkage_data", required = True, type = Path)
    parser.add_argument("--path_to_diagnosis_data", required = True, type = Path)
    parser.add_argument("--path_to_metastatic_disease_data", required = True, type = Path)
    parser.add_argument("--path_to_output_data", required = True, type = Path)
    args = parser.parse_args()

    logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")

    output_data = run_pipeline(
        path_to_clinical_molecular_linkage_data = args.path_to_clinical_molecular_linkage_data,
        path_to_diagnosis_data = args.path_to_diagnosis_data,
        path_to_metastatic_disease_data = args.path_to_metastatic_disease_data
    )

    logging.info(f"{len(output_data)} rows will be written to {args.path_to_output_data}.")
    output_data.to_csv(args.path_to_output_data, index = False)


if __name__ == "__main__":
    main()
