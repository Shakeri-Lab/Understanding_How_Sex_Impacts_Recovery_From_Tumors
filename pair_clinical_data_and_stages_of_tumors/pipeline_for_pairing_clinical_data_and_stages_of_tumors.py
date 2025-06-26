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
                    r"N0|Nx|unknown/not applicable|no tnm applicable for this site/histology combination",
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


def are_within_90_days_after(age_spec: float | None, age_diag: float | None) -> bool:
    if age_spec is None or age_diag is None:
        return False
    diff = age_spec - age_diag
    return 0 <= diff <= 90 / 365.25


def are_greater_than_90_days_after(age_spec: float | None, age_diag: float | None) -> bool:
    if age_spec is None or age_diag is None:
        return False
    diff = age_spec - age_diag
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

    site_coll = str(series_of_clinical_molecular_linkage_data_for_patient.get("SpecimenSiteOfCollection", "")).lower()
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
    
    age_spec = _float(series_of_clinical_molecular_linkage_data_for_patient.get("Age At Specimen Collection"))
    
    
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
    
    Rule #1: Age
        - If AgeAtDiagnosis = "Age 90 or older", then AssignedStage = numerical value of PathGroupStage OR ClinGroupStage (if PathGroupStage is [“Unknown/Not Reported” OR “No TNM applicable for this site/histology combination” OR  “Unknown/Not Applicable”])
            - Total: 5 patients, all cutaneous
                - 4 in group A
                - 1 in group B (D42OPM2PLC)
    '''
    string_representation_of_age_at_diagnosis = str(series_of_diagnosis_data_for_patient.get("AgeAtDiagnosis", "")).strip().lower()
    pathological_group_stage = str(series_of_diagnosis_data_for_patient.get("PathGroupStage", "")).strip().lower()
    clinical_group_stage = str(series_of_diagnosis_data_for_patient.get("ClinGroupStage", "")).strip().lower()
    
    if string_representation_of_age_at_diagnosis == "age 90 or older":
        if pathological_group_stage in ["unknown/not reported", "no tnm applicable for this site/histology combination", "unknown/not applicable"]:
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

    # RULE 2 - PATHIV
    if "IV" in pathological_group_stage.upper():
        return "IV", "PATHIV"

    # RULE 3 - CLINIV
    if "IV" in clinical_group_stage.upper():
        return "IV", "CLINIV"

    # RULE 4 - METSITE
    if not _SITE_LOCAL_RE.search(site_coll):
        return "IV", "METSITE"

    # ────────────────────────────────────────────────────────────────
    # RULE 5 – PRIORDISTANT
    # A specimen is stage IV if -- in the same metastatic disease row --
    #   • MetsDzPrimaryDiagnosisSite matches a cutaneous / ocular / mucosal keyword
    #   • MetastaticDiseaseInd == "yes - distant"
    #   • AgeAtMetastaticSite <= Age At Specimen Collection (fudged)
    # ────────────────────────────────────────────────────────────────
    if (
        not data_frame_of_metastatic_disease_data_for_patient.empty
        and age_spec is not None
    ):
        meta_df = data_frame_of_metastatic_disease_data_for_patient.copy()

        # float-convert the metastatic ages (treat “Age 90 or older” as 90.0)
        meta_df["_age_meta"] = meta_df["AgeAtMetastaticSite"].apply(
            lambda x: 90.0 if str(x).strip() == "Age 90 or older" else _float(x)
        )

        # build one row-wise mask that requires all three criteria simultaneously
        priordistant_mask = (
            meta_df["MetsDzPrimaryDiagnosisSite"].str.contains(
                r"skin|ear|eyelid|vulva|eye|choroid|ciliary body|conjunctiva|sinus|gum|nasal|urethra",
                case = False,
                na = False,
            )
            & meta_df["MetastaticDiseaseInd"].str.strip().str.lower().eq("yes - distant")
            & meta_df["_age_meta"].notna()
            & (meta_df["_age_meta"] <= (age_spec + AGE_FUDGE))
        )

        if priordistant_mask.any():
            return "IV", "PRIORDISTANT"

    # RULE 6 - NOMETS
    if data_frame_of_metastatic_disease_data_for_patient.empty or data_frame_of_metastatic_disease_data_for_patient["MetastaticDiseaseInd"].str.lower().eq("no").all():
        if pathological_group_stage.lower() in ["unknown/not reported", "no tnm applicable for this site/histology combination", "unknown/not applicable"]:
            return (roman_numeral_in(clinical_group_stage) or "Unknown"), "NOMETS"
        return (roman_numeral_in(pathological_group_stage) or "Unknown"), "NOMETS"

    # RULE 7 - NODE
    if ("lymph node" in site_coll) or ("parotid" in site_coll):
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
    
    age_spec_fudged = None if age_spec is None else age_spec + AGE_FUDGE
    within_90d = are_within_90_days(age_spec_fudged, age_diag_f)

    no_row_hits_all_four = True
    if (
        within_90d
        and not data_frame_of_metastatic_disease_data_for_patient.empty
        and age_spec_fudged is not None
    ):
        meta_df = data_frame_of_metastatic_disease_data_for_patient.copy()

        # numeric metastatic age
        meta_df["_age_meta"] = meta_df["AgeAtMetastaticSite"].apply(
            lambda x: 90.0 if str(x).strip() == "Age 90 or older" else _float(x)
        )

        four_crit_mask = (
            meta_df["MetsDzPrimaryDiagnosisSite"].str.contains(
                r"skin|ear|eyelid|vulva", case=False, na=False
            )
            & meta_df["MetastaticDiseaseInd"].str.contains(
                r"yes\s*-\s*(?:regional|nos)", case=False, na=False
            )
            & meta_df["MetastaticDiseaseSite"].str.contains(
                r"skin|ear|eyelid|vulva|breast", case=False, na=False
            )
            & meta_df["_age_meta"].notna()
            & (meta_df["_age_meta"] <= age_spec_fudged)
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
    if not data_frame_of_metastatic_disease_data_for_patient.empty and age_spec is not None:
        meta_df = data_frame_of_metastatic_disease_data_for_patient.copy()

        # numeric version of AgeAtMetastaticSite
        meta_df["_age_meta"] = meta_df["AgeAtMetastaticSite"].apply(
            lambda x: 90.0 if str(x).strip() == "Age 90 or older" else _float(x)
        )

        # ----------  condition A  ----------
        cond_A_mask = (
            meta_df["MetsDzPrimaryDiagnosisSite"].str.contains(
                r"skin|ear|eyelid|vulva", case=False, na=False
            )
            & meta_df["MetastaticDiseaseInd"].str.contains(
                r"yes\s*-\s*(?:regional|nos)", case=False, na=False
            )
            & meta_df["MetastaticDiseaseSite"].str.contains(
                r"skin|ear|eyelid|vulva|head|soft tissues|breast|lymph node",
                case=False,
                na=False,
            )
            & meta_df["_age_meta"].notna()
            & (meta_df["_age_meta"] <= (age_spec + AGE_FUDGE))
        )

        # ----------  condition B  ----------
        cond_B_mask = (
            meta_df["MetsDzPrimaryDiagnosisSite"].str.contains(
                r"skin|ear|eyelid|vulva", case=False, na=False
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
                r"skin|ear|eyelid|vulva", case=False, na=False
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
    pattern = re.compile(r"(IV|III|II|I)(?![IV])", re.I)
    match = pattern.search(str(stage))
    return None if match is None else match.group(1).upper()
    

AGE_FUDGE = 0.005 # years, or approximately 1.8 days.
ICB_PATTERN = re.compile(r"immune checkpoint|pembrolizumab|nivolumab|ipilimumab|atezolizumab|durvalumab|avelumab|cemiplimab|relatlimab", re.I)
CUTANEOUS_RE = re.compile(SITE_KEYWORDS["cutaneous"], re.I)
_SITE_LOCAL_RE = re.compile(r"skin|ear|eyelid|vulva|head|soft tissues|breast|lymph node|parotid|vagina", re.I)



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







def are_within_90_days(age_spec: float | None, age_diag: float | None) -> bool:
    if age_spec is None or age_diag is None:
        return False
    diff = age_spec - age_diag
    return 0 <= abs(diff) <= 90 / 365.25

        
def _hist_clean(txt: str) -> str:
    return re.sub(r"[^A-Za-z]", "", str(txt)).lower()


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
