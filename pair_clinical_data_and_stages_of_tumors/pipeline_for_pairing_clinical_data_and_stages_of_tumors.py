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
        Field Name: Discrepancy (0/1 or No/Yes) 
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
        Field Name: Possible New Primary (0/1 or No/Yes) 
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
    CHANGE FROM PRIOR: Do not exclude any, can still use those with WES only for TMB analysis.
    SpecimenSiteofCollection is the correct field 
For the record: these two patients had the wrong SpecimenID in my file but the correct SpecimenID in by your code according to the rules below. Patient ID 317K6G9N41 should have SpecimenID = 53LFUMZSW8ACOX5IPUZ45ICJ0. Patient ID U2CUPQJ4T1 should have SpecimenID = HIWF190182C5JJE3FHYR5BIXR
    A few clarifications on the rules also in red text below.
        - If RNAseq is available for just one tumor, select the tumor with RNAseq data (even if no WES)
        - If RNAseq data is available for > 1 tumors OR if only WES is available for all tumors:
            - If the patient has a tumor with SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains "skin" and does not also have a tumor with SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains either "lymph node" or "soft tissue", then select the tumor[s] with SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains "skin"
                - If multiple skin specimens meet this rule, then use the one with earliest Age At Specimen Collection
            - If none of the patient's tumors have SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains "skin" or "soft tissue", then select the tumor with SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains "lymph node"
            - If a patient has a tumor with a SpecimenSiteofCollection [SpecimenSiteOfCollection] that contains ["skin" OR "soft tissue"] AND a tumor with a SpecimenSiteofCollection that contains "lymph node", then select the one [the tumor with specimen site of collection containing "skin" or "soft tissue" or the tumor with a specimen site of collection containing "lymph node"] with the earliest Age At Specimen Collection.
            
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
        

AGE_FUDGE = 0.005 # years, or approximately 1.8 days.
_ROMAN_RE = re.compile(r"\b(?:Stage\s*)?([IV]{1,3})(?:[ABCD])?\b", re.I)

ICB_PATTERN = re.compile(r"immune checkpoint|pembrolizumab|nivolumab|ipilimumab|atezolizumab|durvalumab|avelumab|cemiplimab|relatlimab", re.I)

CUTANEOUS_RE = re.compile(SITE_KEYWORDS["cutaneous"], re.I)
_SITE_LOCAL_RE = re.compile(r"skin|ear|eyelid|vulva|head|soft tissues|breast|lymph node|parotid|vagina", re.I)


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

def assign_stage_and_rule(
    spec: pd.Series,
    dx: pd.Series,
    data_frame_of_metastatic_disease_data_for_patient: pd.DataFrame
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
    age_coll = float(90.0 if spec.get("Age At Specimen Collection") == "Age 90 or older" else spec.get("Age At Specimen Collection"))
    
    if data_frame_of_metastatic_disease_data_for_patient.empty:
        MetsDzPrimaryDiagnosisSite = ""
        MetastaticDiseaseInd      = ""
        AgeAtMetastaticSite       = None
    else:
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
    
    age_spec = _float(spec.get("Age At Specimen Collection"))
    
    # Rule 0 - EXCEPTION
    '''
    From "ORIEN Specimen Staging Revised Rules":
    These are the exceptions to the staging rules. I imagine this should be coded before the other staging rules in your script, so I’ve named it Rule#0 EXCEPTION, but you can put it in your code differently if there is a better way.
    
    RULE#0 EXCEPTION
    - If AvatarKey = ["227RDTKST8" or "ILE2DL0KMW" or "YRGE6MVYNK" or "EKZGU61JTP"], then AssignedStage = III
    - If AvatarKey = ["GEM0S42KIH" or "CG6JRI0XIX" or "WDMTMU4SV2"], then AssignedStage = IV 
    - This rule applies to 6 patients, all cutaneous; 5 patients in Group A and 1 patient in Group C (ILE2DL0KMW)
    '''
    
    if spec["ORIENAvatarKey"] in ["227RDTKST8", "ILE2DL0KMW", "YRGE6MVYNK", "EKZGU61JTP", "3Z82S0R5IE", "MXL3WK5YF0"]:#, "DVJMQLZBJV"]:
        return "III", "EXCEPTION"
    elif spec["ORIENAvatarKey"] in ["GEM0S42KIH", "CG6JRI0XIX", "WDMTMU4SV2"]:
        return "IV", "EXCEPTION"
    
    # RULE 1 - AGE
    if age_diag_txt.lower() == "age 90 or older":
        return (_first_roman(path_stg or clin_stg) or "Unknown"), "AGE"

    # RULE 2 - PATHIV
    if "IV" in path_stg.upper():
        return "IV", "PATHIV"

    # RULE 3 - CLINIV
    if "IV" in clin_stg.upper():
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
        if path_stg.lower() in ["unknown/not reported", "no tnm applicable for this site/histology combination", "unknown/not applicable"]:
            return (_first_roman(clin_stg) or "Unknown"), "NOMETS"
        return (_first_roman(path_stg) or "Unknown"), "NOMETS"

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
                r"yes\s*-\s*(regional|nos)", case=False, na=False
            )
            & meta_df["MetastaticDiseaseSite"].str.contains(
                r"skin|ear|eyelid|vulva|breast", case=False, na=False
            )
            & meta_df["_age_meta"].notna()
            & (meta_df["_age_meta"] <= age_spec_fudged)
        )

        no_row_hits_all_four = not four_crit_mask.any()

    if within_90d and no_row_hits_all_four:
        if path_stg.lower() in {
            "unknown/not reported",
            "no tnm applicable for this site/histology combination",
            "unknown/not applicable",
        }:
            return (_first_roman(clin_stg) or "Unknown"), "SKINLESS90D"
        return (_first_roman(path_stg) or "Unknown"), "SKINLESS90D"


    # RULE 9 - SKINREG
    def find_match(pattern: str, text: str):
        """
        Search `text` for the first occurrence of `pattern`.
        Returns a tuple (matched: bool, match_str: str or None).
        """
        m = re.search(pattern, text)
        if m:
            return True, m.group(0) # group(0) is the entire match
        else:
            return False, None
    
    MetsDzPrimaryDiagnosisSite_matched, MetsDzPrimaryDiagnosisSite_match = find_match(r"skin|ear|eyelid|vulva", MetsDzPrimaryDiagnosisSite)
    MetastaticDiseaseInd_matched, MetastaticDiseaseInd_match = find_match(r"yes - regional|yes - nos", MetastaticDiseaseInd)
    MetastaticDiseaseInd_matched_on_yes_distant_or_yes_nos, MetastaticDiseaseInd_match_on_yes_distant_or_yes_nos = find_match(r"yes - distant|yes - nos", MetastaticDiseaseInd)
    AgeAtMetastaticSite_is_less_than_or_equal_to_Age_At_Specimen_Collection_fudged = AgeAtMetastaticSite is not None and age_spec is not None and AgeAtMetastaticSite <= age_spec_fudged
    
    MetastaticDiseaseSite_matched_on_more, MetastaticDiseaseSite_match_on_more = find_match(r"skin|ear|eyelid|vulva|head|soft tissues|breast|lymph node", metastatic_sites_text)
    MetastaticDiseaseInd_matched_on_yes_distant, MetastaticDiseaseInd_match_on_yes_distant = find_match(r"yes - distant", MetastaticDiseaseInd)
    if (
        (
            MetsDzPrimaryDiagnosisSite_matched and
            MetastaticDiseaseInd_matched and
            MetastaticDiseaseSite_matched_on_more and
            AgeAtMetastaticSite_is_less_than_or_equal_to_Age_At_Specimen_Collection_fudged
        ) or not (
            MetsDzPrimaryDiagnosisSite_matched and
            MetastaticDiseaseInd_matched_on_yes_distant and
            ("Age Unknown/Not Recorded" in data_frame_of_metastatic_disease_data_for_patient["AgeAtMetastaticSite"].to_list())
        )
    ):
        return "III", "SKINREG"

    # RULE 10 - SKINUNK
    if (
        MetsDzPrimaryDiagnosisSite_matched and
        MetastaticDiseaseInd_matched_on_yes_distant_or_yes_nos and
        ("Age Unknown/Not Recorded" in data_frame_of_metastatic_disease_data_for_patient["AgeAtMetastaticSite"].to_list())
    ):
        return "IV", "SKINUNK"

    # Fallback (should never be reached according to ORIEN Specimen Staging Revised Rules)
    return "Unknown", "UNMATCHED"


def _within_90_days_after(age_spec: float | None, age_diag: float | None) -> bool:
    if age_spec is None or age_diag is None:
        return False
    diff = age_spec - age_diag
    return 0 <= diff <= 90 / 365.25


def are_within_90_days(age_spec: float | None, age_diag: float | None) -> bool:
    if age_spec is None or age_diag is None:
        return False
    diff = age_spec - age_diag
    return 0 <= abs(diff) <= 90 / 365.25

        
def _hist_clean(txt: str) -> str:
    return re.sub(r"[^A-Za-z]", "", str(txt)).lower()


def select_diagnosis_for_patient_in_C(dx_patient: pd.DataFrame, spec_row: pd.Series, meta_patient: pd.DataFrame) -> pd.Series:
    age_spec = _float(spec_row["Age At Specimen Collection"])
    
    if dx_patient.empty:
        raise ValueError("No diagnosis rows supplied to _select_diagnosis_C")
    dxp = dx_patient.copy()
    age_diag = dxp["AgeAtDiagnosis"].apply(_float)
    
    prox = age_diag.apply(lambda x: _within_90_days_after(age_spec, x))
    
    primary_sites_lower  = dxp["PrimaryDiagnosisSite"].str.lower()
    specimen_site_origin = str(spec_row["SpecimenSiteOfOrigin"]).lower()
    site_match = primary_sites_lower == specimen_site_origin
    
    primary_met = spec_row["Primary/Met"].strip().lower()
    if primary_met == "primary":
        if prox.sum() == 1:
            return dxp[prox].iloc[0]

        unknown_age_exists   = age_diag.isna().any()
        primary_sites_differ = primary_sites_lower.nunique() > 1
        if (prox.sum() > 1 or unknown_age_exists) and primary_sites_differ:
            match_rows = dxp[site_match].copy()
            if not match_rows.empty:
                match_rows["_age"] = match_rows["AgeAtDiagnosis"].apply(_float)
                return match_rows.sort_values("_age", na_position = "last").iloc[0]

    elif primary_met == "metastatic":
        
        site_coll = spec_row["SpecimenSiteOfCollection"].lower()
        if "lymph node" not in site_coll:
            path_stage_raw = dxp["PathGroupStage"].fillna("").str.strip()
            path_stage_U   = path_stage_raw.str.upper()
            clin_stage_U   = dxp["ClinGroupStage"].fillna("").str.strip().str.upper()

            stage_iv_mask = (
                (path_stage_U == "IV") |
                (
                    (clin_stage_U == "IV") &
                    path_stage_raw.str.lower().isin({
                        "unknown/not reported",
                        "no tnm applicable for this site/histology combination",
                        "unknown/not applicable",
                    })
                )
            )

            if stage_iv_mask.sum() == 1:
                return dxp[stage_iv_mask].iloc[0]
            
            if stage_iv_mask.sum() == 0:
                dxp["_age"] = age_diag
                return dxp.sort_values("_age", na_position = "last").iloc[0]

        if "lymph node" in site_coll:
            if prox.sum() == 1:
                return dxp[prox].iloc[0]
            
            if prox.sum() > 1:
                pos_node = ~dxp["PathNStage"].str.contains(
                    r"N0|Nx|unknown/not applicable|no tnm applicable for this site/histology combination",
                    case = False,
                    na = False
                )
                if pos_node.sum() == 1:
                    return dxp[pos_node].iloc[0]
               
            if prox.all():
                
                if site_match.sum() == 1:
                    match_rows = dxp[site_match].copy()
                    if not match_rows.empty:
                        match_rows["_age"] = match_rows["AgeAtDiagnosis"].apply(_float)
                        return match_rows.sort_values("_age", na_position = "last").iloc[0]
                    
                if site_match.sum() == 0:
                    copy_of_dxp = dxp.copy()
                    dxp["_age"] = dxp["AgeAtDiagnosis"].apply(_float)
                    return dxp.sort_values("_age", na_position = "last").iloc[0]
                
    dxp["_age"] = dxp["AgeAtDiagnosis"].astype(float)
    return dxp.sort_values("_age").iloc[0]


def select_tumor_for_patient_in_D(patient_cm: pd.DataFrame) -> pd.Series:
    cm = patient_cm.copy()
    cm_rna = cm[cm["RNASeq"].notna()]
    if len(cm_rna) == 1:
        return cm_rna.iloc[0]
    if len(cm_rna) > 1:
        cm_rna["_age"] = cm_rna["Age At Specimen Collection"].apply(_float)
        return cm_rna.sort_values("_age", na_position = "last").iloc[0]
    cm["_age"] = cm["Age At Specimen Collection"].apply(_float)
    return cm.sort_values("_age", na_position = "last").iloc[0]


def select_diagnosis_for_patient_in_D(dx_patient: pd.DataFrame, spec_row: pd.Series) -> pd.Series:
    site = spec_row["SpecimenSiteOfCollection"].lower()
    dxp = dx_patient.copy()
    dxp["_age"] = dxp["AgeAtDiagnosis"].apply(_float)
    if re.search(r"lymph node", site):
        return dxp.sort_values("_age", ascending = False).iloc[0]
    if re.search(r"skin|soft tissue", site):
        return dxp.sort_values("_age").iloc[0]
    return dxp.sort_values("_age").iloc[0]


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
