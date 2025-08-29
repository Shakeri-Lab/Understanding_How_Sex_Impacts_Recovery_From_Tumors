#!/usr/bin/env python3
'''
Build
"Table 1. Sequencing and clinicopathological characteristics of patient tumor specimens." and
"Table 2. Patient baseline characteristics. Demographic and clinical characteristics at the time of diagnosis."

Usage:
../../miniconda3/envs/ici_sex/bin/python build_tables.py
'''

from pathlib import Path
import numpy as np
import pandas as pd


def classify_ICB_medication(name: str) -> str | None:
    '''
    Classify a medication corresponding to a provided name as "Anti-PD1", "Anti-CTLA4", or None. 
    '''
    lowercase_name = str(name).lower()
    if any (keyword in lowercase_name for keyword in ["nivolumab", "nivolumab-relatlimab-rmbw", "pembrolizumab", "atezolizumab"]):
        return "Anti-PD1"
    if "ipilimumab" in lowercase_name:
        return "Anti-CTLA4"
    return None


def determine_ICB_status(
    patient_ID: str,
    age_at_specimen_collection: float,
    dictionary_of_patients_IDs_and_lists_of_tuples_of_ages_at_medication_start_and_ICB_classes
):
    '''
    Determine whether a patient is
    Naive to ICB medications or
    Experienced with ICB medications and
    Experienced with Anti-PD1 medications only,
    Experienced with Anti-CTLA4 medication only, or
    Experienced with both Anti-PD1 and Anti-CTLA4 medications.
    
    Compare a specimen's age at specimen collection fudged with
    ages at medication start for that patient.
    If there is no age at medication start less than or equal to age at specimen collection fudged,
    then all ages at medication start are greater than age at specimen collection fudged,
    age at specimen collection fudged is before any ages at medication start, and
    patient is Naive.
    Otherwise, patient is Experienced. 
    '''
    list_of_tuples_of_ages_at_medication_start_and_ICB_classes = dictionary_of_patients_IDs_and_lists_of_tuples_of_ages_at_medication_start_and_ICB_classes.get(patient_ID, [])
    if not list_of_tuples_of_ages_at_medication_start_and_ICB_classes or age_at_specimen_collection is None:
        return "Naive"
    list_of_tuples_of_ages_at_medication_start_less_than_or_equal_to_age_at_specimen_collection_fudged_and_ICB_classes = [
        (age, clas)
        for age, clas in list_of_tuples_of_ages_at_medication_start_and_ICB_classes
        if not np.isnan(age) and age <= age_at_specimen_collection + 0.005
    ]
    if not list_of_tuples_of_ages_at_medication_start_less_than_or_equal_to_age_at_specimen_collection_fudged_and_ICB_classes:
        return "Naive"
    set_of_classes = {clas for _, clas in list_of_tuples_of_ages_at_medication_start_less_than_or_equal_to_age_at_specimen_collection_fudged_and_ICB_classes}
    if set_of_classes == {"Anti-PD1"}:
        return "Anti-PD1 only"
    if set_of_classes == {"Anti-CTLA4"}:
        return "Anti-CTLA4 only"
    if set_of_classes == {"Anti-PD1", "Anti-CTLA4"}:
        return "Anti-PD1 and anti-CTLA4"
    return "Experienced"

def classify_primary_diagnosis_site(site: str) -> str | None:
    '''
    Classify a primary diagnosis site as
    "Head and neck",
    "Upper extremity",
    "Trunk",
    "Lower extremity",
    "Vulva", or
    "Skin, NOS"
    '''
    lowercase_site = str(site).lower()
    if any(keyword in lowercase_site for keyword in ["ear", "eyelid", "lip", "scalp", "face"]):
        return "Head and neck"
    if "upper limb" in lowercase_site:
        return "Upper extremity"
    if "trunk" in lowercase_site:
        return "Trunk"
    if "lower limb" in lowercase_site:
        return "Lower extremity"
    if "Vulva, NOS".lower() in lowercase_site:
        return "Vulva"
    if "Skin, NOS".lower() in lowercase_site:
        return "Skin, NOS"

def classify_specimen_site_of_collection(site: str) -> str | None:
    '''
    Classify a specimen site of collection as
    "Skin and other soft tissues",
    "Lymph node",
    "Lung",
    "Abdominal viscera",
    "Brain",
    "Bone", or
    None.
    '''
    lowercase_site = str(site).lower()
    
    if (
        ("skin" in lowercase_site) or
        ("soft tissue" in lowercase_site and not "lymph node" in lowercase_site) or
        ("breast" in lowercase_site) or
        ("ear" in lowercase_site) or
        ("eyelid" in lowercase_site) or
        ("head" in lowercase_site and not "lymph node" in lowercase_site) or
        ("muscle" in lowercase_site) or
        ("thorax" in lowercase_site) or
        ("upper limb, NOS".lower() in lowercase_site) or
        ("vulva" in lowercase_site)
    ):
        return "Skin and other soft tissues"
    
    if any(keyword in lowercase_site for keyword in ["lymph node", "parotid"]):
        return "Lymph node"
    
    if any(keyword in lowercase_site for keyword in ["lower lobe", "lung", "trachea", "upper lobe"]):
        return "Lung"
    
    if any(
        keyword in lowercase_site
        for keyword in [
            "anus",
            "adrenal",
            "colon",
            "gallbladder",
            "ileum",
            "jejunum",
            "kidney",
            "liver",
            "retroperitoneum",
            "small intestine",
            "peritoneum",
            "spleen",
            "ureter",
            "vagina"
        ]
    ):
        return "Abdominal viscera"    
    
    if any(
        keyword in lowercase_site
        for keyword in ["brain", "cerebellum", "frontal lobe", "occipital lobe", "temporal lobe", "parietal lobe"]
    ):
        return "Brain"
    
    if any(keyword in lowercase_site for keyword in ["bone", "spine", "vertebral"]):
        return "Bone"

    return None


def main():
    
    # Read CSV files into data frames.
    PATH_TO_NORMALIZED_FILES = Path("../../../Clinical_Data/24PRJ217UVA_NormalizedFiles")
    PATH_TO_CLINICAL_MOLECULAR_LINKAGE_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv"
    PATH_TO_PATIENT_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_PatientMaster_V4.csv"
    PATH_TO_PATIENT_DATA_WITH_EMILY_NINMERS_ETHNICITIES = "24PRJ217UVA_20241112_PatientMaster_V4_Ethnicity.csv"
    PATH_TO_TUMOR_MARKER_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_TumorMarker_V4.csv"
    PATH_TO_OUTPUT_OF_PIPELINE_FOR_PAIRING_CLINICAL_DATA_AND_STAGES_OF_TUMORS = "../../pair_clinical_data_and_stages_of_tumors/output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv"
    PATH_TO_MEDICATIONS_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_Medications_V4.csv"
    PATH_TO_DIAGNOSIS_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_Diagnosis_V4.csv"
    
    clinical_molecular_linkage_data = pd.read_csv(PATH_TO_CLINICAL_MOLECULAR_LINKAGE_DATA, dtype = str)
    patient_data = pd.read_csv(PATH_TO_PATIENT_DATA, dtype = str)
    patient_data_with_Emily_Ninmers_ethnicities = pd.read_csv(PATH_TO_PATIENT_DATA_WITH_EMILY_NINMERS_ETHNICITIES, dtype = str)
    tumor_marker_data = pd.read_csv(PATH_TO_TUMOR_MARKER_DATA, dtype = str)
    output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors = pd.read_csv(PATH_TO_OUTPUT_OF_PIPELINE_FOR_PAIRING_CLINICAL_DATA_AND_STAGES_OF_TUMORS, dtype = str)
    output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors["index_of_row_of_diagnosis_data_paired_with_specimen"] = pd.to_numeric(
        output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors["index_of_row_of_diagnosis_data_paired_with_specimen"],
        errors = "raise"
    )
    medications_data = pd.read_csv(PATH_TO_MEDICATIONS_DATA, dtype = str)
    diagnosis_data = pd.read_csv(PATH_TO_DIAGNOSIS_DATA, dtype = str)
    diagnosis_data = diagnosis_data.reset_index().rename(columns = {"index": "index_of_row_of_diagnosis_data"})
    
    # Create CSV file with data from output of pipeline and diagnosis data.
    data_from_output_of_pipeline_and_diagnosis_data = output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors[
        ["AvatarKey", "Group", "index_of_row_of_diagnosis_data_paired_with_specimen"]
    ].merge(
        diagnosis_data[
            ["AvatarKey", "index_of_row_of_diagnosis_data", "ClinTStage", "ClinNStage", "ClinMStage", "ClinGroupStage", "PathTStage", "PathNStage", "PathMStage", "PathGroupStage", "TNMEditionNumber"]
        ],
        left_on = ["AvatarKey", "index_of_row_of_diagnosis_data_paired_with_specimen"],
        right_on = ["AvatarKey", "index_of_row_of_diagnosis_data"],
        how = "left"
    )
    data_from_output_of_pipeline_and_diagnosis_data = data_from_output_of_pipeline_and_diagnosis_data[
        ~data_from_output_of_pipeline_and_diagnosis_data["TNMEditionNumber"].str.contains("Eighth Edition", regex = False)
    ]
    data_from_output_of_pipeline_and_diagnosis_data.to_csv("data_from_output_of_pipeline_and_diagnosis_data.csv", index = False)

    # Create a data frame of clinical data, and data in output of pipeline, of cutaneous tumors in output of pipeline.
    tumor_data = clinical_molecular_linkage_data.merge(
        output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors[
            [
                "AvatarKey",
                "ORIENSpecimenID",
                "AssignedPrimarySite",
                "EKN Assigned Stage",
                "index_of_row_of_diagnosis_data_paired_with_specimen"
            ]
        ],
        left_on = ["ORIENAvatarKey", "DeidSpecimenID"],
        right_on = ["AvatarKey", "ORIENSpecimenID"],
        how = "left"
    ).drop(columns = ["AvatarKey", "ORIENSpecimenID"])
    tumor_data = tumor_data.loc[
        (tumor_data["Tumor/Germline"].str.lower() == "tumor") &
        (tumor_data["AssignedPrimarySite"].str.lower() == "cutaneous")
    ].reset_index(drop = True)
    tumor_data = tumor_data.merge(
        diagnosis_data[
            [
                "AvatarKey",
                "index_of_row_of_diagnosis_data",
                "AgeAtDiagnosis",
                "PathGroupStage",
                "ClinGroupStage",
                "TNMEditionNumber",
                "PrimaryDiagnosisSite",
                "PerformStatusAtDiagnosisScale",
                "PerformStatusAtDiagnosis"
            ]
        ],
        left_on = ["ORIENAvatarKey", "index_of_row_of_diagnosis_data_paired_with_specimen"],
        right_on = ["AvatarKey", "index_of_row_of_diagnosis_data"],
        how = "left"
    ).drop(columns = ["AvatarKey", "index_of_row_of_diagnosis_data"])
    INVALID_STAGE_VALUES = {
        "",
        "Unknown/Not Reported",
        "Unknown/Not Applicable",
        "No TNM applicable for this site/histology combination",
    }

    # Print out initial statistics about tumor data.
    number_of_rows_in_tumor_data = len(tumor_data)
    number_of_unique_patients_with_tumor_data = len(tumor_data["ORIENAvatarKey"].unique())
    number_of_unique_patients_with_tumors_with_RNA_sequencing = len(
        tumor_data.loc[
            tumor_data["RNASeq"].notna() & (tumor_data["RNASeq"].str.strip() != ""),
            "ORIENAvatarKey"
        ]
    )
    number_of_unique_patients_with_tumors_with_WES = len(
        tumor_data.loc[
            tumor_data["WES"].notna() & (tumor_data["WES"].str.strip() != ""),
            "ORIENAvatarKey"
        ]
        .unique()
    )
    print(
        "We determined a data frame of clinical data, and " +
        "data from output of pipeline, corresponding to cutaneous tumors in output of pipeline."
    )
    print(f"The number of rows is {number_of_rows_in_tumor_data}.")
    print(f"The number of unique patients is {number_of_unique_patients_with_tumor_data}.")
    print(f"The number of unique patients with tumors with WES is {number_of_unique_patients_with_tumors_with_WES}.")
    print(f"The number of unique patients with tumors with RNA sequencing is {number_of_unique_patients_with_tumors_with_RNA_sequencing}.")
    print("The head of tumor data is")
    print(tumor_data.head(n = 3))

    # Classify specimens as having WES only, RNA sequencing only, or both WES and RNA sequencing.
    series_of_indicators_that_specimen_has_WES = tumor_data["WES"].notna() & (tumor_data["WES"].str.strip() != "")
    series_of_indicators_that_specimen_has_RNA_sequencing = tumor_data["RNASeq"].notna() & tumor_data["RNASeq"].notna() & (tumor_data["RNASeq"].str.strip() != "")
    tumor_data["class_of_sequencing_data"] = np.select(
        [
            series_of_indicators_that_specimen_has_WES & ~series_of_indicators_that_specimen_has_RNA_sequencing,
            ~series_of_indicators_that_specimen_has_WES & series_of_indicators_that_specimen_has_RNA_sequencing,
            series_of_indicators_that_specimen_has_WES & series_of_indicators_that_specimen_has_RNA_sequencing,
            ~series_of_indicators_that_specimen_has_WES & ~series_of_indicators_that_specimen_has_RNA_sequencing
        ],
        [
            "WES only",
            "RNAseq only",
            "WES and RNAseq",
            "not WES and not RNAseq"
        ],
        default = "None"
    )

    print(
        "We determined a class of sequencing data for each specimen. Here is a slice of tumor data with specimen IDs and classes:"
    )
    print(tumor_data[["DeidSpecimenID", "class_of_sequencing_data"]].head(n = 3))
    
    # Classify sites and assign each specimen a class.
    tumor_data["class_of_specimen_site_of_collection"] = tumor_data["SpecimenSiteOfCollection"].apply(classify_specimen_site_of_collection)
    tumor_data["class_of_primary_diagnosis_site"] = tumor_data["PrimaryDiagnosisSite"].apply(classify_primary_diagnosis_site)

    print(
        "We determined classes of sites for each specimen. Here are slices of tumor data with specimen IDs and classes:"
    )
    print(tumor_data[["DeidSpecimenID", "class_of_specimen_site_of_collection"]].head(n = 3))
    print(tumor_data[["DeidSpecimenID", "class_of_primary_diagnosis_site"]].head(n = 3))
    
    # Add melanoma driver genes for each patient.
    tumor_marker_data = tumor_marker_data[
        (
            tumor_marker_data["TMarkerTest"].str.contains("BRAF", regex = False) |
            tumor_marker_data["TMarkerTest"].str.contains("NRAS", regex = False) |
            tumor_marker_data["TMarkerTest"].str.contains("PTEN", regex = False)
        ) &
        tumor_marker_data["TMarkerResult"] == "Positive"
    ]
    tumor_data = tumor_data.merge(
        tumor_marker_data[["AvatarKey", "TMarkerTest"]],
        left_on = "ORIENAvatarKey",
        right_on = "AvatarKey",
        how = "left"
    ).drop(columns = "AvatarKey")

    print(
        "We filter tumor marker data to have " +
        "rows with values in column `TMarkerTest` containing \"BRAF\", \"NRAS\", or \"PTEN\" and " +
        "corresponding values in column `TMarkerResult` containing \"Positive\"."
    )
    print(f"The number of rows in tumor data after merging column `TMarkerTest` from tumor marker data is {len(tumor_data)}.")
    
    # Numericize ages at specimen collection, ages at medication start, and ages at diagnosis.
    tumor_data["age_at_specimen_collection"] = tumor_data["Age At Specimen Collection"].apply(numericize_age)
    medications_data["age_at_med_start"] = medications_data["AgeAtMedStart"].apply(numericize_age)
    tumor_data["age_at_diagnosis"] = tumor_data["AgeAtDiagnosis"].apply(numericize_age)
    
    print("We numericize ages at specimen collection, ages at medication start, and ages at diagnosis.")

    # Assign an ICB class (e.g., "Anti-PD1") to each medication and filter medications data to rows with ICB classes.
    medications_data["ICB_class"] = medications_data["Medication"].map(classify_ICB_medication)
    medications_data = medications_data[medications_data["ICB_class"].notna()]

    print("We assign an ICB class to each medication and filter medications data to rows with ICB classes.")

    '''
    Create a dictionary of patient IDs and lists of tuples of
    ages at medication start and ICB classes of the form
    {
        patient ID: [
            (age at medication start, ICB class),
            (age at medication start, ICB class)
        ]
    }
    '''
    dictionary_of_patient_IDs_and_lists_of_tuples_of_ages_at_medication_start_and_ICB_classes = (
        medications_data
        .dropna(subset = ["age_at_med_start"])
        .groupby("AvatarKey")[["age_at_med_start", "ICB_class"]]
        .apply(
            lambda df: list(
                zip(df["age_at_med_start"], df["ICB_class"])
            )
        )
        .to_dict()
    )

    print("We create a dictionary of patient IDs and lists of tuples of ages at medication start and ICB classes.")

    # Assign an ICB status to each specimen.
    tumor_data["ICB_status"] = [
        determine_ICB_status(
            patient_ID,
            age_at_specimen_collection,
            dictionary_of_patient_IDs_and_lists_of_tuples_of_ages_at_medication_start_and_ICB_classes
        )
        for patient_ID, age_at_specimen_collection
        in zip(tumor_data["ORIENAvatarKey"], tumor_data["age_at_specimen_collection"])
    ]

    print("We assign an ICB status to each specimen.")

    # Add sex for each patient.
    tumor_data = tumor_data.merge(
        patient_data[["AvatarKey", "Sex", "Race", "Ethnicity"]],
        left_on = "ORIENAvatarKey",
        right_on = "AvatarKey",
        how = "left"
    ).drop(columns = "AvatarKey")
    tumor_data["Sex"] = tumor_data["Sex"].str.title()
    
    tumor_data = tumor_data.merge(
        patient_data_with_Emily_Ninmers_ethnicities[["AvatarKey", "EthnicityEKN"]],
        left_on = "ORIENAvatarKey",
        right_on = "AvatarKey",
        how = "left"
    ).drop(columns = "AvatarKey")
    tumor_data["EthnicityEKN"] = tumor_data["EthnicityEKN"].fillna("Unknown")

    print(f"The number of rows in tumor data after merging patient data is {len(tumor_data)}.")

    '''
    Create a data frame of patient IDs and denormalized and normalized ethnicities
    for the 379 patients in patient data.
    '''
    data_frame_of_patient_IDs_and_ethnicities_for_patients_in_patient_data = (
        patient_data[["AvatarKey", "Ethnicity"]]
        .assign(
            Ethnicity = lambda df: df["Ethnicity"].fillna(""),
            normalized_ethnicity = lambda df: df["Ethnicity"].str.split(r"\s*;\s*")
        )
        .explode("normalized_ethnicity")
        .drop_duplicates(ignore_index = True)
        .sort_values(by = "AvatarKey")
        .reset_index(drop = True)
    )
    number_of_unique_patient_IDs = len(data_frame_of_patient_IDs_and_ethnicities_for_patients_in_patient_data["AvatarKey"].unique())
    print(
        "The number of unique patient IDs in " +
        f"data frame of patient IDs and ethnicities for patients in patient data is " +
        f"{number_of_unique_patient_IDs}."
    )
    data_frame_of_patient_IDs_and_ethnicities_for_patients_in_patient_data.to_csv(
        "data_frame_of_patient_IDs_and_ethnicities_for_patients_in_patient_data.csv",
        index = False
    )

    '''
    Create a data frame of patient IDs and denormalized and normalized ethnicities
    for the 366 patients in tumor data.
    '''
    data_frame_of_patient_IDs_and_ethnicities_for_patients_in_tumor_data = tumor_data[["ORIENAvatarKey", "Ethnicity"]].merge(
        data_frame_of_patient_IDs_and_ethnicities_for_patients_in_patient_data,
        left_on = ["ORIENAvatarKey", "Ethnicity"],
        right_on = ["AvatarKey", "Ethnicity"],
        how = "left"
    ).drop(columns = ["AvatarKey"])
    number_of_unique_patient_IDs = len(data_frame_of_patient_IDs_and_ethnicities_for_patients_in_tumor_data["ORIENAvatarKey"].unique())
    print(
        "The number of unique patient IDs in " +
        "data frame of patient IDs and ethnicities for patients in tumor data is " +
        f"{number_of_unique_patient_IDs}."
    )
    data_frame_of_patient_IDs_and_ethnicities_for_patients_in_tumor_data.to_csv(
        "data_frame_of_patient_IDs_and_ethnicities_for_patients_in_tumor_data.csv",
        index = False
    )

    tumor_data["clean_pathological_group_stage"] = tumor_data["PathGroupStage"].fillna("").str.strip()
    tumor_data["clean_clinical_group_stage"] = tumor_data["ClinGroupStage"].fillna("").str.strip()
    series_of_indicators_that_pathological_group_stage_exists = ~tumor_data["clean_pathological_group_stage"].isin(INVALID_STAGE_VALUES)
    series_of_indicators_that_clinical_group_stage_exists = ~tumor_data["clean_clinical_group_stage"].isin(INVALID_STAGE_VALUES)
    tumor_data["Stage"] = np.select(
        [
            series_of_indicators_that_pathological_group_stage_exists,
            ~series_of_indicators_that_pathological_group_stage_exists & series_of_indicators_that_clinical_group_stage_exists
        ],
        [
            tumor_data["clean_pathological_group_stage"],
            tumor_data["clean_clinical_group_stage"]
        ],
        default = "Unknown"
    )

    print("We add column \"Stage\" to tumor data for Table 2.")

    tumor_data["clean_performance_status_at_diagnosis_scale"] = tumor_data["PerformStatusAtDiagnosisScale"].fillna("").str.strip()
    tumor_data["clean_performance_status_at_diagnosis"] = tumor_data["PerformStatusAtDiagnosis"].fillna("").str.strip()
    series_of_indicators_that_performance_status_at_diagnosis_scale_contains_ECOG = tumor_data["clean_performance_status_at_diagnosis_scale"].str.contains("ECOG", regex = False)
    series_of_first_numbers = tumor_data["clean_performance_status_at_diagnosis"].str.extract(r"(\d+)", expand = False).astype(float)
    tumor_data["ECOG PS"] = np.where(
        series_of_indicators_that_performance_status_at_diagnosis_scale_contains_ECOG,
        series_of_first_numbers.map(map_to_class_of_ECOG_performance_status),
        "Unknown"
    )

    print("We add column \"ECOG PS\" to tumor data for Table 2.")

    # Print summary statistics.
    number_of_tumors = len(tumor_data)
    number_of_tumors_of_males = tumor_data["Sex"].eq("Male").sum()
    number_of_tumors_of_females = tumor_data["Sex"].eq("Female").sum()

    print(f"At this point, the number of tumors in our data is {number_of_tumors}.")
    print(f"The number of tumors of males in our data is {number_of_tumors_of_males}.")
    print(f"The number of tumors of females in our data is {number_of_tumors_of_females}.")

    # Create list of rows of statistics re sequencing data.
    list_of_rows_of_statistics_re_sequencing_data = [
        {"Characteristic": "Sequencing data"},
        *(
            {
                "Characteristic": sequencing_data_category,
                **summarize(
                    tumor_data["class_of_sequencing_data"] == sequencing_data_category,
                    tumor_data
                )
            }
            for sequencing_data_category in ["WES only", "RNAseq only", "WES and RNAseq"]
        )
    ]

    # Create lists of rows of statistics re sites.
    list_of_specimen_collection_sites = [
        "Skin and other soft tissues",
        "Lymph node",
        "Lung",
        "Abdominal viscera",
        "Brain",
        "Bone"
    ]
    list_of_rows_of_statistics_re_specimen_collection_sites = [
        {"Characteristic": "Specimen collection site"},
        *(
            {
                "Characteristic": specimen_collection_site,
                **summarize(
                    tumor_data["class_of_specimen_site_of_collection"] == specimen_collection_site,
                    tumor_data
                )
            }
            for specimen_collection_site in list_of_specimen_collection_sites
        )
    ]

    list_of_primary_diagnosis_sites = [
        "Head and neck",
        "Upper extremity",
        "Trunk",
        "Lower extremity",
        "Vulva",
        "Skin, NOS"
    ]
    list_of_rows_of_statistics_re_primary_diagnosis_sites = [
        {"Characteristic": "Primary Diagnosis Site"},
        *(
            {
                "Characteristic": primary_diagnosis_site,
                **summarize(
                    tumor_data["class_of_primary_diagnosis_site"] == primary_diagnosis_site,
                    tumor_data
                )
            }
            for primary_diagnosis_site in list_of_primary_diagnosis_sites
        )
    ]

    # Create list of rows of statistics re melanoma driver mutations.
    list_of_melanoma_driver_mutations = ["BRAF", "NRAS", "PTEN"]
    list_of_rows_of_statistics_re_melanoma_driver_mutations = [
        {"Characteristic": "Melanoma driver mutations"},
        *(
            {
                "Characteristic": melanoma_driver_mutation,
                **summarize(
                    tumor_data["TMarkerTest"].str.contains(melanoma_driver_mutation, regex = False),
                    tumor_data
                )
            }
            for melanoma_driver_mutation in list_of_melanoma_driver_mutations
        )
    ]

    # Create lists of rows of statistics re ages at specimen collection and ages at diagnosis.
    list_of_rows_of_statistics_re_ages_at_specimen_collection = [{"Characteristic": "Age (years)"}]
    list_of_rows_of_statistics_re_ages_at_diagnosis = [{"Characteristic": "Age (years)"}]
    list_of_types_of_ages = ["age_at_specimen_collection", "age_at_diagnosis"]
    dictionary_of_type_of_age_and_list = {
        "age_at_specimen_collection": list_of_rows_of_statistics_re_ages_at_specimen_collection,
        "age_at_diagnosis": list_of_rows_of_statistics_re_ages_at_diagnosis
    }

    list_of_values_of_edges_of_bins = [-np.inf, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
    list_of_labels_of_bins = ["< 20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
    for i, label in enumerate(list_of_labels_of_bins):
        lower_bound = list_of_values_of_edges_of_bins[i]
        upper_bound = list_of_values_of_edges_of_bins[i + 1]
        def evaluate_whether_age_is_in_bin(age, lower_bound = lower_bound, upper_bound = upper_bound):
            return lower_bound <= age < upper_bound

        for type_of_age in list_of_types_of_ages:
            mask = tumor_data[type_of_age].apply(evaluate_whether_age_is_in_bin)
            dictionary_of_type_of_age_and_list[type_of_age].append(
                {
                    "Characteristic": label,
                    **summarize(
                        mask,
                        tumor_data
                    )
                }
            )

    for type_of_age in list_of_types_of_ages:
        array_of_all_ages = tumor_data[type_of_age]
        array_of_ages_of_males = tumor_data.loc[tumor_data["Sex"] == "Male", type_of_age]
        array_of_ages_of_females = tumor_data.loc[tumor_data["Sex"] == "Female", type_of_age]
        dictionary_of_type_of_age_and_list[type_of_age] += [
            {
                "Characteristic": "Mean age",
                "Male": int(round(array_of_ages_of_males.mean(), 0)),
                "Female": int(round(array_of_ages_of_females.mean(), 0)),
                "Total": int(round(array_of_all_ages.mean(), 0))
            },
            {
                "Characteristic": "Median age",
                "Male": int(round(np.median(array_of_ages_of_males), 0)),
                "Female": int(round(np.median(array_of_ages_of_females), 0)),
                "Total": int(round(np.median(array_of_all_ages), 0))
            }
        ]
    
    # Create list of rows of statistics re stages.
    list_of_rows_of_statistics_re_stages_for_Table_1 = [
        {"Characteristic": "Stage"},
        *(
            {
                "Characteristic": stage,
                **summarize(
                    tumor_data["EKN Assigned Stage"] == stage,
                    tumor_data
                )
            }
            for stage in ["II", "III", "IV"]
        )
    ]

    # Create list of rows of statistics re ICB statuses.
    list_of_rows_re_ICB_statuses = list(rows_re_ICB_statuses(tumor_data))
    
    list_of_statistics_re_races = [
        {"Characteristic": "Race"},
        *(
            {
                "Characteristic": race,
                **summarize(
                    tumor_data["Race"] == race,
                    tumor_data
                )
            }
            for race in [
                "American Indian or Alaska Native",
                "Black or African American",
                "Chinese",
                "Korean",
                "Some other race",
                "Unknown by patient",
                "White"
            ]
        )
    ]

    list_of_statistics_re_ethnicities = [
        {"Characteristic": "Ethnicity"},
        *(
            {
                "Characteristic": ethnicity,
                **summarize(
                    tumor_data["EthnicityEKN"] == ethnicity,
                    tumor_data
                )
            }
            for ethnicity in [
                "Hispanic",
                "Non-Hispanic",
                "Unknown"
            ]
        )
    ]

    list_of_statistics_re_stages_for_Table_2 = [
        {"Characteristic": "Stage"},
        *(
            {
                "Characteristic": stage,
                **summarize(
                    tumor_data["Stage"] == stage,
                    tumor_data
                )
            }
            for stage in [
                "IA",
                "IB",
                "IIA",
                "IIB",
                "IIC",
                "III",
                "IIIA",
                "IIIB",
                "IIIC",
                "IIID",
                "IV",
                "IVA",
                "IVB",
                "IVC"
            ]
        )
    ]

    # Create list of rows of statistics re ECOG performance statuses.
    list_of_rows_of_statistics_re_ECOG_performance_statuses = [
        {"Characteristic": "ECOG PS"},
        *(
            {
                "Characteristic": ECOG_PS,
                **summarize(
                    tumor_data["ECOG PS"] == ECOG_PS,
                    tumor_data
                )
            }
            for ECOG_PS in ["0", "1", "2", "3+"]
        )
    ]
    
    data_frame_of_IDs_of_patients_specimens_and_WES = tumor_data[["ORIENAvatarKey", "DeidSpecimenID", "WES", "WES Batch"]]
    data_frame_of_IDs_of_patients_specimens_and_WES.to_csv("data_frame_of_IDs_of_patients_specimens_and_WES.csv", index = False)

    # Assemble tables.
    table_1 = pd.DataFrame(
        list_of_rows_of_statistics_re_sequencing_data +
        list_of_rows_of_statistics_re_specimen_collection_sites + 
        list_of_rows_of_statistics_re_melanoma_driver_mutations +
        list_of_rows_of_statistics_re_ages_at_specimen_collection +
        list_of_rows_of_statistics_re_stages_for_Table_1 +
        list_of_rows_re_ICB_statuses
    )
    table_2 = pd.DataFrame(
        list_of_rows_of_statistics_re_ages_at_diagnosis +
        list_of_statistics_re_races +
        list_of_statistics_re_ethnicities +
        list_of_statistics_re_stages_for_Table_2 +
        list_of_rows_of_statistics_re_primary_diagnosis_sites +
        list_of_rows_of_statistics_re_ECOG_performance_statuses
    )

    dictionary_of_names_of_tables_and_tables = {
        "Table 1. Sequencing and clinicopathological characteristics of patient tumour specimens.": table_1,
        "Table 2. Patient baseline characteristics. Demographic and clinical characteristics at the time of diagnosis.": table_2
    }
    for name_of_table in [
        "Table 1. Sequencing and clinicopathological characteristics of patient tumour specimens.",
        "Table 2. Patient baseline characteristics. Demographic and clinical characteristics at the time of diagnosis."
    ]:
        # Avoid showing NA in tables.
        table = dictionary_of_names_of_tables_and_tables[name_of_table]
        numeric_columns = table.columns.drop("Characteristic")
        table[numeric_columns] = table[numeric_columns].apply(
            lambda column: column.map(
                lambda value: value if pd.notna(value) else ""
            )
        )

        # Rename columns of statistics according to cohort and numbers of patients in cohort.
        table.columns = [
            "Characteristic, N (%)",
            f"Male (N = {number_of_tumors_of_males})",
            f"Female (N = {number_of_tumors_of_females})",
            f"Total (N = {number_of_tumors})"
        ]
    
        # Print tables.
        print(f"\n{name_of_table}\n")
        print(table.to_string(index = False))


def map_to_class_of_ECOG_performance_status(value) -> str:
    '''
    Map an integer or NA to a class of ECOG performance status.
    0 maps to "0".
    1 maps to "1".
    2 maps to "2".
    A number greater than or equal to 3 maps to "3+".
    NA maps to "Unknown".
    '''
    if pd.isna(value):
        return "Unknown"
    if value >= 3:
        return "3+"
    return str(int(value))

    
def numericize_age(age: str):
    if age == "Age 90 or older":
        return 90.0
    if age in ["Age Unknown/Not Recorded", "Unknown/Not Applicable"]:
        return np.nan
    return float(age)


def provide_number_and_percent(n, d):
    return f"{n} ({100 * n / d:.1f})" if d else "0 (0.0)"


def rows_re_ICB_statuses(tumor_data):
    yield {"Characteristic": "ICB Status"}
    tuple_of_categories = (
        ("Naive", tumor_data["ICB_status"] == "Naive"),
        ("Experienced", tumor_data["ICB_status"] != "Naive"),
        ("Anti-PD1 only", tumor_data["ICB_status"] == "Anti-PD1 only"),
        ("Anti-CTLA4 only", tumor_data["ICB_status"] == "Anti-CTLA4 only"),
        ("Anti-PD1 and anti-CTLA4", tumor_data["ICB_status"] == "Anti-PD1 and anti-CTLA4")
    )
    for label, mask in tuple_of_categories:
        yield {
            "Characteristic": label,
            **summarize(
                mask,
                tumor_data
            )
        }


def summarize(mask, tumor_data):
    '''
    Create a series of numbers of patients, females, and males meeting a condition.
    '''
    number_of_tumors = len(tumor_data)
    number_of_tumors_of_males = tumor_data["Sex"].eq("Male").sum()
    number_of_tumors_of_females = tumor_data["Sex"].eq("Female").sum()
    return pd.Series(
        {
            "Male": provide_number_and_percent((mask & (tumor_data["Sex"] == "Male")).sum(), number_of_tumors_of_males),
            "Female": provide_number_and_percent((mask & (tumor_data["Sex"] == "Female")).sum(), number_of_tumors_of_females),
            "Total": provide_number_and_percent(mask.sum(), number_of_tumors)
        }
    )


if __name__ == "__main__":
    main()