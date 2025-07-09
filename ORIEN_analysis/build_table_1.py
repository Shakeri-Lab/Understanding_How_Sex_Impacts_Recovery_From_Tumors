#!/usr/bin/env python3
'''
Build "Table 1. Sequencing and clinicopathological characteristics of patient tumor specimens.".
'''

from pathlib import Path
import numpy as np
import pandas as pd
import re


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


def create_stage_mask(series: pd.Series, stage: str) -> pd.Series:
    pattern = rf"(?:^|\|){re.escape(stage)}(?:\||$)"
    return series.fillna("").str.contains(pattern, regex = True)


def determine_ICB_status(
    patient_ID: str,
    list_of_ages_at_specimen_collection: list[float],
    dictionary_of_patients_IDs_and_lists_of_tuples_of_ages_at_medication_start_and_ICB_class
):
    '''
    Determine whether a patient is
    Naive to ICB medications or
    Experienced with ICB medications and
    Experienced with Anti-PD1 medications only,
    Experienced with Anti-CTLA4 medication only, or
    Experienced with both Anti-PD1 and Anti-CTLA4 medications.
    
    Compare a patient's earliest age at specimen collection fudged with
    ages at medication start for that patient.
    If there is no age at medication start less than or equal to minimum age at specimen collection fudged,
    then all ages at medication start are greater than minimum age at specimen collection fudged, and
    age at specimen collection fudged is before any ages at medication start,
    and patient is Naive.
    Otherwise, patient is Experienced. 
    '''
    minimum_age_at_specimen_collection = min(list_of_ages_at_specimen_collection) if list_of_ages_at_specimen_collection else np.nan
    list_of_tuples_of_ages_at_medication_start_and_ICB_class = dictionary_of_patients_IDs_and_lists_of_tuples_of_ages_at_medication_start_and_ICB_class.get(patient_ID, [])
    if not list_of_tuples_of_ages_at_medication_start_and_ICB_class or np.isnan(minimum_age_at_specimen_collection):
        return "Naive"
    list_of_tuples_of_ages_at_medication_start_less_than_or_equal_to_minimum_age_at_specimen_collection_fudged_and_ICB_class = [
        (age, clas)
        for age, clas in list_of_tuples_of_ages_at_medication_start_and_ICB_class
        if not np.isnan(age) and age <= minimum_age_at_specimen_collection + 0.005
    ]
    if not list_of_tuples_of_ages_at_medication_start_less_than_or_equal_to_minimum_age_at_specimen_collection_fudged_and_ICB_class:
        return "Naive"
    set_of_classes = {clas for _, clas in list_of_tuples_of_ages_at_medication_start_less_than_or_equal_to_minimum_age_at_specimen_collection_fudged_and_ICB_class}
    if set_of_classes == {"Anti-PD1"}:
        return "Anti-PD1 only"
    if set_of_classes == {"Anti-CTLA4"}:
        return "Anti-CTLA4 only"
    if set_of_classes == {"Anti-PD1", "Anti-CTLA4"}:
        return "Anti-PD1 and anti-CTLA4"
    return "Experienced"


def flatten(list_of_lists):
    return [item for sublist in list_of_lists if isinstance(sublist, list) for item in sublist]


def join_strings(series: pd.Series) -> str:
    list_of_strings = [string for string in series.dropna().unique() if str(string).strip()]
    return '|'.join(sorted(list_of_strings)) if list_of_strings else np.nan


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
    
    if any(keyword in lowercase_site for keyword in ["brain", "cerebellum", "frontal lobe", "occipital lobe", "temporal lobe"]):
        return "Brain"
    
    if any(keyword in lowercase_site for keyword in ["bone", "spine", "vertebral"]):
        return "Bone"

    return None


def main():
    
    # Read CSV files into data frames.
    PATH_TO_NORMALIZED_FILES = Path("../../Clinical_Data/24PRJ217UVA_NormalizedFiles")
    PATH_TO_CLINICAL_MOLECULAR_LINKAGE_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv"
    PATH_TO_PATIENT_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_PatientMaster_V4.csv"
    PATH_TO_TUMOR_MARKER_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_TumorMarker_V4.csv"
    PATH_TO_OUTPUT_OF_PIPELINE_FOR_PAIRING_CLINICAL_DATA_AND_STAGES_OF_TUMORS = "../pair_clinical_data_and_stages_of_tumors/output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv"
    PATH_TO_MEDICATIONS_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_Medications_V4.csv"
    
    clinical_molecular_linkage_data = pd.read_csv(PATH_TO_CLINICAL_MOLECULAR_LINKAGE_DATA, dtype = str)
    patient_data = pd.read_csv(PATH_TO_PATIENT_DATA, dtype = str)
    tumor_marker_data = pd.read_csv(PATH_TO_TUMOR_MARKER_DATA, dtype = str)
    output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors = pd.read_csv(PATH_TO_OUTPUT_OF_PIPELINE_FOR_PAIRING_CLINICAL_DATA_AND_STAGES_OF_TUMORS, dtype = str)
    medications_data = pd.read_csv(PATH_TO_MEDICATIONS_DATA, dtype = str)

    # Create a data frame of cutaneous tumors.
    series_of_IDs_of_patients_with_cutaneous_tumors = output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.loc[
        output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors["AssignedPrimarySite"].str.lower() == "cutaneous", "AvatarKey"
    ]
    tumor_data = clinical_molecular_linkage_data.loc[
        (clinical_molecular_linkage_data["Tumor/Germline"].str.lower() == "tumor") &
        clinical_molecular_linkage_data["ORIENAvatarKey"].isin(series_of_IDs_of_patients_with_cutaneous_tumors)
    ].reset_index(drop = True).copy()
    
    print("We determined a data frame of clinical molecular data corresponding to cutaneous tumors.")
    print(f"Number of rows in tumor data initially: {len(tumor_data)}")
    print(f"Number of unique patients with tumor data initially: {len(tumor_data["ORIENAvatarKey"].unique())}")
    print(f"Number of unique patients with WES data initially: {len(tumor_data[tumor_data["WES"].notna()]["ORIENAvatarKey"].unique())}")
    print(tumor_data.head(n = 3))
    
    # Classify patients as having WES only, RNA sequencing only, or both WES and RNA sequencing.
    wes_any = tumor_data["WES"].notna().groupby(tumor_data["ORIENAvatarKey"]).transform("any")
    rna_any = tumor_data["RNASeq"].notna().groupby(tumor_data["ORIENAvatarKey"]).transform("any")
    tumor_data["class_of_sequencing_data"] = np.select(
        [
            wes_any & ~rna_any,
            ~wes_any & rna_any,
            wes_any & rna_any
        ],
        [
            "WES only",
            "RNAseq only",
            "WES and RNAseq"
        ],
        default = "None"
    )

    print(
        "We determined a class of sequencing data for each patient " +
        "and then broadcasted that class back to every row belonging to that patient. " +
        "Here is a slice of tumor data with patient IDs and classes:"
    )
    print(tumor_data[["ORIENAvatarKey", "class_of_sequencing_data"]].head(n = 3))
    
    # Classify specimen sites of collection and assign each patient a string of classes.
    tumor_data["class_of_specimen_site_of_collection"] = tumor_data["SpecimenSiteOfCollection"].apply(classify_specimen_site_of_collection)
    series_of_patient_IDs_and_strings_of_classes_of_specimen_sites_of_collection = tumor_data.groupby("ORIENAvatarKey")["class_of_specimen_site_of_collection"].apply(join_strings)
    tumor_data["string_of_classes_of_specimen_site_of_collection"] = tumor_data["ORIENAvatarKey"].map(series_of_patient_IDs_and_strings_of_classes_of_specimen_sites_of_collection)
    
    print(
        "We determined a string of classes of specimen sites of collection for each patient and " +
        "then broadcasted that string back to every row belonging to that patient. " +
        "Here is a slice of tumor data with patient IDs and strings:"
    )
    print(tumor_data[["ORIENAvatarKey", "string_of_classes_of_specimen_site_of_collection"]].head(n = 3))
    
    # Assign each patient a string of ages at specimen collection.
    series_of_patient_IDs_and_strings_of_ages_at_specimen_collection = tumor_data.groupby("ORIENAvatarKey")["Age At Specimen Collection"].apply(join_strings)
    tumor_data["string_of_ages_at_specimen_collection"] = tumor_data["ORIENAvatarKey"].map(series_of_patient_IDs_and_strings_of_ages_at_specimen_collection)
    tumor_data["list_of_ages_at_specimen_collection"] = tumor_data["string_of_ages_at_specimen_collection"].apply(parse_string_of_ages_at_specimen_collection)

    print(
        "We determined a string and a list of ages at specimen collection for each patient and " +
        "then broadcasted that string and that list back to every row belonging to that patient. " +
        "Here is a slice of tumor data with patient IDs, strings, and lists:"
    )
    print(tumor_data[["ORIENAvatarKey", "string_of_ages_at_specimen_collection", "list_of_ages_at_specimen_collection"]].head(n = 3))
    
    # Add stages for each patient.
    tumor_data = tumor_data.merge(
        output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors[["AvatarKey", "EKN Assigned Stage"]],
        left_on = "ORIENAvatarKey",
        right_on = "AvatarKey",
        how = "left"
    )

    print(
        "The number of rows in tumor data after merging column \"EKN Assigned Stage\" " +
        f"from output of pipeline for pairing clinical data and stages of tumors is {len(tumor_data)}."
    )

    # Assign each patient a string of stages.
    series_of_patient_IDs_and_strings_of_stages = tumor_data.groupby("ORIENAvatarKey")["EKN Assigned Stage"].apply(join_strings)
    tumor_data["string_of_stages"] = tumor_data["ORIENAvatarKey"].map(series_of_patient_IDs_and_strings_of_stages)

    print(
        "We determined a string of stages for each patient and " +
        "then broadcasted that string to every row belonging to that patient. " +
        "Here is a slice of tumor data with patient IDs and strings:"
    )
    print(tumor_data[["ORIENAvatarKey", "string_of_stages"]].head(n = 3))

    # Keep 1 row of denormalized tumor data per patient.
    tumor_data = tumor_data.drop_duplicates(subset = "ORIENAvatarKey", ignore_index = True)
    
    print(
        "After denormalizing classes and strings, " +
        "we drop duplicate rows in tumor data by patient ID. " +
        f"The number of rows of tumor data after dropping duplicates is {len(tumor_data)}."
    )
    
    # Add sex for each patient.
    tumor_data = tumor_data.merge(
        patient_data[["AvatarKey", "Sex"]],
        left_on = "ORIENAvatarKey",
        right_on = "AvatarKey",
        how = "left"
    )
    tumor_data["Sex"] = tumor_data["Sex"].str.title()

    print(f"The number of rows in tumor data after merging column \"Sex\" from patient data is {len(tumor_data)}.")
    
    # Add melanoma driver genes for each patient.
    tumor_marker_data = tumor_marker_data[
        (tumor_marker_data["TMarkerTest"].str.contains("BRAF") | tumor_marker_data["TMarkerTest"].str.contains("NRAS") | tumor_marker_data["TMarkerTest"].str.contains("PTEN")) &
        tumor_marker_data["TMarkerResult"] == "Positive"
    ]

    tumor_data = tumor_data.merge(
        tumor_marker_data[["AvatarKey", "TMarkerTest"]],
        left_on = "ORIENAvatarKey",
        right_on = "AvatarKey",
        how = "left"
    )

    print(
        "We filter tumor marker data to have " +
        "rows with values in column `TMarkerTest` containing \"BRAF\", \"NRAS\", or \"PTEN\" and " +
        "corresponding values in column `TMarkerResult` containing \"Positive\"."
    )
    print(f"The number of rows in tumor data after merging column `TMarkerTest` from tumor marker data is {len(tumor_data)}.")
    
    '''
    # Assign each patient a string of melanoma driver genes.
    series_of_patient_IDs_and_strings_of_melanoma_driver_genes = tumor_data.groupby("ORIENAvatarKey")["TMarkerTest"].apply(join_strings)
    tumor_data["string_of_melanoma_driver_genes"] = tumor_data["ORIENAvatarKey"].map(series_of_patient_IDs_and_strings_of_melanoma_driver_genes)
    
    print(
        "We determined a string of melanoma driver genes for each patient and " +
        "then broadcasted that string back to every row belonging to that patient. " +
        "Here is a slice of tumor data with patient IDs and strings:"
    )
    print(tumor_data[["ORIENAvatarKey", "string_of_melanoma_driver_genes"]].head(n = 3))

    # Keep 1 row of denormalized tumor data per patient.
    tumor_data = tumor_data.drop_duplicates(subset = "ORIENAvatarKey", ignore_index = True)
    
    print(
        "After denormalizing strings, " +
        "we drop duplicate rows in tumor data by patient ID. " +
        f"The number of rows of tumor data after dropping duplicates is {len(tumor_data)}."
    )
    '''
    
    # Assign an ICB class (e.g., "Anti-PD1") to each medication and trim medications data to rows with ICB classes.
    medications_data["ICB_class"] = medications_data["Medication"].map(classify_ICB_medication)
    medications_data = medications_data[medications_data["ICB_class"].notna()]
    medications_data["AgeAtMedStart"] = medications_data["AgeAtMedStart"].apply(numericize_age)

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
    dictionary_of_patients_IDs_and_lists_of_tuples_of_ages_at_medication_start_and_ICB_class = (
        medications_data
        .dropna(subset = ["AgeAtMedStart"])
        .groupby("AvatarKey")[["AgeAtMedStart", "ICB_class"]]
        .apply(
            lambda df: list(
                zip(df["AgeAtMedStart"], df["ICB_class"])
            )
        )
        .to_dict()
    )

    # Assign an ICB status to each patient.
    tumor_data["ICB_status"] = [
        determine_ICB_status(
            patient_ID,
            list_of_ages_at_specimen_collection,
            dictionary_of_patients_IDs_and_lists_of_tuples_of_ages_at_medication_start_and_ICB_class
        )
        for patient_ID, list_of_ages_at_specimen_collection in zip(tumor_data["ORIENAvatarKey"], tumor_data["list_of_ages_at_specimen_collection"])
    ]

    print(f"The number of rows in tumor data after assigning an ICB status to each patient is {len(tumor_data)}.")

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

    # Create list of rows of statistics re specimen collection sites.
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
                    tumor_data["string_of_classes_of_specimen_site_of_collection"].str.contains(specimen_collection_site),
                    tumor_data
                )
            }
            for specimen_collection_site in list_of_specimen_collection_sites
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
                    #tumor_data["string_of_melanoma_driver_genes"].fillna("").str.upper().str.contains(melanoma_driver_mutation),
                    tumor_data["TMarkerTest"].str.contains(melanoma_driver_mutation),
                    tumor_data
                )
            }
            for melanoma_driver_mutation in list_of_melanoma_driver_mutations
        )
    ]

    # Create list of rows of statistics re ages at specimen collection.
    list_of_values_of_edges_of_bins = [-np.inf, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
    list_of_labels_of_bins = ["< 20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
    list_of_rows_of_statistics_re_age_categories = [{"Characteristic": "Age (years)"}]
    for i, label in enumerate(list_of_labels_of_bins):
        lower_bound, upper_bound = list_of_values_of_edges_of_bins[i], list_of_values_of_edges_of_bins[i + 1]
        mask = tumor_data["list_of_ages_at_specimen_collection"].apply(
            lambda list_of_ages, lower_bound = lower_bound, upper_bound = upper_bound: any(
                lower_bound <= age < upper_bound for age in list_of_ages
            )
        )
        list_of_rows_of_statistics_re_age_categories.append(
            {
                "Characteristic": label,
                **summarize(
                    mask,
                    tumor_data
                )
            }
        )
    array_of_all_ages = np.array(flatten(tumor_data["list_of_ages_at_specimen_collection"]))
    array_of_ages_of_males = np.array(flatten(tumor_data.loc[tumor_data["Sex"] == "Male", "list_of_ages_at_specimen_collection"]))
    array_of_ages_of_females = np.array(flatten(tumor_data.loc[tumor_data["Sex"] == "Female", "list_of_ages_at_specimen_collection"]))
    list_of_rows_of_statistics_re_age_categories += [
        {
            "Characteristic": "Mean age",
            "Male": round(array_of_ages_of_males.mean(), 0),
            "Female": round(array_of_ages_of_females.mean(), 0),
            "Total": round(array_of_all_ages.mean(), 0)
        },
        {
            "Characteristic": "Median age",
            "Male": round(np.median(array_of_ages_of_males), 0),
            "Female": round(np.median(array_of_ages_of_females), 0),
            "Total": round(np.median(array_of_all_ages), 0)
        }
    ]
    
    # Create list of rows of statistics re stages.
    list_of_rows_of_statistics_re_stages = [
        {"Characteristic": "Stage"},
        *(
            {
                "Characteristic": stage,
                **summarize(
                    create_stage_mask(tumor_data["string_of_stages"], stage),
                    tumor_data
                )
            }
            for stage in ["II", "III", "IV"]
        )
    ]

    # Create list of rows of statistics re ICB statuses.
    list_of_rows_re_ICB_statuses = list(rows_re_ICB_statuses(tumor_data))
    
    # Assemble Table 1.
    table_1 = pd.DataFrame(
        list_of_rows_of_statistics_re_sequencing_data +
        list_of_rows_of_statistics_re_specimen_collection_sites + 
        list_of_rows_of_statistics_re_melanoma_driver_mutations +
        list_of_rows_of_statistics_re_age_categories +
        list_of_rows_of_statistics_re_stages +
        list_of_rows_re_ICB_statuses
    )

    # Round numeric values to 0 decimal places.
    numeric_columns = table_1.columns.drop("Characteristic")
    table_1[numeric_columns] = table_1[numeric_columns].apply(
        lambda column: column.map(
            lambda value: value if pd.notna(value) else ""
        )
    )

    # Rename columns of statistics according to cohort and numbers of patients in cohort.
    table_1.columns = [
        "Characteristic, N (%)",
        f"Male (N = {number_of_tumors_of_males})",
        f"Female (N = {number_of_tumors_of_females})",
        f"Total (N = {number_of_tumors})"
    ]
    
    # Print Table 1.
    print("\nTable 1. Sequencing and clinicopathological characteristics of patient tumour specimens.\n")
    print(table_1.to_string(index = False))

    
def numericize_age(age: str):
    if age == "Age 90 or older":
        return 90.0
    if age == "Age Unknown/Not Recorded":
        return np.nan
    return float(age)
    

def parse_string_of_ages_at_specimen_collection(string_of_ages_at_specimen_collection: str) -> list[float]:
    list_of_ages = []
    for string_representation_of_age in string_of_ages_at_specimen_collection.split('|'):
        list_of_ages.append(90.0 if string_representation_of_age == "Age 90 or older" else float(string_representation_of_age))
    return list_of_ages


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