#!/usr/bin/env python3
'''
Build "Table 1. Sequencing and clinicopathological characteristics of patient tumor specimens.".
'''

from pathlib import Path
import numpy as np
import pandas as pd


def flatten(list_of_lists):
    return [item for sublist in list_of_lists if isinstance(sublist, list) for item in sublist]


def join_strings(series: pd.Series) -> str:
    list_of_strings = [string for string in series.dropna().unique() if str(string).strip()]
    return '|'.join(sorted(list_of_strings)) if list_of_strings else np.nan


def map_site(site: str) -> str:
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
        ("upper limb, NOS" in lowercase_site) or
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
    PATH_TO_NORMALIZED_FILES = Path("../../Clinical_Data/24PRJ217UVA_NormalizedFiles")
    PATH_TO_CLINICAL_MOLECULAR_LINKAGE_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv"
    PATH_TO_PATIENT_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_PatientMaster_V4.csv"
    PATH_TO_TUMOR_MARKER_DATA = PATH_TO_NORMALIZED_FILES / "24PRJ217UVA_20241112_TumorMarker_V4.csv"
    PATH_TO_OUTPUT_OF_PIPELINE_FOR_PAIRING_CLINICAL_DATA_AND_STAGES_OF_TUMORS = "../pair_clinical_data_and_stages_of_tumors/output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv"
    
    clinical_molecular_linkage_data = pd.read_csv(PATH_TO_CLINICAL_MOLECULAR_LINKAGE_DATA, dtype = str)
    patient_data = pd.read_csv(PATH_TO_PATIENT_DATA, dtype = str)
    tumor_marker_data = pd.read_csv(PATH_TO_TUMOR_MARKER_DATA, dtype = str)
    output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors = pd.read_csv(PATH_TO_OUTPUT_OF_PIPELINE_FOR_PAIRING_CLINICAL_DATA_AND_STAGES_OF_TUMORS, dtype = str)

    series_of_IDs_of_patients_with_cutaneous_tumors = output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.loc[output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors["AssignedPrimarySite"].str.lower() == "cutaneous", "AvatarKey"]
    tumor_data = clinical_molecular_linkage_data.loc[
        (clinical_molecular_linkage_data["Tumor/Germline"].str.lower() == "tumor") &
        clinical_molecular_linkage_data["ORIENAvatarKey"].isin(series_of_IDs_of_patients_with_cutaneous_tumors)
    ].reset_index(drop = True).copy()
    print("We determined a data frame of clinical molecular data corresponding to cutaneous tumors.")
    print(f"Number of rows in tumor data initially: {len(tumor_data)}")
    print(f"Number of unique patients with tumor data initially: {len(tumor_data["ORIENAvatarKey"].unique())}")
    print(f"Number of unique patients with WES data initially: {len(tumor_data[tumor_data["WES"].notna()]["ORIENAvatarKey"].unique())}")
    print(tumor_data.head(n = 3))
    
    wes_any = tumor_data["WES"].notna().groupby(tumor_data["ORIENAvatarKey"]).transform("any")
    rna_any = tumor_data["RNASeq"].notna().groupby(tumor_data["ORIENAvatarKey"]).transform("any")
    tumor_data["sequencing_data_category"] = np.select(
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
    print("We determined an indicator for each patient of whether any of that patient's rows in tumor data contained WES or RNA sequencing data and then broadcasted that indicator back to every row belonging to that patient. Here is a slice of tumor data with patient IDs and indicators:")
    print(tumor_data[["ORIENAvatarKey", "sequencing_data_category"]].head(n = 3))
    
    tumor_data["specimen_site_of_collection"] = tumor_data["SpecimenSiteOfCollection"].apply(map_site)
    patient_to_sites = tumor_data.groupby("ORIENAvatarKey")["specimen_site_of_collection"].apply(join_strings)
    tumor_data["specimen_site_of_collection"] = tumor_data["ORIENAvatarKey"].map(patient_to_sites)
    print("We determined a string of specimen sites of collection for each patient and then broadcasted that string back to every row belonging to that patient. Here is a slice of tumor data with patient IDs and strings:")
    print(tumor_data[["ORIENAvatarKey", "specimen_site_of_collection"]].head(n = 3))
    
    patient_to_ages = tumor_data.groupby("ORIENAvatarKey")["Age At Specimen Collection"].apply(join_strings)
    tumor_data["string_of_ages"] = tumor_data["ORIENAvatarKey"].map(patient_to_ages)
    print("We determined a string of ages at specimen collection for each patient and then broadcasted that string back to every row belonging to that patient. Here is a slice of tumor data with patient IDs and strings:")
    print(tumor_data[["ORIENAvatarKey", "string_of_ages"]].head(n = 3))
    
    tumor_data = tumor_data.drop_duplicates(subset = "ORIENAvatarKey", ignore_index = True)
    print(f"Number of rows in tumor data after dropping duplicates: {len(tumor_data)}")
    
    tumor_data = tumor_data.merge(
        patient_data[["AvatarKey", "Sex"]],
        left_on = "ORIENAvatarKey",
        right_on = "AvatarKey",
        how = "left"
    )
    tumor_data["Sex"] = tumor_data["Sex"].str.title()
    print(f"Number of rows in tumor data after merging column \"Sex\" from patient data: {len(tumor_data)}")
    
    tumor_data = tumor_data.merge(
        tumor_marker_data[["AvatarKey", "TMarkerTest"]],
        left_on = "ORIENAvatarKey",
        right_on = "AvatarKey",
        how = "left"
    )
    print(f"Number of rows in tumor data after merging column \"TMarkerTest\" from tumor marker data: {len(tumor_data)}")
    
    patient_to_melanoma_driver_genes = tumor_data.groupby("ORIENAvatarKey")["TMarkerTest"].apply(join_strings)
    tumor_data["TMarkerTest"] = tumor_data["ORIENAvatarKey"].map(patient_to_melanoma_driver_genes)
    print("We determined a string of melanoma driver genes for each patient and then broadcasted that string back to every row belonging to that patient. Here is a slice of tumor data with patient IDs and strings:")
    print(tumor_data[["ORIENAvatarKey", "TMarkerTest"]].head(n = 3))
    
    tumor_data = tumor_data.drop_duplicates(subset = "ORIENAvatarKey", ignore_index = True)
    print(f"Number of rows in tumor data after dropping duplicates: {len(tumor_data)}")
    
    tumor_data = tumor_data.merge(
        output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors[["AvatarKey", "EKN Assigned Stage"]],
        left_on = "ORIENAvatarKey",
        right_on = "AvatarKey",
        how = "left"
    )
    print(f"Number of rows in tumor data after merging column \"EKN Assigned Stage\" from output of pipeline for pairing clinical data and stages of tumors: {len(tumor_data)}")
    
    number_of_tumors = len(tumor_data)
    number_of_tumors_of_males, number_of_tumors_of_females = tumor_data["Sex"].value_counts().astype(int)
    print(f"Number of tumors: {number_of_tumors}")
    print(f"Number of tumors of males: {number_of_tumors_of_males}")
    print(f"Number of tumors of females: {number_of_tumors_of_females}")

    list_of_rows_of_statistics_re_sequencing_data = [
        dict(Characteristic = "Sequencing data"),
        *(
            {
                "Characteristic": sequencing_data_category,
                **summarize(
                    tumor_data["sequencing_data_category"] == sequencing_data_category,
                    tumor_data,
                    number_of_tumors_of_males,
                    number_of_tumors_of_females,
                    number_of_tumors
                )
            }
            for sequencing_data_category in ["WES only", "RNAseq only", "WES and RNAseq"]
        )
    ]

    list_of_specimen_collection_sites = [
        "Skin and other soft tissues",
        "Lymph node",
        "Lung",
        "Abdominal viscera",
        "Brain",
        "Bone"
    ]
    list_of_rows_of_statistics_re_specimen_collection_sites = [
        dict(Characteristic = "Specimen collection site"),
        *(
            {
                "Characteristic": specimen_collection_site,
                **summarize(
                    tumor_data["specimen_site_of_collection"].str.contains(specimen_collection_site),
                    tumor_data,
                    number_of_tumors_of_males,
                    number_of_tumors_of_females,
                    number_of_tumors
                )
            }
            for specimen_collection_site in list_of_specimen_collection_sites
        )
    ]

    list_of_melanoma_driver_mutations = ["BRAF", "NRAS", "PTEN"]
    list_of_rows_of_statistics_re_melanoma_driver_mutations = [
        dict(Characteristic = "Melanoma driver mutations"),
        *(
            {
                "Characteristic": melanoma_driver_mutation,
                **summarize(
                    tumor_data["TMarkerTest"].str.upper().str.contains(melanoma_driver_mutation),
                    tumor_data,
                    number_of_tumors_of_males,
                    number_of_tumors_of_females,
                    number_of_tumors
                )
            }
            for melanoma_driver_mutation in list_of_melanoma_driver_mutations
        )
    ]

    tumor_data["list_of_ages"] = tumor_data["string_of_ages"].apply(parse_string_of_ages)
    list_of_values_of_edges_of_bins = [-np.inf, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
    list_of_labels_of_bins = ["< 20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
    list_of_rows_of_statistics_re_age_categories = [dict(Characteristic = "Age (years)")]
    for i, label in enumerate(list_of_labels_of_bins):
        lower_bound, upper_bound = list_of_values_of_edges_of_bins[i], list_of_values_of_edges_of_bins[i + 1]
        mask = tumor_data["list_of_ages"].apply(lambda list_of_ages, lower_bound = lower_bound, upper_bound = upper_bound: any(lower_bound <= age < upper_bound for age in list_of_ages))
        list_of_rows_of_statistics_re_age_categories.append(
            {
                "Characteristic": label,
                **summarize(
                    mask,
                    tumor_data,
                    number_of_tumors_of_males,
                    number_of_tumors_of_females,
                    number_of_tumors
                )
            }
        )
    array_of_all_ages = np.array(flatten(tumor_data["list_of_ages"]))
    array_of_ages_of_males = np.array(flatten(tumor_data.loc[tumor_data["Sex"] == "Male", "list_of_ages"]))
    array_of_ages_of_females = np.array(flatten(tumor_data.loc[tumor_data["Sex"] == "Female", "list_of_ages"]))
    list_of_rows_of_statistics_re_age_categories += [
        {
            "Characteristic": "Mean age",
            "Male": array_of_ages_of_males.mean(),
            "Female": array_of_ages_of_females.mean(),
            "Total": array_of_all_ages.mean()
        },
        {
            "Characteristic": "Median age",
            "Male": np.median(array_of_ages_of_males),
            "Female": np.median(array_of_ages_of_females),
            "Total": np.median(array_of_all_ages)
        }
    ]
    
    list_of_rows_of_statistics_re_stages = [
        dict(Characteristic = "Stage"),
        *(
            {
                "Characteristic": stage,
                **summarize(
                    tumor_data["EKN Assigned Stage"] == stage,
                    tumor_data,
                    number_of_tumors_of_males,
                    number_of_tumors_of_females,
                    number_of_tumors
                )
            }
            for stage in ["II", "III", "IV"]
        )
    ]
    
    # TODO: Format all numeric values in table 1 to 1 decimal place.
    table_1 = pd.DataFrame(
        list_of_rows_of_statistics_re_sequencing_data +
        list_of_rows_of_statistics_re_specimen_collection_sites + 
        list_of_rows_of_statistics_re_melanoma_driver_mutations +
        list_of_rows_of_statistics_re_age_categories +
        list_of_rows_of_statistics_re_stages
    )
    table_1.columns = [
        "Characteristic",
        f"Male (N = {number_of_tumors_of_males})",
        f"Female (N = {number_of_tumors_of_females})",
        f"Total (N = {number_of_tumors})"
    ]
    
    print("\nTable 1. Sequencing and clinicopathological characteristics of patient tumour specimens.\n")
    print(table_1.to_string(index = False))


def parse_string_of_ages(string_of_ages: str) -> list[float]:
    list_of_string_representations_of_ages = string_of_ages.split('|')
    list_of_ages = []
    for string_representation_of_age in list_of_string_representations_of_ages:
        if string_representation_of_age == "Age 90 or older":
            list_of_ages.append(90.0)
        else:
            list_of_ages.append(float(string_representation_of_age))
    return list_of_ages


def summarize(mask, tumor_data, number_of_tumors_of_males, number_of_tumors_of_females, number_of_tumors):
    return pd.Series(
        {
            "Male": (mask & (tumor_data["Sex"] == "Male")).sum(),
            "Female": (mask & (tumor_data["Sex"] == "Female")).sum(),
            "Total": mask.sum()
        }
    )


if __name__ == "__main__":
    main()