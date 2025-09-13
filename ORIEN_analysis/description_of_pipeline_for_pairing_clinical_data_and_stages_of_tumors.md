# System Description

*Pipeline for Pairing Clinical Data and Stages of Tumors*

Created on 06/01/2025 by Tom Lever

Updated on 06/30/2025 by Tom Lever


## Context

Shakeri et al. are translating genomic and transcriptomic data that describe melanoma and clinical data that provide context into predictions relating to responses to Immune Checkpoint Blockade therapy and prognoses. Molecular data may include Whole Exome Sequencing, RNA sequencing, or Targeted Gene Panels. Clinical data may include patient identifiers, tumor identifiers, diagnosis identifiers, patient age, descriptions of therapies, descriptions of surgeries, records of metastatic disease, and outcomes. It is important to include in rows of data describing tumors labels of stages when samples of those tumors were collected. Labels of stages of melanoma are "I", "II", "III", "IV", and "Unknown".


## Opportunity Pipeline Addressed

Shakeri et al. will automate pairing clinical data and labels of stages based on "ORIEN Data Rules for Tumor Clinical Pairing" to generate a data set of rows of data describing tumors with the below fields.


## What Pipeline Does

The pipeline implements "ORIEN Specimen Staging Revised Rules v6". The pipeline

1. Loads ORIEN v4 tables supplied as CSV files. Clinical Molecular Linkage, Diagnosis, and Metastatic Disease CSV files are loaded. Only rows of Clinical Molecular Linkage data with a value in column Tumor/Germline of "Tumor" and rows of Diagnosis data whose histology code matches the ICD‑O melanoma pattern `87xx/x` are retained. Examples files are `20250317_UVA_ClinicalMolLinkage_V4.csv`, `20250317_UVA_Diagnosis_V4.csv`, and `20250317_UVA_MetastaticDisease_V4.csv`.

2. Counts melanoma diagnoses and sequenced tumors for each patient.

3. Assigns each patient to group A, B, C, or D based on number of tumors and number of diagnoses. Group A has patients each with 1 tumor and 1 diagnosis. Group B has patients each with more than 1 tumor and 1 diagnosis. Group C has patients each with 1 tumor and more than 1 diagnosis. Group D has patients each with more than 1 tumor and more than 1 diagnosis.

4. Selects a tumor-diagnosis pair for each patient. Selecting a pair for a patient in group A is simple as a patient in group A has 1 tumor and 1 diagnosis. For a patient in group B, a tumor is chosen. For a patient in group C, a diagnosis is chosen. For a patient in group D, a tumor and a diagnosis is chosen.

5. Assigns a primary site of "cutaneous", "ocular", "mucosal", or "unknown" to each selected tumor of each patient.

6. Assigns a stage to each selected tumor by evaluating 11 ordered rules until 1 rule applies.

7. Flags exceptional selected tumors for which there are discrepancies between the tumors' properties and the rules or that may be possible new primary tumors.

8. Writes an output CSV file matching a provided answer key with one row per selected tumor and columns `AvatarKey` representing patient ID, `ORIENSpecimenID` representing specimen ID, `AssignedPrimarySite` representing assigned primary site, Group representing patient group, `EKN Assigned Stage` representing assigned stage, `NEW RULE` representing applied rule, `Discrepancy` representing whether there is a discrepancy between tumor properties and rules, and `Possible New Primary` representing whether tumor is possibly a new primary tumor.


## Iterations of Development

At the end of Iteration...

1.	The pipeline loads tables, pairs patients with tumor and diagnosis information, stages selected tumors of patients, and generates the above CSV file.

2. The pipeline may be extended to assign statuses of ICB therapy.


# Usage

For example,

```
../miniconda3/envs/ici_sex/bin/python pipeline_for_pairing_clinical_data_and_stages_of_tumors.py --path_to_clinical_molecular_linkage_data ../../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_ClinicalMolLinkage_V4.csv --path_to_diagnosis_data ../../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_Diagnosis_V4.csv --path_to_metastatic_disease_data ../../Clinical_Data/24PRJ217UVA_NormalizedFiles/24PRJ217UVA_20241112_MetastaticDisease_V4.csv --path_to_output_data output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.csv > output_of_pipeline_for_pairing_clinical_data_and_stages_of_tumors.txt 2>&1
```