# System Description

*Pipeline for Pairing Clinical Data and Stages of Tumors*

Created on 06/01/2025 by Tom Lever

Updated on 06/02/2025 by Tom Lever


## Context

Shakeri et al. are translating genomic and transcriptomic data that describe melanoma and clinical data that provide context into predictions relating to responses to Immune Checkpoint Blockade therapy and prognoses. Molecular data may include Whole Exome Sequencing, RNA sequencing, or Targeted Gene Panels. Clinical data may include patient identifiers, tumor identifiers, diagnosis identifiers, patient age, descriptions of therapies, descriptions of surgeries, records of metastatic disease, and outcomes. It is important to include in rows of data describing tumors labels of stages when samples of those tumors were collected. Labels of stages of melanoma are I, II, III, and IV.


## Opportunity the Pipeline Will Address

Shakeri et al. will automate pairing clinical data and labels of stages based on "ORIEN Data Rules for Tumor Clinical Pairing" to generate a data set of rows of data describing tumors with the below fields.


## What the Tumor-Stage Pairing Pipeline Will Do

The pipeline will:

1. Ingest Clinical Molecular Linkage, Diagnosis, and Metastatic Disease CSV files and retain rows relating to tumors (as opposed to germ lines). Examples files are `20250317_UVA_ClinicalMolLinkage_V4.csv`, `20250317_UVA_Diagnosis_V4.csv`, and `20250317_UVA_MetastaticDisease_V4.csv`.

2. Assign patients to groups A, B, C, or D based on number of tumors and number of diagnoses. Group A has patients each with 1 tumor and 1 diagnosis. Group B has patients each with more than 1 tumor and 1 diagnosis. Group C has patients each with 1 tumor and more than 1 diagnosis. Group D has patients each with more than 1 tumor and more than 1 diagnosis.

3. Reduce each patient to one tumor-diagnosis pair that best satisfies "ORIEN Data Rules for Tumor Clinical Pairing".

4. If a patient is in Group A, pair the single tumor and single diagnosis of the patient directly.

5. If the patient is in Group B or D, select 1 tumor for a patient, then pair with a single diagnosis of the patient.

    1. Filter out tumors without RNA sequencing. If no sequenced tumor has RNA sequencing data, drop the patient. If only one tumor has RNA sequencing data, select that tumor.
    
    2. Among the remaining tumors,
    
        1. If a site of a tumor contains "skin" and no site contains "lymph node" or "soft tissue", pick a tumor with a site that contains "skin".
        
        2. Otherwise, if no site contains "skin" or "soft tissue", but at least one site contains "lymph node", pick a tumor with a site containing "lymph node".
        
        3. Otherwise, if a site of a tumor contains "skin" or "soft tissue" and a site of a tumor contains "lymph node", pick a tumor with a site that contains "skin" or "soft tissue" and has the earliest value in field "Age At Specimen Collection".
        
        4. Otherwise, select the tumor with the earliest value of "Age At Specimen Collection".

6. If the patient is in Group C or D, select 1 diagnosis for a patient, then pair with a single tumor of the patient.

    1. If a patient only has primary tumors (i.e., the values of field `Primary/Met` is "Primary"),
    
        1. If the collection age is within 90 days after exactly 1 diagnosis, select that diagnosis.
       
        2. If the collection age is within 90 days of more than 1 diagnosis, select the diagnosis whose primary diagnosis site matches the site of the tumor.
        
        3. If no diagnosis passes the 90‑day test or at least one diagnosis age is unknown and histologies differ, choose a diagnosis whose histology code matches the histology of the tumor.

        4. If there is no unique match, choose a diagnosis with the earliest value of `AgeAtDiagnosis`.

    2. If a patient only has metastatic lymph node tumors (i.e., the values of field `Primary/Met` is "Metastatic" and the sites of the tumors contain "lymph node"),

        1. If exactly 1 diagnosis has a positive value of `PathNStage` (i.e., a value that is not "N0", "Nx", or "Unknown"), and the specimen was collected within 90 days after that diagnosis, choose that diagnosis.

        2. Otherwise, if no diagnosis falls within 90 days after specimen collection, if exactly 1 diagnosis has a value of `PrimaryDiagnosisSite` equal to the value of `MetsDzPrimaryDiagnosisSite` from the Metastatic Disease CSV file, select that diagnosis.
        
        3. If the sites of the tumors contain "soft tissue" and the specimen was collected within 90 days after exactly one diagnosis, choose that diagnosis.

    3. If the specimen site is metastatic but is neither lymph-node nor soft-tissue, select the diagnosis that has the earliest value of field `AgeAtDiagnosis`.
    
    4. Otherwise, select the diagnosis with the earliest value for `AgeAtDiagnosis`.

7. Create a CSV with 1 row per tumor and fields `AvatarKey`, `DeidSpecimenID`, `DiagnosisIndex`, `Age At Specimen Collection`, `AssignedPrimarySite`, `AssignedStage`, `StageRuleHit`, and `ICBStatus`.

`AvatarKey` represents patient identifiers. This field is sourced from field `ORIENAvatarKey` in the Molecular Linkage CSV file. This field is required by "ORIEN Data Rules for Tumor Clinical Pairing". This field anchors every row describing a tumor to a patient.

`DeidSpecimenID` represents tumor identifiers. This field is sourced from the Molecular Linkage CSV file. This field is required by "ORIEN Data Rules for Tumor Clinical Pairing". This field uniquely identifiers specimens.

`DiagnosisIndex` represents indices of rows in table Diagnosis. This field is sourced from the Molecular Linkage CSV file. This field is included so that an analyst may review rows in table Diagnosis that were used by the pipeline. A row in table Diagnosis has fields `AvatarKey`, `AgeAtDiagnosis`, `PrimaryDiagnosisSite`, `HistologyCode`, `ClinGroupStage`, `PathGroupStage` that are referenced by "ORIEN Data Rules for Tumor Clinical Pairing". `AgeAtDiagnosis` represents the age of patients when they received diagnoses of melanoma. `PrimaryDiagnosisSite` represents descriptions of anatomic sites of tumors at the basis of diagnoses. `HistologyCode` represents specific cell types of tumors. `ClinGroupStage` represents a stage in I, II, III, and IV of tumors according to clinical assessments. `PathGroupStage` represents a stage in I, II, III, and IV of tumors according to pathological assessments.

"Age At Specimen Collection" represents ages of patients when tumors were collected. This field is sourced from the Molecular Linkage CSV file. This field is essential for determining stages.

`AssignedPrimarySite` represents primary sites of tumors. Primary sites are cutaneous, ocular, mucosal, and unknown. This field is derived by the pipeline by analyzing values of `PrimaryDiagnosisSite`, a field associated with a diagnosis. If a value of field `PrimaryDiagnosisSite` contains "skin", "ear", "eyelid", "vulva", "head", or "scalp", assign "cutaneous". Otherwise, if a value contains "eye", "choroid", "ciliary body", or "conjunctiva", assign "ocular". Otherwise, if a value contains "sinus", "gum", "nasal", "urethra", "anorect", "anus", "rectum", "anal canal", "oropharynx", "oral", "vagina", "esophagus", or "palate", assign "mucosal". If a value does not contain any of these values or contains "unknown", assign "unknown". This field is essential to determining stages.

`AssignedStage` represents labels of stages of tumors. This field is derived by the pipeline based on ordered stage rules for a given primary site (i.e., one of "cutaneous", "ocular", "mucosal", or "unknown"). This field is required by "ORIEN Data Rules for Tumor Clinical Pairing". Stage is the central output.  For melanomas whose primary site is unknown, four ordered rules (UN1, UN2, UN3P, UN3C) and a final UN-UNK fallback mirror the logic used for the other site families.

`StageRuleHit` represents codes (e.g., "CUT4") that each identify which rule was used to determine assigned stage. This field is derived by the pipeline. This field is included so that an analyst may determine how stage was determined. The stage algorithm can also return placeholder IDs CUT‑UNK, OC‑UNK, MU‑UNK, UN‑UNK when no specific rule triggers for a specimen. Every primary site family yields its own "-UNK" value. The codes appear in the output tests and CSV as they do in the pipeline. These values signal to analysts that staging data were insufficient or contradictory.

`ICBStatus` represents statuses of Immune Checkpoint Blockade therapy. This field is derived by the pipeline by comparing specimen ages to earliest ICB therapies as presented in table Medications. Values may be "Pre-ICB", "Post-ICB", "No-ICB", and "Unknown".


## Iterations of Development

At the end of Iteration...

1.	The pipeline will pair clinical data with stages of tumors and generate the above CSV file.