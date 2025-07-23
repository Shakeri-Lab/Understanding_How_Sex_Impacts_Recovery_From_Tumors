#!/usr/bin/env python3
"""
Parse somatic-VCF archives and flag key hotspot mutations.

For every row in a CSV that looks like

ORIENAvatarKey,DeidSpecimenID,WES,WES Batch,path_to_WES,path_to_WES_Batch
078QIEYNUY,W67BNFRO92FVVVAVYXSCQHMGF,FT-SA185777D,FT-SPN04200,../../WES/annotated_somatic_vcfs/TFT-SA185777D_st_t_NFT-SA187862_st_g.ft.M2GEN.PoN.v2.vcf.gz,
…

the script will

1. copy the “ *.vcf.gz ” archive to a temporary folder,
2. gun-zip it to a plain “ *.vcf ” file,
3. scan the file with **pysam**,
4. record whether each of the ten predefined hotspot mutations is present, and
5. write a tidy CSV whose columns are  
   `PatientID, SpecimenID, BRAF_V600E, … , NRAS_Q61P`.

Both GRCh37 and GRCh38 VEP-style annotations are handled (INFO/ANN or
INFO/CSQ).  Adjust `ANN_PROT_INDEX` if your annotation field order differs.

The work is done in pure Python; no external CLI tools are required.
"""

import argparse
import csv
import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pysam


# Map names of mutations to tuples of gene symbol and lists of AA-changes-that-count.
# TODO: What is an AA-change-that-counts?
CATALOGUE: Dict[str, tuple[str, List[str]]] = {
    "BRAF_V600E": ("BRAF", ["p.V600E", "p.Val600Glu"]),
    "BRAF_V600K": ("BRAF", ["p.V600K", "p.Val600Lys"]),
    "NRAS_Q61R": ("NRAS", ["p.Q61R", "p.Gln61Arg"]),
    "NRAS_Q61L": ("NRAS", ["p.Q61L", "p.Gln61Leu"]),
    "NRAS_Q61K": ("NRAS", ["p.Q61K", "p.Gln61Lys"]),
    "NRAS_G12D": ("NRAS", ["p.G12D", "p.Gly12Asp"]),
    "NRAS_G13D": ("NRAS", ["p.G13D", "p.Gly13Asp"]),
    "NRAS_G13R": ("NRAS", ["p.G13R", "p.Gly13Arg"]),
    "NRAS_Q61H": ("NRAS", ["p.Q61H", "p.Gln61His"]),
    "NRAS_Q61P": ("NRAS", ["p.Q61P", "p.Gln61Pro"]),
}


# 0-based index of the “Protein_change” field in a VEP/ANN string.
# TODO: What is a VEP/ANN field?
INDEX_OF_PROTEIN_CHANGE = 10


def create_dictionary_of_mutations_and_indicators_of_presence(variant_record: pysam.libcbcf.VariantRecord) -> Dict[str, bool]:
    """
    Return a dictionary of mutations and indicators of presences for 1 variant record.
    The logic is:
      – Grab gene symbol and protein change(s) from ANN/CSQ/GENE INFO fields.
      – Compare with our catalogue.
    """
    dictionary_of_mutations_and_indicators_of_presence = {name: False for name in CATALOGUE}

    gene_symbols: List[str] = []
    protein_changes: List[str] = []

    if "GENE" in variant_record.info: # Some pipelines add INFO/GENE.
        gene_symbols.extend(record.info["GENE"])

    # VEP or snpEff style:
    ann_key = (
        "ANN"
        if "ANN" in variant_record.info
        else (
            "CSQ"
            if "CSQ" in variant_record.info
            else None
        )
    )
    if ann_key:
        for entry in variant_record.info[ann_key]:
            fields = entry.split("|")
            if len(fields) <= INDEX_OF_PROTEIN_CHANGE:
                continue
            gene_symbols.append(fields[3]) # gene symbol per VEP spec
            protein_changes.append(fields[INDEX_OF_PROTEIN_CHANGE])

    # Check against catalogue.
    for mut_name, (gene, aa_variants) in CATALOGUE.items():
        if dictionary_of_mutations_and_indicators_of_presence[mut_name]:
            continue
        if gene not in gene_symbols:
            continue
        if any(pc in protein_changes for pc in aa_variants):
            dictionary_of_mutations_and_indicators_of_presence[mut_name] = True

    return dictionary_of_mutations_and_indicators_of_presence


def create_dictionary_of_mutations_and_indicators_of_presence_in_any_record(path_to_archive: Path) -> Dict[str, bool]:
    '''
    Return a dictionary of mutations and indicators of presences for 1 archive.
    '''
    dictionary_of_mutations_and_indicators_of_presence_in_any_record = {name: False for name in CATALOGUE}
    with pysam.VariantFile(path_to_archive) as variant_file:
        for variant_record in variant_file:
            dictionary_of_mutations_and_indicators_of_presence_in_record = create_dictionary_of_mutations_and_indicators_of_presence(variant_record)
            dictionary_of_mutations_and_indicators_of_presence_in_any_record = {
                k: dictionary_of_mutations_and_indicators_of_presence_in_any_record[k] or dictionary_of_mutations_and_indicators_of_presence_in_record[k]
                for k in dictionary_of_mutations_and_indicators_of_presence_in_any_record
            }
            if all(dictionary_of_mutations_and_indicators_of_presence_in_any_record.values()):
                break
    return dictionary_of_mutations_and_indicators_of_presence_in_any_record


def process_specimens(path_to_data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES: Path) -> pd.DataFrame:
    '''
    Construct a data frame of patient IDs, specimen IDs, and indicators of mutations.

    Returns
    -------
    pd.DataFrame -- data frame with columns ORIENAvatarKey and DeidSpecimenID and columns of indicators of mutations
    '''
    data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES = pd.read_csv(
        path_to_data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES,
        dtype = str
    ).fillna("")
    
    list_of_row_information: List[Dict[str, str | bool]] = []

    for _, row in data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES.iterrows():
        patient_id = row["ORIENAvatarKey"]
        specimen_id = row["DeidSpecimenID"]
        if row["path_to_WES"] == "":
            print(f"path to WES does not exist for specimen with ID {specimen_id}.")
            continue
        path_to_archive = Path(row["path_to_WES"]).expanduser().resolve()
        dictionary_of_mutations_and_indicators_of_presence = create_dictionary_of_mutations_and_indicators_of_presence_in_any_record(path_to_archive)
        list_of_row_information.append(
            {
                "ORIENAvatarKey": patient_id,
                "DeidSpecimenID": specimen_id,
                **dictionary_of_mutations_and_indicators_of_presence
            }
        )
    return pd.DataFrame(
        list_of_row_information,
        columns = ["ORIENAvatarKey", "DeidSpecimenID", *CATALOGUE]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = "Create a data frame of patient IDs, specimen IDs, and indicators of whether mutations are present corresponding to a data frame of IDs of patients, specimens, and WES and paths to WES."
    )
    parser.add_argument(
        "input_csv",
        type = Path,
        help = "data frame of IDs of patients, specimens, and WES and paths to WES"
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type = Path,
        default = Path("data_frame_of_patient_IDs_specimen_IDs_and_indicators_of_mutations.csv"),
        help = "path to data frame of patient IDs, specimen IDs, and indicators of mutations",
    )
    args = parser.parse_args()

    data_frame_of_patient_IDs_specimen_IDs_and_indicators_of_mutations = process_specimens(args.input_csv)
    data_frame_of_patient_IDs_specimen_IDs_and_indicators_of_mutations.to_csv(args.output_csv, index = False)
    print(f"Data frame of patient IDs, specimen IDs, and indicators of mutations was written to {args.output_csv}.")
