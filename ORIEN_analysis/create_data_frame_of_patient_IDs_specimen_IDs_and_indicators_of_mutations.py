#!/usr/bin/env python3
'''
Parse annotated somatic and tumor FT VCF archives corresponding to WES IDs and
create a data frame of patient IDs, specimen IDs, and indicators of whether mutations are present.

For every row in a CSV that looks like

ORIENAvatarKey,DeidSpecimenID,WES,WES Batch,path_to_WES,path_to_WES_Batch
078QIEYNUY,W67BNFRO92FVVVAVYXSCQHMGF,FT-SA185777D,FT-SPN04200,../../WES/annotated_somatic_vcfs/TFT-SA185777D_st_t_NFT-SA187862_st_g.ft.M2GEN.PoN.v2.vcf.gz

the script will

3. scan the file corresponding to the path to WES with pysam,
4. record whether each mutation is present, and
5. write a tidy CSV whose columns are `ORIENAvatarKey`, `DeidSpecimenID`, `BRAF_V600E`, ..., and `NRAS_Q61P`.
'''

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


def get_indices_of_gene_symbol_and_protein_change(vcf_header):
    '''
    Return a tuple of an index of a HUGO gene symbol in a variant record and an index of a protein change.
    '''
    string_of_Funcotator_fields = vcf_header.info["FUNCOTATION"].description.split("Funcotation fields are: ")[1]
    list_of_Funcotator_fields = string_of_Funcotator_fields.split('|')
    index_of_gene_symbol = list_of_Funcotator_fields.index("Gencode_32_hugoSymbol")
    index_of_protein_change = list_of_Funcotator_fields.index("Gencode_32_proteinChange")
    return index_of_gene_symbol, index_of_protein_change


def create_dictionary_of_mutations_and_indicators_of_presence(
    variant_record: pysam.libcbcf.VariantRecord,
    index_of_gene: int,
    index_of_protein_change: int,
    row
) -> Dict[str, bool]:
    '''
    Return a dictionary of mutations and indicators of presences for 1 variant record.
    '''
    '''
    print("Variant record info is the following.")
    print(variant_record.info)
    input()
    '''
    dictionary_of_mutations_and_indicators_of_presence = {name: False for name in CATALOGUE}
    if "FUNCOTATION" in variant_record.info:
        '''
        print("Value corresponding to key FUNCOTATION in variant record info is the following.")
        print(variant_record.info["FUNCOTATION"])
        input()
        '''
        for entry in variant_record.info["FUNCOTATION"]:
            '''
            print("Entry is the following.")
            print(entry)
            input()
            '''
            fields = entry.lstrip('[').rstrip(']').split('|')
            '''
            print("Fields is the following.")
            print(fields)
            input()
            '''
            gene = fields[index_of_gene]
            '''
            print("Gene is the following.")
            print(gene)
            input()
            '''
            protein_change = fields[index_of_protein_change]
            '''
            print("Protein change is the following.")
            print(protein_change)
            input()
            '''
            for mutation, (expected_gene, aa_variants) in CATALOGUE.items():
                '''
                print(f"Mutation is {mutation}.")
                print(f"Expected gene is {expected_gene}.")
                input()
                '''
                if dictionary_of_mutations_and_indicators_of_presence[mutation]:
                    continue
                if gene != expected_gene:
                    continue
                if any(p == protein_change or p == protein_change.lstrip("p.") for p in aa_variants):
                    patient_ID = row["ORIENAvatarKey"]
                    specimen_ID = row["DeidSpecimenID"]
                    print(f"Patient ID is {patient_ID}. specimen ID is {specimen_ID}. gene is {gene}. protein change is {protein_change}.")
                    dictionary_of_mutations_and_indicators_of_presence[mutation] = True
    return dictionary_of_mutations_and_indicators_of_presence


def create_dictionary_of_mutations_and_indicators_of_presence_in_any_record(row) -> Dict[str, bool]:
    '''
    Return a dictionary of mutations and indicators of presences for 1 archive.
    '''
    path_to_archive = Path(row["path_to_WES"]).expanduser().resolve()
    with pysam.VariantFile(path_to_archive) as variant_file:
        index_of_gene_symbol, index_of_protein_change = get_indices_of_gene_symbol_and_protein_change(variant_file.header)
        '''
        print(f"Index of gene symbol is {index_of_gene_symbol}.")
        print(f"Index of protein change is {index_of_protein_change}.")
        input()
        '''
        dictionary_of_mutations_and_indicators_of_presence_in_any_record = {name: False for name in CATALOGUE}
        for variant_record in variant_file:
            '''
            print("Variant record is the following.")
            print(variant_record)
            input()
            '''
            dictionary_of_mutations_and_indicators_of_presence_in_record = create_dictionary_of_mutations_and_indicators_of_presence(variant_record, index_of_gene_symbol, index_of_protein_change, row)
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
        dictionary_of_mutations_and_indicators_of_presence = create_dictionary_of_mutations_and_indicators_of_presence_in_any_record(row)
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
