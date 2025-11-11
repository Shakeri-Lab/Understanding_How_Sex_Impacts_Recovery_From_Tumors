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

It will also print the numbers of female, male, and total patients with BRAF and NRAS mutations,
using the same logic as in `create_summary_of_driver_mutations.py`.
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

from ORIEN_analysis.config import paths


# Map names of mutations to tuples of gene symbol and lists of equivalent protein changes.
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
        wl: Dict[str, set[str]] = {}
        for mut, (_, aa_variants) in CATALOGUE.items():
            s = set()
            for p in aa_variants:
                p = p.strip()
                if not p:
                    continue
                s.add(p)
                s.add(p.lstrip("p."))
            wl[mut] = s
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
            tokens = set(normalize_protein_tokens(fields[index_of_protein_change]))
            if not tokens:
                continue
            tokens_no_p = {t.lstrip("p.") for t in tokens}
            '''
            print("Protein change is the following.")
            print(protein_change)
            input()
            '''
            for mutation, (expected_gene, _) in CATALOGUE.items():
                '''
                print(f"Mutation is {mutation}.")
                print(f"Expected gene is {expected_gene}.")
                input()
                '''
                if dictionary_of_mutations_and_indicators_of_presence[mutation] or gene != expected_gene:
                    continue
                if tokens & wl[mutation] or tokens_no_p & wl[mutation]:
                    patient_ID = row["ORIENAvatarKey"]
                    specimen_ID = row["DeidSpecimenID"]
                    print(f"Patient ID is {patient_ID}. Specimen ID is {specimen_ID}. Gene is {gene}. Protein change is {sorted(tokens)}.")
                    dictionary_of_mutations_and_indicators_of_presence[mutation] = True
    return dictionary_of_mutations_and_indicators_of_presence


def normalize_protein_tokens(raw: str) -> list[str]:
    core = raw.split(';', 1)[0].strip()
    parts = []
    for piece in core.split(','):
        parts.extend(piece.split('/'))
    return [p.strip() for p in parts if p.strip() and p.strip() not in {'.', 'NA'}]


def collect_tokens_for_gene(fields, idx_gene, idx_pchg, target_gene, bucket: set[str]):
    gene = fields[idx_gene]
    if gene == target_gene:
        raw = fields[idx_pchg]
        for token in normalize_protein_tokens(raw):
            bucket.add(token)


_AUDIT_ROWS: list[dict] = []

def create_dictionary_of_mutations_and_indicators_of_presence_in_any_record(row) -> Dict[str, bool]:
    '''
    Return a dictionary of mutations and indicators of presences for 1 archive.
    '''
    path_to_archive = Path(row["path_to_WES"]).expanduser().resolve()
    pid, sid = row["ORIENAvatarKey"], row["DeidSpecimenID"]

    seen_braf: set[str] = set()
    seen_nras: set[str] = set()

    with pysam.VariantFile(path_to_archive) as variant_file:
        index_of_gene_symbol, index_of_protein_change = get_indices_of_gene_symbol_and_protein_change(variant_file.header)
        '''
        print(f"Index of gene symbol is {index_of_gene_symbol}.")
        print(f"Index of protein change is {index_of_protein_change}.")
        input()
        '''
        dictionary_of_mutations_and_indicators_of_presence_in_any_record = {name: False for name in CATALOGUE}
        for variant_record in variant_file:
            if "FUNCOTATION" not in variant_record.info:
                continue
            for entry in variant_record.info["FUNCOTATION"]:
                fields = entry.lstrip('[').rstrip(']').split('|')
                collect_tokens_for_gene(fields, index_of_gene_symbol, index_of_protein_change, "BRAF", seen_braf)
                collect_tokens_for_gene(fields, index_of_gene_symbol, index_of_protein_change, "NRAS", seen_nras)
            dictionary_of_mutations_and_indicators_of_presence_in_record = create_dictionary_of_mutations_and_indicators_of_presence(
                variant_record,
                index_of_gene_symbol,
                index_of_protein_change,
                row
            )
            dictionary_of_mutations_and_indicators_of_presence_in_any_record = {
                k: dictionary_of_mutations_and_indicators_of_presence_in_any_record[k] or dictionary_of_mutations_and_indicators_of_presence_in_record[k]
                for k in dictionary_of_mutations_and_indicators_of_presence_in_any_record
            }
    _AUDIT_ROWS.append(
        {
            "ORIENAvatarKey": pid,
            "DeidSpecimenID": sid,
            "BRAF_protein_changes": '|'.join(sorted(seen_braf)) if seen_braf else "",
            "NRAS_protein_changes": '|'.join(sorted(seen_nras)) if seen_nras else ""
        }
    )
    return dictionary_of_mutations_and_indicators_of_presence_in_any_record


def process_specimens(path_to_data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES: Path) -> pd.DataFrame:
    '''
    Construct a data frame of patient IDs, specimen IDs, and indicators of mutations,
    then merge Sex from paths.patient_data.

    Returns
    -------
    pd.DataFrame -- data frame with columns ORIENAvatarKey, DeidSpecimenID, Sex, and columns of indicators of mutations
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
            print(f"Path to WES does not exist for specimen with ID {specimen_id}.")
            continue
        dictionary_of_mutations_and_indicators_of_presence = create_dictionary_of_mutations_and_indicators_of_presence_in_any_record(row)
        list_of_row_information.append(
            {
                "ORIENAvatarKey": patient_id,
                "DeidSpecimenID": specimen_id,
                **dictionary_of_mutations_and_indicators_of_presence
            }
        )
    df = pd.DataFrame(
        list_of_row_information,
        columns = ["ORIENAvatarKey", "DeidSpecimenID", *CATALOGUE]
    )
    patient_data = pd.read_csv(paths.patient_data)
    df = df.merge(
        patient_data[["AvatarKey", "Sex"]],
        how = "left",
        left_on = "ORIENAvatarKey",
        right_on = "AvatarKey"
    ).drop(columns = "AvatarKey")
    df = df[["ORIENAvatarKey", "DeidSpecimenID", "Sex", *CATALOGUE]]
    return df


def print_numbers_of_patients_with_mutations_in_gene(df_with_sex: pd.DataFrame, gene: str) -> None:
    '''
    Compute presence from the indicator columns in this wide table.
    Counting is at the patient level.
    Specimens are collapsed using logical OR.
    '''
    gene_prefix = f"{gene}_"
    gene_cols = [c for c in df_with_sex.columns if c.startswith(gene_prefix)]
    row_has_gene = df_with_sex[gene_cols].any(axis = 1)
    df_tmp = df_with_sex.assign(_HAS_GENE = row_has_gene)
    per_patient = (
        df_tmp.groupby(["ORIENAvatarKey", "Sex"], as_index = False)["_HAS_GENE"].max()
    )
    total = int(per_patient["_HAS_GENE"].sum())
    female = int(per_patient.loc[(per_patient["Sex"] == "Female") & (per_patient["_HAS_GENE"]), "_HAS_GENE"].sum())
    male = int(per_patient.loc[(per_patient["Sex"] == "Male") & (per_patient["_HAS_GENE"]), "_HAS_GENE"].sum())
    if gene == "BRAF":
        print(f"Number of female patients with a BRAF mutation present: {female}")
        print(f"Number of male patients with a BRAF mutation present: {male}")
        print(f"Total number of patients with a BRAF mutation present: {total}")
    elif gene == "NRAS":
        print(f"Number of female patients with an NRAS mutation present: {female}")
        print(f"Number of male patients with an NRAS mutation present: {male}")
        print(f"Total number of patients with an NRAS mutation present: {total}")
    else:
        raise Exception("Gene should be BRAF or NRAS.")


def main():
    paths.ensure_dependencies_for_creating_data_frame_of_patient_IDs_specimen_IDs_and_indicators_of_mutations_exist()
    data_frame_of_patient_and_specimen_information_and_indicators_of_mutations = process_specimens(
        paths.data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES
    )
    for gene in ("BRAF", "NRAS"):
        print_numbers_of_patients_with_mutations_in_gene(
            data_frame_of_patient_and_specimen_information_and_indicators_of_mutations,
            gene
        )
    if _AUDIT_ROWS:
        audit_df = pd.DataFrame(_AUDIT_ROWS).drop_duplicates().sort_values(["ORIENAvatarKey", "DeidSpecimenID", "BRAF_protein_changes"])
        out = paths.data_frame_of_patient_and_specimen_information_and_indicators_of_mutations.with_name(
            f"{paths.data_frame_of_patient_and_specimen_information_and_indicators_of_mutations.stem}_BRAF_audit.csv"
        )
        audit_df.to_csv(out, index = False)
        print(f"[DIAG] Wrote BRAF audit to: {out}")
    data_frame_of_patient_and_specimen_information_and_indicators_of_mutations.to_csv(
        paths.data_frame_of_patient_and_specimen_information_and_indicators_of_mutations,
        index = False
    )


if __name__ == "__main__":
    main()