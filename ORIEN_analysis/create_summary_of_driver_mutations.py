#!/usr/bin/env python3
'''
Parse annotated somatic and tumor FT VCF archives corresponding to WES IDs and
create a summary table that reports, for every patient and melanoma driver mutation,
ORIENAvatarKey,
DeidSpecimenID,
Mutation,
Present,
Gene Name,
Chromosome,
Genomic Position,
Reference Allele, and
Alternate Allele.

General Usage:
../miniconda3/envs/ici_sex/bin/python create_summary_of_driver_mutations.py <input_csv> [--output_csv <output_csv>]

Example usage:
../miniconda3/envs/ici_sex/bin/python create_data_frame_of_patient_IDs_specimen_IDs_and_indicators_of_mutations.py data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES.csv

Input CSV file contains a data frame with columns `ORIENAvatarKey`, `DeidSpecimenID`, and `path_to_WES`.
'''

from typing import Dict
from typing import List
from pathlib import Path
from typing import Tuple
import argparse
import pandas as pd
import pysam


CATALOGUE: dict[str, tuple[str, list[str], str, int, str, list[str]]] = {
    "BRAF_V600E": (
        "BRAF",
        ["p.V600E", "p.Val600Glu"],
        "chr7",
        140_753_336,
        "T",
        ["A"],
    ),
    "BRAF_V600K": (
        "BRAF",
        ["p.V600K", "p.Val600Lys"],
        "chr7",
        140_753_336,
        "GT",
        ["AA"],
    ),
    "NRAS_Q61R": (
        "NRAS",
        ["p.Q61R", "p.Gln61Arg"],
        "chr1",
        114_713_908,
        "T",
        ["C"],
    ),
    "NRAS_Q61L": (
        "NRAS",
        ["p.Q61L", "p.Gln61Leu"],
        "chr1",
        114_713_908,
        "T",
        ["A"],
    ),
    "NRAS_Q61K": (
        "NRAS",
        ["p.Q61K", "p.Gln61Lys"],
        "chr1",
        114_713_909,
        "G",
        ["T"],
    ),
    "NRAS_Q61H": (
        "NRAS",
        ["p.Q61H", "p.Gln61His"],
        "chr1",
        114_713_907,
        "T",
        ["G", "A"],
    ),
    "NRAS_Q61P": (
        "NRAS",
        ["p.Q61P", "p.Gln61Pro"],
        "chr1",
        114_713_908,
        "T",
        ["G"],
    ),
    "NRAS_G12D": (
        "NRAS",
        ["p.G12D", "p.Gly12Asp"],
        "chr1",
        114_716_126,
        "C",
        ["T"],
    ),
    "NRAS_G13D": (
        "NRAS",
        ["p.G13D", "p.Gly13Asp"],
        "chr1",
        114_716_123,
        "C",
        ["T"],
    ),
    "NRAS_G13R": (
        "NRAS",
        ["p.G13R", "p.Gly13Arg"],
        "chr1",
        114_716_124,
        "C",
        ["G"],
    ),
}


def get_tuple_of_indices_of_gene_symbol_and_protein_change(variant_file_header) -> Tuple[int, int]:
    '''
    Return a tuple of an index of a HUGO gene symbol in a variant record and an index of a protein change.
    '''
    string_of_Funcotator_fields = variant_file_header.info["FUNCOTATION"].description.split("Funcotation fields are: ")[1]
    list_of_Funcotator_fields = string_of_Funcotator_fields.split('|')
    index_of_gene_symbol = list_of_Funcotator_fields.index("Gencode_32_hugoSymbol")
    index_of_protein_change = list_of_Funcotator_fields.index("Gencode_32_proteinChange")
    return index_of_gene_symbol, index_of_protein_change


def create_list_of_dictionaries_of_mutations(row: pd.Series) -> List[dict]:
    '''
    For 1 specimen, provide a list of dictionaries each corresponding to a mutation.
    A dictionary will have keys equal to the names of columns in the final data frame.
    '''
    patient_ID = row["ORIENAvatarKey"]
    specimen_ID = row["DeidSpecimenID"]
    
    results = {}
    for mutation, (
        expected_gene,
        list_of_protein_changes,
        chromosome,
        genomic_position,
        reference_allele,
        alternate_allele
    ) in CATALOGUE.items():
        results[mutation] = {
            "ORIENAvatarKey": patient_ID,
            "DeidSpecimenID": specimen_ID,
            "Mutation": mutation,
            "Present": False,
            "Gene Name": expected_gene,
            "Chromosome": chromosome,
            "Genomic Position": genomic_position,
            "Reference Allele": reference_allele,
            "Alternate Allele": ','.join(alternate_allele)
        }
    
    if row["path_to_WES"] == "":
        specimen_ID = row["DeidSpecimenID"]
        print(f"Path to WES does not exist for specimen with ID {specimen_ID}.")
    else:
        path = Path(row["path_to_WES"]).expanduser().resolve()
        with pysam.VariantFile(path) as variant_file:
            index_of_gene, index_of_protein_change = get_tuple_of_indices_of_gene_symbol_and_protein_change(variant_file.header)

            for variant_record in variant_file:
                if "FUNCOTATION" not in variant_record.info:
                    continue

                for entry in variant_record.info["FUNCOTATION"]:
                    fields = entry.lstrip('[').rstrip(']').split('|')
                    gene = fields[index_of_gene]
                    protein_change = fields[index_of_protein_change]

                    for mutation, lookup in CATALOGUE.items():
                        expected_gene, list_of_protein_changes, *_ = lookup
                        if (
                            gene == expected_gene and
                            any(p == protein_change or p.lstrip("p.") == protein_change for p in list_of_protein_changes)
                        ):
                            print(f"Patient ID is {patient_ID}. Specimen ID is {specimen_ID}. Gene is {gene}. Protein change is {protein_change}.")
                            results[mutation].update(
                                {
                                    "Present": True,
                                    "Gene Name": gene,
                                    "Chromosome": variant_record.chrom,
                                    "Genomic Position": variant_record.pos,
                                    "Reference Allele": variant_record.ref,
                                    "Alternate Allele": ','.join(variant_record.alts)
                                }
                            )
                            break
            
    return list(results.values())


def create_summary(path_to_data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES: Path) -> pd.DataFrame:
    '''
    Iterate over data frame of IDs of patients, specimens, and WES and paths to WES and
    accumulate dictionaries each corresponding to a specimen and a mutation.
    '''
    data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES = pd.read_csv(path_to_data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES, dtype = str).fillna("")
    
    list_of_dictionaries: List[dict] = []
    for _, row in data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES.iterrows():
        list_of_dictionaries.extend(create_list_of_dictionaries_of_mutations(row))
    
    return pd.DataFrame(
        list_of_dictionaries,
        columns = [
            "ORIENAvatarKey",
            "DeidSpecimenID",
            "Mutation",
            "Present",
            "Gene Name",
            "Chromosome",
            "Genomic Position",
            "Reference Allele",
            "Alternate Allele"
        ]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Create a summary of driver mutations.")
    parser.add_argument(
        "input_csv",
        type = Path,
        help = "data frame of IDs of patients, specimens, and WES and paths to WES"
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type = Path,
        default = Path("summary_of_driver_mutations.csv"),
        help = "path to summary of driver mutations",
    )
    args = parser.parse_args()

    summary = create_summary(args.input_csv)
    summary.to_csv(args.output_csv, index = False)
    print(f"Summary of driver mutations was written to {args.output_csv}.")

