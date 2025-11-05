#!/usr/bin/env python3
'''
Add paths of WES to a data frame of IDs of patients, specimens, and WES.

Usage:
python -m ORIEN_analysis.add_paths_to_data_frame_of_IDs_of_patients_specimens_and_WES
'''
from pathlib import Path
import pandas as pd
from ORIEN_analysis.config import paths


def join_matching_paths(
    value: str,
    list_of_paths_to_WES: list[str]
) -> str:
    if not value or pd.isna(value):
        return ""
    value = value.strip()
    list_of_matching_paths = [
        path
        for path in list_of_paths_to_WES
        if (
            ("annotated" in path) and
            ("somatic" in path) and
            (value in path) and
            (".ft" in path) and
            (".vcf" in path) and
            path.endswith(".gz")
        )
    ]
    if len(list_of_matching_paths) == 1:
        return "|".join(list_of_matching_paths)
    if len(list_of_matching_paths) < 1:
        
        list_of_matching_paths = [
            path
            for path in list_of_paths_to_WES
            if (
                ("annotated" in path) and
                ("tumor" in path) and
                (value in path) and
                (".ft" in path) and
                (".vcf" in path) and
                path.endswith(".gz")
            )
        ]
        if len(list_of_matching_paths) == 1:
            return "|".join(list_of_matching_paths)
        if len(list_of_matching_paths) < 1:
            print(f"A path containing \"annotated\", \"somatic\", \"{value}\", \".ft\", and \".vcf\", and ending with \".gz\" does not exist.")
            print(f"Paths containing {value} and \".vcf\" are the following.")
            list_of_matching_paths = [
                path
                for path in list_of_paths_to_WES
                if (value in path) and (".vcf" in path)
            ]
            print('\n'.join(list_of_matching_paths) + '\n')
            return ""
        if len(list_of_matching_paths) > 1:
            raise Exception(f"Multiple paths containing \"annotated\", \"tumor\", \"{value}\", \".ft\", and \".vcf\", and ending with \".gz\" exist.")    
    
    if len(list_of_matching_paths) > 1:
        raise Exception(f"Multiple paths containing \"annotated\", \"somatic\", \"{value}\", \".ft\", and \".vcf\", and ending with \".gz\" exist.")
    return ""


def main() -> None:
    paths.ensure_dependencies_for_adding_paths_to_data_frame_of_IDs_of_patients_specimens_and_WES_exist()
    data_frame_of_IDs_of_patients_specimens_and_WES = pd.read_csv(
        paths.data_frame_of_IDs_of_patients_specimens_and_WES,
        dtype = str
    ).fillna("")

    tuple_of_paths = [str(path) for path in paths.WES.rglob('*') if path.is_file()]

    data_frame_of_IDs_of_patients_specimens_and_WES["path_to_WES"] = data_frame_of_IDs_of_patients_specimens_and_WES["WES"].apply(
        lambda value: join_matching_paths(value, tuple_of_paths)
    )
    
    data_frame_of_IDs_of_patients_specimens_and_WES["path_to_WES_Batch"] = data_frame_of_IDs_of_patients_specimens_and_WES["WES Batch"].apply(
        lambda value: join_matching_paths(value, tuple_of_paths)
    )

    data_frame_of_IDs_of_patients_specimens_and_WES.to_csv(
        paths.data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES,
        index = False
    )


if __name__ == "__main__":
    main()
