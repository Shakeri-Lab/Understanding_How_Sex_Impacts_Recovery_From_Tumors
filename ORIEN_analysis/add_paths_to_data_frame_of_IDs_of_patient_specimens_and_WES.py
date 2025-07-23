#!/usr/bin/env python3
'''
Add paths of WES to a data frame of IDs of patients, specimens, and WES.

Usage:
../miniconda3/envs/ici_sex/bin/python add_paths_to_data_frame_of_IDs_of_patient_specimens_and_WES.py
'''
from pathlib import Path
import pandas as pd


PATH_TO_DATA_FRAME_OF_IDS_OF_PATIENTS_SPECIMENS_AND_WES = Path(
    "data_frame_of_IDs_of_patients_specimens_and_WES.csv"
)

PATH_TO_PATHS_OF_WES = Path("paths_of_WES.txt")

PATH_TO_DATA_FRAME_OF_IDS_OF_PATIENTS_SPECIMENS_AND_WES_AND_PATHS_TO_WES = PATH_TO_DATA_FRAME_OF_IDS_OF_PATIENTS_SPECIMENS_AND_WES.with_name(
    f"{PATH_TO_DATA_FRAME_OF_IDS_OF_PATIENTS_SPECIMENS_AND_WES.stem}_and_paths_to_WES.csv"
)


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
        raise Exception(f"Multiple paths containing \"annotated\", \"somatic\", \"{value}\", \".ft\", and \".vcf\", and ending with \".gz\" exist.")


def main() -> None:
    data_frame_of_IDs_of_patients_specimens_and_WES = pd.read_csv(PATH_TO_DATA_FRAME_OF_IDS_OF_PATIENTS_SPECIMENS_AND_WES, dtype = str).fillna("")

    list_of_paths_of_WES = []
    with PATH_TO_PATHS_OF_WES.open() as f:
        list_of_paths_of_WES = [line.strip() for line in f if line.strip()]

    data_frame_of_IDs_of_patients_specimens_and_WES["path_to_WES"] = data_frame_of_IDs_of_patients_specimens_and_WES["WES"].apply(
        lambda value: join_matching_paths(value, list_of_paths_of_WES)
    )
    
    data_frame_of_IDs_of_patients_specimens_and_WES["path_to_WES_Batch"] = data_frame_of_IDs_of_patients_specimens_and_WES["WES Batch"].apply(
        lambda value: join_matching_paths(value, list_of_paths_of_WES)
    )

    data_frame_of_IDs_of_patients_specimens_and_WES.to_csv(
        PATH_TO_DATA_FRAME_OF_IDS_OF_PATIENTS_SPECIMENS_AND_WES_AND_PATHS_TO_WES,
        index = False
    )


if __name__ == "__main__":
    main()
