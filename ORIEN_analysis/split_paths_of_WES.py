#!/usr/bin/env python3
"""
Read a CSV that already contains paths to WES, then write—one file per
patient—newline-separated lists of those paths.

Input
-----
data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES.csv
    Must contain at least these columns:
        • ORIENAvatarKey   – patient identifier
        • path_to_WES      – pipe-delimited (“|”) list of WES paths

Output
------
A directory named “paths_of_WES_for_patients/” (created if necessary)
containing one text file per ORIENAvatarKey, e.g. “12345.txt”, with one
WES path per line.

Example
-------
$ python write_wes_paths_per_patient.py
"""

from pathlib import Path
import pandas as pd

# --------------------------------------------------------------------------- #
# ― CONFIGURATION ― adjust if your paths differ
# --------------------------------------------------------------------------- #
PATH_TO_CSV = Path(
    "data_frame_of_IDs_of_patients_specimens_and_WES_and_paths_to_WES.csv"
)
PATH_TO_OUTPUT_DIR = Path("paths_of_WES_for_patients")  # will be created
COLUMN_PATIENT_ID = "ORIENAvatarKey"
COLUMN_WES_PATHS = "path_to_WES"
DELIMITER = "|"  # delimiter in the CSV column
# --------------------------------------------------------------------------- #


def main() -> None:
    # Ensure the output directory exists
    PATH_TO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the CSV (keep dtype=str to avoid accidental numeric casting)
    df = pd.read_csv(PATH_TO_CSV, dtype=str)

    if COLUMN_PATIENT_ID not in df.columns or COLUMN_WES_PATHS not in df.columns:
        missing = {COLUMN_PATIENT_ID, COLUMN_WES_PATHS} - set(df.columns)
        raise ValueError(f"Missing expected column(s): {', '.join(missing)}")

    # Group by patient in case the CSV has multiple rows per patient
    for patient_id, group in df.groupby(COLUMN_PATIENT_ID):
        # Collect the pipe-delimited strings, ignore NaNs/empties
        raw_strings = group[COLUMN_WES_PATHS].dropna().astype(str)

        # Split, strip whitespace, and discard blank tokens
        paths = (
            path.strip()
            for string in raw_strings
            for path in string.split(DELIMITER)
        )
        cleaned_paths = [p for p in paths if p]  # remove empty strings

        # Skip patients with no paths
        if not cleaned_paths:
            continue

        # Write one line per path
        output_path = PATH_TO_OUTPUT_DIR / f"{patient_id}.txt"
        output_path.write_text("\n".join(cleaned_paths) + "\n")

        print(f"Wrote {len(cleaned_paths):3d} path(s) → {output_path}")


if __name__ == "__main__":
    main()
