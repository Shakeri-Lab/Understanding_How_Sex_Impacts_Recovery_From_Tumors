"""
compare_CSV_files.py
Verify that each value in a CSV file is equal to the corresponding value in another CSV file.

Usage
-----
$ python compare_CSV_files.py first.csv second.csv
"""

import pandas as pd
import sys


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype = str, keep_default_na = False)


def main(file_a: str, file_b: str) -> None:
    df_a, df_b = map(load_csv, (file_a, file_b))

    if df_a.shape != df_b.shape:
        print(f"Shape mismatch: {file_a} is {df_a.shape}, {file_b} is {df_b.shape}")
        sys.exit(1)
    if list(df_a.columns) != list(df_b.columns):
        print("Column order / names differ.")
        sys.exit(1)

    equal_mask = df_a.values == df_b.values
    if equal_mask.all():
        print("✅  All cells match.")
        return

    diff_indices = (~equal_mask).nonzero()
    for row, col in zip(*diff_indices):
        col_name = df_a.columns[col]
        val_a, val_b = df_a.iat[row, col], df_b.iat[row, col]
        print(f"Row {row} | Column '{col_name}': {val_a!r}  !=  {val_b!r}")

    print(f"❌  {len(diff_indices[0])} cell(s) differ.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python csv_diff.py <file1.csv> <file2.csv>")
    main(sys.argv[1], sys.argv[2])