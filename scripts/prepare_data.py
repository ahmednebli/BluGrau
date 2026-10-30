#!/usr/bin/env python
"""Convert raw data to the format used by the code: adjacency.npy (G,N,N) and labels.npy (G,).

Two input formats:

  --format sex-csv   clinics.csv (';'-separated, column 'gender') + subjects/S001.csv, ... (sex cohort)
  --format arrays    --adjacency file.npy (G,N,N) and --labels file.npy or file.csv (G,)
                     (e.g. the cannabis cohort exported from Grenol-Net: 35 x 35 matrices)
"""
import argparse

import numpy as np
import pandas as pd

from blugrau.data import from_sex_cohort_csv, save_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", choices=("sex-csv", "arrays"), required=True)
    p.add_argument("--raw-dir", help="sex-csv: folder with clinics.csv and subjects/")
    p.add_argument("--positive", default="male", help="sex-csv: gender mapped to label 1")
    p.add_argument("--adjacency", help="arrays: .npy file with shape (G, N, N)")
    p.add_argument("--labels", help="arrays: .npy or single-column .csv with G binary labels")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    if a.format == "sex-csv":
        A, y = from_sex_cohort_csv(a.raw_dir, a.positive)
    else:
        A = np.load(a.adjacency)
        y = np.load(a.labels) if a.labels.endswith(".npy") else pd.read_csv(a.labels, header=None).values.ravel()
    print(f"adjacency {A.shape}, labels {np.bincount(y.astype(int))}")
    save_dataset(A, y.astype(int), a.out_dir)


if __name__ == "__main__":
    main()
