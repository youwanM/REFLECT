#!/usr/bin/env python3

import pandas as pd
import argparse
from scipy.stats import wilcoxon

def load_csv(file_path):
    df = pd.read_csv(file_path)
    # Optional: drop rows with zero lesion volume
    df = df[df["LesionVolume"] > 0].reset_index(drop=True)
    return df

def compute_wilcoxon(df1, df2, size=None):
    if size:
        d1 = df1[df1["SizeCategory"] == size]["Dice"]
        d2 = df2[df2["SizeCategory"] == size]["Dice"]
        label = f"Size {size}"
    else:
        d1 = df1["Dice"]
        d2 = df2["Dice"]
        label = "All sizes"

    if not (df1["SubjectID"].tolist() == df2["SubjectID"].tolist()):
        raise ValueError(f"SubjectID columns do not match for {label}. Wilcoxon test cannot be performed.")

    
    stat, p = wilcoxon(d1, d2)
    print(f"Wilcoxon test ({label}): statistic={stat:.4f}, p-value={p:.4e}")
    return stat, p

def main(csv1, csv2):
    df1 = load_csv(csv1)
    df2 = load_csv(csv2)

    print(f"\nComparing {csv1} vs {csv2}...\n")

    # All Dice
    compute_wilcoxon(df1, df2, size=None)

    # By size category
    for size in ["S", "M", "L"]:
        compute_wilcoxon(df1, df2, size=size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wilcoxon test for Dice scores between two CSVs")
    parser.add_argument("csv1", type=str, help="Path to first CSV file")
    parser.add_argument("csv2", type=str, help="Path to second CSV file")
    args = parser.parse_args()

    main(args.csv1, args.csv2)
