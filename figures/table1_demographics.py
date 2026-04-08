#!/usr/bin/env python3
"""
table1_demographics.py

Generate Table 1: demographic and dataset characteristics of the
BSCR and control cohorts.

Usage:
    python table1_demographics.py --bscr patients_bscr.xlsx --controls patients_controls.xlsx
"""

import argparse
import pandas as pd


def cohort_stats(df, label):
    n = len(df)
    age_str = f"{df['Âge'].mean():.2f} ± {df['Âge'].std():.2f}"
    sex = df["Sexe"].value_counts()
    f_n, m_n = sex.get("F", 0), sex.get("M", 0)
    sex_str = f"Female: {f_n} ({f_n/n*100:.1f}%) / Male: {m_n} ({m_n/n*100:.1f}%)"
    od_str = f"{df['Nombre d\\'images OD'].mean():.2f} ± {df['Nombre d\\'images OD'].std():.2f}"
    og_str = f"{df['Nombre d\\'images OG'].mean():.2f} ± {df['Nombre d\\'images OG'].std():.2f}"
    return {f"{label} (n={n})": [age_str, sex_str, od_str, og_str]}


def main(args):
    bscr = pd.read_excel(args.bscr)
    ctrl = pd.read_excel(args.controls)

    table = pd.DataFrame({
        "Characteristic": ["Age, mean ± SD", "Sex, n (%)", "OD images, mean ± SD", "OG images, mean ± SD"],
        **cohort_stats(bscr, "BSCR"),
        **cohort_stats(ctrl, "Control"),
    })

    print("\nTable 1. Demographic and dataset characteristics\n")
    print(table.to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Table 1")
    parser.add_argument("--bscr", required=True)
    parser.add_argument("--controls", required=True)
    main(parser.parse_args())
