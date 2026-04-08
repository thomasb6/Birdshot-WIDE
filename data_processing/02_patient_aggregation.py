#!/usr/bin/env python3
"""
02_patient_aggregation.py

Aggregate image-level metadata into a patient-level summary table
with counts of right-eye (OD) and left-eye (OG) images per patient.

Usage:
    python 02_patient_aggregation.py --input metadata.xlsx --output patients.xlsx
"""

import argparse
import pandas as pd
import numpy as np


def main(args):
    df = pd.read_excel(args.input)
    df = df[df["Catégorie"].astype(str).str.strip() == "Retinophotographie"]

    df["Patient_Unique_ID"] = (
        df["Nom"].astype(str) + "_" + df["Prénom"].astype(str) + "_" + df["Date de Naissance"].astype(str)
    )

    summary = df.groupby("Patient_Unique_ID").agg(
        Nom=("Nom", "first"),
        Prénom=("Prénom", "first"),
        Sexe=("Sexe", "first"),
        Âge=("Âge", "first"),
        OD_Count=("Latéralité Œil", lambda x: (x == "R").sum()),
        OG_Count=("Latéralité Œil", lambda x: (x == "L").sum()),
    ).reset_index(drop=True)

    summary.columns = ["Nom", "Prénom", "Sexe", "Âge", "Nombre d'images OD", "Nombre d'images OG"]

    conditions = [
        (summary["Nombre d'images OD"] > 0) & (summary["Nombre d'images OG"] > 0),
        (summary["Nombre d'images OD"] > 0) & (summary["Nombre d'images OG"] == 0),
        (summary["Nombre d'images OD"] == 0) & (summary["Nombre d'images OG"] > 0),
    ]
    summary["Origine"] = np.select(conditions, ["ODG", "OD", "OG"], default="N/A")

    summary.to_excel(args.output, index=False)
    print(f"Done. {len(summary)} patients saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate metadata to patient level")
    parser.add_argument("--input", required=True, help="Image-level metadata (.xlsx)")
    parser.add_argument("--output", required=True, help="Output patient-level file (.xlsx)")
    main(parser.parse_args())
