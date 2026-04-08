#!/usr/bin/env python3
"""
05_anonymize_metadata.py

Produce the final, publication-ready metadata file:
  - Generate pseudonymized patient IDs via SHA-256 hashing
  - Compute age at acquisition
  - Rename and format columns
  - Strip all identifying information (name, date of birth)

Usage:
    python 05_anonymize_metadata.py --input metadata_full.xlsx --output metadata_final.xlsx
"""

import argparse
import hashlib
import pathlib
import pandas as pd


def generate_anonymous_id(row) -> str:
    """SHA-256 hash of Nom + Prénom + Date de Naissance, truncated to 16 hex chars."""
    raw = f"{str(row['Nom']).strip().upper()}_{str(row['Prénom']).strip().upper()}_{str(row['Date de Naissance']).strip()}"
    return "P_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def calculate_age(row) -> int | None:
    """Compute age at the date of the examination."""
    try:
        dob = pd.to_datetime(str(row["Date de Naissance"]), format="%Y%m%d")
        acq = pd.to_datetime(str(row["Date Étude"]), format="%Y%m%d")
        return acq.year - dob.year - ((acq.month, acq.day) < (dob.month, dob.day))
    except Exception:
        return None


def main(args):
    df = pd.read_excel(args.input)
    print(f"Loaded {len(df)} rows.")

    df["patient_id"] = df.apply(generate_anonymous_id, axis=1)
    df["age_at_acquisition_date"] = df.apply(calculate_age, axis=1)

    df.rename(columns={
        "Nom Fichier Original": "filename",
        "Date Étude": "acquisition_date",
        "Sexe": "sex",
        "Latéralité Œil": "laterality",
    }, inplace=True)

    df["sex"] = df["sex"].map({"M": "Male", "F": "Female"}).fillna(df["sex"])
    df["acquisition_date"] = pd.to_datetime(
        df["acquisition_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    df["filename"] = df["filename"].astype(str).apply(lambda x: pathlib.Path(x).stem + ".jpg")

    final_columns = [
        "filename", "patient_id", "acquisition_date", "age_at_acquisition_date",
        "sex", "laterality", "lesion_type", "cohort",
    ]
    df_final = df[final_columns].copy()
    df_final.to_excel(args.output, index=False)
    print(f"Done. {len(df_final)} rows saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize metadata for publication")
    parser.add_argument("--input", required=True, help="Full metadata with identifiers (.xlsx)")
    parser.add_argument("--output", required=True, help="Anonymized output metadata (.xlsx)")
    main(parser.parse_args())
