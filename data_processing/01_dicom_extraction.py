#!/usr/bin/env python3
"""
01_dicom_extraction.py

Extract DICOM files from a source directory, convert colour fundus images
to JPEG using dcmj2pnm (dcmtk), and generate an image-level metadata CSV.

Requirements:
    - pydicom
    - dcmtk system package (provides dcmj2pnm)

Usage:
    python 01_dicom_extraction.py --input /path/to/dicoms --output /path/to/output --csv metadata.csv
"""

import os
import csv
import argparse
import subprocess
from datetime import datetime

import pydicom


def calculate_age(dob_str: str) -> int | None:
    """Calculate age from a DICOM-formatted date of birth (YYYYMMDD)."""
    if not dob_str:
        return None
    try:
        dob = datetime.strptime(dob_str, "%Y%m%d")
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except ValueError:
        return None


def main(args):
    retino_dir = os.path.join(args.output, "Retinophotographie")
    other_dir = os.path.join(args.output, "Autres")
    os.makedirs(retino_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)

    # Verify dcmj2pnm availability
    try:
        subprocess.run(["dcmj2pnm", "--version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("dcmj2pnm (dcmtk) is required. Install with: brew install dcmtk / apt install dcmtk")

    fieldnames = [
        "Nom Fichier Original", "SOPInstanceUID", "StudyInstanceUID",
        "Patient ID", "Catégorie", "Nom", "Prénom", "Sexe",
        "Date de Naissance", "Âge", "Latéralité Œil", "Date Étude",
    ]

    with open(args.csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for root, _, files in os.walk(args.input):
            for filename in files:
                if not filename.lower().endswith(".dcm"):
                    continue

                dcm_path = os.path.join(root, filename)
                try:
                    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                except Exception:
                    continue

                if "PhotometricInterpretation" not in ds:
                    continue

                pi = ds.get("PhotometricInterpretation", "").strip().upper()
                category = "Retinophotographie" if pi in ("RGB", "YBR_FULL_422") else "Autres"

                # Extract metadata
                name = ds.get("PatientName", "")
                laterality_elem = ds.get((0x0020, 0x0062))
                laterality = laterality_elem.value.strip().upper() if laterality_elem and hasattr(laterality_elem, "value") else "N/A"

                row = {
                    "Nom Fichier Original": filename,
                    "SOPInstanceUID": ds.get("SOPInstanceUID", "N/A"),
                    "StudyInstanceUID": ds.get("StudyInstanceUID", "N/A"),
                    "Patient ID": ds.get("PatientID", "N/A"),
                    "Catégorie": category,
                    "Nom": name.family_name.upper() if hasattr(name, "family_name") else "",
                    "Prénom": name.given_name.capitalize() if hasattr(name, "given_name") else "",
                    "Sexe": ds.get("PatientSex", ""),
                    "Date de Naissance": ds.get("PatientBirthDate", ""),
                    "Âge": calculate_age(ds.get("PatientBirthDate", "")),
                    "Latéralité Œil": laterality,
                    "Date Étude": ds.get("StudyDate", ""),
                }

                # Convert to JPEG
                base = os.path.splitext(filename)[0]
                target_dir = retino_dir if category == "Retinophotographie" else other_dir
                output_path = os.path.join(target_dir, f"{base}.jpg")

                try:
                    subprocess.run(["dcmj2pnm", "--write-jpeg", dcm_path, output_path],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError:
                    continue

                writer.writerow(row)

    print(f"Done. Metadata saved to {args.csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DICOM images and metadata")
    parser.add_argument("--input", required=True, help="Source directory containing .dcm files")
    parser.add_argument("--output", required=True, help="Output directory for converted images")
    parser.add_argument("--csv", required=True, help="Output CSV path for metadata")
    main(parser.parse_args())
