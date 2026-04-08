#!/usr/bin/env python3
"""
04_add_lesion_type.py

Assign a dominant lesion-type label to each BSCR image based on
manually sorted folder structure.

Expected folder layout:
    images_root/
    ├── 1_Excluded/
    ├── 2_Pseudo-albinism/
    ├── 3_Pseudo-melanocytosis/
    ├── 4_Pigmented plaque and spots/
    ├── 5_Pigmented spots without plaque/
    ├── 6_Non pigmented plaque and spots/
    ├── 7_Non pigmented plaque/
    ├── 8_Non pigmented atrophic and non atrophic spots/
    └── 9_Non pigmented non atrophic spots/

Usage:
    python 04_add_lesion_type.py --input metadata.xlsx --images /path/to/sorted --output metadata_lesions.xlsx
"""

import argparse
import os
import pathlib
import pandas as pd

FOLDER_TO_LABEL = {
    "1_Excluded": "Excluded",
    "2_Pseudo-albinism": "Pseudo-albinism",
    "3_Pseudo-melanocytosis": "Pseudo-melanocytosis",
    "4_Pigmented plaque and spots": "Pigmented plaque and spots",
    "5_Pigmented spots without plaque": "Pigmented spots without plaque",
    "6_Non pigmented plaque and spots": "Non pigmented plaque and spots",
    "7_Non pigmented plaque": "Non pigmented plaque",
    "8_Non pigmented atrophic and non atrophic spots": "Non pigmented atrophic and non-atrophic spots",
    "9_Non pigmented non atrophic spots": "Non pigmented non atrophic spots",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm", ".tif"}


def build_mapping(root_folder: str) -> dict:
    """Scan sorted folders and return {filename_stem: label}."""
    mapping = {}
    for folder_name, label in FOLDER_TO_LABEL.items():
        folder_path = os.path.join(root_folder, folder_name)
        if not os.path.exists(folder_path):
            print(f"  Warning: folder '{folder_name}' not found, skipping.")
            continue
        count = 0
        for f in os.listdir(folder_path):
            if pathlib.Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                mapping[pathlib.Path(f).stem] = label
                count += 1
        print(f"  {folder_name}: {count} images")
    return mapping


def main(args):
    df = pd.read_excel(args.input)
    mapping = build_mapping(args.images)
    print(f"Total classified images: {len(mapping)}")

    df["lesion_type"] = df["Nom Fichier Original"].apply(
        lambda x: mapping.get(pathlib.Path(str(x)).stem)
    )

    filled = df["lesion_type"].notna().sum()
    print(f"Rows with lesion type: {filled} / {len(df)}")

    df.to_excel(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add lesion-type labels from sorted folders")
    parser.add_argument("--input", required=True, help="Metadata file (.xlsx)")
    parser.add_argument("--images", required=True, help="Root folder with sorted lesion subfolders")
    parser.add_argument("--output", required=True, help="Output metadata file (.xlsx)")
    main(parser.parse_args())
