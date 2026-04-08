#!/usr/bin/env python3
"""
03_add_cohort_label.py

Add a 'cohort' column with a fixed value (e.g. 'BCR' or 'Control')
to every row of a metadata spreadsheet.

Usage:
    python 03_add_cohort_label.py --input metadata.xlsx --label Control
"""

import argparse
import os
import pandas as pd


def main(args):
    df = pd.read_excel(args.input)
    df["cohort"] = args.label

    base, ext = os.path.splitext(args.input)
    output = args.output or f"{base}_cohort{ext}"
    df.to_excel(output, index=False)
    print(f"Done. Column 'cohort' = '{args.label}' added. Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add cohort column to metadata")
    parser.add_argument("--input", required=True, help="Input metadata file (.xlsx)")
    parser.add_argument("--label", required=True, help="Cohort label (e.g. BCR or Control)")
    parser.add_argument("--output", default=None, help="Output file path (default: input_cohort.xlsx)")
    main(parser.parse_args())
