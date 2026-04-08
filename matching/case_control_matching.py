#!/usr/bin/env python3
"""
case_control_matching.py

Iterative case-control matching algorithm for the Birdshot-WIDE dataset.

For each BSCR patient, a control is sought based on two criteria:
  - Identical sex
  - Age difference ≤ 10 years

The algorithm prioritises a single bilateral (ODG) control. If unavailable,
it composes a match from one OD-only and one OG-only control, each meeting
the matching criteria independently. Each control is used at most once.

Usage:
    python case_control_matching.py \
        --bscr patients_bscr.xlsx \
        --controls patients_controls.xlsx \
        --output matching_results.xlsx
"""

import argparse
import pandas as pd


def main(args):
    bscr_df = pd.read_excel(args.bscr)
    ctrl_df = pd.read_excel(args.controls)

    ctrl_pool = ctrl_df.dropna(subset=["Sexe", "Âge", "Origine"]).copy()
    ctrl_pool["ID"] = ctrl_pool["Nom"].str.upper().str.strip() + "_" + ctrl_pool["Prénom"].str.upper().str.strip()

    bscr_valid = bscr_df.dropna(subset=["Sexe", "Âge"]).copy()
    bscr_valid["ID"] = bscr_valid["Nom"].str.upper().str.strip() + "_" + bscr_valid["Prénom"].str.upper().str.strip()

    matched_rows = []
    matched_ids = set()

    for _, bs in bscr_valid.iterrows():
        result = {
            "Birdshot_ID": bs["ID"], "Birdshot_Age": bs["Âge"], "Birdshot_Sexe": bs["Sexe"],
            "Control_OD_ID": None, "Control_OD_Age": None,
            "Control_OG_ID": None, "Control_OG_Age": None,
            "Control_Origine": None,
        }

        age_mask = ctrl_pool["Âge"].between(bs["Âge"] - 10, bs["Âge"] + 10)
        sex_mask = ctrl_pool["Sexe"] == bs["Sexe"]

        # 1. Try bilateral match
        odg = ctrl_pool[(ctrl_pool["Origine"] == "ODG") & sex_mask & age_mask]
        if not odg.empty:
            c = odg.iloc[0]
            result.update(Control_OD_ID=c["ID"], Control_OD_Age=c["Âge"],
                          Control_OG_ID=c["ID"], Control_OG_Age=c["Âge"], Control_Origine="ODG")
            matched_ids.add(c["ID"])
            ctrl_pool = ctrl_pool[ctrl_pool["ID"] != c["ID"]]
        else:
            # 2. Try separate OD + OG
            od_cands = ctrl_pool[(ctrl_pool["Origine"] == "OD") & sex_mask & age_mask]
            og_cands = ctrl_pool[(ctrl_pool["Origine"] == "OG") & sex_mask & age_mask]

            if not od_cands.empty:
                c = od_cands.iloc[0]
                result["Control_OD_ID"], result["Control_OD_Age"] = c["ID"], c["Âge"]
                matched_ids.add(c["ID"])
                ctrl_pool = ctrl_pool[ctrl_pool["ID"] != c["ID"]]

            if not og_cands.empty:
                c = og_cands.iloc[0]
                result["Control_OG_ID"], result["Control_OG_Age"] = c["ID"], c["Âge"]
                matched_ids.add(c["ID"])
                ctrl_pool = ctrl_pool[ctrl_pool["ID"] != c["ID"]]

            if result["Control_OD_ID"] and result["Control_OG_ID"]:
                result["Control_Origine"] = "OD+OG"
            elif result["Control_OD_ID"]:
                result["Control_Origine"] = "OD only"
            elif result["Control_OG_ID"]:
                result["Control_Origine"] = "OG only"

        matched_rows.append(result)

    matched_df = pd.DataFrame(matched_rows)
    unmatched = matched_df[matched_df["Control_OD_ID"].isna() & matched_df["Control_OG_ID"].isna()]

    n_full = matched_df["Control_Origine"].notna().sum()
    print(f"Matched: {n_full} / {len(matched_df)} BSCR patients")
    print(f"Unmatched: {len(unmatched)}")

    with pd.ExcelWriter(args.output) as writer:
        matched_df.to_excel(writer, sheet_name="Matches", index=False)
        unmatched.to_excel(writer, sheet_name="Unmatched BSCR", index=False)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Case-control matching for Birdshot-WIDE")
    parser.add_argument("--bscr", required=True, help="BSCR patient-level file (.xlsx)")
    parser.add_argument("--controls", required=True, help="Control patient-level file (.xlsx)")
    parser.add_argument("--output", required=True, help="Output matching results (.xlsx)")
    main(parser.parse_args())
