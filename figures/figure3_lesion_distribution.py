#!/usr/bin/env python3
"""
figure3_lesion_distribution.py

Generate a horizontal bar chart showing the frequency of lesion types
in the BSCR cohort.

Usage:
    python figure3_lesion_distribution.py --metadata metadata_final.xlsx --output figure_lesions.png
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd


def main(args):
    df = pd.read_excel(args.metadata)
    bscr = df[df["cohort"] == "BCR"].copy()
    total = len(bscr)

    counts = bscr["lesion_type"].value_counts()
    counts = counts.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hlines(y=counts.index, xmin=0, xmax=counts.values, color="#4c72b0", alpha=0.6, linewidth=2)
    ax.scatter(counts.values, counts.index, color="#4c72b0", s=150, zorder=3)

    for label, value in counts.items():
        pct = value / total * 100
        ax.text(value + total * 0.01, label, f" {value}  ({pct:.1f}%)", va="center", fontsize=11)

    ax.set_title(f"Lesion-type distribution in the BSCR cohort (N={total})", fontsize=14, pad=20)
    ax.set_xlabel("Number of images", fontsize=12)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot lesion-type distribution")
    parser.add_argument("--metadata", required=True, help="Final metadata file (.xlsx)")
    parser.add_argument("--output", default="figure_lesion_distribution.png", help="Output image path")
    main(parser.parse_args())
