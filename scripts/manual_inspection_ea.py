"""
In this script a balanced manual inspection sample from the AllSides 45 error analysis export is created
Input: results/allsides_45_error_analysis_export.csv
Output: results/manual_inspection_ea.csv
"""
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("results/allsides_45_error_analysis_export.csv")
OUTPUT_PATH = Path("results/manual_inspection_ea.csv")

SAMPLE_PER_ERROR_TYPE = 12
RANDOM_SEED = 42


def main():
    df = pd.read_csv(INPUT_PATH)

    sampled_rows = []

    for error_type, group in df.groupby("error_type"):
        sample_size = min(SAMPLE_PER_ERROR_TYPE, len(group))

        sampled_group = group.sample(
            n=sample_size,
            random_state=RANDOM_SEED,
        )

        sampled_rows.append(sampled_group)

    sample_df = pd.concat(sampled_rows, ignore_index=True)

    sample_df = sample_df.sort_values(
        by=["error_type", "doc_id", "gold_start", "pred_start"],
        na_position="last",
    )

    sample_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved manual-inspection sample to {OUTPUT_PATH} ({len(sample_df)} rows)")
    print(sample_df["error_type"].value_counts())


if __name__ == "__main__":
    main()

