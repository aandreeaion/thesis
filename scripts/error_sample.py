"""
Create a label-balanced manual sample from the AllSides 45 model-gold disagreement export.

This script does not run the model and does not change the evaluation results.
It takes the full disagreement export created by export_allsides_45_error_analysis.py
and creates a smaller CSV for qualitative manual error analysis.

The manual sample includes cases involving each final label:
- testimony
- anecdote
- assumption
- statistics
- other
"""

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("results/allsides_45_error_analysis_export.csv")
OUTPUT_PATH = Path("results/allsides_45_error_analysis_manual_sample.csv")


def sample_rows(df, condition, n, random_state=None):
    """
    Sample up to n rows matching a condition.
    """
    subset = df[condition]

    if len(subset) == 0:
        return subset

    return subset.sample(
        n=min(n, len(subset)),
        random_state=random_state,
    )


def main():
    print("Loading full model-gold disagreement export...")
    df = pd.read_csv(INPUT_PATH)

    print("Full disagreement rows:", len(df))

    if "error_type" in df.columns:
        print("\nAutomatic retrieval-type breakdown in full export:")
        print(df["error_type"].value_counts())

    labels_to_sample = [
        "testimony",
        "anecdote",
        "assumption",
        "statistics",
        "other",
    ]

    samples = []

    # Label-balanced sample:
    # A row is selected for a label if that label appears either in the gold
    # annotation or in the model prediction.
    for i, label in enumerate(labels_to_sample, start=1):
        samples.append(
            sample_rows(
                df=df,
                condition=(
                    (df["gold_label"] == label)
                    | (df["pred_label"] == label)
                ),
                n=10,
                random_state=i,
            )
        )

    label_sample_df = pd.concat(samples).drop_duplicates()

    # Small general random supplement from rows not already selected.
    remaining_df = df.drop(index=label_sample_df.index)
    random_sample_df = remaining_df.sample(
        n=min(10, len(remaining_df)),
        random_state=10,
    )

    sample_df = pd.concat(
        [label_sample_df, random_sample_df]
    ).drop_duplicates().reset_index(drop=True)

    # Only keep columns needed.
    inspection_columns = [
        "doc_id",
        "gold_label",
        "pred_label",
        "gold_text",
        "pred_text",
        "context",
        "manual_category",
        "note",
    ]

    sample_df = sample_df[inspection_columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\nSaved manual-review sample to:")
    print(OUTPUT_PATH)
    print("Sample rows:", len(sample_df))

    print("\nGold-label breakdown in manual sample:")
    print(sample_df["gold_label"].value_counts(dropna=False))

    print("\nPredicted-label breakdown in manual sample:")
    print(sample_df["pred_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()