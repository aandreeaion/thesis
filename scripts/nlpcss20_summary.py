"""
This script creates a summary from the nplcss20 prediction file

Input:
results/model_predictions/nlpcss20_model_predictions_filtered.csv

Output: 
results/model_predictions/nlpcss20_article_summary.csv

The input file has one row per model predicted argumentative span
the output file will have one row per article, calculate counts, shares and proportions

"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    PROJECT_ROOT / "results" / "model_predictions" / "nlpcss20_model_predictions_filtered.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "model_predictions" / "nlpcss20_article_summary.csv"
)

LABELS = [
    "assumption",
    "testimony",
    "anecdote",
    "statistics",
    "other",
]

EVIDENCE_LABELS = [
    "testimony",
    "anecdote",
    "statistics",
]

METADATA_COLUMNS = [
    "article_position",
    "original_index",
    "event_id",
    "source",
    "title",
    "adfontes_fair",
    "adfontes_political",
    "allsides_bias",
    "time",
    "topics",
    "author",
    "word_count",
]


def load_predictions(path):
    """
    Load the span-level prediction file and check that the expected columns exist.
    """
    df = pd.read_csv(path)

    expected_columns = METADATA_COLUMNS + ["label"]

    missing_columns = [
        column for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    return df

METADATA_COLUMNS = [
    "article_position",
    "original_index",
    "event_id",
    "source",
    "title",
    "adfontes_fair",
    "adfontes_political",
    "allsides_bias",
    "time",
    "topics",
    "author",
    "word_count",
]


def load_predictions(path):
    """
    Load the span-level prediction file.
    """
    df = pd.read_csv(path)

    required_columns = METADATA_COLUMNS + ["label"]

    missing_columns = set(required_columns) - set(df.columns)

    return df

def create_article_base(df):
    """
    Creating one row per article with article metadata and total predicted span count
    """
    metadata_to_keep = [
        column for column in METADATA_COLUMNS
        if column != "article_position"
    ]

    article_base = (
        df.groupby("article_position", as_index=False)
        .agg({column: "first" for column in metadata_to_keep})
    )

    total_spans = (
        df.groupby("article_position")
        .size()
        .reset_index(name="total_predicted_spans")
    )

    article_base = article_base.merge(
        total_spans,
        on="article_position",
        how="left",
    )

    return article_base

def add_label_counts(article_base, df):
    """
    Adding one count column per predicted label
    """
    label_counts = (
        df.groupby(["article_position", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for label in LABELS:
        if label not in label_counts.columns:
            label_counts[label] = 0

    label_counts = label_counts[
        ["article_position"] + LABELS
    ]

    label_counts = label_counts.rename(
        columns={label: f"{label}_count" for label in LABELS}
    )

    article_summary = article_base.merge(
        label_counts,
        on="article_position",
        how="left",
    )

    return article_summary

def add_label_shares(article_summary):
    """
    Adding one share column per predicted label

    The share is calculated within each article:
    label_count / total_predicted_spans
    """
    for label in LABELS:
        count_column = f"{label}_count"
        share_column = f"{label}_share"

        article_summary[share_column] = (
            article_summary[count_column]
            / article_summary["total_predicted_spans"]
        )

    return article_summary

def add_label_densities(article_summary):
    """
    Add one density column per predicted label.

    The density is calculated as:
    label_count / word_count * 1000
    """
    article_summary["total_spans_per_1000_words"] = (
        article_summary["total_predicted_spans"]
        / article_summary["word_count"]
        * 1000
    )

    for label in LABELS:
        count_column = f"{label}_count"
        density_column = f"{label}_per_1000_words"

        article_summary[density_column] = (
            article_summary[count_column]
            / article_summary["word_count"]
            * 1000
        )

    return article_summary

def add_derived_measures(article_summary):
    """
    Add combined evidence and non-evidence measures.

    Evidence labels:
    testimony, anecdote, statistics

    Non-evidence labels:
    assumption, other
    """
    article_summary["evidence_count"] = sum(
        article_summary[f"{label}_count"]
        for label in EVIDENCE_LABELS
    )

    article_summary["non_evidence_count"] = (
        article_summary["assumption_count"]
        + article_summary["other_count"]
    )

    article_summary["evidence_share"] = (
        article_summary["evidence_count"]
        / article_summary["total_predicted_spans"]
    )

    article_summary["non_evidence_share"] = (
        article_summary["non_evidence_count"]
        / article_summary["total_predicted_spans"]
    )

    article_summary["evidence_per_1000_words"] = (
        article_summary["evidence_count"]
        / article_summary["word_count"]
        * 1000
    )

    article_summary["non_evidence_per_1000_words"] = (
        article_summary["non_evidence_count"]
        / article_summary["word_count"]
        * 1000
    )

    article_summary["evidence_to_assumption_ratio"] = (
        article_summary["evidence_count"]
        / article_summary["assumption_count"].replace(0, pd.NA)
    )

    return article_summary

def main():
    """
    Creating and saving the afile.
    """
    print("Loading span-level predictions:")
    print(INPUT_PATH)

    df = load_predictions(INPUT_PATH)

    print("Loaded prediction rows:", len(df))
    print("Unique articles:", df["article_position"].nunique())

    article_base = create_article_base(df)
    article_summary = add_label_counts(article_base, df)
    article_summary = add_label_shares(article_summary)
    article_summary = add_label_densities(article_summary)
    article_summary = add_derived_measures(article_summary)

    article_summary.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved article-level summary:")
    print(OUTPUT_PATH)
    print("Rows:", len(article_summary))
    print("Columns:", len(article_summary.columns))


if __name__ == "__main__":
    main()