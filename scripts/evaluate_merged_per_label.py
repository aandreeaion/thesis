"""
Evaluate the merged-label relaxed-selected ModernBERT model per label on the Webis test split.

This script loads the saved merged-label ModernBERT model,
predicts BIO labels for the merged Webis test set, and computes:
- strict per-label precision/recall/F1 using seqeval
- relaxed per-label precision/recall/F1 using span overlap

The merged label scheme treats common-ground as assumption.
"""

from pathlib import Path
import json
import csv

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import classification_report

from relaxed_F1 import (
    gold_spans_from_bio_records,
    spans_from_bio_predictions,
    relaxed_f1_for_dataset,
)

MODEL_VERSION = "relaxed_selected"

DATA_DIR = Path("data/bio_modernbert_tokens")
TEST_FILE = DATA_DIR / "webis_editorials_test_bio_modernbert_merged.jsonl"

MODEL_DIR = Path("outputs/modernbert_webis_final_relaxed_selected_merged/best_model")
OUTPUT_DIR = Path(f"outputs/modernbert_webis_final_{MODEL_VERSION}_merged")

CSV_OUTPUT = Path("results/webis_test_per_label_results_merged.csv")
JSON_OUTPUT = Path("results/webis_test_per_label_results_merged.json")

LABEL_LIST = [
    "O",
    "B-anecdote",
    "I-anecdote",
    "B-testimony",
    "I-testimony",
    "B-assumption",
    "I-assumption",
    "B-statistics",
    "I-statistics",
    "B-other",
    "I-other",
]

ARGUMENT_LABELS = [
    "anecdote",
    "testimony",
    "assumption",
    "statistics",
    "other",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def convert_labels_to_ids(record):
    label_ids = []

    for label in record["ner_tags"]:
        if label == -100:
            label_ids.append(-100)
        else:
            label_ids.append(LABEL2ID[label])

    return {
        "input_ids": record["input_ids"],
        "attention_mask": record["attention_mask"],
        "labels": label_ids,
    }


def load_dataset_from_jsonl(path):
    records = load_jsonl(path)
    converted_records = [convert_labels_to_ids(record) for record in records]
    return Dataset.from_list(converted_records)

def predictions_to_bio_sequences(predictions, label_ids):
    """
    Convert numeric predictions and gold label IDs into BIO label strings.

    Returns:
    - predicted_sequences: without -100 positions, for seqeval strict F1
    - gold_sequences: without -100 positions, for seqeval strict F1
    - relaxed_predictions: with -100 positions kept, for relaxed F1 alignment
    """
    predicted_sequences = []
    gold_sequences = []
    relaxed_predictions = []

    for prediction_doc, gold_doc in zip(predictions, label_ids):
        predicted_doc_labels = []
        gold_doc_labels = []
        relaxed_doc_labels = []

        for predicted_label_id, gold_label_id in zip(prediction_doc, gold_doc):
            if gold_label_id == -100:
                relaxed_doc_labels.append(-100)
                continue

            predicted_label = ID2LABEL[int(predicted_label_id)]
            gold_label = ID2LABEL[int(gold_label_id)]

            predicted_doc_labels.append(predicted_label)
            gold_doc_labels.append(gold_label)
            relaxed_doc_labels.append(predicted_label)

        predicted_sequences.append(predicted_doc_labels)
        gold_sequences.append(gold_doc_labels)
        relaxed_predictions.append(relaxed_doc_labels)

    return predicted_sequences, gold_sequences, relaxed_predictions

def compute_relaxed_per_label(records, relaxed_predictions):
    """
    Compute relaxed precision/recall/F1 separately for each argument label.
    """
    gold_spans_by_doc = gold_spans_from_bio_records(records)
    pred_spans_by_doc = spans_from_bio_predictions(
        records=records,
        predicted_label_sequences=relaxed_predictions,
    )

    results = {}

    for label in ARGUMENT_LABELS:
        gold_label_spans_by_doc = {}
        pred_label_spans_by_doc = {}

        for doc_id, gold_spans in gold_spans_by_doc.items():
            gold_label_spans_by_doc[doc_id] = [
                span for span in gold_spans
                if span["label"] == label
            ]

            pred_spans = pred_spans_by_doc.get(doc_id, [])
            pred_label_spans_by_doc[doc_id] = [
                span for span in pred_spans
                if span["label"] == label
            ]

        label_results = relaxed_f1_for_dataset(
            gold_spans_by_doc=gold_label_spans_by_doc,
            pred_spans_by_doc=pred_label_spans_by_doc,
        )

        results[label] = label_results

    return results

def build_per_label_table(strict_report, relaxed_per_label):
    """
    Combine strict seqeval results and relaxed per-label results.
    """
    rows = []

    for label in ARGUMENT_LABELS:
        strict_label_results = strict_report.get(label, {})
        relaxed_label_results = relaxed_per_label.get(label, {})

        row = {
            "label": label,

            "strict_precision": strict_label_results.get("precision", 0.0),
            "strict_recall": strict_label_results.get("recall", 0.0),
            "strict_f1": strict_label_results.get("f1-score", 0.0),
            "strict_support": strict_label_results.get("support", 0),

            "relaxed_precision": relaxed_label_results.get("precision", 0.0),
            "relaxed_recall": relaxed_label_results.get("recall", 0.0),
            "relaxed_f1": relaxed_label_results.get("f1", 0.0),
            "relaxed_gold_spans": relaxed_label_results.get("gold_spans", 0),
            "relaxed_predicted_spans": relaxed_label_results.get("predicted_spans", 0),
        }

        rows.append(row)

    return rows

def save_rows_to_csv(rows, output_path):
    """
    Save per-label result rows to a CSV file.
    """
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    print("Evaluating Model B per label on Webis test set")
    print("Model directory:", MODEL_DIR)
    print("Test file:", TEST_FILE)

    print("\nLoading test data...")
    test_records = load_jsonl(TEST_FILE)
    test_dataset = load_dataset_from_jsonl(TEST_FILE)
    print("Test examples:", len(test_dataset))

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "per_label_tmp"),
        per_device_eval_batch_size=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\nPredicting on test set...")
    predictions_output = trainer.predict(test_dataset)

    logits = predictions_output.predictions
    label_ids = predictions_output.label_ids

    predictions = np.argmax(logits, axis=2)

    predicted_sequences, gold_sequences, relaxed_predictions = predictions_to_bio_sequences(
        predictions=predictions,
        label_ids=label_ids,
    )

    print("\nComputing strict per-label results...")
    strict_report = classification_report(
        gold_sequences,
        predicted_sequences,
        output_dict=True,
        zero_division=0,
    )

    print("\nComputing relaxed per-label results...")
    relaxed_per_label = compute_relaxed_per_label(
        records=test_records,
        relaxed_predictions=relaxed_predictions,
    )

    rows = build_per_label_table(
        strict_report=strict_report,
        relaxed_per_label=relaxed_per_label,
    )

    save_rows_to_csv(rows, CSV_OUTPUT)

    results = {
        "model_version": MODEL_VERSION,
        "model_dir": str(MODEL_DIR),
        "test_file": str(TEST_FILE),
        "per_label_results": rows,
        "strict_report": strict_report,
        "relaxed_per_label": relaxed_per_label,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)

    print("\nSaved CSV results to:")
    print(CSV_OUTPUT)

    print("\nSaved JSON results to:")
    print(JSON_OUTPUT)

    print("\nPer-label results:")
    for row in rows:
        print(row)

    print("\nDone.")

if __name__ == "__main__":
    main()