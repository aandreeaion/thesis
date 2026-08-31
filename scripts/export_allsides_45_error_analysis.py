"""
Evaluate the final merged-label ModernBERT model on the 45 manually annotated
AllSides news articles.

This script loads the saved merged-label ModernBERT model trained on Webis editorials,
predicts BIO labels for the out-of-domain AllSides evaluation set, and computes:
- strict precision/recall/F1 using seqeval
- relaxed span-level precision/recall/F1
- per-label strict and relaxed results

Note: Merged label scheme treats common-ground as assumption.
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
    spans_from_bio_predictions,
    relaxed_f1_for_dataset,
)

MODEL_VERSION = "relaxed_selected"

DATA_DIR = Path("data/bio_modernbert_tokens")
TEST_FILE = DATA_DIR / "allsides_45_bio_modernbert_merged.jsonl"
GOLD_SPAN_FILE = Path("data/bio_modernbert_tokens/allsides_45_gold_merged.jsonl")
MODEL_DIR = Path("outputs/modernbert_webis_final_relaxed_selected_merged/best_model")
OUTPUT_DIR = Path("outputs/allsides_45_out_of_domain_evaluation")

CSV_OUTPUT = Path("results/allsides_45_out_of_domain_per_label_results.csv")
JSON_OUTPUT = Path("results/allsides_45_out_of_domain_results.json")
ERROR_ANALYSIS_OUTPUT = Path("results/allsides_45_error_analysis_export.csv")
PREDICTED_SPANS_OUTPUT = Path("results/allsides_45_predicted_spans.csv")
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

def load_gold_span_records(path):
    """
    Load the original span-level gold annotations.

    These are used for relaxed F1 so that gold spans come directly
    from the manual annotations
    """
    records = load_jsonl(path)

    gold_spans_by_doc = {}

    for record in records:
        gold_spans_by_doc[record["id"]] = record["spans"]

    return gold_spans_by_doc

def compute_relaxed_per_label(gold_spans_by_doc, records, relaxed_predictions):
    """
    Compute relaxed precision/recall/F1 separately for each label

    Gold spans come from the original manual span-level annotations
    Predicted spans come from the model BIO predictions
    """
    pred_spans_by_doc = spans_from_bio_predictions(
        records=records,
        predicted_label_sequences=relaxed_predictions,
    )

    per_label_results = {}

    for label in ARGUMENT_LABELS:
        gold_for_label = {}
        pred_for_label = {}

        for doc_id, gold_spans in gold_spans_by_doc.items():
            gold_for_label[doc_id] = [
                span for span in gold_spans
                if span["label"] == label
            ]

            pred_for_label[doc_id] = [
                span for span in pred_spans_by_doc.get(doc_id, [])
                if span["label"] == label
            ]

        per_label_results[label] = relaxed_f1_for_dataset(
            gold_spans_by_doc=gold_for_label,
            pred_spans_by_doc=pred_for_label,
        )

    return per_label_results

def compute_relaxed_overall(gold_spans_by_doc, records, relaxed_predictions):
    """
    Compute overall relaxed precision/recall/F1 across all labels.

    Gold spans come from the original manual span-level annotations.
    Predicted spans come from the model BIO predictions.
    """
    pred_spans_by_doc = spans_from_bio_predictions(
        records=records,
        predicted_label_sequences=relaxed_predictions,
    )

    return relaxed_f1_for_dataset(
        gold_spans_by_doc=gold_spans_by_doc,
        pred_spans_by_doc=pred_spans_by_doc,
    )

def load_gold_texts(path):
    """
    Load the original article texts from the gold span file.
    """
    records = load_jsonl(path)
    return {record["id"]: record["text"] for record in records}


def clean_for_csv(text):
    """
    Make span/context text easier to inspect in a spreadsheet.
    """
    if text is None:
        return ""

    return " ".join(text.split())


def get_span_text(text, span):
    """
    Extract the text covered by a character span.
    """
    if not text or span is None:
        return ""

    start = max(0, int(span["start"]))
    end = min(len(text), int(span["end"]))

    return clean_for_csv(text[start:end])


def get_context(text, start, end, window=180):
    """
    Extract surrounding context.
    """
    if not text:
        return ""

    context_start = max(0, int(start) - window)
    context_end = min(len(text), int(end) + window)

    return clean_for_csv(text[context_start:context_end])


def spans_overlap(span_a, span_b):
    """
    Check whether two character spans overlap.
    """
    return span_a["start"] < span_b["end"] and span_b["start"] < span_a["end"]


def overlap_length(span_a, span_b):
    """
    Compute the number of overlapping characters between two spans.
    """
    if not spans_overlap(span_a, span_b):
        return 0

    return min(span_a["end"], span_b["end"]) - max(span_a["start"], span_b["start"])


def predicted_spans_from_bio_for_error_analysis(records, predicted_label_sequences):
    """
    Convert predicted BIO labels back into character spans.

    This uses the token offsets in the AllSides BIO file.
    """
    pred_spans_by_doc = {}

    for record, predicted_labels in zip(records, predicted_label_sequences):
        doc_id = record["id"]
        offsets = record["offsets"]

        spans = []
        current_span = None

        for label, offset in zip(predicted_labels, offsets):
            if label == -100 or label == "O":
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
                continue

            if not isinstance(label, str) or "-" not in label:
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
                continue

            start, end = offset

            if start == end:
                continue

            prefix, argument_label = label.split("-", 1)

            if argument_label not in ARGUMENT_LABELS:
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None
                continue

            if (
                prefix == "B"
                or current_span is None
                or current_span["label"] != argument_label
            ):
                if current_span is not None:
                    spans.append(current_span)

                current_span = {
                    "start": int(start),
                    "end": int(end),
                    "label": argument_label,
                }

            elif prefix == "I" and current_span["label"] == argument_label:
                current_span["end"] = int(end)

        if current_span is not None:
            spans.append(current_span)

        pred_spans_by_doc[doc_id] = spans

    return pred_spans_by_doc

def build_predicted_span_rows(pred_spans_by_doc, texts_by_doc):
    """
    Build a CSV table with all predicted spans.
    """
    rows = []

    for doc_id in sorted(pred_spans_by_doc.keys()):
        text = texts_by_doc.get(doc_id, "")
        pred_spans = pred_spans_by_doc.get(doc_id, [])

        for pred_index, pred_span in enumerate(pred_spans):
            rows.append({
                "doc_id": doc_id,
                "prediction_index": pred_index,
                "pred_label": pred_span["label"],
                "pred_start": pred_span["start"],
                "pred_end": pred_span["end"],
                "pred_text": get_span_text(text, pred_span),
            })

    return rows

def find_best_overlapping_span(target_span, candidate_spans, used_indices, same_label=None):
    """
    Find the unused candidate span with the largest overlap.
    Optionally require the same label.
    """
    best_index = None
    best_span = None
    best_overlap = 0

    for index, candidate_span in enumerate(candidate_spans):
        if index in used_indices:
            continue

        if same_label is True and candidate_span["label"] != target_span["label"]:
            continue

        if same_label is False and candidate_span["label"] == target_span["label"]:
            continue

        current_overlap = overlap_length(target_span, candidate_span)

        if current_overlap > best_overlap:
            best_index = index
            best_span = candidate_span
            best_overlap = current_overlap

    return best_index, best_span, best_overlap

def find_all_overlapping_spans(target_span, candidate_spans, used_indices, same_label=None):
    """
    Find all unused candidate spans that overlap with the target span.
    Optionally require the same label or a different label.
    """
    matches = []

    for index, candidate_span in enumerate(candidate_spans):
        if index in used_indices:
            continue

        if same_label is True and candidate_span["label"] != target_span["label"]:
            continue

        if same_label is False and candidate_span["label"] == target_span["label"]:
            continue

        current_overlap = overlap_length(target_span, candidate_span)

        if current_overlap > 0:
            matches.append((index, candidate_span, current_overlap))

    return matches

def build_error_analysis_rows(gold_spans_by_doc, pred_spans_by_doc, texts_by_doc):
    """
    Build a table of model-gold disagreement cases for qualitative error analysis.

    The automatic error_type values are retrieval categories, not final manual
    interpretations. Final categories are assigned during manual inspection.
    """
    rows = []

    for doc_id in sorted(gold_spans_by_doc.keys()):
        text = texts_by_doc.get(doc_id, "")
        gold_spans = gold_spans_by_doc.get(doc_id, [])
        pred_spans = pred_spans_by_doc.get(doc_id, [])

        used_gold_indices = set()
        used_pred_indices = set()

        # First pass: same-label overlaps.
        # This allows one gold span to match multiple same-label predicted spans.
        # Exact matches and same-clean-text matches are not exported as errors.
        for gold_index, gold_span in enumerate(gold_spans):
            same_label_matches = find_all_overlapping_spans(
                target_span=gold_span,
                candidate_spans=pred_spans,
                used_indices=used_pred_indices,
                same_label=True,
            )

            if not same_label_matches:
                continue

            same_label_matches = sorted(
                same_label_matches,
                key=lambda match: (match[1]["start"], match[1]["end"]),
            )

            used_gold_indices.add(gold_index)

            for pred_index, _, _ in same_label_matches:
                used_pred_indices.add(pred_index)

            matched_pred_spans = [
                pred_span for _, pred_span, _ in same_label_matches
            ]

            gold_text = get_span_text(text, gold_span)

            pred_text = " || ".join(
                get_span_text(text, pred_span)
                for pred_span in matched_pred_spans
            )

            pred_start = min(pred_span["start"] for pred_span in matched_pred_spans)
            pred_end = max(pred_span["end"] for pred_span in matched_pred_spans)
            overlap = sum(overlap for _, _, overlap in same_label_matches)

            if len(matched_pred_spans) == 1:
                pred_span = matched_pred_spans[0]

                exact_match = (
                    gold_span["start"] == pred_span["start"]
                    and gold_span["end"] == pred_span["end"]
                    and gold_span["label"] == pred_span["label"]
                )

                same_clean_text = (
                    gold_span["label"] == pred_span["label"]
                    and gold_text == get_span_text(text, pred_span)
                )

                if exact_match or same_clean_text:
                    continue

                error_type = "same_label_boundary_difference"

            else:
                error_type = "same_label_segmentation_difference"

            start = min(gold_span["start"], pred_start)
            end = max(gold_span["end"], pred_end)

            rows.append({
                "doc_id": doc_id,
                "error_type": error_type,
                "gold_label": gold_span["label"],
                "pred_label": gold_span["label"],
                "gold_start": gold_span["start"],
                "gold_end": gold_span["end"],
                "pred_start": pred_start,
                "pred_end": pred_end,
                "overlap_chars": overlap,
                "gold_text": gold_text,
                "pred_text": pred_text,
                "context": get_context(text, start, end),
                "manual_category": "",
                "note": "",
            })

        # Second pass: overlapping spans with different labels.
        for gold_index, gold_span in enumerate(gold_spans):
            if gold_index in used_gold_indices:
                continue

            pred_index, pred_span, overlap = find_best_overlapping_span(
                target_span=gold_span,
                candidate_spans=pred_spans,
                used_indices=used_pred_indices,
                same_label=False,
            )

            if pred_span is None:
                continue

            used_gold_indices.add(gold_index)
            used_pred_indices.add(pred_index)

            start = min(gold_span["start"], pred_span["start"])
            end = max(gold_span["end"], pred_span["end"])

            rows.append({
                "doc_id": doc_id,
                "error_type": "overlapping_different_label",
                "gold_label": gold_span["label"],
                "pred_label": pred_span["label"],
                "gold_start": gold_span["start"],
                "gold_end": gold_span["end"],
                "pred_start": pred_span["start"],
                "pred_end": pred_span["end"],
                "overlap_chars": overlap,
                "gold_text": get_span_text(text, gold_span),
                "pred_text": get_span_text(text, pred_span),
                "context": get_context(text, start, end),
                "manual_category": "",
                "note": "",
            })

        # Third pass: remaining unmatched gold spans.
        for gold_index, gold_span in enumerate(gold_spans):
            if gold_index in used_gold_indices:
                continue

            rows.append({
                "doc_id": doc_id,
                "error_type": "unmatched_gold_span",
                "gold_label": gold_span["label"],
                "pred_label": "",
                "gold_start": gold_span["start"],
                "gold_end": gold_span["end"],
                "pred_start": "",
                "pred_end": "",
                "overlap_chars": 0,
                "gold_text": get_span_text(text, gold_span),
                "pred_text": "",
                "context": get_context(text, gold_span["start"], gold_span["end"]),
                "manual_category": "",
                "note": "",
            })

        # Fourth pass: remaining unmatched predicted spans.
        for pred_index, pred_span in enumerate(pred_spans):
            if pred_index in used_pred_indices:
                continue

            rows.append({
                "doc_id": doc_id,
                "error_type": "unmatched_prediction",
                "gold_label": "",
                "pred_label": pred_span["label"],
                "gold_start": "",
                "gold_end": "",
                "pred_start": pred_span["start"],
                "pred_end": pred_span["end"],
                "overlap_chars": 0,
                "gold_text": "",
                "pred_text": get_span_text(text, pred_span),
                "context": get_context(text, pred_span["start"], pred_span["end"]),
                "manual_category": "",
                "note": "",
            })

    return rows

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
    print("Evaluating Model B per label on out-of-domain AllSides 45 news articles")
    print("Model directory:", MODEL_DIR)
    print("Test file:", TEST_FILE)

    print("\nLoading test data...")
    test_records = load_jsonl(TEST_FILE)
    gold_spans_by_doc = load_gold_span_records(GOLD_SPAN_FILE)
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
        gold_spans_by_doc=gold_spans_by_doc,
        records=test_records,
        relaxed_predictions=relaxed_predictions,
    )
    
    print("\nComputing relaxed overall results...")
    relaxed_overall = compute_relaxed_overall(
        gold_spans_by_doc=gold_spans_by_doc,
        records=test_records,
        relaxed_predictions=relaxed_predictions,
    )

    print("Overall relaxed precision:", relaxed_overall["precision"])
    print("Overall relaxed recall:", relaxed_overall["recall"])
    print("Overall relaxed F1:", relaxed_overall["f1"])
    print(f"\nBuilding error-analysis export...")

    texts_by_doc = load_gold_texts(GOLD_SPAN_FILE)

    pred_spans_by_doc = spans_from_bio_predictions(
        records=test_records,
        predicted_label_sequences=relaxed_predictions,
    )

    predicted_span_rows = build_predicted_span_rows(
        pred_spans_by_doc=pred_spans_by_doc,
        texts_by_doc=texts_by_doc,
    )

    save_rows_to_csv(predicted_span_rows, PREDICTED_SPANS_OUTPUT)

    print(f"Saved predicted spans to {PREDICTED_SPANS_OUTPUT} ({len(predicted_span_rows)} rows)")

    error_rows = build_error_analysis_rows(
        gold_spans_by_doc=gold_spans_by_doc,
        pred_spans_by_doc=pred_spans_by_doc,
        texts_by_doc=texts_by_doc,
    )

    save_rows_to_csv(error_rows, ERROR_ANALYSIS_OUTPUT)

    print(f"Saved error-analysis export to {ERROR_ANALYSIS_OUTPUT} ({len(error_rows)} rows)")
    rows = build_per_label_table(
        strict_report=strict_report,
        relaxed_per_label=relaxed_per_label,
    )

    save_rows_to_csv(rows, CSV_OUTPUT)

    results = {
        "model_version": MODEL_VERSION,
        "model_dir": str(MODEL_DIR),
        "test_file": str(TEST_FILE),
        "overall_relaxed": relaxed_overall,
        "per_label_results": rows,
        "strict_report": strict_report,
        "relaxed_per_label": relaxed_per_label,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)

    print(f"\nSaved CSV results to: {CSV_OUTPUT}")

    print(f"\nSaved JSON results to: {JSON_OUTPUT}")

    print("\nPer-label results:")
    for row in rows:
        print(row)

    print("\nDone.")

if __name__ == "__main__":
    main()