"""
Temperature calibration and validation-threshold check for the final merged-label ModernBERT model.

This script loads the final merged-label ModernBERT token-classification model and
runs it on the Webis validation split. It learns one global scalar temperature on
the validation logits and reports calibration quality before and after temperature
scaling.

The script also checks whether applying a global confidence threshold to non-O
predictions improves validation F1. Thresholding is evaluated as a diagnostic
step only; the selected threshold is saved, but thresholding is not automatically
applied to later corpus-level predictions unless a separate prediction script
explicitly uses it.

Outputs:
- negative log likelihood and expected calibration error before/after temperature scaling;
- strict and relaxed validation F1 before/after temperature scaling;
- strict and relaxed validation F1 for candidate confidence thresholds;
- a JSON file with the learned temperature and validation-threshold results.
"""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from relaxed_F1 import (
    relaxed_f1_from_bio_predictions,
)

PROJECT_ROOT = Path(".")
MODEL_VERSION = "relaxed_selected_merged"

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "modernbert_webis_final_relaxed_selected_merged"
    / "best_model"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "bio_modernbert_tokens"
    / "webis_editorials_validation_bio_modernbert_merged.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "modernbert_webis_final_relaxed_selected_merged"
)

CALIBRATION_OUTPUT_FILE = (
    OUTPUT_DIR
    / "temp_calibration_threshold_f1_validation_results.json"
)

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
    """
    Convert BIO string labels to numeric IDs
    Special-token labels are already -100 and stay -100
    """
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

def ids_to_bio_sequences(predictions, label_ids):
    """
    Convert numeric prediction IDs and numeric gold label IDs into BIO label strings
    """
    predicted_sequences = []
    gold_sequences = []

    for pred_doc, gold_doc in zip(predictions, label_ids):
        predicted_doc_labels = []
        gold_doc_labels = []

        for pred_id, gold_id in zip(pred_doc, gold_doc):
            if gold_id == -100:
                continue

            predicted_doc_labels.append(ID2LABEL[int(pred_id)])
            gold_doc_labels.append(ID2LABEL[int(gold_id)])

        predicted_sequences.append(predicted_doc_labels)
        gold_sequences.append(gold_doc_labels)

    return predicted_sequences, gold_sequences

def compute_strict_and_relaxed_f1(records, predicted_label_sequences, gold_label_sequences):
    """
    Compute strict seqeval metrics and relaxed span-overlap F1.
    """
    strict_precision = precision_score(gold_label_sequences, predicted_label_sequences)
    strict_recall = recall_score(gold_label_sequences, predicted_label_sequences)
    strict_f1 = f1_score(gold_label_sequences, predicted_label_sequences)
    strict_accuracy = accuracy_score(gold_label_sequences, predicted_label_sequences)

    relaxed_results = relaxed_f1_from_bio_predictions(
        records=records,
        predicted_label_sequences=predicted_label_sequences,
    )

    return {
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": strict_f1,
        "strict_accuracy": strict_accuracy,
        "relaxed_precision": relaxed_results["precision"],
        "relaxed_recall": relaxed_results["recall"],
        "relaxed_f1": relaxed_results["f1"],
    }

def apply_confidence_threshold(logits, temperature, threshold):
    """
    Apply temperature scaling and then replace low-confidence non-O predictions with O.

    The O label is kept as O regardless of confidence.
    Non-O labels are kept only if their calibrated confidence is >= threshold.
    """
    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)

    confidences, predictions = torch.max(probabilities, dim=-1)

    o_label_id = LABEL2ID["O"]

    thresholded_predictions = predictions.clone()

    low_confidence_non_o = (predictions != o_label_id) & (confidences < threshold)
    thresholded_predictions[low_confidence_non_o] = o_label_id

    return thresholded_predictions

def compute_nll(logits, labels, temperature=1.0):

    # Compute negative log likelihood
    scaled_logits = logits / temperature
    loss_function = nn.CrossEntropyLoss()
    return loss_function(scaled_logits, labels).item()


def compute_ece(logits, labels, temperature=1.0, n_bins=10):
    
    # Compute Expected Calibration Error
    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)

    confidences, predictions = torch.max(probabilities, dim=-1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)

    for bin_start, bin_end in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_start) * confidences.le(bin_end)
        proportion_in_bin = in_bin.float().mean()

        if proportion_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(confidence_in_bin - accuracy_in_bin) * proportion_in_bin

    return ece.item()

def learn_temperature(logits, labels):

    # Learn one scalar temperature on the validation set.
    temperature = torch.ones(1, device=logits.device, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=0.01,
        max_iter=50,
    )

    loss_function = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_function(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    learned_temperature = temperature.detach().item()

    # Safety: temperature should be positive.
    if learned_temperature <= 0:
        learned_temperature = 1.0

    return learned_temperature

def collect_validation_logits_and_labels(model, trainer, validation_dataset):
    """
    Run the model on the validation set and collect logits and gold labels.
    """
    predictions_output = trainer.predict(validation_dataset)

    logits = torch.tensor(predictions_output.predictions)
    labels = torch.tensor(predictions_output.label_ids)

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)

    valid_positions = flat_labels != -100

    valid_logits = flat_logits[valid_positions]
    valid_labels = flat_labels[valid_positions]

    return valid_logits, valid_labels

def collect_prediction_output(trainer, dataset):
    """
    Run the model on a dataset and return full logits and label IDs.

    This keeps the original document/token structure, so it can compute BIO F1.
    """
    predictions_output = trainer.predict(dataset)

    logits = torch.tensor(predictions_output.predictions)
    label_ids = torch.tensor(predictions_output.label_ids)

    return logits, label_ids

def main():
    print("Temperature calibration + validation threshold tuning")
    print("Model version:", MODEL_VERSION)
    print("Model directory:", MODEL_DIR)
    print("Validation file:", VALIDATION_FILE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nCUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading validation data...")
    validation_records = load_jsonl(VALIDATION_FILE)
    validation_dataset = load_dataset_from_jsonl(VALIDATION_FILE)

    print("Validation examples:", len(validation_dataset))

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model.to(device)
    model.eval()

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "calibration_threshold_tmp"),
        per_device_eval_batch_size=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\nCollecting validation logits for calibration...")
    logits, labels = collect_validation_logits_and_labels(
        model=model,
        trainer=trainer,
        validation_dataset=validation_dataset,
    )

    logits = logits.to(device)
    labels = labels.to(device)

    before_nll = compute_nll(logits, labels, temperature=1.0)
    before_ece = compute_ece(logits, labels, temperature=1.0)

    temperature = learn_temperature(logits, labels)

    after_nll = compute_nll(logits, labels, temperature=temperature)
    after_ece = compute_ece(logits, labels, temperature=temperature)

    print("\nCalibration results:")
    print("Temperature:", temperature)
    print("NLL before:", before_nll)
    print("NLL after:", after_nll)
    print("ECE before:", before_ece)
    print("ECE after:", after_ece)

    print("\nComputing validation F1...")
    validation_logits_full, validation_label_ids_full = collect_prediction_output(
        trainer=trainer,
        dataset=validation_dataset,
    )

    # F1 before calibration
    validation_predictions_before = torch.argmax(validation_logits_full, dim=-1)

    validation_pred_sequences_before, validation_gold_sequences = ids_to_bio_sequences(
        predictions=validation_predictions_before,
        label_ids=validation_label_ids_full,
    )

    validation_f1_before = compute_strict_and_relaxed_f1(
        records=validation_records,
        predicted_label_sequences=validation_pred_sequences_before,
        gold_label_sequences=validation_gold_sequences,
    )

    # F1 after calibration only
    validation_predictions_after_calibration = torch.argmax(
        validation_logits_full / temperature,
        dim=-1,
    )

    validation_pred_sequences_after_calibration, _ = ids_to_bio_sequences(
        predictions=validation_predictions_after_calibration,
        label_ids=validation_label_ids_full,
    )

    validation_f1_after_calibration = compute_strict_and_relaxed_f1(
        records=validation_records,
        predicted_label_sequences=validation_pred_sequences_after_calibration,
        gold_label_sequences=validation_gold_sequences,
    )

    print("\nValidation F1 before calibration:")
    print(validation_f1_before)

    print("\nValidation F1 after calibration only:")
    print(validation_f1_after_calibration)

    print("\nTuning confidence threshold on validation set...")

    thresholds = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]

    threshold_results = []
    best_threshold = None
    best_validation_relaxed_f1 = -1.0
    best_validation_threshold_metrics = None

    for threshold in thresholds:
        thresholded_validation_predictions = apply_confidence_threshold(
            logits=validation_logits_full,
            temperature=temperature,
            threshold=threshold,
        )

        thresholded_validation_pred_sequences, _ = ids_to_bio_sequences(
            predictions=thresholded_validation_predictions,
            label_ids=validation_label_ids_full,
        )

        threshold_metrics = compute_strict_and_relaxed_f1(
            records=validation_records,
            predicted_label_sequences=thresholded_validation_pred_sequences,
            gold_label_sequences=validation_gold_sequences,
        )

        threshold_result = {
            "threshold": threshold,
            **threshold_metrics,
        }

        threshold_results.append(threshold_result)

        if threshold_metrics["relaxed_f1"] > best_validation_relaxed_f1:
            best_validation_relaxed_f1 = threshold_metrics["relaxed_f1"]
            best_threshold = threshold
            best_validation_threshold_metrics = threshold_metrics

    print("\nBest validation threshold:", best_threshold)
    print("Best validation threshold metrics:")
    print(best_validation_threshold_metrics)

    results = {
        "model_version": MODEL_VERSION,
        "model_dir": str(MODEL_DIR),
        "validation_file": str(VALIDATION_FILE),
        "temperature": temperature,
        "n_calibration_tokens": int(len(labels)),

        "before_nll": before_nll,
        "after_nll": after_nll,
        "before_ece": before_ece,
        "after_ece": after_ece,

        "validation_f1_before_calibration": validation_f1_before,
        "validation_f1_after_calibration_only": validation_f1_after_calibration,

        "threshold_selection_metric": "validation_relaxed_f1",
        "threshold_results_validation": threshold_results,
        "best_threshold_selected_on_validation_relaxed_f1": best_threshold,
        "best_validation_threshold_metrics": best_validation_threshold_metrics,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CALIBRATION_OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSaved results to:")
    print(CALIBRATION_OUTPUT_FILE)

    print("\nDone.")

if __name__ == "__main__":
    main()