"""
Temperature calibration for final ModernBERT token-classification models.

This script loads one final trained model, runs it on the Webis validation split,
learns one temperature parameter on the validation logits, and reports
calibration metrics before and after temperature scaling.

Set MODEL_VERSION to either:
- "strict_selected"
- "relaxed_selected"
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


MODEL_VERSION = "relaxed_selected"


DATA_DIR = Path("data/bio_modernbert_tokens")
VALIDATION_FILE = DATA_DIR / "webis_editorials_validation_bio_modernbert.jsonl"

MODEL_DIR = Path(f"outputs/modernbert_webis_final_{MODEL_VERSION}/best_model")
OUTPUT_DIR = Path(f"outputs/modernbert_webis_final_{MODEL_VERSION}")

CALIBRATION_OUTPUT_FILE = OUTPUT_DIR / "temperature_calibration_results.json"


LABEL_LIST = [
    "O",
    "B-anecdote",
    "I-anecdote",
    "B-testimony",
    "I-testimony",
    "B-common-ground",
    "I-common-ground",
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

    Labels with -100 are removed.
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

def main():
    print("Temperature calibration")
    print("Model version:", MODEL_VERSION)
    print("Model directory:", MODEL_DIR)
    print("Validation file:", VALIDATION_FILE)

    print("\nChecking GPU...")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading validation dataset...")
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
        output_dir=str(OUTPUT_DIR / "calibration_tmp"),
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

    print("\nCollecting validation logits...")
    logits, labels = collect_validation_logits_and_labels(
        model=model,
        trainer=trainer,
        validation_dataset=validation_dataset,
    )

    logits = logits.to(device)
    labels = labels.to(device)

    print("Calibration tokens:", len(labels))

    print("\nMetrics before calibration...")
    before_nll = compute_nll(logits, labels, temperature=1.0)
    before_ece = compute_ece(logits, labels, temperature=1.0)

    print("Before NLL:", before_nll)
    print("Before ECE:", before_ece)

    print("\nLearning temperature...")
    temperature = learn_temperature(logits, labels)

    print("Learned temperature:", temperature)

    print("\nMetrics after calibration...")
    after_nll = compute_nll(logits, labels, temperature=temperature)
    after_ece = compute_ece(logits, labels, temperature=temperature)

    print("After NLL:", after_nll)
    print("After ECE:", after_ece)

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
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with CALIBRATION_OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSaved calibration results to:")
    print(CALIBRATION_OUTPUT_FILE)

    print("\nDone.")

if __name__ == "__main__":
    main()