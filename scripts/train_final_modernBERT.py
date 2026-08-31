"""
This script is used for training a final ModernBERT token-classification model using the best
hyperparameters from Optuna, then evaluate it on the Webis Editorials test set.

Set MODEL_VERSION to either:
- strict_selected
- relaxed_selected
"""
from pathlib import Path
import json
import numpy as np
import torch
from datasets import Dataset
import evaluate
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from relaxed_F1 import relaxed_f1_from_bio_predictions

MODEL_VERSION = "relaxed_selected"
MODEL_NAME = "answerdotai/ModernBERT-base"

DATA_DIR = Path("data/bio_modernbert_tokens")
OUTPUT_DIR = Path(f"outputs/modernbert_webis_final_{MODEL_VERSION}")

TRAIN_FILE = DATA_DIR / "webis_editorials_train_bio_modernbert.jsonl"
VALIDATION_FILE = DATA_DIR / "webis_editorials_validation_bio_modernbert.jsonl"
TEST_FILE = DATA_DIR / "webis_editorials_test_bio_modernbert.jsonl"


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

# Selection block for the best hyperparameters from Optuna
FINAL_CONFIGS = {
    "strict_selected": {
        "selection_metric": "f1",
        "learning_rate": 4.2691226724443725e-05,
        "weight_decay": 0.00047042574893661065,
        "num_train_epochs": 4,
        "warmup_ratio": 0.17278407753461164,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "cosine",
    },
    "relaxed_selected": {
        "selection_metric": "relaxed_f1",
        "learning_rate": 4.906314988493981e-05,
        "weight_decay": 0.06830142941975816,
        "num_train_epochs": 5,
        "warmup_ratio": 0.15822895971894255,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "linear",
    },
}

CONFIG = FINAL_CONFIGS[MODEL_VERSION]

# Loading functions
def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def convert_labels_to_ids(record):
    """
    Convert BIO string labels to numeric label IDs
    Special tokens already have the label -100 and are kept as -100
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

def compute_metrics(eval_prediction):
    """
    Compute both strict seqeval F1 and relaxed span F1.

    Strict F1 evaluates exact BIO/entity matches.
    Relaxed F1 keeps the full token sequence, including -100 positions,
    so predictions stay aligned with the ModernBERT offset mappings.
    """
    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []
    relaxed_predictions = []

    for prediction, label_sequence in zip(predictions, labels):
        document_predictions = []
        document_labels = []
        document_relaxed_predictions = []

        for predicted_label_id, gold_label_id in zip(prediction, label_sequence):
            if gold_label_id == -100:
                document_relaxed_predictions.append(-100)
                continue

            predicted_label = ID2LABEL[predicted_label_id]
            gold_label = ID2LABEL[gold_label_id]

            document_predictions.append(predicted_label)
            document_labels.append(gold_label)

            document_relaxed_predictions.append(predicted_label)

        true_predictions.append(document_predictions)
        true_labels.append(document_labels)
        relaxed_predictions.append(document_relaxed_predictions)

    seqeval = evaluate.load("seqeval")
    strict_results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels,
        zero_division=0,
    )

    relaxed_results = relaxed_f1_from_bio_predictions(
        records=EVAL_RECORDS,
        predicted_label_sequences=relaxed_predictions,
    )

    return {
        "precision": strict_results["overall_precision"],
        "recall": strict_results["overall_recall"],
        "f1": strict_results["overall_f1"],
        "accuracy": strict_results["overall_accuracy"],
        "relaxed_precision": relaxed_results["precision"],
        "relaxed_recall": relaxed_results["recall"],
        "relaxed_f1": relaxed_results["f1"],
    }

def main():
    global EVAL_RECORDS

    print("Final model version:", MODEL_VERSION)
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    print("\nChecking PyTorch/GPU...")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading datasets...")
    train_dataset = load_dataset_from_jsonl(TRAIN_FILE)
    validation_dataset = load_dataset_from_jsonl(VALIDATION_FILE)
    test_dataset = load_dataset_from_jsonl(TEST_FILE)

    print("Train examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))
    print("Test examples:", len(test_dataset))

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        num_train_epochs=CONFIG["num_train_epochs"],
        weight_decay=CONFIG["weight_decay"],
        warmup_ratio=CONFIG["warmup_ratio"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model=CONFIG["selection_metric"],
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    EVAL_RECORDS = load_jsonl(VALIDATION_FILE)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting final training...")
    trainer.train()

    print("\nSaving best validation-selected model...")
    best_model_dir = OUTPUT_DIR / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    print("\nEvaluating on Webis test split...")
    EVAL_RECORDS = load_jsonl(TEST_FILE)
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print("\nWebis test results:")
    for key, value in test_results.items():
        print(f"  {key}: {value}")

    results_path = OUTPUT_DIR / "webis_test_results.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    print("\nSaved test results to:")
    print(results_path)

if __name__ == "__main__":
    main()

