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

MODEL_NAME = "answerdotai/ModernBERT-base"

DATA_DIR = Path("data/bio_modernbert_tokens")
OUTPUT_DIR = Path("outputs/modernbert_webis_token_classification")

TRAIN_FILE = DATA_DIR / "webis_editorials_train_bio_modernbert.jsonl"
VALIDATION_FILE = DATA_DIR / "webis_editorials_validation_bio_modernbert.jsonl"


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

# Note: maybe later on when code is complete I can move some repetitive functions to a helper.py script so I do not write them again and angain...
def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def convert_labels_to_ids(record):
    """
    Converting BIO string labels to numeric label IDs.
    Special tokens already have the label -100 and are kept as -100.
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
    Computing seqeval precision, recall, F1, and accuracy.
    Labels with ID -100 are ignored because they correspond to special tokens.
    """
    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label_sequence in zip(predictions, labels):
        document_predictions = []
        document_labels = []

        for predicted_label_id, gold_label_id in zip(prediction, label_sequence):
            if gold_label_id == -100:
                continue

            document_predictions.append(ID2LABEL[predicted_label_id])
            document_labels.append(ID2LABEL[gold_label_id])

        true_predictions.append(document_predictions)
        true_labels.append(document_labels)

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels,
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading datasets...")
    train_dataset = load_dataset_from_jsonl(TRAIN_FILE)
    validation_dataset = load_dataset_from_jsonl(VALIDATION_FILE)

    print("Train examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))

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
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving best model...")
    trainer.save_model(str(OUTPUT_DIR / "best_model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "best_model"))

    print("\nDone.")
    print("Best model saved to:", OUTPUT_DIR / "best_model")


if __name__ == "__main__":
    main()