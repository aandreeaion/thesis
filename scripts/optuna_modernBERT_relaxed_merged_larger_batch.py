from pathlib import Path
import json
import shutil
import numpy as np
import torch
import optuna
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

MODEL_NAME = "answerdotai/ModernBERT-base"
DATA_DIR = Path("data/bio_modernbert_tokens")
OUTPUT_DIR = Path("outputs/modernbert_webis_token_classification_optuna_relaxed_larger_batch_merged")
TRAIN_FILE = DATA_DIR / "webis_editorials_train_bio_modernbert_merged.jsonl"
VALIDATION_FILE = DATA_DIR / "webis_editorials_validation_bio_modernbert_merged.jsonl"

N_TRIALS = 30
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
    Converting BIO string labels to numeric label IDs
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
    Compute both strict seqeval metrics and relaxed span-level F1.

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

            # Filtered version for seqeval
            document_predictions.append(predicted_label)
            document_labels.append(gold_label)

            # Full aligned version for relaxed F1
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
        records=VALIDATION_RECORDS,
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

def objective(trial):
    """
    Running one Optuna trial. For each trial, Optuna chooses a set of hyperparameters, 
    the model is fine-tuned with those values, and the validation F1 is returned so Optuna can compare 
    it with the other trials
    """

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        5e-5,
        log=True,
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        0.0,
        0.1,
    )

    num_train_epochs = trial.suggest_int(
        "num_train_epochs",
        2,
        5,
    )

    warmup_ratio = trial.suggest_float(
        "warmup_ratio",
        0.0,
        0.2,
    )

    per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size",
        [4, 8],
    )

    gradient_accumulation_steps = trial.suggest_categorical(
        "gradient_accumulation_steps",
        [1, 2, 4],
    )

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps

    lr_scheduler_type = trial.suggest_categorical(
        "lr_scheduler_type",
        ["linear", "cosine", "cosine_with_restarts"],
    )

    trial_output_dir = OUTPUT_DIR / f"trial_{trial.number}"

    print("\n" + "_" * 60)
    print(f"Starting trial {trial.number}")
    print("Hyperparameters:")
    print(f"  learning_rate: {learning_rate}")
    print(f"  weight_decay: {weight_decay}")
    print(f"  num_train_epochs: {num_train_epochs}")
    print(f"  warmup_ratio: {warmup_ratio}")
    print(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"  per_device_train_batch_size: {per_device_train_batch_size}")
    print(f"  effective_batch_size: {effective_batch_size}")
    print(f"  lr_scheduler_type: {lr_scheduler_type}")
    print("_" * 60)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(trial_output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="relaxed_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        lr_scheduler_type=lr_scheduler_type,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TRAIN_DATASET,
        eval_dataset=VALIDATION_DATASET,
        processing_class=TOKENIZER,
        data_collator=DATA_COLLATOR,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    validation_f1 = eval_results["eval_relaxed_f1"]

    print(f"Trial {trial.number} relaxed validation F1: {validation_f1}")

    # Storing useful values inside the Optuna trial
    trial.set_user_attr("eval_precision", eval_results["eval_precision"])
    trial.set_user_attr("eval_recall", eval_results["eval_recall"])
    trial.set_user_attr("eval_f1", eval_results["eval_f1"])
    trial.set_user_attr("eval_accuracy", eval_results["eval_accuracy"])

    trial.set_user_attr(
        "eval_relaxed_precision",
        eval_results["eval_relaxed_precision"],
    )
    trial.set_user_attr(
        "eval_relaxed_recall",
        eval_results["eval_relaxed_recall"],
    )
    trial.set_user_attr(
        "eval_relaxed_f1",
        eval_results["eval_relaxed_f1"],
    )

    trial.set_user_attr("eval_loss", eval_results["eval_loss"])
    trial.set_user_attr("effective_batch_size", effective_batch_size)

    # Deleting trial checkpoints after evaluation
    # Optuna keeps the score and hyperparameters in memory/study results
    if trial_output_dir.exists():
        shutil.rmtree(trial_output_dir)

    # Cleaning GPU memory before the next trial
    del model
    del trainer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return validation_f1

def save_best_results(study):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_trial = study.best_trial

    results = {
        "best_trial_number": best_trial.number,
        "best_validation_relaxed_f1": best_trial.value,
        "best_params": best_trial.params,
        "best_user_attrs": best_trial.user_attrs,
        "n_trials": len(study.trials),
    }

    output_path = OUTPUT_DIR / "best_optuna_results.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nBest Optuna results saved to:")
    print(output_path)

    print("\nBest trial:")
    print("  Trial number:", best_trial.number)
    print("  Validation relaxed F1:", best_trial.value)
    print("  Parameters:")

    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print("  Other validation metrics:")

    for key, value in best_trial.user_attrs.items():
        print(f"    {key}: {value}")

def main():
    global TOKENIZER
    global DATA_COLLATOR
    global TRAIN_DATASET
    global VALIDATION_DATASET
    global VALIDATION_RECORDS

    print("Checking PyTorch/GPU...")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading datasets...")
    TRAIN_DATASET = load_dataset_from_jsonl(TRAIN_FILE)
    VALIDATION_DATASET = load_dataset_from_jsonl(VALIDATION_FILE)
    VALIDATION_RECORDS = load_jsonl(VALIDATION_FILE)
    print("Train examples:", len(TRAIN_DATASET))
    print("Validation examples:", len(VALIDATION_DATASET))

    print("\nLoading tokenizer...")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

    DATA_COLLATOR = DataCollatorForTokenClassification(
        tokenizer=TOKENIZER,
        padding=True,
    )

    print("\nStarting Optuna hyperparameter tuning...")
    print("Number of trials:", N_TRIALS)

    study = optuna.create_study(
        direction="maximize",
        study_name="modernbert_webis_token_classification_relaxed_larger_batch_merged",
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
    )

    save_best_results(study)
    print("\nDone.")

if __name__ == "__main__":
    main()