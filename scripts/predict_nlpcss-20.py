"""
Apply the final merged-label ModernBERT evidence model to an unannotated corpus.

This script loads the cleaned NLPCSS-20 corpus and applies the final merged-label
ModernBERT token-classification model. It also loads the temperature-calibration
results from the Webis validation split and uses the learned global temperature
to produce calibrated probabilities.

The script saves predicted evidence spans with:
- article metadata;
- character-level span boundaries;
- predicted label;
- span text;
- calibrated confidence;
- calibrated per-label probabilities.

By default, this script runs on only the first few articles as a test. Set
MAX_ARTICLES_TO_PROCESS = None before running the full corpus.

Input:
data/inputs/nlpcss20_clean.jsonl
outputs/modernbert_webis_final_relaxed_selected_merged/temp_calibration_threshold_f1_validation_results.json

Output:
nlpcss20_model_predictions_first5_raw.jsonl
nlpcss20_model_predictions_first5_raw.csv
nlpcss20_model_predictions_first5_filtered.jsonl
nlpcss20_model_predictions_first5_filtered.csv

For the full run:
results/model_predictions/nlpcss20_model_predictions_raw.jsonl
results/model_predictions/nlpcss20_model_predictions_raw.csv
results/model_predictions/nlpcss20_model_predictions_filtered.jsonl
results/model_predictions/nlpcss20_model_predictions_filtered.csv
"""

from pathlib import Path
import json
import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "modernbert_webis_final_relaxed_selected_merged"
    / "best_model"
)

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "modernbert_webis_final_relaxed_selected_merged"
    / "temp_calibration_threshold_f1_validation_results.json"
)

INPUT_PATH = PROJECT_ROOT / "data" / "inputs" / "nlpcss20_clean.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "results" / "model_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 8192

# Keeping this to 5 for testing purposes and 
# setting it to None only when running the full corpus
MAX_ARTICLES_TO_PROCESS = None

USE_CONFIDENCE_THRESHOLD = False

BASE_LABELS = [
    "anecdote",
    "testimony",
    "assumption",
    "statistics",
    "other",
]


def clean_for_csv(text):
    """
    Make text easier to inspect in CSV.
    """
    return " ".join(str(text).split())


def get_output_paths():

    if MAX_ARTICLES_TO_PROCESS is None:
        base_name = "nlpcss20_model_predictions"
    else:
        base_name = f"nlpcss20_model_predictions_first{MAX_ARTICLES_TO_PROCESS}"

    paths = {
        "raw_jsonl": OUTPUT_DIR / f"{base_name}_raw.jsonl",
        "raw_csv": OUTPUT_DIR / f"{base_name}_raw.csv",
        "filtered_jsonl": OUTPUT_DIR / f"{base_name}_filtered.jsonl",
        "filtered_csv": OUTPUT_DIR / f"{base_name}_filtered.csv",
    }

    return paths

def load_calibration_results(path):
    """
    Load the learned temperature from the calibration results.
    """
    with path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    temperature = float(results["temperature"])
    best_threshold = results.get("best_threshold_selected_on_validation_relaxed_f1")

    return temperature, best_threshold


def base_label_probabilities(probabilities, label_to_id):
    """
    Collapse BIO probabilities into base-label probabilities.

    Example:
        prob_assumption = P(B-assumption) + P(I-assumption)
    """
    collapsed = {}

    for label in BASE_LABELS:
        b_label = f"B-{label}"
        i_label = f"I-{label}"

        probability = 0.0

        if b_label in label_to_id:
            probability += float(probabilities[label_to_id[b_label]])

        if i_label in label_to_id:
            probability += float(probabilities[label_to_id[i_label]])

        collapsed[label] = probability

    return collapsed


def word_level_predictions(
    predicted_ids,
    probabilities,
    offsets,
    word_ids,
    id_to_label,
    label_to_id,
):
    """
    Convert token-level predictions to word-level predictions.

    The model was trained with labels only on the first subword of each word.
    Therefore, during model application, only the first subword prediction is
    used as the word-level prediction. The character span of the full word is
    reconstructed from all subword offsets.
    """
    word_to_token_indices = {}
    word_order = []

    for token_index, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        start, end = offsets[token_index]

        if start == end:
            continue

        if word_id not in word_to_token_indices:
            word_to_token_indices[word_id] = []
            word_order.append(word_id)

        word_to_token_indices[word_id].append(token_index)

    word_items = []

    for word_id in word_order:
        token_indices = word_to_token_indices[word_id]
        first_token_index = token_indices[0]

        word_start = min(int(offsets[index][0]) for index in token_indices)
        word_end = max(int(offsets[index][1]) for index in token_indices)

        predicted_id = int(predicted_ids[first_token_index])
        predicted_label = id_to_label[predicted_id]

        token_probabilities = probabilities[first_token_index]
        collapsed_probabilities = base_label_probabilities(
            probabilities=token_probabilities,
            label_to_id=label_to_id,
        )

        word_items.append(
            {
                "start": word_start,
                "end": word_end,
                "label": predicted_label,
                "base_probabilities": collapsed_probabilities,
            }
        )

    return word_items


def close_current_span(spans, current_span, text):
    """
    Finalize the current span and add text/probability summaries.
    """
    if current_span is None:
        return

    span_label = current_span["label"]
    token_probabilities = current_span.pop("token_probabilities")

    for label in BASE_LABELS:
        values = [probs[label] for probs in token_probabilities]
        current_span[f"prob_{label}"] = sum(values) / len(values)

    current_span["confidence"] = current_span[f"prob_{span_label}"]
    current_span["text"] = clean_for_csv(text[current_span["start"]:current_span["end"]])

    spans.append(current_span)


def bio_words_to_spans(word_items, text):
    """
    Convert word-level BIO labels into character-level spans.
    """
    spans = []
    current_span = None

    for item in word_items:
        label = item["label"]

        if label == "O":
            close_current_span(spans, current_span, text)
            current_span = None
            continue

        if "-" not in label:
            close_current_span(spans, current_span, text)
            current_span = None
            continue

        prefix, base_label = label.split("-", 1)

        if base_label not in BASE_LABELS:
            close_current_span(spans, current_span, text)
            current_span = None
            continue

        starts_new_span = (
            prefix == "B"
            or current_span is None
            or current_span["label"] != base_label
        )

        if starts_new_span:
            close_current_span(spans, current_span, text)

            current_span = {
                "start": item["start"],
                "end": item["end"],
                "label": base_label,
                "token_probabilities": [item["base_probabilities"]],
            }

        elif prefix == "I" and current_span["label"] == base_label:
            current_span["end"] = item["end"]
            current_span["token_probabilities"].append(item["base_probabilities"])

    close_current_span(spans, current_span, text)

    return spans


def predict_article_spans(
    text,
    tokenizer,
    model,
    device,
    id_to_label,
    label_to_id,
    temperature,
    threshold,
):
    """
    Predict evidence spans for one article.
    """
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    article_spans = []

    for chunk_index in range(len(encoded["input_ids"])):
        model_inputs = {
            "input_ids": torch.tensor(
                [encoded["input_ids"][chunk_index]],
                device=device,
            ),
            "attention_mask": torch.tensor(
                [encoded["attention_mask"][chunk_index]],
                device=device,
            ),
        }

        with torch.no_grad():
            outputs = model(**model_inputs)

        scaled_logits = outputs.logits.squeeze(0) / temperature
        probabilities = torch.softmax(scaled_logits, dim=-1)
        confidences, predicted_ids = torch.max(probabilities, dim=-1)

        if USE_CONFIDENCE_THRESHOLD and threshold is not None:
            o_label_id = label_to_id["O"]
            low_confidence_non_o = (
                (predicted_ids != o_label_id)
                & (confidences < float(threshold))
            )
            predicted_ids = predicted_ids.clone()
            predicted_ids[low_confidence_non_o] = o_label_id

        word_ids = encoded.word_ids(batch_index=chunk_index)

        word_items = word_level_predictions(
            predicted_ids=predicted_ids.detach().cpu().tolist(),
            probabilities=probabilities.detach().cpu(),
            offsets=encoded["offset_mapping"][chunk_index],
            word_ids=word_ids,
            id_to_label=id_to_label,
            label_to_id=label_to_id,
        )

        chunk_spans = bio_words_to_spans(
            word_items=word_items,
            text=text,
        )

        article_spans.extend(chunk_spans)

    return article_spans


def get_metadata_value(article, column):
    """
    Safely get metadata from an article row.
    """
    if column not in article.index:
        return ""

    value = article[column]

    if value is None:
        return ""

    if isinstance(value, (list, dict)):
        return value

    if pd.isna(value):
        return ""

    return value

def get_misc_value(article, key):
    """
    Safely get metadata stored inside the nested misc field.
    """
    misc = get_metadata_value(article, "misc")

    if not isinstance(misc, dict):
        
        return ""

    value = misc.get(key, "")

    if value is None:
        return ""

    if value == "None":
        return ""

    if isinstance(value, list):
        return "; ".join(str(item) for item in value if str(item).strip())

    return value

def build_prediction_rows(df, tokenizer, model, device, id_to_label, label_to_id, temperature, threshold):
    """
    Apply the model to the selected articles and build one row per predicted span.
    """
    rows = []

    if MAX_ARTICLES_TO_PROCESS is None:
        articles_to_process = df
    else:
        articles_to_process = df.head(MAX_ARTICLES_TO_PROCESS)

    for article_position, (_, article) in enumerate(articles_to_process.iterrows(), start=1):
        text = article["content"]

        spans = predict_article_spans(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            id_to_label=id_to_label,
            label_to_id=label_to_id,
            temperature=temperature,
            threshold=threshold,
        )

        for span_id, span in enumerate(spans):
            row = {
                "article_position": article_position,
                "original_index": get_metadata_value(article, "original_index"),
                "event_id": get_metadata_value(article, "event_id"),
                "source": get_metadata_value(article, "source"),
                "title": get_metadata_value(article, "title"),
                "adfontes_fair": get_metadata_value(article, "adfontes_fair"),
                "adfontes_political": get_metadata_value(article, "adfontes_political"),
                "allsides_bias": get_metadata_value(article, "allsides_bias"),
                "time": get_misc_value(article, "time"),
                "topics": get_misc_value(article, "topics"),
                "author": get_misc_value(article, "author"),
                "word_count": get_metadata_value(article, "word_count"),
                "span_id": span_id,
                "span_start": span["start"],
                "span_end": span["end"],
                "label": span["label"],
                "confidence": span["confidence"],
                "prob_anecdote": span["prob_anecdote"],
                "prob_testimony": span["prob_testimony"],
                "prob_assumption": span["prob_assumption"],
                "prob_statistics": span["prob_statistics"],
                "prob_other": span["prob_other"],
                "text": span["text"],
            }

            rows.append(row)

        print(
            f"Processed article {article_position}/{len(articles_to_process)} "
            f"| predicted spans: {len(spans)}"
        )

    return rows

def has_alphanumeric_content(text):
    """
    Check whether a predicted span contains at least one letter or number.

    Punctuation-only spans cannot satisfy the ADU definition and are removed
    from the filtered analysis output.
    """
    return any(char.isalnum() for char in str(text))


def filter_prediction_rows(rows):
    """
    Remove only punctuation-only or formatting-only predicted spans.

    This is deliberately conservative: short spans are kept when they contain
    letters or numbers, because they may reflect model boundary/unitization
    errors that should remain visible.
    """
    return [
        row for row in rows
        if has_alphanumeric_content(row["text"])
    ]

def save_prediction_rows(rows, output_jsonl, output_csv):
    """
    Save prediction rows to JSONL and CSV.
    """
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main():
    output_paths = get_output_paths()

    print("Applying final merged-label ModernBERT model to unannotated corpus")
    print("Model path:", MODEL_PATH)
    print("Input path:", INPUT_PATH)
    print("Calibration path:", CALIBRATION_PATH)
    print("Raw output JSONL:", output_paths["raw_jsonl"])
    print("Raw output CSV:", output_paths["raw_csv"])
    print("Filtered output JSONL:", output_paths["filtered_jsonl"])
    print("Filtered output CSV:", output_paths["filtered_csv"])
    print("Max articles to process:", MAX_ARTICLES_TO_PROCESS)
    print("Use confidence threshold:", USE_CONFIDENCE_THRESHOLD)

    temperature, best_threshold = load_calibration_results(CALIBRATION_PATH)

    print("\nLoaded calibration results:")
    print("Temperature:", temperature)
    print("Best validation threshold:", best_threshold)

    df = pd.read_json(INPUT_PATH, lines=True)

    print("\nLoaded articles:", len(df))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

    id_to_label = {int(key): value for key, value in model.config.id2label.items()}
    label_to_id = {value: key for key, value in id_to_label.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    print("Using device:", device)
    print("Model labels:", id_to_label)

    rows = build_prediction_rows(
        df=df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        temperature=temperature,
        threshold=best_threshold,
    )

    filtered_rows = filter_prediction_rows(rows)

    save_prediction_rows(
        rows=rows,
        output_jsonl=output_paths["raw_jsonl"],
        output_csv=output_paths["raw_csv"],
    )

    save_prediction_rows(
        rows=filtered_rows,
        output_jsonl=output_paths["filtered_jsonl"],
        output_csv=output_paths["filtered_csv"],
    )

    print("\nSaved raw prediction rows:", len(rows))
    print("Saved filtered prediction rows:", len(filtered_rows))
    print("Removed punctuation-only rows:", len(rows) - len(filtered_rows))

    print("Saved raw JSONL:", output_paths["raw_jsonl"])
    print("Saved raw CSV:", output_paths["raw_csv"])
    print("Saved filtered JSONL:", output_paths["filtered_jsonl"])
    print("Saved filtered CSV:", output_paths["filtered_csv"])


if __name__ == "__main__":
    main()