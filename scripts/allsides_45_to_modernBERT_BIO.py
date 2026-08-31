
# Importing
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer

MODEL_NAME = "answerdotai/ModernBERT-base"

# Defining argumentation labels pre-BIO conversion
ARG_LABELS = {
    "anecdote",
    "testimony",
    "assumption",
    "statistics",
    "other",
}

# Defining a function that reads a `.jsonl` file and returns a list of records
def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records

# Saving data in `.jsonl` format
def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def trim_token_offset(text, start, end):
    """
    Trim whitespace from the beginning and end of a token offset.
    Some tokenizer offsets include the preceding space for tokens marked with Ġ. For BIO labeling, I compare the actual non-whitespace token span to the
    annotated span boundaries. This helps because annotations are based on exact character positionns. Therefore, BIO labels should be assigned to the actual token, not the extra whitespace.
    """
    while start < end and text[start].isspace():
        start += 1

    while end > start and text[end - 1].isspace():
        end -= 1

    return start, end


def assign_bio_labels_to_modernbert_tokens(text, offsets, spans):
    """
    Assign BIO labels to ModernBERT tokens using character offset mappings.

    Special tokens (like [CLS] and [SEP]) with offset (0, 0) receive -100.
    Normal tokens receive O, B-label, or I-label.
    """
    labels = []

    for token_start, token_end in offsets:
        if token_start == 0 and token_end == 0:
            labels.append(-100)
        else:
            labels.append("O")

    for span in spans:
        span_start = span["start"]
        span_end = span["end"]
        span_label = span["label"]

        if span_label not in ARG_LABELS:
            raise ValueError(f"Unexpected label: {span_label}")

        token_indices_in_span = []

        for i, (token_start, token_end) in enumerate(offsets):
            if labels[i] == -100:
                continue

            trimmed_start, trimmed_end = trim_token_offset(
                text,
                token_start,
                token_end,
            )

            if trimmed_start == trimmed_end:
                continue

            token_inside_span = (
                trimmed_start >= span_start
                and trimmed_end <= span_end
            )

            if token_inside_span:
                token_indices_in_span.append(i)

        if token_indices_in_span:
            first_index = token_indices_in_span[0]
            labels[first_index] = f"B-{span_label}"

            for index in token_indices_in_span[1:]:
                labels[index] = f"I-{span_label}"

    return labels

# Assigning numeric label IDs to BIO labels, while keeping special tokens as -100
def count_bio_starts(labels):
    counts = Counter()

    for label in labels:
        if isinstance(label, str) and label.startswith("B-"):
            counts[label[2:]] += 1

    return counts

# Converting one span-level document record to ModernBERT-tokenized BIO format
def convert_record_to_modernbert_bio(record, tokenizer):
    encoding = tokenizer(
        record["text"],
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False, 
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offsets = encoding["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    bio_labels = assign_bio_labels_to_modernbert_tokens(
        record["text"],
        offsets,
        record["spans"],
    )

    original_span_counts = Counter(span["label"] for span in record["spans"])
    b_label_counts = count_bio_starts(bio_labels)

    if original_span_counts != b_label_counts:
        print("This may happen if an annotated span does not fully contain any tokenizer token.")

    if len(input_ids) != len(bio_labels):
        raise ValueError(
            f"Token/label length mismatch for document {record['id']}: "
            f"{len(input_ids)} input_ids vs {len(bio_labels)} labels"
        )

    return {
        "id": record["id"],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokens": tokens,
        "offsets": offsets,
        "ner_tags": bio_labels,
    }

# Converting a full split to ModernBERT-tokenized BIO format
def convert_split_to_modernbert_bio(records, tokenizer):
    return [
        convert_record_to_modernbert_bio(record, tokenizer)
        for record in records
    ]

# Printing basic BIO label statistics for one split; not needed for training but maybe useful?
def summarize_bio_split(split_name, bio_records):
    label_counts = Counter()

    for record in bio_records:
        label_counts.update(record["ner_tags"])

    print(split_name)
    print("-" * len(split_name))
    print("Documents:", len(bio_records))
    print("Total ModernBERT tokens:", sum(len(record["tokens"]) for record in bio_records))
    print("Label counts:")

    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

    print()


def main():
    input_path = Path("data/bio_modernbert_tokens/allsides_45_gold_merged.jsonl")
    output_dir = Path("data/bio_modernbert_tokens")
    output_path = output_dir / "allsides_45_bio_modernbert_merged.jsonl"

    print("Loading AllSides 45 gold records...")
    records = load_jsonl(input_path)

    print("Records:", len(records))

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nConverting AllSides 45 articles to ModernBERT BIO...")
    bio_records = convert_split_to_modernbert_bio(records, tokenizer)

    print("\nSaving output file...")
    save_jsonl(bio_records, output_path)

    print("\nSaved file:")
    print(output_path)

    print("\nBIO statistics:")
    summarize_bio_split("AllSides 45 ModernBERT BIO", bio_records)


if __name__ == "__main__":
    main()