import json
from pathlib import Path


DATA_DIR = Path("data/bio_modernbert_tokens")

SPLITS = ["train", "validation", "test"]

LABEL_MERGE_MAP = {
    "B-common-ground": "B-assumption",
    "I-common-ground": "I-assumption",
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records, path):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_labels(record):
    record = record.copy()

    record["ner_tags"] = [
        LABEL_MERGE_MAP.get(label, label)
        for label in record["ner_tags"]
    ]

    return record


def main():
    for split in SPLITS:
        input_path = DATA_DIR / f"webis_editorials_{split}_bio_modernbert.jsonl"
        output_path = DATA_DIR / f"webis_editorials_{split}_bio_modernbert_merged.jsonl"

        records = load_jsonl(input_path)
        merged_records = [merge_labels(record) for record in records]

        save_jsonl(merged_records, output_path)

        if remaining_common_ground != 0:
            raise ValueError(
                f"{split}: common-ground labels still found"
            )

        print(f"Saved {output_path} | remaining common-ground labels: {remaining_common_ground}")

    print("Done.")

if __name__ == "__main__":
    main()