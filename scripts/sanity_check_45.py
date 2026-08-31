import json
from pathlib import Path
from collections import Counter

SPAN_FILE = Path("data/bio_modernbert_tokens/allsides_45_gold_merged.jsonl")
BIO_FILE = Path("data/bio_modernbert_tokens/allsides_45_bio_modernbert_merged.jsonl")

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

span_records = load_jsonl(SPAN_FILE)
bio_records = load_jsonl(BIO_FILE)

print("Span records:", len(span_records))
print("BIO records:", len(bio_records))

invalid_spans = []
span_counts = Counter()
bio_b_counts = Counter()

for record in span_records:
    text = record["text"]

    for span in record["spans"]:
        start = span["start"]
        end = span["end"]
        label = span["label"]

        span_counts[label] += 1

        if not (0 <= start < end <= len(text)):
            invalid_spans.append((record["id"], start, end, label))

print("\nInvalid spans:", len(invalid_spans))

if invalid_spans:
    for item in invalid_spans[:10]:
        print(item)

for record in bio_records:
    for label in record["ner_tags"]:
        if isinstance(label, str) and label.startswith("B-"):
            bio_b_counts[label[2:]] += 1

print("\nGold span counts:")
for label, count in span_counts.most_common():
    print(f"  {label}: {count}")

print("\nBIO B-label counts:")
for label, count in bio_b_counts.most_common():
    print(f"  {label}: {count}")
