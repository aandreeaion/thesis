"""
Compute relaxed span-level F1 for ModernBERT BIO token-classification outputs.

The model predicts BIO labels for ModernBERT tokens. This script converts
gold and predicted BIO labels back into character spans using token offsets,
then computes relaxed span-level precision, recall, and F1.

A predicted span receives partial credit when it overlaps with a gold span
of the same label. Matching is one-to-one: each predicted span can match at
most one gold span, and each gold span can be matched at most once.
"""

import json


def bio_to_spans(labels, offsets):
    """
    Convert BIO labels and token offsets into character spans.
    Unexpected I-labels are treated as new spans to make evaluation robust.
    """
    spans = []
    current_span = None

    for label, offset in zip(labels, offsets):
        start, end = offset

        if label == -100 or start == end:
            continue

        if label == "O":
            if current_span is not None:
                spans.append(current_span)
                current_span = None
            continue

        prefix, span_label = label.split("-", 1)

        if prefix == "B":
            if current_span is not None:
                spans.append(current_span)

            current_span = {
                "start": start,
                "end": end,
                "label": span_label,
            }

        elif prefix == "I":
            if current_span is not None and current_span["label"] == span_label:
                current_span["end"] = end
            else:
                current_span = {
                    "start": start,
                    "end": end,
                    "label": span_label,
                }

        else:
            raise ValueError(f"Unexpected BIO prefix: {prefix}")

    if current_span is not None:
        spans.append(current_span)

    return spans


def span_length(span):
    return span["end"] - span["start"]


def span_overlap_length(span_a, span_b):
    overlap_start = max(span_a["start"], span_b["start"])
    overlap_end = min(span_a["end"], span_b["end"])

    if overlap_end <= overlap_start:
        return 0

    return overlap_end - overlap_start


def score_all(precision_credit, precision_denominator, recall_credit, recall_denominator):
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    if precision_denominator > 0:
        precision = precision_credit / precision_denominator

    if recall_denominator > 0:
        recall = recall_credit / recall_denominator

    if precision_denominator == 0 and recall_denominator == 0:
        f1 = 1.0
    elif precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def relaxed_f1_for_spans(gold_spans, pred_spans):
    """
    Compute relaxed precision, recall, and F1 for one document.
    Uses one-to-one matching between predicted and gold spans.
    """
    precision_credit = 0.0
    recall_credit = 0.0
    matched_gold_indices = set()

    for pred_span in pred_spans:
        pred_len = span_length(pred_span)

        if pred_len == 0:
            continue

        best_gold_index = None
        best_overlap = 0
        best_match_score = 0.0

        for gold_index, gold_span in enumerate(gold_spans):
            if gold_index in matched_gold_indices:
                continue

            if pred_span["label"] != gold_span["label"]:
                continue

            gold_len = span_length(gold_span)

            if gold_len == 0:
                continue

            overlap = span_overlap_length(pred_span, gold_span)

            if overlap == 0:
                continue

            match_score = overlap / max(pred_len, gold_len)

            if match_score > best_match_score:
                best_match_score = match_score
                best_gold_index = gold_index
                best_overlap = overlap

        if best_gold_index is not None:
            matched_gold_indices.add(best_gold_index)

            matched_gold_span = gold_spans[best_gold_index]
            gold_len = span_length(matched_gold_span)

            precision_credit += best_overlap / pred_len
            recall_credit += best_overlap / gold_len

    return score_all(
        precision_credit=precision_credit,
        precision_denominator=len(pred_spans),
        recall_credit=recall_credit,
        recall_denominator=len(gold_spans),
    )


def relaxed_f1_for_dataset(gold_spans_by_doc, pred_spans_by_doc):
    """
    Compute relaxed precision, recall, and F1 over a full dataset.
    """
    total_precision_credit = 0.0
    total_recall_credit = 0.0

    total_predicted_spans = 0
    total_gold_spans = 0

    for doc_id, gold_spans in gold_spans_by_doc.items():
        pred_spans = pred_spans_by_doc.get(doc_id, [])

        precision, recall, _ = relaxed_f1_for_spans(
            gold_spans=gold_spans,
            pred_spans=pred_spans,
        )

        total_precision_credit += precision * len(pred_spans)
        total_recall_credit += recall * len(gold_spans)

        total_predicted_spans += len(pred_spans)
        total_gold_spans += len(gold_spans)

    precision, recall, f1 = score_all(
        precision_credit=total_precision_credit,
        precision_denominator=total_predicted_spans,
        recall_credit=total_recall_credit,
        recall_denominator=total_gold_spans,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_spans": total_gold_spans,
        "predicted_spans": total_predicted_spans,
    }


def gold_spans_from_bio_records(records):
    """
    Convert gold BIO labels from JSONL records into spans by document.
    """
    spans_by_doc = {}

    for record in records:
        spans_by_doc[record["id"]] = bio_to_spans(
            labels=record["ner_tags"],
            offsets=record["offsets"],
        )

    return spans_by_doc


def spans_from_bio_predictions(records, predicted_label_sequences):
    """
    Convert predicted BIO label sequences into spans by document.
    """
    spans_by_doc = {}

    for record, predicted_labels in zip(records, predicted_label_sequences):
        spans_by_doc[record["id"]] = bio_to_spans(
            labels=predicted_labels,
            offsets=record["offsets"],
        )

    return spans_by_doc


def relaxed_f1_from_bio_predictions(records, predicted_label_sequences):
    """
    Compute relaxed F1 from BIO ModernBERT records and predicted BIO labels.
    """
    gold_spans_by_doc = gold_spans_from_bio_records(records)
    pred_spans_by_doc = spans_from_bio_predictions(
        records=records,
        predicted_label_sequences=predicted_label_sequences,
    )

    return relaxed_f1_for_dataset(
        gold_spans_by_doc=gold_spans_by_doc,
        pred_spans_by_doc=pred_spans_by_doc,
    )


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def count_unexpected_i_labels(records, max_examples=10):
    """
    Diagnostic helper: count I-labels that appear without a matching B-label.
    """
    total_unexpected = 0
    examples = []

    for record in records:
        current_label = None

        for token_index, (label, offset) in enumerate(
            zip(record["ner_tags"], record["offsets"])
        ):
            start, end = offset

            if label == -100 or start == end:
                continue

            if label == "O":
                current_label = None
                continue

            prefix, span_label = label.split("-", 1)

            if prefix == "B":
                current_label = span_label

            elif prefix == "I":
                if current_label == span_label:
                    continue

                total_unexpected += 1

                if len(examples) < max_examples:
                    examples.append({
                        "doc_id": record["id"],
                        "token_index": token_index,
                        "label": label,
                        "offset": offset,
                    })

                current_label = span_label

    return total_unexpected, examples


if __name__ == "__main__":
    validation_file = (
        "data/bio_modernbert_tokens/"
        "webis_editorials_validation_bio_modernbert.jsonl"
    )

    records = load_jsonl(validation_file)

    unexpected_count, examples = count_unexpected_i_labels(records)
    print("Unexpected I-labels:", unexpected_count)

    for example in examples:
        print(example)

    fake_predictions = [record["ner_tags"] for record in records]

    results = relaxed_f1_from_bio_predictions(
        records=records,
        predicted_label_sequences=fake_predictions,
    )

    print()
    print("Relaxed precision:", results["precision"])
    print("Relaxed recall:", results["recall"])
    print("Relaxed F1:", results["f1"])
    print("Gold spans:", results["gold_spans"])
    print("Predicted spans:", results["predicted_spans"])