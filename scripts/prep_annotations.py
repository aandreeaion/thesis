import json
import shutil
from pathlib import Path

# Folder exported from INCEpTION
PROJECT_DIR = Path("project-andreea_thesis-2026-06-17-202838")

ANNOTATION_DIR = PROJECT_DIR / "annotation"
SOURCE_DIR = PROJECT_DIR / "source"

# Clean output folder that we will create
OUTPUT_DIR = Path("45annotations_merged")
OUTPUT_ANNOTATION_DIR = OUTPUT_DIR / "annotation"
OUTPUT_SOURCE_DIR = OUTPUT_DIR / "source"
OUTPUT_JSONL_SPAN = Path("allsides_45_gold_merged.jsonl")

LABEL_MAP = {
    "Common Ground": "Assumption",
}

STRATEGY_TO_LABEL = {
    "Anecdote": "anecdote",
    "Testimony": "testimony",
    "Assumption": "assumption",
    "Statistics": "statistics",
    "Other": "other",
}

def selected_annotator_for_article(article_number):
    """
    Selects which annotator file to use for each article.

    Articles 1-5:
    - 1-2: Andreea
    - 3-5: Sara

    Articles 6-45:
    - even: Andreea
    - odd: Sara
    """

    if article_number in [1, 2]:
        return "Andreea.json"

    if article_number in [3, 4, 5]:
        return "nabhani.json"

    if article_number % 2 == 0:
        return "Andreea.json"

    return "nabhani.json"

def merge_common_ground_labels(obj):
    """
    Recursively search through the JSON object and change
    Common Ground labels to Assumption.

    Returns the number of labels changed.
    """

    changed_count = 0

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "strategy" and isinstance(value, str) and value in LABEL_MAP:
                obj[key] = LABEL_MAP[value]
                changed_count += 1
            else:
                changed_count += merge_common_ground_labels(value)

    elif isinstance(obj, list):
        for item in obj:
            changed_count += merge_common_ground_labels(item)

    return changed_count

def extract_argument_spans(obj):
    """
    Extract argumentative-unit spans from an INCEpTION CAS JSON object.

    Returns spans in the format:
    {"start": ..., "end": ..., "label": ...}
    """

    spans = []

    if isinstance(obj, dict):
        if obj.get("%TYPE") == "webanno.custom.ARG":
            strategy = obj.get("strategy")

            if isinstance(strategy, str) and strategy in STRATEGY_TO_LABEL:
                spans.append(
                    {
                        "start": obj["begin"],
                        "end": obj["end"],
                        "label": STRATEGY_TO_LABEL[strategy],
                    }
                )

        for value in obj.values():
            spans.extend(extract_argument_spans(value))

    elif isinstance(obj, list):
        for item in obj:
            spans.extend(extract_argument_spans(item))

    return spans

def extract_sofa_string(obj):
    """
    Extract the sofaString from an INCEpTION CAS JSON object.
    Annotation offsets are relative to this text.
    """

    if isinstance(obj, dict):
        if isinstance(obj.get("sofaString"), str):
            return obj["sofaString"]

        for value in obj.values():
            result = extract_sofa_string(value)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = extract_sofa_string(item)
            if result is not None:
                return result

    return None

def prepare_gold_annotations():
    """
    Create a clean 45-article gold annotation folder and one span-level JSONL file.

    For each article:
    - select one annotator JSON file
    - recode Common Ground to Assumption
    - save it as selected.json
    - copy the source text file
    - add one record to the span-level JSONL
    """

    OUTPUT_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    total_changed = 0
    jsonl_records = []

    for article_number in range(1, 46):
        article_name = f"article_{article_number:02d}.txt"
        annotator_file = selected_annotator_for_article(article_number)

        input_annotation_path = ANNOTATION_DIR / article_name / annotator_file
        input_source_path = SOURCE_DIR / article_name

        output_article_dir = OUTPUT_ANNOTATION_DIR / article_name
        output_annotation_path = output_article_dir / "selected.json"
        output_source_path = OUTPUT_SOURCE_DIR / article_name

        output_article_dir.mkdir(parents=True, exist_ok=True)

        with input_annotation_path.open("r", encoding="utf-8") as f:
            annotation_data = json.load(f)

        total_changed += merge_common_ground_labels(annotation_data)

        text = extract_sofa_string(annotation_data)
        spans = extract_argument_spans(annotation_data)

        jsonl_records.append(
            {
                "id": article_name.replace(".txt", ""),
                "text": text,
                "spans": spans,
            }
        )

        with output_annotation_path.open("w", encoding="utf-8") as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=2)

        shutil.copy2(input_source_path, output_source_path)

    with OUTPUT_JSONL_SPAN.open("w", encoding="utf-8") as f:
        for record in jsonl_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Common Ground labels changed: {total_changed}")

if __name__ == "__main__":
    prepare_gold_annotations()