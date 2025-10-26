# Workflow helper: merge_and_clean_summaries.py
# Required files / artifacts:
# - output/*.json (input files with summaries, e.g., output/summaries_{bucket}.json)
# - writes: output/all_cleaned_summaries.json

"""
Moved from tools/merge_and_clean_summaries.py into tools/workflow.
Provides main(input_dir, output_file) which merges summary JSONs into a cleaned list.
"""
import os
import json
from glob import glob
from zenml import step


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def clean_summary(summary):
    # Example cleaning: strip whitespace, remove empty entries
    return summary.strip() if isinstance(summary, str) and summary.strip() else None


def main(input_dir="output", output_file="output/all_cleaned_summaries.json"):
    files = glob(os.path.join(input_dir, "*.json"))
    all_summaries = []
    for file in files:
        data = load_json(file)
        # If data is a list, clean each summary and add metadata
        if isinstance(data, list):
            for idx, s in enumerate(data):
                cleaned = clean_summary(s)
                if cleaned:
                    all_summaries.append({
                        "source_file": os.path.basename(file),
                        "index": idx,
                        "summary": cleaned
                    })
        # If data is a dict, check for 'artifact_data' key
        elif isinstance(data, dict):
            if "artifact_data" in data and isinstance(data["artifact_data"], list):
                for idx, s in enumerate(data["artifact_data"]):
                    cleaned = clean_summary(s)
                    if cleaned:
                        all_summaries.append({
                            "source_file": os.path.basename(file),
                            "index": idx,
                            "summary": cleaned
                        })
            else:
                for key, v in data.items():
                    cleaned = clean_summary(v)
                    if cleaned:
                        all_summaries.append({
                            "source_file": os.path.basename(file),
                            "key": key,
                            "summary": cleaned
                        })
    with open(output_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Merged and cleaned summaries with metadata written to {output_file}")
    # Return explicit artifact path + metadata for ZenML tracing
    try:
        count = len(all_summaries)
    except Exception:
        count = 0
    return output_file, {"count": count}


if __name__ == "__main__":
    main()


@step
def merge_and_clean_summaries_step(input_dir: str = "output", output_file: str = "output/all_cleaned_summaries.json"):
    """ZenML step wrapper for the merge-and-clean helper.

    Calls the existing `main` and returns (output_file, metadata).
    """
    main(input_dir=input_dir, output_file=output_file)
    # Load the output to compute count
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = len(data) if isinstance(data, list) else 1
    except Exception:
        count = 0
    return output_file, {"count": count}
