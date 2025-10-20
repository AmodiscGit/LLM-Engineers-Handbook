import os
import json
from glob import glob

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

if __name__ == "__main__":
    main()
