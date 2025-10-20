import json
from glob import glob
import os

# Directory where summary files are stored
SUMMARY_DIR = "output"
# Pattern to match summary files (customize if needed)
SUMMARY_PATTERN = os.path.join(SUMMARY_DIR, "summaries_*.json")
# Output file for combined summaries
COMBINED_FILE = os.path.join(SUMMARY_DIR, "all_summaries.json")


def combine_summaries():
    summary_files = glob(SUMMARY_PATTERN)
    all_summaries = []
    for file in summary_files:
        with open(file, "r") as f:
            summaries = json.load(f)
            all_summaries.extend(summaries)
    with open(COMBINED_FILE, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Combined {len(summary_files)} files into {COMBINED_FILE} with {len(all_summaries)} summaries.")

if __name__ == "__main__":
    combine_summaries()
