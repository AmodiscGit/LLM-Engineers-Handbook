#!/usr/bin/env python3
"""Run the full local workflow end-to-end:

- s3 ETL (fetch raw docs with source links)
- s3 summarization ETL (create per-document summaries)
- merge/ensure cleaned summaries exist
- create hybrid dataset (pair raw -> summary)
- produce cleaned training JSONL (source -> summary) ready for seq2seq finetuning

This script calls the repository helper scripts and performs light post-processing.
"""

import os
import subprocess
import glob
import json
import sys


def run_cmd(cmd, env=None):
    # if the command is run via 'poetry run', don't substitute python executable
    if not cmd.strip().startswith("poetry run"):
        # ensure we invoke the same python interpreter used to run this script
        if cmd.strip().startswith("python "):
            cmd = cmd.replace("python ", f"{sys.executable} ", 1)
        if " python " in cmd:
            cmd = cmd.replace(" python ", f" {sys.executable} ")
    print(f"\n>>> Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise SystemExit(f"Command failed: {cmd} (exit {res.returncode})")


def merge_summary_outputs(output_dir="output", merged_path="output/all_cleaned_summaries.json"):
    # If merged already exists, keep it
    if os.path.exists(merged_path):
        print(f"Found existing merged summaries at {merged_path}")
        return merged_path

    # Otherwise, merge any summaries_*.json files
    files = sorted(glob.glob(os.path.join(output_dir, "summaries_*.json")))
    if not files:
        print("No summaries_*.json files found to merge.")
        return None

    merged = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
                if isinstance(data, list):
                    # convert simple list of strings to summary objects
                    for i, s in enumerate(data):
                        if isinstance(s, str):
                            merged.append({"source_file": os.path.basename(f), "index": i, "summary": s})
                        elif isinstance(s, dict):
                            merged.append(s)
                elif isinstance(data, dict):
                    # try to find top-level list
                    for v in data.values():
                        if isinstance(v, list):
                            for i, s in enumerate(v):
                                if isinstance(s, str):
                                    merged.append({"source_file": os.path.basename(f), "index": i, "summary": s})
                                else:
                                    merged.append(s)
            except Exception as e:
                print(f"Failed to parse {f}: {e}")

    if merged:
        os.makedirs(os.path.dirname(merged_path), exist_ok=True)
        with open(merged_path, "w", encoding="utf-8") as out:
            json.dump(merged, out, ensure_ascii=False, indent=2)
        print(f"Wrote merged summaries to {merged_path} ({len(merged)} entries)")
        return merged_path
    else:
        print("No summaries merged.")
        return None


def produce_training_jsonl(hybrid_jsonl_path="data/artifacts/hybrid_summaries.jsonl", out_path="data/artifacts/training_data.jsonl", min_source_chars=200, min_summary_chars=20, min_score=0.05):
    if not os.path.exists(hybrid_jsonl_path):
        raise SystemExit(f"Hybrid dataset not found: {hybrid_jsonl_path}")

    kept = 0
    total = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(hybrid_jsonl_path, "r", encoding="utf-8") as inf, open(out_path, "w", encoding="utf-8") as outf:
        for line in inf:
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            source = rec.get("source")
            summary = rec.get("summary")
            score = float(rec.get("matched_score") or 0.0)
            # simple filters to remove license/boilerplate
            if not source or not summary:
                continue
            if len(source) < min_source_chars or len(summary) < min_summary_chars:
                continue
            if score < min_score:
                continue

            out_rec = {"source": source, "target": summary, "provenance": {"matched_raw_doc_id": rec.get("matched_raw_doc_id"), "matched_score": score}}
            outf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Produced training JSONL at {out_path}: kept {kept}/{total} records")
    return out_path, kept, total


def main():
    # 1) Run ETL to fetch raw documents
    run_cmd("poetry run python tools/run_s3_etl.py")

    # 2) Run summarization ETL to produce per-document summaries
    run_cmd("poetry run python tools/run_s3_summarization_etl.py")

    # 3) Ensure merged summaries exist (merge outputs if necessary)
    merged = merge_summary_outputs()
    if not merged:
        print("No merged summaries available; aborting.")
        sys.exit(1)

    # 4) Create hybrid dataset (pair raw -> summary)
    run_cmd("python tools/create_hybrid_dataset.py")

    # 5) Produce cleaned training JSONL
    training_path, kept, total = produce_training_jsonl()

    print("\nFull workflow complete.")
    print(f"Training file: {training_path} ({kept}/{total} kept)")


if __name__ == "__main__":
    main()
