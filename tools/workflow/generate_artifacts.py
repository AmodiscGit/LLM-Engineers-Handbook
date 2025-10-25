#!/usr/bin/env python3
"""Programmatic generator for required artifacts.

This script runs the minimal set of steps (programmatically, not via subprocess)
that produce the files the UI and downstream tools expect:

- output/all_cleaned_summaries.json
- data/artifacts/raw_documents.json
- data/artifacts/cleaned_documents.json
- data/artifacts/hybrid_summaries.jsonl
- data/artifacts/hybrid_summaries_report.json

It prefers to call importable helpers where available to avoid shell and CWD issues.

Usage:
  poetry run python tools/workflow/generate_artifacts.py

Flags:
  --skip-etl        Skip raw document ETL
  --skip-summarize  Skip summarization (assumes summaries already exist under output/)
  --skip-merge      Skip merge of summaries
  --skip-cleaned    Skip cleaned_documents generation
  --skip-hybrid     Skip hybrid generation

"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
import sys

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)


def write_json_safe(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path}")


def run_etl_generate_raw(out_path: Path = Path("data/artifacts/raw_documents.json")):
    try:
        from tools.workflow.run_s3_etl import generate_raw_documents
    except Exception as e:
        print("tools.run_s3_etl.generate_raw_documents not importable:", e)
        return False

    print("Running generate_raw_documents()...")
    docs = generate_raw_documents(out_dir=str(out_path.parent))
    if isinstance(docs, list):
        try:
            write_json_safe(out_path, docs)
        except Exception:
            pass
        return True
    elif isinstance(docs, dict):
        try:
            write_json_safe(out_path, docs)
        except Exception:
            pass
        return True
    else:
        print("generate_raw_documents() returned no documents")
        return False


def run_summarization_and_export(configs_path: str = "configs/s3_etl.yaml", output_dir: Path = Path("output")):
    # load general env (contains OPENAI key) and S3-specific env
    load_dotenv(".env")
    load_dotenv(".env.s3")
    with open(configs_path, "r") as f:
        cfg = yaml.safe_load(f)
    bucket_name = cfg.get("bucket_name", "")
    prefix = cfg.get("prefix", "")

    # First try to run ZenML pipeline (may return run/metadata rather than a raw list)
    summaries = None
    try:
        from pipelines.s3_summarization_etl_pipeline import s3_summarization_etl_pipeline

        print(f"Running summarization pipeline for bucket={bucket_name} prefix={prefix} ...")
        summaries = s3_summarization_etl_pipeline(bucket_name=bucket_name, aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), prefix=prefix)
        if not isinstance(summaries, list):
            print("Warning: pipeline did not return a list; will attempt a direct summarization fallback")
            summaries = None
    except Exception as e:
        print("Could not import or run summarization pipeline:", e)
        summaries = None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"summaries_{bucket_name}.json"

    # If the pipeline didn't yield a list, fall back to summarizing raw documents directly using OpenAI
    if not isinstance(summaries, list):
        print("Falling back to direct summarization from raw_documents.json using OpenAI API (env OPENAI_API_KEY)")

        # try to reuse the project's summarization helper (preferred)
        try:
            from steps.summarization.summarize_documents import summarize_documents as summarize_helper

            print("Using steps.summarization.summarize_documents() for direct summarization fallback")
            raw_path = Path("data/artifacts/raw_documents.json")
            docs = []
            if raw_path.exists():
                try:
                    with open(raw_path, "r", encoding="utf-8") as rf:
                        data = json.load(rf)
                        if isinstance(data, dict) and "artifact_data" in data and isinstance(data["artifact_data"], list):
                            docs = [d.get("content") or d.get("text") or "" for d in data["artifact_data"]]
                        elif isinstance(data, list):
                            tmp = []
                            for el in data:
                                if isinstance(el, dict):
                                    tmp.append(el.get("content") or el.get("text") or "")
                                elif isinstance(el, str):
                                    tmp.append(el)
                            docs = tmp
                except Exception as e:
                    print("Failed to load raw documents for fallback summarization:", e)

            if not docs:
                print("No raw documents available for summarization fallback")
                summaries = []
            else:
                try:
                    # call the helper (it is a zenml-decorated step but callable)
                    summaries = summarize_helper(docs)
                except Exception as e:
                    print("summarize_documents() helper failed, falling back to simple OpenAI loop:", e)
                    summaries = None
        except Exception as e:
            print("Could not import steps.summarization.summarize_documents:", e)
            summaries = None

        # if helper unavailable or failed, use the simpler OpenAI loop
        if summaries is None:
            raw_path = Path("data/artifacts/raw_documents.json")
            docs = []
            if raw_path.exists():
                try:
                    with open(raw_path, "r", encoding="utf-8") as rf:
                        data = json.load(rf)
                        if isinstance(data, dict) and "artifact_data" in data and isinstance(data["artifact_data"], list):
                            docs = [d.get("content") or d.get("text") or "" for d in data["artifact_data"]]
                        elif isinstance(data, list):
                            tmp = []
                            for el in data:
                                if isinstance(el, dict):
                                    tmp.append(el.get("content") or el.get("text") or "")
                                elif isinstance(el, str):
                                    tmp.append(el)
                            docs = tmp
                except Exception as e:
                    print("Failed to load raw documents for fallback summarization:", e)

            if not docs:
                print("No raw documents available for summarization fallback")
                summaries = []
            else:
                try:
                    import openai

                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        print("OPENAI_API_KEY not set; cannot run direct summarization")
                        summaries = []
                    else:
                        openai.api_key = api_key
                        model_id = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
                        fallback_model = os.getenv("SUMMARIZE_FALLBACK_MODEL", "gpt-3.5-turbo")
                        max_tokens = int(os.getenv("SUMMARIZE_MAX_TOKENS", "400"))
                        temperature = float(os.getenv("SUMMARIZE_TEMPERATURE", "0.2"))
                        summaries = []
                        for doc in docs:
                            prompt = f"Summarize the following document in 3 concise bullets and a provenance line:\n\n{(doc or '')[:4000]}"
                            try:
                                resp = openai.chat.completions.create(
                                    model=model_id,
                                    messages=[{"role": "user", "content": prompt}],
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                )
                            except Exception as e:
                                # try fallback model
                                try:
                                    resp = openai.chat.completions.create(
                                        model=fallback_model,
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                    )
                                except Exception:
                                    print("OpenAI summarization failed for a document; appending empty summary")
                                    summaries.append("")
                                    continue
                            try:
                                summary = resp.choices[0].message.content.strip()
                            except Exception:
                                summary = ""
                            summaries.append(summary)
                except Exception as e:
                    print("Direct summarization fallback failed:", e)
                    summaries = []

    write_json_safe(out_path, summaries or [])
    return out_path


def merge_summaries_into_all(output_dir: Path = Path("output"), merged_path: Path = Path("output/all_cleaned_summaries.json")):
    # prefer calling the helper
    try:
        from tools.workflow import merge_and_clean_summaries
        # module provides main() that writes the merged file
    except Exception:
        merge_and_clean_summaries = None

    if merge_and_clean_summaries and hasattr(merge_and_clean_summaries, "main"):
        print("Running tools.merge_and_clean_summaries.main()")
        # call with defaults
        merge_and_clean_summaries.main(input_dir=str(output_dir), output_file=str(merged_path))
        return merged_path if merged_path.exists() else None

    # fallback: do a simple merge of any output/summaries_*.json
    merged = []
    import glob
    files = sorted(glob.glob(str(output_dir / "summaries_*.json")))
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    for i, s in enumerate(data):
                        if isinstance(s, str):
                            merged.append({"source_file": os.path.basename(f), "index": i, "summary": s})
                        elif isinstance(s, dict):
                            merged.append(s)
                elif isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list):
                            for i, s in enumerate(v):
                                if isinstance(s, str):
                                    merged.append({"source_file": os.path.basename(f), "index": i, "summary": s})
                                else:
                                    merged.append(s)
        except Exception as e:
            print("Failed to parse", f, e)

    if merged:
        write_json_safe(merged_path, merged)
        return merged_path
    else:
        print("No summaries merged.")
        return None


def generate_cleaned_documents(raw_path: Path = Path("data/artifacts/raw_documents.json"), summaries_path: Path = Path("output/all_cleaned_summaries.json"), out_path: Path = Path("data/artifacts/cleaned_documents.json")):
    try:
        from tools.workflow.filter_and_reindex import generate_cleaned_documents as gen
        print("Running tools.filter_and_reindex.generate_cleaned_documents()")
        gen(raw_path=str(raw_path), summaries_path=str(summaries_path), out_path=str(out_path))
        return out_path if out_path.exists() else None
    except Exception as e:
        print("Failed to run generate_cleaned_documents:", e)
        return None


def generate_hybrid(summaries_path: Path = Path("output/all_cleaned_summaries.json"), cleaned_docs_path: Path = Path("data/artifacts/cleaned_documents.json"), raw_docs_path: Path = Path("data/artifacts/raw_documents.json"), out_jsonl: Path = Path("data/artifacts/hybrid_summaries.jsonl"), report_path: Path = Path("data/artifacts/hybrid_summaries_report.json")):
    try:
        from tools.workflow.create_hybrid_dataset import generate_hybrid_dataset
        print("Running tools.workflow.create_hybrid_dataset.generate_hybrid_dataset()")
        recs = generate_hybrid_dataset(summaries_path=str(summaries_path), cleaned_docs_path=str(cleaned_docs_path), raw_docs_path=str(raw_docs_path), out_jsonl=str(out_jsonl), report_path=str(report_path))
        return out_jsonl if out_jsonl.exists() else None
    except Exception as e:
        print("Failed to run generate_hybrid_dataset:", e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Programmatically generate artifacts used by the UI and downstream steps")
    parser.add_argument("--skip-etl", action="store_true")
    parser.add_argument("--skip-summarize", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-cleaned", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    args = parser.parse_args()

    output_dir = Path("output")
    raw_path = Path("data/artifacts/raw_documents.json")
    merged_path = Path("output/all_cleaned_summaries.json")
    cleaned_path = Path("data/artifacts/cleaned_documents.json")
    hybrid_path = Path("data/artifacts/hybrid_summaries.jsonl")
    hybrid_report = Path("data/artifacts/hybrid_summaries_report.json")

    # Step 1: ETL raw docs
    if not args.skip_etl:
        ok = run_etl_generate_raw(out_path=raw_path)
        if not ok:
            print("ETL failed or produced no docs; continuing but some steps may fail")
    else:
        print("Skipping ETL per flag")

    # Step 2: Summarization
    if not args.skip_summarize:
        summ_path = run_summarization_and_export()
        if summ_path:
            print("Summaries exported to", summ_path)
        else:
            print("Summarization did not produce an export file")
    else:
        print("Skipping summarization per flag")

    # Step 3: Merge summaries -> output/all_cleaned_summaries.json
    if not args.skip_merge:
        merged = merge_summaries_into_all(output_dir=output_dir, merged_path=merged_path)
        if merged:
            print("Merged summaries at", merged)
        else:
            print("No merged summaries produced")
    else:
        print("Skipping merge per flag")

    # Step 4: generate cleaned_documents.json
    if not args.skip_cleaned:
        cleaned = generate_cleaned_documents(raw_path=raw_path, summaries_path=merged_path, out_path=cleaned_path)
        if cleaned:
            print("Cleaned documents written to", cleaned)
        else:
            print("No cleaned documents produced")
    else:
        print("Skipping cleaned docs per flag")

    # Step 5: create hybrid dataset
    if not args.skip_hybrid:
        hybrid = generate_hybrid(summaries_path=merged_path, cleaned_docs_path=cleaned_path, raw_docs_path=raw_path, out_jsonl=hybrid_path, report_path=hybrid_report)
        if hybrid:
            print("Hybrid dataset written to", hybrid)
        else:
            print("Hybrid generation failed")
    else:
        print("Skipping hybrid per flag")

    print("Done. Artifacts status:")
    for p in [merged_path, raw_path, cleaned_path, hybrid_path, hybrid_report]:
        print(p, "exists=" , p.exists())


if __name__ == "__main__":
    main()
