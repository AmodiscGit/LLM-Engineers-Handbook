# Workflow helper: create_hybrid_dataset.py
# Required files / artifacts:
# - output/all_cleaned_summaries.json (input)
# - data/artifacts/cleaned_documents.json (optional input)
# - data/artifacts/raw_documents.json (input)
# - writes: data/artifacts/hybrid_summaries.jsonl and data/artifacts/hybrid_summaries_report.json

"""
Moved from tools/create_hybrid_dataset.py into tools/workflow. This module creates
hybrid summary pairs by matching cleaned summaries to raw docs.
"""
#!/usr/bin/env python3
"""Create a hybrid (source -> summary) dataset by pairing cleaned summaries
with local raw/cleaned documents using simple heuristics.

Outputs:
 - data/artifacts/hybrid_summaries.jsonl  (one JSON per line)
 - data/artifacts/hybrid_summaries_report.json (diagnostics)

This is a best-effort heuristic matcher: it first attempts long-substring
matches, then falls back to token-overlap scoring.
"""

import json
import os
import uuid
import re
from collections import Counter
from typing import Any, Tuple

from loguru import logger
from zenml import step


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def long_sentences(text, min_len=60):
    # split on punctuation, return sentences longer than min_len
    parts = re.split(r"[\.\n\?\!]", text)
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


def token_overlap_score(a: str, b: str) -> float:
    # simple token overlap (no stopword removal for simplicity)
    a_toks = set(re.findall(r"\w{3,}", a.lower()))
    b_toks = set(re.findall(r"\w{3,}", b.lower()))
    if not a_toks or not b_toks:
        return 0.0
    overlap = a_toks & b_toks
    return len(overlap) / max(1, min(len(a_toks), len(b_toks)))


def find_best_match(summary_text, docs):
    norm_summary = normalize_text(summary_text)
    # try long-sentence exact substring match first
    for sent in long_sentences(norm_summary, min_len=60):
        for doc in docs:
            content = normalize_text(doc.get("content", ""))
            if sent in content:
                return doc, 1.0, "substring"

    # fallback: token overlap scoring
    best = None
    best_score = 0.0
    for doc in docs:
        content = normalize_text(doc.get("content", ""))
        score = token_overlap_score(norm_summary, content)
        if score > best_score:
            best_score = score
            best = doc

    # Require a non-trivial overlap to consider a match. Raise threshold to avoid
    # spuriously matching very short or generic summaries.
    if best is not None and best_score > 0.20:
        return best, best_score, "overlap"

    return None, 0.0, None


def main():
    # default paths
    summaries_path = "output/all_cleaned_summaries.json"
    cleaned_docs_path = "data/artifacts/cleaned_documents.json"
    raw_docs_path = "data/artifacts/raw_documents.json"
    out_jsonl = "data/artifacts/hybrid_summaries.jsonl"
    report_path = "data/artifacts/hybrid_summaries_report.json"

    return generate_hybrid_dataset(
        summaries_path=summaries_path,
        cleaned_docs_path=cleaned_docs_path,
        raw_docs_path=raw_docs_path,
        out_jsonl=out_jsonl,
        report_path=report_path,
    )


@step
def generate_hybrid_dataset(
    summaries_path: str = "output/all_cleaned_summaries.json",
    cleaned_docs_path: str = "data/artifacts/cleaned_documents.json",
    raw_docs_path: str = "data/artifacts/raw_documents.json",
    out_jsonl: str = "data/artifacts/hybrid_summaries.jsonl",
    report_path: str = "data/artifacts/hybrid_summaries_report.json",
):
    """Create the hybrid dataset and return the list of records.

    The function writes `out_jsonl` and `report_path` like the original script.
    Returns: list of record dicts (the same objects written to the jsonl).
    """

    if not os.path.exists(summaries_path):
        # Raise a clear error so ZenML records the failure in the run
        raise FileNotFoundError(f"Summaries file not found: {summaries_path}")

    summaries = load_json(summaries_path)
    # ensure it's a list
    if isinstance(summaries, dict):
        # try to find a top-level list value
        for v in summaries.values():
            if isinstance(v, list):
                summaries = v
                break

    docs = []
    # If cleaned docs are missing, run the filter-and-reindex step to produce them
    if not os.path.exists(cleaned_docs_path):
        logger.info(
            "cleaned_docs_path %s not found; attempting to generate it via tools.workflow.filter_and_reindex.generate_cleaned_documents()...",
            cleaned_docs_path,
        )
        # Prefer importing the lightweight helper to avoid running an expensive pipeline
        try:
            from tools.workflow.filter_and_reindex import generate_cleaned_documents

            try:
                generate_cleaned_documents(raw_path=raw_docs_path, summaries_path=summaries_path, out_path=cleaned_docs_path)
            except Exception:
                # fall back to running the script as a separate run (best-effort)
                import runpy

                runpy.run_path(os.path.join("tools", "workflow", "filter_and_reindex.py"), run_name="__main__")
        except Exception as e:
            logger.warning("Failed to generate cleaned documents via filter_and_reindex: %s", e)
    # Load candidate docs from cleaned and raw paths and deduplicate them.
    seen_ids = set()
    seen_content = set()
    for p in (cleaned_docs_path, raw_docs_path):
        if os.path.exists(p):
            data = load_json(p)
            items = []
            if isinstance(data, dict) and "artifact_data" in data and isinstance(data["artifact_data"], list):
                items = data["artifact_data"]
            elif isinstance(data, list):
                items = data
            for d in items:
                # prefer stable id or link for deduplication
                doc_id = d.get("id") or d.get("link")
                content = d.get("content", "")
                norm = normalize_text(content)[:1000] if content else None
                if doc_id and doc_id in seen_ids:
                    continue
                if norm and norm in seen_content:
                    continue
                if doc_id:
                    seen_ids.add(doc_id)
                if norm:
                    seen_content.add(norm)
                docs.append(d)

    logger.info("Loaded %d summaries and %d candidate raw docs for matching.", len(summaries), len(docs))

    paired = 0
    unmatched = 0
    written = 0
    results = []

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    try:
        with open(out_jsonl, "w", encoding="utf-8") as out_f:
            for i, s in enumerate(summaries):
                summary_text = s.get("summary") or s.get("text") or ""
                source_file = s.get("source_file")
                index = s.get("index")
                rec = {
                    "id": str(uuid.uuid4()),
                    "summary": summary_text,
                    "source_file": source_file,
                    "index": index,
                }

                best_doc, score, method = find_best_match(summary_text, docs)
                if best_doc:
                    rec["matched_raw_doc_id"] = best_doc.get("id") or best_doc.get("link") or None
                    rec["matched_score"] = float(score)
                    rec["matched_method"] = method
                    rec["source"] = best_doc.get("content")
                    paired += 1
                else:
                    rec["matched_raw_doc_id"] = None
                    rec["matched_score"] = 0.0
                    rec["matched_method"] = None
                    rec["source"] = None
                    unmatched += 1

                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                results.append(rec)
                written += 1
    except Exception as e:
        logger.exception("Failed while writing hybrid jsonl to %s: %s", out_jsonl, e)
        # Re-raise so pipeline sees the failure
        raise

    report = {
        "summaries_total": len(summaries),
        "docs_total": len(docs),
        "paired": paired,
        "unmatched": unmatched,
        "output_file": out_jsonl,
    }

    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2)
    except Exception as e:
        logger.exception("Failed to write report to %s: %s", report_path, e)
        raise

    logger.info("Done. Report: %s", json.dumps(report, indent=2))

    # Explicit outputs: return the report dict and the path to the generated JSONL
    return report, out_jsonl


if __name__ == "__main__":
    main()
