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

    if best is not None and best_score > 0.05:
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
        raise SystemExit(f"Summaries file not found: {summaries_path}")

    summaries = load_json(summaries_path)
    # ensure it's a list
    if isinstance(summaries, dict):
        # try to find a top-level list value
        for v in summaries.values():
            if isinstance(v, list):
                summaries = v
                break

    docs = []
    for p in (cleaned_docs_path, raw_docs_path):
        if os.path.exists(p):
            data = load_json(p)
            if isinstance(data, dict) and "artifact_data" in data and isinstance(data["artifact_data"], list):
                for d in data["artifact_data"]:
                    docs.append(d)
            elif isinstance(data, list):
                docs.extend(data)

    print(f"Loaded {len(summaries)} summaries and {len(docs)} candidate raw docs for matching.")

    paired = 0
    unmatched = 0
    written = 0
    results = []

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

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

    report = {
        "summaries_total": len(summaries),
        "docs_total": len(docs),
        "paired": paired,
        "unmatched": unmatched,
        "output_file": out_jsonl,
    }

    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    print("Done. Report:")
    print(json.dumps(report, indent=2))

    return results


if __name__ == "__main__":
    main()
