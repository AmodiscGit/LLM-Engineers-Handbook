#!/usr/bin/env python3
"""Filter obvious boilerplate/license/robot files from raw artifacts and run a re-index.

Usage: poetry run python tools/filter_and_reindex.py
"""
from __future__ import annotations
import json
import os
import re
from typing import List
from pathlib import Path
import subprocess


BOILERPLATE_PATTERNS = [
    r"robots\.txt",
    r"copyright",
    r"permission to use, copy",
    r"license",
    r"@license",
    r"all rights reserved",
    r"^\s*/\*\!",  # license header starts
    r"^#!",  # script shebangs
]


def is_boilerplate(text: str) -> bool:
    if not text or len(text.strip()) < 60:
        # too short to be useful
        return True
    lower = text.lower()
    # code-like ratio heuristic: many braces, semicolons
    code_chars = sum(1 for c in text if c in '{};<>/\\')
    if code_chars / max(1, len(text)) > 0.01 and len(text) < 1000:
        # short but mostly code => likely license/js bundle, treat as boilerplate
        return True

    for p in BOILERPLATE_PATTERNS:
        if re.search(p, lower):
            return True

    # too repetitive (lots of repeated words)
    tokens = re.findall(r"\w+", lower)
    if tokens:
        most_common = max((tokens.count(t) for t in set(tokens)))
        if most_common / len(tokens) > 0.2:
            return True

    return False


def load_raw(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_cleaned(docs: List[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def main():
    raw_path = 'data/artifacts/raw_documents.json'
    summaries_path = 'output/all_cleaned_summaries.json'
    out_path = 'data/artifacts/cleaned_documents.json'

    raw = load_raw(raw_path)
    cleaned = []
    for d in raw:
        content = d.get('content') or ''
        if not is_boilerplate(content):
            cleaned.append(d)

    # Also include cleaned summaries if they look non-boilerplate
    if os.path.exists(summaries_path):
        with open(summaries_path, 'r', encoding='utf-8') as f:
            try:
                arr = json.load(f)
            except Exception:
                arr = []
        for s in arr:
            text = s.get('summary') or s.get('text') or ''
            if text and not is_boilerplate(text):
                rec = {'id': s.get('index') or None, 'content': text, 'link': s.get('source_file')}
                cleaned.append(rec)

    print(f'Raw docs: {len(raw)}, Cleaned docs: {len(cleaned)}')
    write_cleaned(cleaned, out_path)

    # Run the feature-engineering pipeline with no-cache to re-index cleaned docs
    print('Running feature-engineering pipeline (no-cache) to re-index cleaned docs...')
    cmd = ['poetry', 'run', 'python', 'tools/run.py', '--no-cache', '--run-feature-engineering']
    subprocess.run(cmd, check=True)

    print('Re-index complete. You can verify with the Streamlit UI or run sample queries.')


if __name__ == '__main__':
    main()
