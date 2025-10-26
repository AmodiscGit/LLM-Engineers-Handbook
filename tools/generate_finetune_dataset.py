"""Utility to scaffold a supervised fine-tuning dataset from annotated interactions.

Expect input annotations as JSONL with fields:
  - query
  - answer
  - sources: list of {link, summary}

This tool converts each annotation into an OpenAI-style training example JSONL:
  {"prompt": <prompt text>, "completion": <completion text>}

Usage:
  poetry run python tools/generate_finetune_dataset.py --input annotations.jsonl --out ft_dataset.jsonl

If annotations are not present, you can produce them by saving `tools/rag_tools.generate_answer` results and editing/curating answers.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def build_prompt(query: str, sources: List[Dict[str, Any]]) -> str:
    parts = [f"Question: {query}", "Use the following sources to answer:"]
    for i, s in enumerate(sources, start=1):
        parts.append(f"Source {i}: {s.get('link')}\nSummary: {s.get('summary')}")
    parts.append("\nAnswer concisely:")
    return "\n\n".join(parts)


def convert_annotation_to_example(ann: Dict[str, Any]) -> Dict[str, str]:
    query = ann.get("query")
    answer = ann.get("answer")
    sources = ann.get("sources") or []
    prompt = build_prompt(query, sources)
    # For OpenAI fine-tuning, completions should begin with a space and end with the special token if required.
    completion = " " + answer.strip()
    return {"prompt": prompt, "completion": completion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input annotations JSONL")
    parser.add_argument("--out", required=True, help="Output fine-tune JSONL")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    if not inp.exists():
        print("Input file not found:", inp)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with inp.open("r", encoding="utf-8") as r, out.open("w", encoding="utf-8") as w:
        for line in r:
            if not line.strip():
                continue
            ann = json.loads(line)
            ex = convert_annotation_to_example(ann)
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("Wrote fine-tune dataset to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
