#!/usr/bin/env python3
"""Verify and run quick tests against a local fine-tuned model.

Usage examples:
  # Run a few example prompts and print outputs
  poetry run python scripts/verify_model.py

  # Run prompts with a specified model directory
  poetry run python scripts/verify_model.py --model-dir output/trained_model

  # Run a quick evaluation comparing generated outputs to references in a JSON file
  poetry run python scripts/verify_model.py --eval --eval-file output/all_cleaned_summaries.json --n-samples 50

The evaluation uses a simple token-overlap (precision/recall/F1) metric to avoid
adding external dependencies. It expects the eval JSON to contain a field
called 'summary' (reference) and a field with the source text (common names: 'text', 'content', 'summary' etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def simple_overlap_metrics(pred: str, ref: str) -> Tuple[float, float, float]:
    # Tokenize by whitespace and simple punctuation splitting
    def tok(s: str) -> List[str]:
        return [t.strip().lower() for t in s.replace("\n", " ").split() if t.strip()]

    p_tokens = tok(pred)
    r_tokens = tok(ref)
    if not p_tokens or not r_tokens:
        return 0.0, 0.0, 0.0

    p_set = set(p_tokens)
    r_set = set(r_tokens)
    overlap = p_set & r_set
    if len(p_set) == 0:
        prec = 0.0
    else:
        prec = len(overlap) / len(p_set)
    if len(r_set) == 0:
        rec = 0.0
    else:
        rec = len(overlap) / len(r_set)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def load_eval_examples(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # If file is a mapping, try to extract a list
    if isinstance(data, dict):
        # try to find a top-level list field
        for v in data.values():
            if isinstance(v, list):
                return v
        raise ValueError("Eval JSON is an object but no top-level list found")
    if isinstance(data, list):
        return data
    raise ValueError("Eval JSON must be a list or a dict with a list value")


def guess_source_field(example: dict) -> str:
    # prefer common names
    for name in ("text", "content", "document", "input", "source", "summary"):
        if name in example:
            return name
    # fallback: first non-empty string field
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            return k
    raise ValueError("Couldn't guess source field in eval example")


def guess_reference_field(example: dict) -> str:
    for name in ("summary", "output", "reference", "target"):
        if name in example:
            return name
    # fallback: None
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="output/trained_model")
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected by default)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--eval-file", default="output/all_cleaned_summaries.json")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = torch.device("cuda" if (args.device is None and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    print(f"Loading tokenizer & model from: {model_dir} on device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to(device)

    # Example prompts to inspect behavior
    examples = [
        "Summarize: non-accidental trauma",
        "Summarize: potential non-accidental trauma in infants.",
    ]

    print("\n=== Example generations ===")
    for i, prompt in enumerate(examples, 1):
        out = generate(model, tokenizer, prompt, device, max_new_tokens=args.max_new_tokens)
        print(f"\n[{i}] PROMPT: {prompt}\n----\n{out}\n")

    if args.eval:
        print("\n=== Running simple evaluation (token-overlap) ===")
        examples_data = load_eval_examples(args.eval_file)
        random.seed(args.seed)
        samples = random.sample(examples_data, min(args.n_samples, len(examples_data)))

        # try to guess source/ref fields
        src_field = guess_source_field(samples[0])
        ref_field = guess_reference_field(samples[0])
        if ref_field is None:
            print("No reference field like 'summary'/'output' found in examples; skipping evaluation.")
            return

        print(f"Using source field: {src_field}; reference field: {ref_field}")

        precs, recs, f1s = [], [], []
        for ex in samples:
            src = ex.get(src_field, "")
            ref = ex.get(ref_field, "")
            if not src or not ref:
                continue
            prompt = f"Summarize: {src}"
            gen = generate(model, tokenizer, prompt, device, max_new_tokens=args.max_new_tokens)
            prec, rec, f1 = simple_overlap_metrics(gen, ref)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)

        def mean(l):
            return sum(l) / len(l) if l else 0.0

        print(f"N eval samples: {len(f1s)}")
        print(f"Average Precision: {mean(precs):.4f}")
        print(f"Average Recall: {mean(recs):.4f}")
        print(f"Average F1 (overlap): {mean(f1s):.4f}")


if __name__ == "__main__":
    main()
