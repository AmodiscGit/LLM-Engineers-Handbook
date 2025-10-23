"""Prepare simple instruction/target training pairs from chunked documents.

Writes `data/processed/training_data.jsonl` with fields:
 - instruction: e.g., "Summarize the following chunk and include a one-line provenance:"
 - input: the chunk text
 - target: the extractive provenance summary produced by ui_helpers.summarize_with_provenance

This is a low-cost way to produce fine-tuning examples for a seq2seq model.
"""
from __future__ import annotations
import json
from pathlib import Path
import os
from tools import ui_helpers

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
OUT_FILE = Path("data/processed/training_data.jsonl")
MAX_EXAMPLES = int(os.getenv("MAX_TRAIN_EXAMPLES", "500"))


def main():
    if not CHUNKS_FILE.exists():
        print(f"Chunks file not found: {CHUNKS_FILE}. Run tools/prepare_chunks.py first.")
        return

    out_dir = OUT_FILE.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with CHUNKS_FILE.open("r", encoding="utf-8") as fin, OUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text") or ""
            if not text.strip():
                continue

            instruction = "Summarize the following text in 3 short bullets and provide a one-line provenance indicating the source file."
            # use the repo's extractive summarizer to create a target
            summary = ui_helpers.summarize_text(text)
            # fallback to simple truncation
            if not summary or len(summary.strip()) < 20:
                summary = (text[:300] + "...") if len(text) > 300 else text

            example = {"instruction": instruction, "input": text, "target": summary}
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1
            if written >= MAX_EXAMPLES:
                break

    print(f"Wrote {written} training examples to {OUT_FILE}")


if __name__ == "__main__":
    main()
