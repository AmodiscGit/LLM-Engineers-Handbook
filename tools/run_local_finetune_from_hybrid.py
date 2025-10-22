#!/usr/bin/env python3
"""Convert hybrid dataset to the expected format and run a short local finetune.

This script:
 - Reads data/artifacts/hybrid_summaries.jsonl
 - Produces a temporary json dataset where each record has a `text` field combining source and target
 - Calls the `finetune_local` function to run a short finetune and save to output/trained_model
"""

import os
import json
import tempfile
from llm_engineering.model.finetuning.local_finetune import finetune_local


def prepare_dataset(hybrid_path="data/artifacts/hybrid_summaries.jsonl", out_path="data/artifacts/hybrid_for_finetune.json"):
    records = []
    with open(hybrid_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            source = rec.get("source")
            target = rec.get("summary")
            score = rec.get("matched_score", 0.0)
            # simple formatting: include both source and target in a single text field
            if not source or not target:
                continue
            text = f"Source:\n{source}\n\nSummary:\n{target}"
            records.append({"text": text})

    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(records, out, ensure_ascii=False)
    print(f"Wrote {len(records)} records to {out_path}")
    return out_path


def main():
    hybrid = "data/artifacts/hybrid_summaries.jsonl"
    if not os.path.exists(hybrid):
        raise SystemExit("Hybrid dataset not found; run create_hybrid_dataset.py first")

    ds_path = prepare_dataset(hybrid)

    output_dir = "output/trained_model"
    # Run a tiny finetune: 1 epoch, small batch
    model, tokenizer = finetune_local(
        model_name="distilgpt2",
        output_dir=output_dir,
        dataset_path=ds_path,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        is_dummy=True,
    )

    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
