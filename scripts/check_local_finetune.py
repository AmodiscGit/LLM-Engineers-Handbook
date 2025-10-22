#!/usr/bin/env python3
"""Standalone import-check + optional local finetune runner.

This script intentionally avoids importing the `llm_engineering` package so it
won't trigger `sentence-transformers` imports. Use it to verify the environment
(can we import transformers, datasets, torch, etc.) and to run a small local
finetune using the same code as `local_finetune.finetune_local` if you pass
`--run-train`.

Usage examples:
  # just run import checks
  poetry run python scripts/check_local_finetune.py

  # run a short local train (will use tiny subset)
  poetry run python scripts/check_local_finetune.py --run-train --dataset output/all_cleaned_summaries.json --output output/trained_model --model distilgpt2 --epochs 1 --batch 2
"""

import argparse
import sys
import os


def import_checks():
    ok = True
    try:
        import importlib
        hf = importlib.import_module("transformers")
        print("transformers:", getattr(hf, "__version__", "unknown"))
    except Exception as e:
        print("ERROR importing transformers:", type(e).__name__, e)
        ok = False

    try:
        ds = importlib.import_module("datasets")
        print("datasets:", getattr(ds, "__version__", "unknown"))
    except Exception as e:
        print("ERROR importing datasets:", type(e).__name__, e)
        ok = False

    try:
        th = importlib.import_module("torch")
        print("torch:", getattr(th, "__version__", "unknown"))
    except Exception as e:
        print("ERROR importing torch:", type(e).__name__, e)
        ok = False

    try:
        hfh = importlib.import_module("huggingface_hub")
        print("huggingface_hub:", getattr(hfh, "__version__", "unknown"))
    except Exception as e:
        print("ERROR importing huggingface_hub:", type(e).__name__, e)
        ok = False

    return ok


def do_train(model_name, dataset_path, output_dir, epochs, batch_size, dummy=True):
    # Copy of the critical logic from local_finetune.finetune_local but standalone
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Local dataset_path '{dataset_path}' does not exist.")

    ds = load_dataset("json", data_files=dataset_path, split="train")

    def make_text(example):
        if "instruction" in example and "output" in example:
            return {"text": f"Below is an instruction.\n\nInstruction:\n{example['instruction']}\n\nResponse:\n{example['output']}"}
        if "text" in example:
            return {"text": example["text"]}
        return {"text": " ".join([str(v) for v in example.values()])}

    ds = ds.map(make_text, remove_columns=ds.column_names)

    if dummy:
        n = min(200, len(ds))
        ds = ds.select(range(n))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=5e-5,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training finished. Artifacts saved to:", output_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-train", action="store_true", help="If set, run a short local training run")
    p.add_argument("--model", default="distilgpt2", help="HF model id to use")
    p.add_argument("--dataset", default="output/all_cleaned_summaries.json")
    p.add_argument("--output", default="output/trained_model")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--no-dummy", dest="dummy", action="store_false")

    args = p.parse_args()

    print("Running import checks...")
    ok = import_checks()
    if not ok:
        print("Some imports failed. See errors above. You can still try to run training but it will likely fail.")

    if args.run_train:
        if not ok:
            print("Warning: proceeding despite import check failures.")
        do_train(args.model, args.dataset, args.output, args.epochs, args.batch, dummy=args.dummy)
    else:
        print("Import check complete. To run a small training job, re-run with --run-train")
