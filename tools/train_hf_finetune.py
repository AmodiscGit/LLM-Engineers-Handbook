#!/usr/bin/env python3
"""
Small Hugging Face fine-tune script for causal LMs.

Usage (from repo root):
  poetry run python tools/train_hf_finetune.py \
      --data tools/ft_dataset.jsonl \
      --model distilgpt2 \
      --output models/hf-finetuned-small \
      --epochs 3 \
      --batch_size 8

This script expects the dataset JSONL to have one record per line with keys
"prompt" and "completion" (the OpenAI fine-tune format produced earlier).

It tokenizes prompt and completion separately, concatenates them, and sets labels
so that the loss is computed only on the completion tokens (prompt tokens set to -100).

Notes:
- For small datasets prefer few epochs and a small model (distilgpt2 or gpt2).
- For production or larger datasets prefer PEFT/LoRA to avoid full fine-tuning.
"""

import argparse
import os
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="JSONL training file (lines=1)")
    p.add_argument("--model", default="distilgpt2", help="Base model to fine-tune")
    p.add_argument("--output", "--output_dir", dest="output", default="models/hf-finetuned-small")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", "--per_device_train_batch_size", dest="batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--validation_split", type=float, default=0.0, help="Fraction of data to reserve for validation (0 to disable)")
    p.add_argument("--early_stopping_patience", type=int, default=0, help="If >0 and validation enabled, use EarlyStopping with this patience (in eval steps/epochs)")
    return p.parse_args()


def build_example(example: Dict, tokenizer, max_length: int):
    # Expect keys 'prompt' and 'completion' (fall back to common alternatives)
    prompt = example.get("prompt") or example.get("instruction") or ""
    completion = example.get("completion") or example.get("output") or example.get("answer") or ""

    # Ensure prompt ends with whitespace/newline to avoid token merging with completion
    if not prompt.endswith("\n"):
        prompt_text = prompt + "\n"
    else:
        prompt_text = prompt

    # Tokenize separately
    p_tok = tokenizer(prompt_text, add_special_tokens=False)
    c_tok = tokenizer(completion, add_special_tokens=False)

    input_ids = p_tok["input_ids"] + c_tok["input_ids"]

    # Labels: mask prompt tokens with -100
    labels = [-100] * len(p_tok["input_ids"]) + c_tok["input_ids"]

    # Truncate if needed
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def main():
    args = parse_args()

    data_path = args.data
    if not Path(data_path).exists():
        raise SystemExit(f"Training file not found: {data_path}")

    print("Loading dataset:", data_path)
    ds = load_dataset("json", data_files={"train": data_path}, split="train")
    print("Records:", len(ds))

    # Optionally create a validation split
    if args.validation_split and args.validation_split > 0.0:
        print(f"Creating validation split: {args.validation_split}")
        split = ds.train_test_split(test_size=args.validation_split)
        ds_train = split["train"]
        ds_eval = split["test"]
        print("Train records:", len(ds_train), "Eval records:", len(ds_eval))
    else:
        ds_train = ds
        ds_eval = None

    print("Loading tokenizer and model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Map examples to token ids and labels
    def fn_map(example):
        return build_example(example, tokenizer, args.max_length)

    print("Tokenizing and preparing labels...")
    ds_proc = ds_train.map(fn_map, remove_columns=ds_train.column_names)
    ds_proc_eval = None
    if ds_eval is not None:
        ds_proc_eval = ds_eval.map(fn_map, remove_columns=ds_eval.column_names)

    # Data collator will pad input_ids and labels (labels already contain -100 for masked tokens)
    data_collator = DataCollatorWithPadding(tokenizer)

    print("Loading model (this may download weights)...")
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Resize token embeddings if tokenizer was changed (pad token added)
    model.resize_token_embeddings(len(tokenizer))

    # Enable evaluation and save strategy depending on whether we have a validation dataset
    if ds_proc_eval is not None:
        evaluation_strategy = "epoch"
        save_strategy = "epoch"
        load_best = True
    else:
        evaluation_strategy = "no"
        save_strategy = "no"
        load_best = False

    # Build TrainingArguments kwargs dynamically to support older/newer transformers versions
    import inspect

    train_kwargs = {
        "output_dir": args.output,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        # prefer saving/eval per epoch when supported
        "save_strategy": save_strategy,
        "logging_strategy": "epoch",
        "evaluation_strategy": evaluation_strategy,
        "remove_unused_columns": False,
        "fp16": False,
        # best model handling
        "load_best_model_at_end": load_best,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 3,
    }

    # Filter kwargs to only those accepted by the installed transformers.TrainingArguments
    try:
        sig = inspect.signature(TrainingArguments)
        accepted = set(sig.parameters.keys())
        # If the installed transformers doesn't accept evaluation_strategy but does accept save_strategy
        # we must avoid enabling load_best_model_at_end to prevent a mismatch error (eval=no vs save=epoch)
        if "evaluation_strategy" not in accepted and train_kwargs.get("load_best_model_at_end"):
            # disable load_best behavior when evaluation can't be enabled
            train_kwargs["load_best_model_at_end"] = False
            if "save_strategy" in accepted:
                # prefer not to save per-epoch when eval is disabled to avoid mismatches
                train_kwargs["save_strategy"] = "no"

        filtered = {k: v for k, v in train_kwargs.items() if k in accepted}
    except Exception:
        # If signature inspection fails for any reason, fall back to using the original dict
        filtered = train_kwargs

    training_args = TrainingArguments(**filtered)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_proc,
        eval_dataset=ds_proc_eval,
        data_collator=data_collator,
    )

    # Attach EarlyStopping if requested and evaluation is enabled
    attach_early = False
    if args.early_stopping_patience and args.early_stopping_patience > 0 and ds_proc_eval is not None:
        # Check whether the TrainingArguments actually enabled evaluation (some transformers versions may not accept eval args)
        try:
            es = getattr(training_args, "evaluation_strategy", None)
            es_s = str(es).lower() if es is not None else "no"
            if "no" not in es_s:
                attach_early = True
        except Exception:
            attach_early = False

    if attach_early:
        print(f"Attaching EarlyStoppingCallback with patience={args.early_stopping_patience}")
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    else:
        if args.early_stopping_patience and args.early_stopping_patience > 0:
            print("EarlyStopping requested but evaluation is not enabled; skipping EarlyStopping attachment.")

    print("Starting training...")
    trainer.train()

    print("Saving model to:", args.output)
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
