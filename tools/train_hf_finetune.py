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

    # Enable evaluation if we have a validation dataset
    evaluation_strategy = "epoch" if ds_proc_eval is not None else "no"

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy="epoch",
        logging_strategy="epoch",
        evaluation_strategy=evaluation_strategy,
        remove_unused_columns=False,
        fp16=False,
        load_best_model_at_end=True if ds_proc_eval is not None else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_proc,
        eval_dataset=ds_proc_eval,
        data_collator=data_collator,
    )

    # Attach EarlyStopping if requested and we have an eval set
    if args.early_stopping_patience and args.early_stopping_patience > 0 and ds_proc_eval is not None:
        print(f"Attaching EarlyStoppingCallback with patience={args.early_stopping_patience}")
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    print("Starting training...")
    trainer.train()

    print("Saving model to:", args.output)
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
