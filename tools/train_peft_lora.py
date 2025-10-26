#!/usr/bin/env python3
"""
PEFT / LoRA fine-tuning script (lightweight)

Usage:
  poetry run python tools/train_peft_lora.py \
    --data tools/ft_dataset.jsonl \
    --model distilgpt2 \
    --output models/hf-peft-lora \
    --epochs 3 \
    --batch_size 8

This script:
- Loads JSONL dataset with 'prompt' and 'completion' fields.
- Tokenizes and builds input_ids + labels (prompt tokens masked with -100).
- Wraps the base model with PEFT/LoRA adapters and trains the adapter weights only.

Notes:
- For best performance with large models, install bitsandbytes and run with --use_8bit.
- Default target modules are appropriate for GPT-style causal models; you can override via --target_modules.
"""

import argparse
from pathlib import Path
from typing import Dict, List
import os

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:
    raise SystemExit("peft library is required. Install with: pip install peft")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--output", default="models/hf-peft-lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--target_modules", default=None,
                   help="Comma-separated module names to apply LoRA to (default suited for GPT-style models)")
    p.add_argument("--use_8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    return p.parse_args()


def build_example(example: Dict, tokenizer, max_length: int):
    prompt = example.get("prompt") or example.get("instruction") or ""
    completion = example.get("completion") or example.get("output") or example.get("answer") or ""

    if not prompt.endswith("\n"):
        prompt_text = prompt + "\n"
    else:
        prompt_text = prompt

    p_tok = tokenizer(prompt_text, add_special_tokens=False)
    c_tok = tokenizer(completion, add_special_tokens=False)

    input_ids = p_tok["input_ids"] + c_tok["input_ids"]
    labels = [-100] * len(p_tok["input_ids"]) + c_tok["input_ids"]

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    print("Loading dataset:", data_path)
    ds = load_dataset("json", data_files={"train": str(data_path)}, split="train")
    print("Examples:", len(ds))

    print("Loading tokenizer:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    print("Tokenizing dataset...")
    def map_fn(ex):
        return build_example(ex, tokenizer, args.max_length)

    ds_proc = ds.map(map_fn, remove_columns=ds.column_names)

    # Custom data collator: use tokenizer.pad for inputs and pad/truncate labels manually
    def data_collator(features):
        labels = [f.get("labels") for f in features]
        features_wo_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = tokenizer.pad(features_wo_labels, padding=True, return_tensors="pt")

        max_len = batch["input_ids"].size(1)
        padded_labels = []
        for lab in labels:
            if isinstance(lab, list):
                if len(lab) > max_len:
                    lab = lab[-max_len:]
                padded = lab + [-100] * (max_len - len(lab))
            elif isinstance(lab, torch.Tensor):
                lab = lab.tolist()
                if len(lab) > max_len:
                    lab = lab[-max_len:]
                padded = lab + [-100] * (max_len - len(lab))
            else:
                padded = [-100] * max_len
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    print("Loading base model:", args.model)
    load_kwargs = {}
    if args.use_8bit:
        # prepare for k-bit training requires bitsandbytes and proper model type
        try:
            load_kwargs["load_in_8bit"] = True
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    # Prepare model for k-bit if requested
    if args.use_8bit:
        try:
            model = prepare_model_for_kbit_training(model)
        except Exception:
            print("Warning: prepare_model_for_kbit_training not available or failed; continuing without it")

    # Default target modules for GPT-style models
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    else:
        target_modules = ["c_attn", "c_proj"]

    print("Applying LoRA adapters: r=%s alpha=%s target_modules=%s" % (args.lora_r, args.lora_alpha, target_modules))
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_proc,
        data_collator=data_collator,
    )

    print("Starting LoRA training...")
    trainer.train()

    print("Saving LoRA adapters to:", args.output)
    model.save_pretrained(args.output)


if __name__ == "__main__":
    main()
