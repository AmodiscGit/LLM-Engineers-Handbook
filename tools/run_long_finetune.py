#!/usr/bin/env python3
"""Run a longer local finetune and verify generation.

Saves model to output/trained_model_long
"""
import os
from llm_engineering.model.finetuning.local_finetune import finetune_local
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    dataset_path = "data/artifacts/hybrid_for_finetune.json"
    output_dir = "output/trained_model_long"

    print("Starting longer finetune: epochs=5, batch=4, full dataset")
    model, tokenizer = finetune_local(
        model_name="distilgpt2",
        output_dir=output_dir,
        dataset_path=dataset_path,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        is_dummy=False,
        max_seq_length=512,
    )

    print(f"Training finished. Model saved to {output_dir}")

    # quick verification
    print('Loading model for generation...')
    tok = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    prompt = 'Source:\nThis short example document describes approaches to manage a fracture in a child.\n\nSummary:\n'
    inputs = tok(prompt, return_tensors='pt').to(device)
    out = model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.7, top_p=0.9)
    print('\n--- GENERATED SAMPLE AFTER LONGER TRAINING ---\n')
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
