#!/usr/bin/env python3
"""Evaluate a saved causal or seq2seq model on the hybrid dataset and compute ROUGE.

Default input: data/artifacts/hybrid_summaries.jsonl (expects fields 'source' and 'summary')
"""
import argparse
import json
import os
import math
from tqdm import tqdm

def load_hybrid(path):
    data = []
    if not os.path.exists(path):
        raise SystemExit(f"Input file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            src = rec.get('source')
            tgt = rec.get('summary')
            if src and tgt:
                data.append({'source': src, 'target': tgt})
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--input', default='data/artifacts/hybrid_summaries.jsonl')
    parser.add_argument('--max_examples', type=int, default=100)
    parser.add_argument('--outfile', default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    import torch
    try:
        from evaluate import load as load_metric
    except Exception:
        load_metric = None

    data = load_hybrid(args.input)
    if not data:
        raise SystemExit('No eval examples found')

    # trim
    if args.max_examples and len(data) > args.max_examples:
        data = data[:args.max_examples]

    # load tokenizer + model (try seq2seq then causal)
    model = None
    tokenizer = None
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Try seq2seq first
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
        is_seq2seq = True
    except Exception:
        # fallback causal
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
            is_seq2seq = False
        except Exception as e:
            raise SystemExit(f"Failed to load model from {args.model_dir}: {e}")

    model.to(device)
    model.eval()

    preds = []
    refs = []
    examples = []

    for rec in tqdm(data, desc='Generating'):
        src = rec['source']
        tgt = rec['target']
        prompt = f"Source:\n{src}\n\nSummary:\n"
        # Determine model context window and reserve space for generation
        try:
            if hasattr(model.config, 'n_ctx'):
                ctx = int(model.config.n_ctx)
            else:
                ctx = int(getattr(model.config, 'max_position_embeddings', 1024))
        except Exception:
            ctx = 1024
        max_new_tokens = 150
        allowed_input_len = max(1, ctx - max_new_tokens)

        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=allowed_input_len).to(device)
        with torch.no_grad():
            try:
                if is_seq2seq:
                    out = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'), max_new_tokens=max_new_tokens)
                else:
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            except Exception as gen_e:
                # On failure, attempt a more aggressive truncation and retry once
                try:
                    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max(64, allowed_input_len//2)).to(device)
                    if is_seq2seq:
                        out = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask'), max_new_tokens=64)
                    else:
                        out = model.generate(**inputs, max_new_tokens=64)
                except Exception:
                    # Give up and record an empty generation
                    out = None

        gen = ""
        if out is not None:
            # For seq2seq models the output is the decoded prediction. For causal models
            # the returned ids include the input prompt followed by generated tokens, so
            # slice off the input length to decode only the newly generated tokens.
            try:
                if is_seq2seq:
                    gen = tokenizer.decode(out[0], skip_special_tokens=True)
                else:
                    input_len = inputs['input_ids'].shape[-1]
                    gen_ids = out[0][input_len:]
                    gen = tokenizer.decode(gen_ids, skip_special_tokens=True)
            except Exception:
                # fallback to naive decode
                gen = tokenizer.decode(out[0], skip_special_tokens=True)
        # If seq2seq model returns full text, potentially includes prompt; try to strip the prompt prefix
        if gen.startswith(prompt):
            gen = gen[len(prompt):].strip()
        preds.append(gen)
        refs.append(tgt)
        examples.append({'source': src, 'pred': gen, 'ref': tgt})

    # compute ROUGE if available
    scores = {}
    if load_metric is not None:
        rouge = load_metric('rouge')
        scores = rouge.compute(predictions=preds, references=refs)

    report = {'model_dir': args.model_dir, 'n': len(preds), 'rouge': scores}
    if args.outfile:
        with open(args.outfile, 'w', encoding='utf-8') as f:
            json.dump({'report': report, 'examples': examples[:20]}, f, indent=2)
        print(f"Wrote report to {args.outfile}")

    print('\nEvaluation report:')
    print(json.dumps(report, indent=2))
    print('\nExample outputs (first 3):')
    for e in examples[:3]:
        print('\n---')
        print('SOURCE:', e['source'][:200].replace('\n',' '))
        print('PRED:', e['pred'][:300].replace('\n',' '))
        print('REF:', e['ref'][:200].replace('\n',' '))


if __name__ == '__main__':
    main()
