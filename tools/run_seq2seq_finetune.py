#!/usr/bin/env python3
"""Fine-tune a seq2seq model (T5/BART) on the hybrid training data.

Writes model to output/trained_seq2seq
"""
import argparse
import json
import os

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            src = rec.get('source')
            tgt = rec.get('summary')
            if src and tgt:
                data.append({'input': src, 'target': tgt})
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='t5-small')
    parser.add_argument('--data', default='data/artifacts/hybrid_summaries.jsonl')
    parser.add_argument('--output', default='output/trained_seq2seq')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--bs', type=int, default=4)
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )

    data = load_data(args.data)
    if not data:
        raise SystemExit('No data found')

    ds = Dataset.from_list([{'input': d['input'], 'target': d['target']} for d in data])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def preprocess(batch):
        inputs = tokenizer(batch['input'], truncation=True, padding='longest', max_length=512)
        targets = tokenizer(batch['target'], truncation=True, padding='longest', max_length=150)
        inputs['labels'] = targets['input_ids']
        return inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        logging_steps=10,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output)
    print(f"Seq2seq training finished. Model saved to {args.output}")


if __name__ == '__main__':
    main()
