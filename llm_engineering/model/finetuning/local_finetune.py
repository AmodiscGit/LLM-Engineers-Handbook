from typing import Tuple
import os

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from torch.utils.data import DataLoader, Dataset

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

### Change these levers to tweak the training process !!!!!!!!!!!!!!!!!!!!!!

## eg: num_train_epochs (e.g., 3, 5, 10)
# per_device_train_batch_size (increase if GPU permits)
# learning_rate (try 3e-5 or 1e-5)
# is_dummy=False to use full dataset
# max_seq_length if you need longer context

def finetune_local(
    model_name: str,
    output_dir: str,
    dataset_path: str,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 5e-5,
    is_dummy: bool = False,
    max_seq_length: int = 512,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """A lightweight local finetune using Hugging Face Trainer.

    This is intended for small-scale local experiments (CPU / small GPU). It uses a small causal LM
    (distilgpt2 by default) and saves the final model and tokenizer to ``output_dir``.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Local dataset_path '{dataset_path}' does not exist.")

    # load json file (supports jsonlines or single json list)
    ds = load_dataset("json", data_files=dataset_path, split="train")

    # normalize to a `text` field suitable for causal LM training
    def make_text(example):
        if "instruction" in example and "output" in example:
            return {"text": alpaca_template.format(example["instruction"], example["output"])}
        if "text" in example:
            return {"text": example["text"]}
        # fallback: join all values
        return {"text": " ".join([str(v) for v in example.values()])}

    ds = ds.map(make_text, remove_columns=ds.column_names)

    if is_dummy:
        n = min(200, len(ds))
        ds = ds.select(range(n))

    # choose a small default model to make local runs feasible
    if model_name is None:
        model_name = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_length)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    # Minimal PyTorch training loop to avoid importing HF Trainer and its
    # integrations (which can pull in heavy, platform-dependent packages).
    class TokenizedDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings['input_ids'])

        def __getitem__(self, idx):
            # Return plain Python lists here; the collate function will pad and
            # convert to tensors using the tokenizer.pad utility.
            return {k: v[idx] for k, v in self.encodings.items()}

    model = AutoModelForCausalLM.from_pretrained(model_name)
    # resize embeddings if we added a pad token
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    # prepare dataloader
    encodings = {k: tokenized[k] for k in tokenized.features.keys() if k in tokenized.column_names or k in tokenized}
    # The datasets object exposes columns; we'll build encodings dict with input_ids and attention_mask
    encodings = {k: tokenized[k] for k in tokenized.column_names if k in tokenized.column_names and k in tokenized.features}
    # But easier: rely on tokenized to return 'input_ids' and 'attention_mask' keys
    if 'input_ids' not in tokenized.column_names:
        # tokenized is a Dataset; create encodings via tokenizer.pad on batches
        input_ids = [x['input_ids'] for x in tokenized]
        attention_mask = [x['attention_mask'] for x in tokenized]
        encodings = {'input_ids': input_ids, 'attention_mask': attention_mask}
    else:
        encodings = {k: tokenized[k] for k in ['input_ids', 'attention_mask']}

    dataset = TokenizedDataset(encodings)
    def collate_fn(batch: list[dict]):
        # batch is a list of dicts with keys 'input_ids' and 'attention_mask',
        # where values are lists of ints. Use the tokenizer to pad and return
        # PyTorch tensors.
        return tokenizer.pad(batch, padding=True, return_tensors='pt')

    dataloader = DataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_train_epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch+1}/{num_train_epochs} - avg loss: {avg_loss:.4f}")

    # save final model/tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer
