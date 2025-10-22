#!/usr/bin/env python3
"""Ad-hoc indexer: chunk cleaned or raw docs, compute embeddings with
transformers mean-pooling, and bulk-insert into Qdrant collections.

This is a safe alternative when `sentence-transformers` fails to load.

Usage: poetry run python tools/index_to_qdrant_alt.py --limit 20
"""
from __future__ import annotations
import argparse
import json
import os
import uuid
from itertools import islice
from typing import List

import numpy as np
import torch

from llm_engineering.application.preprocessing.dispatchers import ChunkingDispatcher
from llm_engineering.domain.chunks import ArticleChunk
from llm_engineering.domain.embedded_chunks import EmbeddedArticleChunk
from llm_engineering.domain.cleaned_documents import CleanedArticleDocument


def load_candidates(summaries_path: str, raw_path: str) -> List[dict]:
    docs = []
    if os.path.exists(summaries_path):
        with open(summaries_path, 'r', encoding='utf-8') as f:
            arr = json.load(f)
            # summaries may be list of dicts with 'summary' and 'source_file'
            for i, rec in enumerate(arr):
                text = rec.get('summary') or rec.get('text') or ''
                docs.append({'id': str(uuid.uuid4()), 'content': text, 'link': rec.get('source_file', '')})

    if os.path.exists(raw_path):
        with open(raw_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            if isinstance(raw, list):
                for r in raw:
                    docs.append({'id': r.get('id') or str(uuid.uuid4()), 'content': r.get('content', ''), 'link': r.get('link', '')})

    return docs


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # first element of model_output contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).cpu().numpy()


def embed_texts(texts: List[str], model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    embeddings = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embs = mean_pooling(outputs, attention_mask)
        for e in embs:
            embeddings.append(e.tolist())

    return embeddings


def main(limit: int = 20, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    summaries_path = 'output/all_cleaned_summaries.json'
    raw_path = 'data/artifacts/raw_documents.json'

    docs = load_candidates(summaries_path, raw_path)
    if not docs:
        raise SystemExit('No documents found to index. Ensure output/all_cleaned_summaries.json or data/artifacts/raw_documents.json exist.')

    docs = docs[:limit]
    print(f'Indexing {len(docs)} documents (limit={limit})')

    # For each doc create a minimal CleanedArticleDocument and chunk it
    all_chunks = []
    doc_map = {}
    for d in docs:
        doc_id = uuid.UUID(d['id']) if (isinstance(d['id'], str) and len(d['id']) == 36) else uuid.uuid4()
        cleaned = CleanedArticleDocument(
            id=doc_id,
            content=d['content'] or '',
            platform='s3',
            author_id=uuid.uuid4(),
            author_full_name='unknown',
            link=d.get('link', ''),
        )
        chunks = ChunkingDispatcher.dispatch(cleaned)
        for c in chunks:
            all_chunks.append(c)
        doc_map[str(doc_id)] = d

    print(f'Created {len(all_chunks)} chunks')

    # Embed in batches
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        texts = [c.content for c in batch_chunks]
        embs = embed_texts(texts, model_name=model_name)

        embedded_models = []
        for c, emb in zip(batch_chunks, embs):
            embedded = EmbeddedArticleChunk(
                id=c.id,
                content=c.content,
                embedding=emb,
                platform=c.platform,
                link=getattr(c, 'link', ''),
                document_id=c.document_id,
                author_id=c.author_id,
                author_full_name=c.author_full_name,
                metadata=c.metadata,
            )
            embedded_models.append(embedded)

        # Bulk insert into Qdrant
        EmbeddedArticleChunk.bulk_insert(embedded_models)
        print(f'Inserted batch {i//batch_size + 1} ({len(embedded_models)} chunks)')

    print('Indexing complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()
    main(limit=args.limit, model_name=args.model)
