#!/usr/bin/env python3
"""Index chunked data into Qdrant using sentence-transformers embeddings.

Reads `data/processed/chunks.jsonl` (expected fields: id, content, source_file, maybe metadata).
Computes embeddings with `sentence-transformers` (all-MiniLM-L6-v2) and upserts points
to a Qdrant collection named `llm_engineering_chunks`.

Usage: python tools/index_chunks_qdrant.py [--chunks data/processed/chunks.jsonl] [--collection llm_engineering_chunks]
"""

import argparse
import json
from pathlib import Path
import sys


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="data/processed/chunks.jsonl")
    p.add_argument("--collection", default="llm_engineering_chunks")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"Chunks file not found: {chunks_path}")
        sys.exit(1)

    chunks = load_jsonl(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest
    except Exception as e:
        print("Missing required packages: please install sentence-transformers and qdrant-client via poetry.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(url="http://127.0.0.1:6333")

    collection_name = args.collection
    # create collection if missing
    vector_size = model.get_sentence_embedding_dimension()
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )
    except Exception:
        # fall back to create if recreate not supported
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
            )
        except Exception:
            pass

    # upsert in batches
    batch = []
    import uuid

    for i, ch in enumerate(chunks, start=1):
        raw_id = ch.get("id") or f"chunk_{i}"
        # Qdrant requires point IDs to be either integers or UUID strings.
        # Try to use an existing UUID-like id; otherwise deterministically map to a UUID.
        try:
            # accept existing UUID strings
            _ = uuid.UUID(str(raw_id))
            chunk_id = str(raw_id)
        except Exception:
            # create a deterministic UUID5 from the raw id to guarantee valid UUID format
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_OID, str(raw_id)))
        text = ch.get("content") or ch.get("text") or ch.get("summary") or ""
        emb = model.encode(text)
        # include both source_file and source_link when available for robust resolution downstream
        payload = {
            "content": text,
            "source_file": ch.get("source_file"),
            "source_link": ch.get("source_link") or ch.get("link") or ch.get("source") ,
            "chunk_index": ch.get("index") or ch.get("chunk_index") or i,
        }
        batch.append((chunk_id, emb, payload))

        if len(batch) >= args.batch_size:
            ids = [b[0] for b in batch]
            vectors = [b[1].tolist() if hasattr(b[1], 'tolist') else list(b[1]) for b in batch]
            payloads = [b[2] for b in batch]
            client.upsert(collection_name=collection_name, points=[rest.PointStruct(id=ids[j], vector=vectors[j], payload=payloads[j]) for j in range(len(batch))])
            print(f"Upserted {len(batch)} vectors (total {i})")
            batch = []

    if batch:
        ids = [b[0] for b in batch]
        vectors = [b[1].tolist() if hasattr(b[1], 'tolist') else list(b[1]) for b in batch]
        payloads = [b[2] for b in batch]
        client.upsert(collection_name=collection_name, points=[rest.PointStruct(id=ids[j], vector=vectors[j], payload=payloads[j]) for j in range(len(batch))])
        print(f"Upserted final {len(batch)} vectors.")

    print("Indexing complete.")


if __name__ == "__main__":
    main()
