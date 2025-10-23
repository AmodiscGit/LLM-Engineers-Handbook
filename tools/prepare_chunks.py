"""Prepare document chunks from local artifact files.

Writes newline-delimited JSON to `data/processed/chunks.jsonl` with fields:
 - id: original document id
 - chunk_id: incremental id per chunk (e.g., {id}_0)
 - text: chunk text
 - source_link: original s3:// link when available
 - source_file: filename or other metadata

Usage:
  poetry run python tools/prepare_chunks.py

This is a character-based chunker (safe for mixed content). Configurable
via CHUNK_SIZE and CHUNK_OVERLAP environment variables.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import hashlib
import re

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

DATA_DIR = Path("data") / "artifacts"
OUT_DIR = Path("data") / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "chunks.jsonl"


def _load_json_safe(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # try jsonl
        items = []
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        except Exception:
            return []
        return items


def normalize_docs():
    docs = []
    if not DATA_DIR.exists():
        print(f"No artifacts dir at {DATA_DIR}")
        return docs

    for p in DATA_DIR.glob("*.json"):
        try:
            data = _load_json_safe(p)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                if not isinstance(item, dict):
                    continue
                doc_id = item.get("id") or item.get("source_file") or hashlib.sha1((item.get("content") or "").encode('utf-8')).hexdigest()
                content = item.get("content") or item.get("text") or item.get("body") or item.get("summary") or ""
                link = item.get("link") or item.get("source_file") or item.get("url") or ""
                docs.append({"id": str(doc_id), "content": content, "link": link})
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue

    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == L:
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def main():
    docs = normalize_docs()
    total_docs = len(docs)
    if total_docs == 0:
        print("No documents found to chunk.")
        return

    written = 0
    by_doc = {}
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for d in docs:
            doc_id = d["id"]
            content = d.get("content") or ""
            link = d.get("link") or ""
            chunks = chunk_text(content)
            by_doc[doc_id] = len(chunks)
            for i, c in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                rec = {"id": doc_id, "chunk_id": chunk_id, "text": c, "source_link": link}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Processed {total_docs} docs -> wrote {written} chunks")
    # show a small sample
    sample = []
    with OUT_FILE.open("r", encoding="utf-8") as f:
        for _ in range(3):
            line = f.readline()
            if not line:
                break
            sample.append(json.loads(line))
    print("Sample chunks:")
    for s in sample:
        print(json.dumps({"chunk_id": s["chunk_id"], "source_link": s.get("source_link")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
