#!/usr/bin/env python3
"""Rank hybrid summary -> source pairs by relevance to a query.

Creates `data/artifacts/hybrid_summaries.jsonl` (by invoking
`tools/create_hybrid_dataset.py`) if it doesn't exist. Then ranks the
hybrid records by similarity to the provided query and prints the top-K
results including the original s3:// link resolved from
`data/artifacts/raw_documents.json`.

Usage: python tools/rank_hybrid.py --query "your question" [--topk 10]
"""

import argparse
import json
import os
import sys
import math
import re
from pathlib import Path


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def token_overlap_score(a: str, b: str) -> float:
    a_toks = set(re.findall(r"\w{3,}", (a or "").lower()))
    b_toks = set(re.findall(r"\w{3,}", (b or "").lower()))
    if not a_toks or not b_toks:
        return 0.0
    overlap = a_toks & b_toks
    # normalized by geometric mean of sizes to reduce bias toward short texts
    return len(overlap) / math.sqrt(max(1, len(a_toks)) * max(1, len(b_toks)))


def compute_scores(hybrid, query, use_embeddings=False):
    # Try to use sentence-transformers if requested and available
    if use_embeddings:
        try:
            from sentence_transformers import SentenceTransformer, util

            model = SentenceTransformer("all-MiniLM-L6-v2")
            corpus = [h.get("summary") or "" for h in hybrid]
            corpus_emb = model.encode(corpus, convert_to_tensor=True)
            q_emb = model.encode(query, convert_to_tensor=True)
            sims = util.cos_sim(q_emb, corpus_emb)[0].tolist()
            return sims
        except Exception as e:
            print("Embedding model unavailable or failed, falling back to token-overlap.", file=sys.stderr)

    # fallback: token overlap
    scores = []
    for h in hybrid:
        txt = (h.get("summary") or "") + "\n" + (h.get("source") or "")
        scores.append(token_overlap_score(query, txt))
    return scores


def resolve_raw_link(matched_id, raw_docs):
    if matched_id is None:
        return None
    # raw_docs is a list; try to match by id or link
    for d in raw_docs:
        if d.get("id") == matched_id:
            return d.get("link")
        if d.get("link") == matched_id:
            return d.get("link")
    # fallback: if matched_id looks like a substring of link
    for d in raw_docs:
        if isinstance(d.get("link"), str) and matched_id in d.get("link"):
            return d.get("link")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="Query/question to rank hybrid summaries against")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--use-embeddings", action="store_true", help="Use sentence-transformers embeddings if available")
    p.add_argument("--out-json", help="Path to save ranked results as JSON")
    p.add_argument("--out-csv", help="Path to save ranked results as CSV")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    hybrid_path = repo_root / "data" / "artifacts" / "hybrid_summaries.jsonl"
    raw_docs_path = repo_root / "data" / "artifacts" / "raw_documents.json"

    # create hybrid file if missing by invoking the existing script
    if not hybrid_path.exists():
        print("Hybrid dataset not found; creating it using tools/create_hybrid_dataset.py...")
        script = repo_root / "tools" / "create_hybrid_dataset.py"
        if not script.exists():
            print("create_hybrid_dataset.py not found. Cannot build hybrid dataset.", file=sys.stderr)
            sys.exit(1)
        # run the script in-process to avoid subprocess overhead
        try:
            import runpy

            runpy.run_path(str(script), run_name="__main__")
        except SystemExit:
            # script may call sys.exit; ignore
            pass
        except Exception as e:
            print(f"Failed to run create_hybrid_dataset.py: {e}", file=sys.stderr)
            sys.exit(1)

    if not hybrid_path.exists():
        print(f"Expected hybrid file at {hybrid_path} but it does not exist.", file=sys.stderr)
        sys.exit(1)

    hybrid = load_jsonl(hybrid_path)
    raw_docs = []
    if raw_docs_path.exists():
        raw_docs = load_json(raw_docs_path)

    print(f"Loaded {len(hybrid)} hybrid records and {len(raw_docs)} raw docs.")

    scores = compute_scores(hybrid, args.query, use_embeddings=args.use_embeddings)

    # attach scores and resolved links
    for h, s in zip(hybrid, scores):
        h["score"] = float(s)
        h["resolved_link"] = resolve_raw_link(h.get("matched_raw_doc_id"), raw_docs)

    # sort by score desc
    hybrid_sorted = sorted(hybrid, key=lambda x: x.get("score", 0.0), reverse=True)

    topk = hybrid_sorted[: args.topk]

    # print results
    rows = []
    for rank, rec in enumerate(topk, start=1):
        summary = (rec.get("summary") or "").strip().replace("\n", " ")
        snippet = summary[:400] + ("..." if len(summary) > 400 else "")
        link = rec.get("resolved_link") or rec.get("source_file") or rec.get("matched_raw_doc_id")
        print(f"\nRank {rank}  score={rec.get('score'):.4f}")
        print(f"Link: {link}")
        print(f"Summary snippet: {snippet}\n")
        rows.append(
            {
                "rank": rank,
                "score": float(rec.get("score", 0.0)),
                "link": link,
                "summary": summary,
                "matched_raw_doc_id": rec.get("matched_raw_doc_id"),
            }
        )

    # optionally save
    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as jf:
            json.dump(rows, jf, ensure_ascii=False, indent=2)
        print(f"Wrote JSON output to {outp}")

    if args.out_csv:
        import csv

        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=["rank", "score", "link", "matched_raw_doc_id", "summary"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote CSV output to {outp}")


if __name__ == "__main__":
    main()
