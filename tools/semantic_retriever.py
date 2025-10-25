#!/usr/bin/env python3
"""Simple CLI to query the Qdrant collection `llm_engineering_chunks`.

Returns top-K chunks for a query and resolves source links from
`data/artifacts/raw_documents.json` when possible.

Usage: python tools/semantic_retriever.py --query "..." --topk 5
"""

import argparse
import json
from pathlib import Path
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def token_overlap_score(a: str, b: str) -> float:
    import re
    a_toks = set(re.findall(r"\w{3,}", (a or "").lower()))
    b_toks = set(re.findall(r"\w{3,}", (b or "").lower()))
    if not a_toks or not b_toks:
        return 0.0
    overlap = a_toks & b_toks
    # normalize by geometric mean to avoid bias toward short texts
    import math
    return len(overlap) / math.sqrt(max(1, len(a_toks)) * max(1, len(b_toks)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--collection", default="llm_engineering_chunks")
    p.add_argument("--output-json", help="Path to write JSON results")
    p.add_argument("--hybrid-fuzzy-threshold", type=float, default=0.05, help="If no direct hybrid match, attach best hybrid summary with score >= threshold (token-overlap). Set 0 to disable")
    p.add_argument("--hybrid-fuzzy-method", choices=["overlap"], default="overlap", help="Fuzzy matching method to use (currently only 'overlap')")
    args = p.parse_args()

    try:
        from sentence_transformers import SentenceTransformer, util
        from qdrant_client import QdrantClient
    except Exception as e:
        print("Missing required packages. Install sentence-transformers and qdrant-client.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q = QdrantClient(url="http://127.0.0.1:6333")

    q_emb = model.encode(args.query)
    hits = q.search(collection_name=args.collection, query_vector=list(q_emb), limit=args.topk)

    raw_docs_path = Path("data/artifacts/raw_documents.json")
    raw_docs = []
    if raw_docs_path.exists():
        raw_docs = load_json(raw_docs_path)
    # build mapping link -> id for raw docs
    link_to_raw_id = {}
    for d in raw_docs:
        if isinstance(d, dict):
            l = d.get("link")
            rid = d.get("id")
            if l and rid:
                link_to_raw_id[str(l)] = str(rid)

    # load hybrid summaries to attach the paired summary when available
    hybrid_path = Path("data/artifacts/hybrid_summaries.jsonl")
    hybrid_map_by_id = {}
    hybrid_map_by_source_file = {}
    hybrid_list = []
    if hybrid_path.exists():
        try:
            with open(hybrid_path, "r", encoding="utf-8") as hf:
                for line in hf:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    hybrid_list.append(rec)
                    mid = rec.get("matched_raw_doc_id")
                    sf = rec.get("source_file")
                    if mid:
                        hybrid_map_by_id[str(mid)] = rec
                    if sf:
                        hybrid_map_by_source_file[str(sf)] = rec
        except Exception:
            pass

    def resolve_link_from_payload(payload):
        # payload may contain source_file or other info
        if not payload:
            return None
        # prefer explicit source_link
        sl = payload.get("source_link") or payload.get("source")
        if sl:
            for d in raw_docs:
                if d.get("link") == sl or d.get("id") == sl or (isinstance(d.get("link"), str) and sl in d.get("link")):
                    return d.get("link")
            return sl
        sf = payload.get("source_file")
        if sf:
            # try to find in raw_docs
            for d in raw_docs:
                if d.get("link") == sf or d.get("id") == sf or (isinstance(d.get("link"), str) and sf in d.get("link")):
                    return d.get("link")
            return sf
        # fallback: try to fuzzy-match payload content to raw documents content
        content = payload.get("content") or ""
        if content:
            short = content.strip()[:200]
            for d in raw_docs:
                doc_content = (d.get("content") or "")
                if short and short in doc_content:
                    return d.get("link")
        return None

    rows = []
    for i, h in enumerate(hits, start=1):
        payload = h.payload or {}
        content = payload.get("content") or ""
        snippet = content[:400] + ("..." if len(content) > 400 else "")
        link = resolve_link_from_payload(payload) or payload.get("source_file")
        print(f"\nRank {i}  id={h.id}  score={h.score}")
        print(f"Link: {link}")
        print(f"Snippet: {snippet}\n")
        # attempt to attach a hybrid summary using matched_raw_doc_id or source_file
        attached_summary = None
        attached_score = None
        # check payload source_link first
        plink = payload.get("source_link") or payload.get("source_file") or link
        if plink:
            # direct match where hybrid.matched_raw_doc_id is stored as the link
            if str(plink) in hybrid_map_by_id:
                attached_summary = hybrid_map_by_id[str(plink)].get("summary")
                attached_score = 1.0
            else:
                # if we have a raw_doc mapping, map link -> raw_id and try to find hybrid by that id
                mapped_raw_id = link_to_raw_id.get(str(plink))
                if mapped_raw_id and mapped_raw_id in hybrid_map_by_id:
                    attached_summary = hybrid_map_by_id[mapped_raw_id].get("summary")
                    attached_score = 1.0
        # fallback: try matching by payload.source_file
        if not attached_summary and payload.get("source_file") and str(payload.get("source_file")) in hybrid_map_by_source_file:
            attached_summary = hybrid_map_by_source_file[str(payload.get("source_file"))].get("summary")
            attached_score = 1.0

        # Final fallback: fuzzy match against all hybrid summaries (adjustable threshold)
        if not attached_summary and args.hybrid_fuzzy_threshold and args.hybrid_fuzzy_threshold > 0.0 and hybrid_list:
            # compute best token-overlap between this hit content (or snippet) and hybrid summaries
            query_text = content or snippet or ""
            best_score = 0.0
            best_rec = None
            for rec in hybrid_list:
                summ = rec.get("summary") or ""
                if not summ:
                    continue
                if args.hybrid_fuzzy_method == "overlap":
                    sc = token_overlap_score(query_text, summ)
                else:
                    sc = 0.0
                if sc > best_score:
                    best_score = sc
                    best_rec = rec
            if best_rec and best_score >= args.hybrid_fuzzy_threshold:
                attached_summary = best_rec.get("summary")
                attached_score = float(best_score)

        rows.append({
            "rank": i,
            "id": str(h.id),
            "score": float(h.score) if h.score is not None else None,
            "link": link,
            "summary": attached_summary,
            "summary_match_score": attached_score,
            "snippet": snippet,
            "payload": payload,
        })

    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as jf:
            json.dump(rows, jf, ensure_ascii=False, indent=2)
        print(f"Wrote semantic results to {outp}")


if __name__ == "__main__":
    main()
