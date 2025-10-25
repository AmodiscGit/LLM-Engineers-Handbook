#!/usr/bin/env python3
"""Streamlit UI for semantic retriever.

Run with:
  poetry run streamlit run tools/semantic_retriever_ui.py --server.port 8502

Features:
- Query input, top-K, fuzzy threshold slider
- Run retrieval against local Qdrant using sentence-transformers
- Display results table (rank, score, link, summary, match score, snippet)
- Save results to JSON and CSV files and provide download buttons
"""

from __future__ import annotations
import streamlit as st
import json
from pathlib import Path
import csv
import math


def token_overlap_score(a: str, b: str) -> float:
    import re
    a_toks = set(re.findall(r"\w{3,}", (a or "").lower()))
    b_toks = set(re.findall(r"\w{3,}", (b or "").lower()))
    if not a_toks or not b_toks:
        return 0.0
    overlap = a_toks & b_toks
    return len(overlap) / math.sqrt(max(1, len(a_toks)) * max(1, len(b_toks)))


def resolve_link_from_payload(payload, raw_docs):
    if not payload:
        return None
    sl = payload.get("source_link") or payload.get("source")
    if sl:
        for d in raw_docs:
            if d.get("link") == sl or d.get("id") == sl or (isinstance(d.get("link"), str) and sl in d.get("link")):
                return d.get("link")
        return sl
    sf = payload.get("source_file")
    if sf:
        for d in raw_docs:
            if d.get("link") == sf or d.get("id") == sf or (isinstance(d.get("link"), str) and sf in d.get("link")):
                return d.get("link")
        return sf
    content = payload.get("content") or ""
    if content:
        short = content.strip()[:200]
        for d in raw_docs:
            if short and short in (d.get("content") or ""):
                return d.get("link")
    return None


def run_retrieval(query: str, topk: int, threshold: float, use_embeddings: bool = True, collection: str = "llm_engineering_chunks"):
    # lazy imports
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
    except Exception as e:
        st.error(f"Missing packages: {e}. Install sentence-transformers and qdrant-client in the project environment.")
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(url="http://127.0.0.1:6333")

    q_emb = model.encode(query)
    hits = client.search(collection_name=collection, query_vector=list(q_emb), limit=topk)

    raw_docs_path = Path("data/artifacts/raw_documents.json")
    raw_docs = []
    if raw_docs_path.exists():
        with open(raw_docs_path, "r", encoding="utf-8") as f:
            try:
                raw_docs = json.load(f)
            except Exception:
                raw_docs = []

    hybrid_path = Path("data/artifacts/hybrid_summaries.jsonl")
    hybrid_list = []
    hybrid_map_by_id = {}
    hybrid_map_by_source_file = {}
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
            hybrid_list = []

    # map raw docs link->id
    link_to_raw_id = {}
    for d in raw_docs:
        if isinstance(d, dict):
            l = d.get("link")
            rid = d.get("id")
            if l and rid:
                link_to_raw_id[str(l)] = str(rid)

    results = []
    for i, h in enumerate(hits, start=1):
        payload = h.payload or {}
        content = payload.get("content") or ""
        snippet = content[:400] + ("..." if len(content) > 400 else "")
        link = resolve_link_from_payload(payload, raw_docs) or payload.get("source_file")

        attached_summary = None
        attached_score = None
        plink = payload.get("source_link") or payload.get("source_file") or link
        if plink:
            if str(plink) in hybrid_map_by_id:
                attached_summary = hybrid_map_by_id[str(plink)].get("summary")
                attached_score = 1.0
            else:
                mapped_raw_id = link_to_raw_id.get(str(plink))
                if mapped_raw_id and mapped_raw_id in hybrid_map_by_id:
                    attached_summary = hybrid_map_by_id[mapped_raw_id].get("summary")
                    attached_score = 1.0

        if not attached_summary and payload.get("source_file") and str(payload.get("source_file")) in hybrid_map_by_source_file:
            attached_summary = hybrid_map_by_source_file[str(payload.get("source_file"))].get("summary")
            attached_score = 1.0

        # fuzzy fallback
        if not attached_summary and threshold and threshold > 0.0 and hybrid_list:
            query_text = content or snippet or query
            best_score = 0.0
            best_rec = None
            for rec in hybrid_list:
                summ = rec.get("summary") or ""
                if not summ:
                    continue
                sc = token_overlap_score(query_text, summ)
                if sc > best_score:
                    best_score = sc
                    best_rec = rec
            if best_rec and best_score >= threshold:
                attached_summary = best_rec.get("summary")
                attached_score = float(best_score)

        results.append(
            {
                "rank": i,
                "id": str(h.id),
                "score": float(h.score) if getattr(h, "score", None) is not None else None,
                "link": link,
                "summary": attached_summary,
                "summary_match_score": attached_score,
                "snippet": snippet,
                "payload": payload,
            }
        )

    return results


def save_json(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "score", "link", "summary_match_score", "summary", "snippet", "id"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})


def main():
    st.set_page_config(page_title="Semantic Retriever UI", layout="wide")
    st.title("Semantic Retriever (Qdrant) — UI")

    with st.sidebar.form("params"):
        query = st.text_area("Query", value="methods to fix a fracture", height=120)
        topk = st.slider("Top K", min_value=1, max_value=50, value=8)
        threshold = st.slider("Hybrid fuzzy threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        collection = st.text_input("Qdrant collection", value="llm_engineering_chunks")
        out_json = st.text_input("Output JSON path", value="data/artifacts/semantic_results_ui.json")
        out_csv = st.text_input("Output CSV path", value="data/artifacts/semantic_results_ui.csv")
        run = st.form_submit_button("Run retrieval")

    if run:
        with st.spinner("Running retrieval..."):
            rows = run_retrieval(query=query, topk=topk, threshold=threshold, collection=collection)

        st.success(f"Retrieved {len(rows)} results")

        # show table
        import pandas as pd

        df = pd.DataFrame(rows)
        if "snippet" in df.columns:
            df["snippet"] = df["snippet"].str.replace("\n", " ")
        st.dataframe(df[["rank", "score", "link", "summary_match_score", "summary", "snippet"]])

        # save buttons
        if st.button("Save JSON"):
            save_json(Path(out_json), rows)
            st.success(f"Wrote JSON to {out_json}")

        if st.button("Save CSV"):
            save_csv(Path(out_csv), rows)
            st.success(f"Wrote CSV to {out_csv}")

        # provide download
        st.download_button("Download JSON", json.dumps(rows, ensure_ascii=False, indent=2), file_name=Path(out_json).name, mime="application/json")
        # CSV download
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["rank", "score", "link", "summary_match_score", "summary", "snippet", "id"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})
        st.download_button("Download CSV", buf.getvalue(), file_name=Path(out_csv).name, mime="text/csv")


if __name__ == "__main__":
    main()
