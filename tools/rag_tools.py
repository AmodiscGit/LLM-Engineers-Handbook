"""RAG helper utilities for provenance-backed Q&A.

Usage:
    from tools.rag_tools import generate_answer

    res = generate_answer(query="What is robots.txt?", collection="embedded_articles", topk=5)
    print(res['answer'])
    for s in res['sources']:
        print(s['summary'])
        print(s['link'])

Behavior:
- Queries Qdrant for top-k nearest points in `collection`.
- Loads `data/artifacts/hybrid_summaries.jsonl` to map summaries to source links.
- Builds a context that includes the top summaries and calls OpenAI (if OPENAI_API_KEY is set) to produce a final answer.
- Returns a dict: {
    'answer': str,
    'sources': [{'summary': str, 'link': str, 'match_score': float}],
  }

Note: This is a small utility for development and testing. For production use
add batching, retry, rate-limit handling, and secure credential management.
"""
from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Load environment variables from common dotenv files so OPENAI_API_KEY and other secrets
# present in the repo's .env or .env.s3 are available to this module at import time.
try:
    from dotenv import load_dotenv
    # prefer explicit .env, then .env.s3, then default find
    if Path('.env').exists():
        load_dotenv('.env')
    elif Path('.env.s3').exists():
        load_dotenv('.env.s3')
    else:
        # fallback to any findable dotenv
        load_dotenv()
except Exception:
    # python-dotenv is optional; if missing, environment variables must be set externally
    logger.debug('python-dotenv not available; skipping .env load')


def _load_hybrid_map(hybrid_path: str = "data/artifacts/hybrid_summaries.jsonl") -> Dict[str, Dict[str, Any]]:
    """Load hybrid JSONL and return map keyed by source id or source_file."""
    m = {}
    p = Path(hybrid_path)
    if not p.exists():
        return m
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # try multiple keys so we can lookup by raw id or source_file
                mid = rec.get("matched_raw_doc_id")
                sf = rec.get("source_file")
                if mid is not None:
                    m.setdefault(str(mid), rec)
                if sf is not None:
                    m.setdefault(str(sf), rec)
    except Exception as e:
        logger.exception("Failed to load hybrid summaries: %s", e)
    return m


def _query_qdrant(collection: str, query: str, topk: int = 5):
    """Return search hits from Qdrant (raw records)."""
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
    except Exception as e:
        raise RuntimeError("Missing dependencies for Qdrant/querying: %s" % e)

    model = SentenceTransformer(os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"))
    q_emb = model.encode(query)
    hits = client.search(collection_name=collection, query_vector=list(q_emb), limit=topk)
    return hits


def _call_openai(prompt: str) -> str:
    # prefer the new openai client if available
    try:
        import openai
    except Exception:
        openai = None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or (openai is None):
        # fallback: return the prompt as a crude "answer" so the developer sees context
        logger.info("OPENAI_API_KEY not set or openai package missing; returning extractive context instead of LLM answer.")
        return "\n\n".join([p.strip() for p in prompt.split("\n\n### SOURCE:\n")[:3]])

    try:
        # use ChatCompletion if available
        if hasattr(openai, "ChatCompletion"):
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini") if os.environ.get("OPENAI_MODEL") else "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an assistant that answers questions using the provided source summaries and always includes provenance links with the answer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            return resp["choices"][0]["message"]["content"].strip()
        else:
            # older API
            openai.api_key = api_key
            resp = openai.Completion.create(
                engine=os.environ.get("OPENAI_MODEL", "text-davinci-003"),
                prompt=prompt,
                temperature=0.0,
                max_tokens=512,
            )
            return resp["choices"][0]["text"].strip()
    except Exception as e:
        logger.exception("OpenAI call failed: %s", e)
        # fallback
        return "(OpenAI call failed)" + str(e)


def generate_answer(query: str, collection: str = "llm_engineering_chunks", topk: int = 5, include_summaries: bool = True) -> Dict[str, Any]:
    """Run a RAG-style retrieval and generate an answer with provenance.

    Returns:
      {
        'answer': str,
        'sources': [{'summary': str, 'link': str, 'match_score': float}],
      }
    """
    hybrid_map = _load_hybrid_map()
    try:
        hits = _query_qdrant(collection=collection, query=query, topk=topk)
    except Exception as e:
        logger.exception("Query to Qdrant failed: %s", e)
        return {"answer": "(Qdrant query failed) %s" % e, "sources": []}

    # build source list
    sources: List[Dict[str, Any]] = []
    for h in hits:
        payload = getattr(h, "payload", None) or {}
        # attempt to find a matching hybrid summary, prioritized by matched_raw_doc_id then source_file then source_link
        sf = payload.get("source_file") or payload.get("source_link") or None
        mid = payload.get("matched_raw_doc_id") or None
        candidate = None
        if mid and str(mid) in hybrid_map:
            candidate = hybrid_map.get(str(mid))
        if not candidate and sf and str(sf) in hybrid_map:
            candidate = hybrid_map.get(str(sf))

        # fallback synthesised summary from payload content
        summ = None
        link = None
        if candidate:
            summ = candidate.get("summary")
            # prefer explicit link in candidate then payload
            link = candidate.get("source_link") or candidate.get("source_file") or payload.get("source_link") or payload.get("source_file")
        else:
            # use snippet from payload
            summ = (payload.get("summary") or (payload.get("content") or ""))[:800]
            link = payload.get("source_link") or payload.get("source_file") or None

        sources.append({"summary": summ, "link": link, "match_score": float(getattr(h, "score", 0.0))})

    # Optional reranking by embedding similarity (improves precision of top sources)
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        rerank = True
    except Exception:
        rerank = False

    if rerank and sources:
        try:
            emb_model = SentenceTransformer(os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
            qvec = emb_model.encode(query)
            # compute similarity between query and each source summary
            sims = []
            for s in sources:
                text = (s.get("summary") or "")[:1000]
                vec = emb_model.encode(text)
                # cosine similarity
                sc = float(np.dot(qvec, vec) / (np.linalg.norm(qvec) * max(1e-8, np.linalg.norm(vec))))
                sims.append(sc)
            # attach and sort
            for idx, s in enumerate(sources):
                s["rerank_score"] = sims[idx]
            sources = sorted(sources, key=lambda x: x.get("rerank_score", x.get("match_score", 0.0)), reverse=True)
        except Exception:
            # if rerank fails, keep original order
            pass

    # Build a context prompt that lists the top summaries and links with a few-shot example
    few_shot = []
    # small generic examples to guide the model (kept short)
    few_shot.append({
        "q": "What is robots.txt?",
        "a": "Robots.txt is a website file that instructs web crawlers which pages may be crawled. See source(s): <link>.",
        "sources": ["https://example.com/robots.txt"]
    })

    prompt_parts = [
        "You are a helpful assistant. Answer the question using ONLY the provided source summaries. Always include the list of sources used (link or identifier) at the end under 'Sources:'.",
        "",
    ]

    # insert few-shot examples
    for ex in few_shot:
        prompt_parts.append(f"EXAMPLE QUESTION: {ex['q']}")
        prompt_parts.append(f"EXAMPLE ANSWER: {ex['a']}")
        prompt_parts.append(f"EXAMPLE SOURCES: {', '.join(ex['sources'])}")
        prompt_parts.append("")

    prompt_parts.append(f"QUESTION: {query}")
    prompt_parts.append("")
    for i, s in enumerate(sources, start=1):
        prompt_parts.append(f"### SOURCE {i}\nLink: {s.get('link')}\nSummary:\n{s.get('summary')}\n")

    prompt_parts.append("\nProvide a concise answer (3-6 sentences) and then list Sources: with links.")
    prompt = "\n\n".join(prompt_parts)

    answer = _call_openai(prompt)

    # ensure we include the summaries and links in the returned structure
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    # Quick local smoke example
    q = input("Query: ")
    out = generate_answer(q, collection=os.environ.get("DEFAULT_QDRANT_COLLECTION", "llm_engineering_chunks"), topk=5)
    print("\n=== ANSWER ===\n")
    print(out.get("answer"))
    print("\n=== SOURCES ===\n")
    for s in out.get("sources", []):
        print(s.get("link"))
        print(s.get("summary"))
        print("---")
