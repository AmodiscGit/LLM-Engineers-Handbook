"""Helpers for the interactive UI: retrieval, fallback search and summarization.

The helpers try to reuse the repo's ContextRetriever and call_llm_service when
available. If the vector DB or inference stack is not configured, they fall back
to keyword scan over local artifact files (`data/artifacts/raw_documents.json` and
`output/all_cleaned_summaries.json`).
"""
from __future__ import annotations
import json
import os
import re
from typing import List, Dict

DEFAULT_RAW = "data/artifacts/raw_documents.json"
DEFAULT_SUMMARIES = "output/all_cleaned_summaries.json"


def _load_json(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            # try jsonl
            f.seek(0)
            data = [json.loads(l) for l in f if l.strip()]
    return data


def fallback_keyword_search(query: str, max_results: int = 10) -> List[Dict]:
    """Simple keyword-based search over local artifacts.

    Returns list of dicts: {'id','source','snippet','score','metadata'}
    """
    tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    if not tokens:
        return []

    candidates = []
    for path in (DEFAULT_RAW, DEFAULT_SUMMARIES):
        docs = _load_json(path)
        for d in docs:
            # content fields may vary
            content = d.get("content") or d.get("text") or d.get("summary") or d.get("body") or ""
            if not isinstance(content, str):
                continue
            text = content.lower()
            score = sum(text.count(tok) for tok in tokens)
            if score > 0:
                snippet_start = text.find(tokens[0])
                if snippet_start == -1:
                    snippet = content[:300]
                else:
                    snippet = content[max(0, snippet_start - 80): snippet_start + 220]

                candidates.append(
                    {
                        "id": d.get("id") or d.get("source_file") or d.get("link") or None,
                        "source": path,
                        "snippet": snippet.strip(),
                        "score": score,
                        "metadata": {k: v for k, v in d.items() if k in ("link", "source_file")},
                    }
                )

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:max_results]
    return candidates


def try_retriever(query: str, k: int = 5) -> List[Dict]:
    """Try to use ContextRetriever from the repo. Returns same shaped list as fallback.

    If the vector DB or retriever isn't configured, raises ImportError.
    """
    try:
        from llm_engineering.application.rag.retriever import ContextRetriever
    except Exception as e:
        raise ImportError("ContextRetriever unavailable or misconfigured") from e

    retriever = ContextRetriever(mock=False)
    docs = retriever.search(query, k=k)
    results = []
    for doc in docs:
        # EmbeddedChunk likely exposes payload/text attributes; be defensive
        text = getattr(doc, "content", None) or getattr(doc, "text", None) or getattr(doc, "payload", {}).get("content") if hasattr(doc, "payload") else None
        text = text or str(doc)
        metadata = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        results.append({"id": getattr(doc, "id", None), "source": getattr(doc, "source", None), "snippet": text[:500], "score": None, "metadata": metadata})
    return results


def summarize_text(text: str, prefer_remote: bool = False) -> str:
    """Summarize text.

    Behavior:
    - By default we DO NOT call the repo remote inference (SageMaker) because that
      may be unavailable or require cloud permissions. Use an extractive fallback.
    - If the environment variable USE_REMOTE_INFERENCE is set to truthy, we will
      attempt to call the repo `call_llm_service` as before.
    """
    use_remote = os.getenv("USE_REMOTE_INFERENCE", "false").lower() in ("1", "true", "yes") or prefer_remote
    if use_remote:
        try:
            from llm_engineering.infrastructure.inference_pipeline_api import call_llm_service
            prompt = f"Summarize the following text in 3 short bullet points:\n\n{text[:4000]}"
            return call_llm_service(prompt, context=None)
        except Exception:
            # Fall through to extractive fallback if remote fails
            pass

    # Extractive fallback: first 3 sentences (safe and fast, no external deps)
    sentences = re.split(r"(?<=[.!?])\\s+", text.strip())
    return "\n".join(s for s in sentences if s)[:2000]


def summarize_with_provenance(query: str, results: List[Dict], max_sources: int = 5) -> str:
    """Create a short provenance-backed extractive summary from search results.

    The function is intentionally extractive and deterministic (no remote LLM calls)
    so it works offline and avoids cloud permissions. It will:
    - select up to `max_sources` documents from `results`
    - extract sentences from each document that contain query keywords
    - produce an overall short summary and a numbered list of source-backed bullets

    Returns a string suitable for display in the UI.
    """
    if not results:
        return "No documents to summarize."

    # tokens from query for simple matching
    tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]

    sources = results[:max_sources]
    collected_sentences = []
    provenance_entries = []

    for i, src in enumerate(sources, start=1):
        text = src.get("snippet") or src.get("content") or src.get("metadata", {}).get("content") or ""
        # make sure it's a simple string
        if not isinstance(text, str) or not text.strip():
            continue

        # split into sentences and pick those containing tokens
        sents = re.split(r"(?<=[.!?])\\s+", text.strip())
        matched = [s for s in sents if any(tok in s.lower() for tok in tokens)]
        if not matched:
            # fallback: take first two sentences
            matched = sents[:2]

        # dedupe matched sentences
        seen = set()
        matched_clean = []
        for s in matched:
            t = s.strip()
            if t and t not in seen:
                matched_clean.append(t)
                seen.add(t)

        if matched_clean:
            # record for overall summary
            collected_sentences.extend(matched_clean)

            # provenance info: prefer explicit link/filename
            link = src.get("metadata", {}).get("link") or src.get("metadata", {}).get("source_file") or src.get("source") or src.get("id")
            if not link:
                # try common fallback keys
                link = src.get("link") or src.get("source_file") or src.get("id") or "unknown"

            provenance_entries.append({"index": i, "link": link, "snippets": matched_clean})

    # Create a concise overall summary: pick up to 5 unique sentences
    unique_overall = []
    seen = set()
    for s in collected_sentences:
        short = s.strip()
        if short and short not in seen:
            unique_overall.append(short)
            seen.add(short)
        if len(unique_overall) >= 5:
            break

    overall = "\n".join(f"- {s}" for s in unique_overall)

    # Build provenance-backed bullets
    bullets = []
    for entry in provenance_entries:
        i = entry["index"]
        link = entry["link"]
        # join up to 2 snippets per source
        snippets_joined = " ".join(entry["snippets"][:2])
        bullets.append(f"[{i}] {snippets_joined} (source: {link})")

    if not overall and bullets:
        # fallback: use bullets as summary
        return "\n".join(bullets)

    summary = "Summary:\n" + (overall if overall else "No concise sentences extracted.") + "\n\nSources:\n" + "\n".join(bullets)

    # Optionally synthesize with the local fine-tuned model when requested by env var
    use_local = os.getenv("USE_LOCAL_SYNTH", "false").lower() in ("1", "true", "yes")
    if use_local:
        try:
            from tools.local_synthesizer import synthesize_from_provenance

            synthesized = synthesize_from_provenance(bullets)
            return synthesized + "\n\n" + summary
        except Exception:
            # Fall back to extractive summary if local synthesis fails
            pass

    return summary


def simple_cli(query: str):
    """Simple CLI helper used in smoke tests: run retriever or fallback and print results."""
    try:
        results = try_retriever(query, k=5)
        source = "vector_db"
    except Exception:
        results = fallback_keyword_search(query, max_results=5)
        source = "fallback"

    print(f"Using {source}, found {len(results)} results")
    for i, r in enumerate(results, 1):
        print("---")
        print(i, r.get("id"), r.get("source"), r.get("metadata"))
        print(r.get("snippet")[:400].replace("\n", " "))

    if results:
        print("\nSummarizing top result:\n")
        print(summarize_text(results[0]["snippet"]))
