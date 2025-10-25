"""Minimal Streamlit UI for browsing S3-derived artifacts and asking questions.

Run with: poetry run streamlit run tools/streamlit_ui.py
"""
from __future__ import annotations
import streamlit as st
from tools import ui_helpers
import yaml
import os
from dotenv import load_dotenv
import uuid
import traceback


def _extract_link_from_result(r: dict) -> str:
    """Try several common places to find an original source link for a result."""
    if not isinstance(r, dict):
        return ""
    # direct top-level useful fields
    for key in ("link", "source", "source_file", "id"):
        v = r.get(key)
        if v:
            return str(v)

    # metadata field may contain nested payloads
    meta = r.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("link", "source", "source_file", "url"):
            v = meta.get(key)
            if v:
                return str(v)
        # nested payload
        payload = meta.get("payload") or {}
        if isinstance(payload, dict):
            for key in ("link", "source", "source_file", "url"):
                v = payload.get(key)
                if v:
                    return str(v)

    return ""


def _resolve_link_from_local_artifacts(r: dict, snippet: str | None = None) -> str:
    """Try to find a source link by looking up local ETL artefacts when the result lacks one.

    Strategy:
    - If result has an 'id', search data/artifacts/*.json(, .jsonl) for that id and return its 'link'
    - Otherwise, search raw documents for a doc whose content contains a snippet substring and return its link
    """
    import json
    from pathlib import Path

    data_dir = Path("data") / "artifacts"
    if not data_dir.exists():
        return ""

    # try id match first
    rid = r.get("id") or r.get("metadata", {}).get("matched_raw_doc_id")
    if rid:
        for p in data_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                # doc may be a list
                if isinstance(doc, list):
                    for item in doc:
                        if isinstance(item, dict) and (item.get("id") == rid or item.get("matched_raw_doc_id") == rid):
                            link = item.get("link") or item.get("source_file") or item.get("url")
                            if link:
                                return str(link)
                elif isinstance(doc, dict):
                    if doc.get("id") == rid:
                        link = doc.get("link") or doc.get("source_file") or doc.get("url")
                        if link:
                            return str(link)
            except Exception:
                continue

    # fallback: try fuzzy content match in raw_documents.json and hybrid_summaries.jsonl
    snippet_search = None
    if snippet:
        snippet_search = snippet.strip()[:120]

    # check raw_documents.json
    raw_path = data_dir / "raw_documents.json"
    if raw_path.exists():
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # first pass: substring match
            for item in raw:
                try:
                    content = (item.get("content") or "")
                    if snippet_search and snippet_search in content:
                        link = item.get("link") or item.get("source_file") or item.get("url")
                        if link:
                            return str(link).strip()
                except Exception:
                    continue

            # second pass: token overlap heuristic (robust to snippet variations)
            if snippet_search:
                tokens = [t.lower() for t in __import__('re').findall(r"\w+", snippet_search) if len(t) > 3]
                if tokens:
                    best = (None, 0)  # (link, score)
                    for item in raw:
                        try:
                            content = (item.get("content") or "").lower()
                            score = sum(1 for tok in tokens if tok in content)
                            if score > best[1]:
                                link = item.get("link") or item.get("source_file") or item.get("url")
                                if link:
                                    best = (str(link).strip(), score)
                        except Exception:
                            continue
                    if best[0] and best[1] > 0:
                        return best[0]
        except Exception:
            pass

    # check hybrid_summaries.jsonl
    hybrid_path = data_dir / "hybrid_summaries.jsonl"
    if hybrid_path.exists() and snippet_search:
        try:
            with open(hybrid_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        summary = item.get("summary") or ""
                        source_file = item.get("source_file")
                        if snippet_search in summary and source_file:
                            return str(source_file)
                    except Exception:
                        continue
        except Exception:
            pass

    # As a last resort, try to match by a filename token present in the snippet (e.g., "MCQs")
    if snippet:
        try:
            filename_tokens = [t for t in __import__('re').findall(r"[A-Za-z0-9_\-]+", snippet) if len(t) > 2]
            if filename_tokens:
                # search all json artifacts for links containing a token
                for token in filename_tokens:
                    for p in data_dir.glob("*.json"):
                        try:
                            with open(p, 'r', encoding='utf-8') as f:
                                doc = json.load(f)
                            candidates = doc if isinstance(doc, list) else [doc]
                            for item in candidates:
                                link = (item.get('link') or item.get('source_file') or item.get('url') or '')
                                if link and token.lower() in str(link).lower():
                                    return str(link).strip()
                        except Exception:
                            continue
        except Exception:
            pass

    return ""


def _get_content_from_local_artifacts(r: dict, snippet: str | None = None) -> str:
    """Return the full content for a result by looking up local artifacts by id or snippet.

    Falls back to the provided snippet or r.get('snippet').
    """
    import json
    from pathlib import Path

    data_dir = Path("data") / "artifacts"
    if not data_dir.exists():
        return snippet or r.get("snippet", "") or ""

    rid = r.get("id") or r.get("metadata", {}).get("matched_raw_doc_id")
    # try id match
    if rid:
        for p in data_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                items = doc if isinstance(doc, list) else [doc]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("id") == rid or item.get("matched_raw_doc_id") == rid:
                        return item.get("content") or item.get("text") or item.get("summary") or snippet or r.get("snippet", "") or ""
            except Exception:
                continue

    # fallback: search raw_documents.json by snippet substring
    raw_path = data_dir / "raw_documents.json"
    if raw_path.exists() and snippet:
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for item in raw:
                try:
                    content = item.get("content") or item.get("text") or ""
                    if snippet.strip() and snippet.strip()[:120] in (content or ""):
                        return content
                except Exception:
                    continue
        except Exception:
            pass

    return snippet or r.get("snippet", "") or ""


st.set_page_config(page_title="S3 Explorer & QA", layout="wide")

st.title("S3 Explorer & QA")


def _do_search_callback():
    """Callback used by the query text_input to auto-run a search when the query changes."""
    try:
        q = st.session_state.get("query_input", "")
        method = st.session_state.get("search_method", "keyword_fallback")
        k = int(st.session_state.get("search_k", 5))
        if method == "vector_retriever":
            try:
                results = ui_helpers.try_retriever(q, k=k)
            except Exception:
                results = ui_helpers.fallback_keyword_search(q, max_results=k)
        else:
            results = ui_helpers.fallback_keyword_search(q, max_results=k)

        st.session_state["results"] = results
    except Exception as e:
        st.error(f"Search callback failed: {e}")


query = st.text_input("Ask a question about the S3 buckets:", "What are the methods to fix a fracture?", key="query_input", on_change=_do_search_callback)
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Search method")
    # initialize session-state keys used by the auto-search callback
    if "search_method" not in st.session_state:
        st.session_state["search_method"] = "keyword_fallback"
    if "search_k" not in st.session_state:
        st.session_state["search_k"] = 5

    # bind widgets to session state so changes trigger the search callback
    method = st.radio("Use:", ("vector_retriever", "keyword_fallback"), key="search_method", on_change=_do_search_callback)
    k = st.slider("Number of results", min_value=1, max_value=10, value=st.session_state.get("search_k", 5), key="search_k", on_change=_do_search_callback)
    # Score filter controls: allow users to show only results above a threshold
    if "score_filter_threshold" not in st.session_state:
        st.session_state["score_filter_threshold"] = 35
    thresh = st.number_input("Min score to show (inclusive)", min_value=0, value=int(st.session_state["score_filter_threshold"]), step=1)
    st.session_state["score_filter_threshold"] = int(thresh)
    if "score_filter_active" not in st.session_state:
        st.session_state["score_filter_active"] = False
    col_sf = st.columns([1, 1])
    with col_sf[0]:
        if st.button("Apply score filter"):
            st.session_state["score_filter_active"] = True
            st.success(f"Score filter enabled: >= {st.session_state['score_filter_threshold']}")
    with col_sf[1]:
        if st.button("Clear score filter"):
            st.session_state["score_filter_active"] = False
            st.info("Score filter cleared")
    st.markdown("---")
    st.header("S3 actions")
    st.write("You can scan the S3 bucket configured in `configs/s3_etl.yaml` (credentials loaded from `.env.s3`).")
    # Debug option: show raw result payloads in the results pane
    if "show_raw_payloads" not in st.session_state:
        st.session_state["show_raw_payloads"] = False
    show_raw = st.checkbox("Show raw result payloads (debug)", value=st.session_state.get("show_raw_payloads", False))
    st.session_state["show_raw_payloads"] = bool(show_raw)
    if st.button("Scan S3 and find relevant files"):
        with st.spinner("Fetching documents from S3..."):
            try:
                # load credentials and config (matches tools/run_s3_etl.py behavior)
                load_dotenv(".env.s3")
                aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
                aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

                cfg = {"bucket_name": "", "prefix": ""}
                try:
                    with open("configs/s3_etl.yaml", "r") as f:
                        cfg = yaml.safe_load(f) or cfg
                except Exception:
                    # fall through - allow the step to raise if missing
                    pass

                bucket_name = cfg.get("bucket_name", "")
                prefix = cfg.get("prefix", "")

                # Try to call the project's crawl_s3_bucket step directly
                try:
                    from steps.etl.crawl_s3_bucket import crawl_s3_bucket
                except Exception as e:
                    st.error(f"Could not import crawl helper: {e}")
                    raise

                docs = crawl_s3_bucket(bucket_name=bucket_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, prefix=prefix)

                # normalize documents to dicts with id/content/link
                normalized = []
                if isinstance(docs, list):
                    for d in docs:
                        if isinstance(d, dict):
                            cid = d.get("id") or str(uuid.uuid4())
                            content = d.get("content") or d.get("text") or d.get("body") or ""
                            link = d.get("link") or d.get("source_file")
                        else:
                            cid = str(uuid.uuid4())
                            content = d or ""
                            link = None
                        normalized.append({"id": cid, "content": content, "link": link})

                # Decide whether to use semantic ranking or keyword heuristics.
                # If many documents are present, prefer semantic ranking (embedding) for accuracy.
                results = []
                try:
                    if len(normalized) > 20:
                        # Attempt semantic ranking using sentence-transformers
                        try:
                            from sentence_transformers import SentenceTransformer, util
                        except Exception:
                            SentenceTransformer = None

                        if SentenceTransformer is not None:
                            model = SentenceTransformer('all-MiniLM-L6-v2')
                            docs_texts = [(d.get('content') or '') for d in normalized]
                            # compute embeddings
                            doc_embs = model.encode(docs_texts, convert_to_tensor=True)
                            query_emb = model.encode(query, convert_to_tensor=True)
                            # cosine similarities
                            sims = util.cos_sim(query_emb, doc_embs)[0].cpu().tolist()
                            scored = []
                            for doc, sim in zip(normalized, sims):
                                if sim and sim > 0:
                                    text = doc.get('content') or ''
                                    # create a snippet around the first match of a keyword or start
                                    snippet = text[:300]
                                    scored.append({"id": doc.get("id"), "source": doc.get("link") or f"s3://{bucket_name}", "snippet": snippet.strip(), "score": float(sim), "metadata": {"link": doc.get("link") or ""}})
                            results = sorted(scored, key=lambda x: x["score"], reverse=True)[:k]
                        else:
                            # sentence-transformers not installed: fall back to keyword
                            raise RuntimeError("sentence-transformers unavailable")
                    else:
                        raise RuntimeError("few_docs")
                except Exception:
                    # Fallback: Score each document against the current query using keyword heuristic
                    tokens = [t.lower() for t in __import__('re').findall(r"\w+", query) if len(t) > 2]
                    scored = []
                    for doc in normalized:
                        text = (doc.get("content") or "")
                        low = text.lower()
                        score = sum(low.count(tok) for tok in tokens) if tokens else 0
                        if score > 0:
                            snippet_start = low.find(tokens[0]) if tokens else 0
                            if snippet_start == -1:
                                snippet = text[:300]
                            else:
                                snippet = text[max(0, snippet_start - 80): snippet_start + 220]
                            scored.append({"id": doc.get("id"), "source": doc.get("link") or f"s3://{bucket_name}", "snippet": snippet.strip(), "score": score, "metadata": {"link": doc.get("link") or ""}})
                    results = sorted(scored, key=lambda x: x["score"], reverse=True)[:k]
                st.session_state["results"] = results
                st.success(f"Found {len(results)} candidate files from S3")
            except Exception as e:
                st.error(f"S3 scan failed: {e}\n{traceback.format_exc()}")
    if st.button("Search"):
        with st.spinner("Searching..."):
            if method == "vector_retriever":
                try:
                    results = ui_helpers.try_retriever(query, k=k)
                except Exception as e:
                    st.warning(f"Retriever unavailable, falling back to keyword search: {e}")
                    results = ui_helpers.fallback_keyword_search(query, max_results=k)
            else:
                results = ui_helpers.fallback_keyword_search(query, max_results=k)

            st.session_state["results"] = results

with col2:
    st.header("Results & Summaries")
    results = st.session_state.get("results", [])
    # apply score filter if active
    if st.session_state.get("score_filter_active"):
        try:
            thr = int(st.session_state.get("score_filter_threshold", 0))
            filtered = [r for r in results if (r.get("score") is not None and float(r.get("score", 0)) >= thr)]
            results = filtered
        except Exception:
            # if something goes wrong, leave results unchanged
            pass
    if not results:
        st.info("No results yet — run a search on the left.")
    else:
        # Selection controls
        if "select_all" not in st.session_state:
            st.session_state["select_all"] = False

        col_select_controls = st.columns([1, 1])
        with col_select_controls[0]:
            if st.button("Select all"):
                st.session_state["select_all"] = True
        with col_select_controls[1]:
            if st.button("Clear selection"):
                st.session_state["select_all"] = False

        # Render results with checkboxes
        for i, r in enumerate(results, 1):
            rid = r.get("id") or f"result_{i}"
            st.subheader(f"Result {i} — score: {r.get('score')}")
            metadata = r.get("metadata", {}) or {}
            st.write(metadata)
            # find the best candidate for a source link
            source_link = _extract_link_from_result(r)
            # if link missing/empty or it's only a bucket (e.g., 's3://manna-public'), try resolving
            try:
                need_resolve = False
                if not source_link:
                    need_resolve = True
                else:
                    # detect bucket-only s3://... (no object key)
                    if isinstance(source_link, str) and source_link.lower().startswith("s3://"):
                        rest = source_link.split("s3://", 1)[1]
                        if "/" not in rest or rest.strip().endswith(":"):
                            need_resolve = True

                if need_resolve:
                    resolved = _resolve_link_from_local_artifacts(r, snippet=r.get("snippet"))
                    if resolved:
                        source_link = resolved
            except Exception:
                pass

            if source_link:
                sl = str(source_link)
                if sl.lower().startswith("http"):
                    st.markdown(f"**Source:** [{sl}]({sl})")
                elif sl.lower().startswith("s3://"):
                    # render s3 console link (best effort) and, when possible, a raw https object URL
                    try:
                        # prefer quote_plus so spaces become '+' which many S3 UIs accept
                        from urllib.parse import quote_plus

                        _, rest = sl.split("s3://", 1)
                        if "/" in rest:
                            bucket, key = rest.split("/", 1)
                        else:
                            bucket, key = rest, ""

                        # try to read region from env (AWS_REGION or AWS_DEFAULT_REGION)
                        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_S3_REGION") or ""

                        # build console link; include region param when available for more accurate console landing
                        # For human-friendly raw URLs we preserve '/' in the key and replace spaces with '+'
                        from urllib.parse import quote
                        if key:
                            encoded_key = quote(key, safe='/').replace('%20', '+')
                        else:
                            encoded_key = ""

                        if region:
                            console = f"https://s3.console.aws.amazon.com/s3/object/{bucket}?region={region}&prefix={encoded_key}"
                        else:
                            console = f"https://s3.console.aws.amazon.com/s3/object/{bucket}?prefix={encoded_key}"

                        # raw HTTPS object URL (public buckets). If region known, prefer region-specific endpoint.
                        if key:
                            if region:
                                raw = f"https://{bucket}.s3.{region}.amazonaws.com/{encoded_key}"
                            else:
                                raw = f"https://{bucket}.s3.amazonaws.com/{encoded_key}"
                        else:
                            raw = f"https://{bucket}.s3.amazonaws.com/"

                        st.markdown(f"**Source (S3)** — [Open in AWS Console]({console}) | [Open raw object]({raw})")
                        st.text_input("S3 URI (copy)", value=sl, key=f"s3_copy_{rid}")
                    except Exception:
                        st.text_input("S3 URI (copy)", value=sl, key=f"s3_copy_{rid}")
                else:
                    input_key = f"source_input_{rid}"
                    st.text_input("Source link (copy)", value=sl, key=input_key)
            # ensure unique key per text_area to avoid duplicate element id errors
            ta_key = f"snippet_{rid}"
            st.text_area("Snippet", r.get("snippet", ""), height=150, key=ta_key)
            # checkbox key: use id when available to persist selection across reruns
            key = f"select_{rid}"
            default = bool(st.session_state.get("select_all", False))
            selected = st.checkbox("Select for combined summary", value=default, key=key)
            # individual quick summary (unique key via button id)
            summarize_btn_key = f"summarize_btn_{rid}"
            if st.button(f"Summarize {i}", key=summarize_btn_key):
                with st.spinner("Summarizing with provenance..."):
                    # try to fetch full content for a better summary
                    full_text = _get_content_from_local_artifacts(r, snippet=r.get("snippet"))
                    fake_result = {"id": r.get("id"), "snippet": full_text, "metadata": {"link": r.get("metadata", {}).get("link") or r.get("source") or r.get("link")}}
                    summary = ui_helpers.summarize_with_provenance(query, [fake_result], max_sources=1)
                    st.success(summary)

            # optional raw payload debug view
            if st.session_state.get("show_raw_payloads", False):
                exp_key = f"raw_expander_{rid}"
                # NOTE: some Streamlit versions do not accept a `key` arg for expander
                with st.expander("Show raw payload", expanded=False):
                    try:
                        import json as _json

                        st.code(_json.dumps(r, indent=2, ensure_ascii=False), language="json")
                    except Exception:
                        st.text(str(r))

        # Summarize selected files action
        if st.button("Summarize selected files"):
            # gather selected results in original order
            selected_results = []
            for r in results:
                rid = r.get("id") or ""
                key = f"select_{rid}"
                if st.session_state.get(key, False):
                    selected_results.append(r)

            if not selected_results:
                st.warning("No files selected — check the boxes for files you want summarized.")
            else:
                with st.spinner("Building combined provenance-backed summary from selected files..."):
                    combined = ui_helpers.summarize_with_provenance(query, selected_results, max_sources=len(selected_results))
                    st.success(combined)

        if st.button("Summarize with provenance"):
            with st.spinner("Building provenance-backed summary..."):
                summary = ui_helpers.summarize_with_provenance(query, results, max_sources=k)
                st.success(summary)
