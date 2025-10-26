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
import streamlit.components.v1 as components
from pathlib import Path
import csv
import math
import subprocess
import shlex
import datetime
import os


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


def run_retrieval(query: str, topk: int, threshold: float, use_embeddings: bool = True, collection: str = "llm_engineering_chunks", no_etl: bool = False, no_hybrid: bool = False):
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
    else:
        if not no_etl:
            # Try to generate/load raw_documents programmatically using the project's ETL function
            try:
                from tools.workflow.run_s3_etl import generate_raw_documents

                docs = generate_raw_documents()
                if isinstance(docs, list):
                    raw_docs = docs
                else:
                    raw_docs = docs or []
                # attempt to persist the file for other tools
                try:
                    with open(raw_docs_path, "w", encoding="utf-8") as f:
                        json.dump(raw_docs, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                st.info(f"Loaded {len(raw_docs)} raw documents via generate_raw_documents()")
            except Exception as e:
                st.warning(
                    "raw_documents.json not found and generate_raw_documents() failed; continuing without raw doc link resolution."
                )
                st.debug(str(e))
        else:
            st.info("Skipping ETL generation of raw documents (no_etl=True)")

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
    else:
        if not no_hybrid:
            # generate hybrid dataset programmatically (so the UI can work without a prior step)
            try:
                from tools.workflow.create_hybrid_dataset import generate_hybrid_dataset

                hybrid_list = generate_hybrid_dataset()
                # populate lookup maps
                for rec in hybrid_list:
                    mid = rec.get("matched_raw_doc_id")
                    sf = rec.get("source_file")
                    if mid:
                        hybrid_map_by_id[str(mid)] = rec
                    if sf:
                        hybrid_map_by_source_file[str(sf)] = rec
                st.info(f"Generated {len(hybrid_list)} hybrid records via create_hybrid_dataset.generate_hybrid_dataset()")
            except Exception as e:
                st.warning("hybrid_summaries.jsonl not found and generation failed; continuing without hybrid summaries")
                st.debug(str(e))
        else:
            st.info("Skipping hybrid generation (no_hybrid=True)")

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

    # Artifacts we care about
    ARTIFACT_PATHS = [
        "output/all_cleaned_summaries.json",
        "data/artifacts/cleaned_documents.json",
        "data/artifacts/raw_documents.json",
        "data/artifacts/hybrid_summaries.jsonl",
        "data/artifacts/hybrid_summaries_report.json",
    ]

    def check_artifacts():
        missing = []
        for p in ARTIFACT_PATHS:
            if not Path(p).exists():
                missing.append(p)
        return missing

    def run_full_workflow(in_process: bool = True):
        # Prefer running the in-repo generator in-process for faster feedback and to avoid
        # an extra subprocess; fall back to the poetry subprocess if import fails.
        # If requested, attempt an in-process run which is faster and avoids a shell/Poetry subprocess.
        if in_process:
            try:
                # capture stdout/stderr
                import io
                import contextlib
                import traceback

                buf = io.StringIO()
                try:
                    # import the generator from the workflow package
                    from tools.workflow import generate_artifacts
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                        try:
                            generate_artifacts.main()
                            return 0, buf.getvalue()
                        except SystemExit as se:
                            # generator may call sys.exit()
                            code = se.code if isinstance(se.code, int) else 0
                            return code, buf.getvalue()
                        except Exception:
                            traceback.print_exc(file=buf)
                            return 1, buf.getvalue()
                except Exception as imp_e:
                    buf.write(f"In-process run failed: {imp_e}\n")
                    buf.write("Falling back to subprocess 'poetry run python tools/workflow/generate_artifacts.py'\n")
            except Exception:
                # if capturing machinery fails, fall back to plain subprocess
                pass

        # fallback or explicit subprocess run: run via poetry subprocess
        cmd = "poetry run python tools/workflow/generate_artifacts.py"
        try:
            proc = subprocess.run(cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return proc.returncode, proc.stdout
        except Exception as e:
            return 1, str(e)

    with st.sidebar.expander("Artifacts / regenerate", expanded=False):
        missing_now = check_artifacts()
        if not missing_now:
            st.success("All required artifacts present")
        else:
            st.warning(f"Missing {len(missing_now)} artifact(s):")
            for m in missing_now:
                st.write(f"- {m}")

        # Option: let the user choose whether to run the generator in-process or as a subprocess
        run_in_process = st.checkbox("Run generator in-process (faster, uses current environment)", value=True, help="When checked the UI will import and call the in-repo generator directly. Uncheck to force running under 'poetry run' as a subprocess.")

        if st.button("Regenerate artifacts (run full workflow)"):
            with st.spinner("Running full workflow (this may take a while)..."):
                code, output = run_full_workflow(in_process=run_in_process)
            if code == 0:
                st.success("Full workflow completed (exit 0). Re-checking artifacts...")
            else:
                st.error(f"Full workflow finished with exit code {code}")
            # show output and re-check
            st.text_area("Workflow output (stdout/stderr)", output, height=300)
            missing_now = check_artifacts()
            if not missing_now:
                st.success("Artifacts now present")
            else:
                st.warning(f"Still missing: {len(missing_now)} files")

    # Training panel: allow user to kick off HF or PEFT/LoRA training via subprocess (safe, non-blocking option)
    with st.sidebar.expander("Training / Fine-tune (beta)", expanded=False):
        st.markdown("Small UI to start local Hugging Face fine-tuning. Training can be long — use background mode to avoid blocking the UI.")
        tg_exists = Path("tools/train_hf_finetune.py").exists()
        peft_exists = Path("tools/train_peft_lora.py").exists()
        if not tg_exists and not peft_exists:
            st.warning("No local training scripts found (tools/train_hf_finetune.py or tools/train_peft_lora.py).")
        train_backend = st.selectbox("Backend", options=[o for o in ["HF-full" if tg_exists else None, "PEFT-LoRA" if peft_exists else None] if o])
        model_name = st.text_input("Base model", value="distilgpt2")
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=3)
        batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=8)
        use_8bit = False
        if train_backend == "PEFT-LoRA":
            use_8bit = st.checkbox("Load in 8-bit (requires bitsandbytes)", value=False)
        output_dir = st.text_input("Output dir", value="models/hf-finetuned")
        background = st.checkbox("Run in background (recommended)", value=True, help="If checked the training will be launched as a detached subprocess and logs will be written to a file in logs/")
        confirm_train = st.checkbox("I understand training can be long and resource intensive", value=False)

        # Small editor for the fine-tune JSONL dataset so the user can tweak prompts/completions
        st.markdown("---")
        st.markdown("**Edit fine-tune dataset (tools/ft_dataset.jsonl)**")
        ft_path = Path("tools/ft_dataset.jsonl")
        initial_text = ""
        if ft_path.exists():
            try:
                with open(ft_path, "r", encoding="utf-8") as f:
                    initial_text = f.read()
            except Exception:
                initial_text = ""
        else:
            # provide a tiny template
            initial_text = json.dumps({"prompt": "<INSTRUCTION>", "completion": "<RESPONSE>"}, ensure_ascii=False) + "\n"

        edited = st.text_area("ft_dataset.jsonl (one JSON per line)", value=initial_text, height=200)
        if st.button("Validate & Save dataset"):
            # Quick validation: each non-empty line must be valid JSON with prompt/completion
            bad = []
            lines = [l for l in edited.splitlines() if l.strip()]
            parsed = []
            for i, ln in enumerate(lines, start=1):
                try:
                    obj = json.loads(ln)
                    if not (obj.get("prompt") or obj.get("instruction")) or not (obj.get("completion") or obj.get("output") or obj.get("answer")):
                        bad.append((i, "missing prompt or completion fields"))
                    else:
                        parsed.append(obj)
                except Exception as e:
                    bad.append((i, str(e)))

            if bad:
                st.error(f"Validation failed: {len(bad)} problem(s)")
                for row, msg in bad[:10]:
                    st.write(f"Line {row}: {msg}")
                if len(bad) > 10:
                    st.write(f"...and {len(bad)-10} more issues")
            else:
                try:
                    ft_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(ft_path, "w", encoding="utf-8") as f:
                        f.write("\n".join([json.dumps(o, ensure_ascii=False) for o in parsed]) + "\n")
                    st.success(f"Wrote {len(parsed)} records to {ft_path}")
                except Exception as e:
                    st.error(f"Failed to write dataset: {e}")

            # --- Single-record form to append one example to tools/ft_dataset.jsonl ---
            st.markdown("---")
            st.markdown("**Add a single example to the fine-tune dataset**")
            st.info("Use these fields to add one prompt/completion pair to tools/ft_dataset.jsonl. This is helpful for quick iterative edits before training.")

            # Prefill the Question box with a short helpful placeholder
            question_input = st.text_area(
                "Question / Prompt",
                value="",
                height=120,
                placeholder="Question: ...\n\nUse the following sources to answer:\n\nSource 1: s3://...\nSummary: ...\n\nAnswer concisely:",
            )
            completion_input = st.text_area(
                "Answer / Completion",
                value="",
                height=120,
                placeholder="The concise answer text here (what the model should generate).",
            )

            # stacked buttons (one per line)
            append_single = st.button("Append single record to ft_dataset.jsonl", key="append_single")
            clear_fields = st.button("Clear fields", key="clear_fields")

            # If the button doesn't expose a stable aria-label (some Streamlit versions leave it empty),
            # inject a small JS snippet to find the button by its visible text and style it orange.
            try:
                components.html(
                    """
                    <script>
                    (function(){
                      const label = "Append single record to ft_dataset.jsonl";
                      function styleBtn(){
                        const buttons = Array.from(document.querySelectorAll('button'));
                        for(const b of buttons){
                          // normalize inner text and trim
                          const txt = (b.innerText || '').trim();
                          if(txt === label || txt.startsWith(label)){
                            b.style.backgroundColor = 'orange';
                            b.style.color = 'white';
                            b.style.border = 'none';
                            b.style.boxShadow = 'none';
                            b.style.transition = 'background-color 120ms ease-in-out';
                            b.addEventListener('mouseover', ()=> b.style.backgroundColor = 'darkorange');
                            b.addEventListener('mouseout', ()=> b.style.backgroundColor = 'orange');
                          }
                        }
                      }
                      // run after short delay and watch for DOM changes
                      setTimeout(styleBtn, 200);
                      const obs = new MutationObserver(styleBtn);
                      obs.observe(document.body, {childList:true, subtree:true});
                    })();
                    </script>
                    """,
                    height=0,
                )
            except Exception:
                # components may not be available in some environments, silently continue
                pass

            if clear_fields:
                # simply re-rendering with empty defaults will clear (Streamlit will reset on rerun)
                st.experimental_rerun()

            if append_single:
                # Basic validation
                q = (question_input or "").strip()
                c = (completion_input or "").strip()
                if not q:
                    st.error("Question/prompt is empty — please provide text to append.")
                elif not c:
                    st.error("Completion/answer is empty — please provide the target completion.")
                else:
                    # Auto-wrap the question into a standard prompt template if it doesn't already contain an "answer concisely" hint
                    try:
                        import re
                        if not re.search(r"answer\s+concisely", q, flags=re.IGNORECASE):
                            # Ensure we append a double-newline before the hint for readability
                            if not q.endswith("\n"):
                                q = q + "\n\nAnswer concisely:"
                            else:
                                q = q + "\nAnswer concisely:"
                    except Exception:
                        # If regex import fails (very unlikely), fall back to a simple lowercase check
                        if "answer concisely" not in q.lower():
                            q = q + "\n\nAnswer concisely:"

                    rec = {"prompt": q, "completion": c}
                    try:
                        ft_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(ft_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        st.success(f"Appended 1 record to {ft_path}")
                        st.write(rec)
                    except Exception as e:
                        st.error(f"Failed to append record: {e}")

        if st.button("Start training"):
            if not confirm_train:
                st.error("Please check the confirmation checkbox to proceed.")
            else:
                # build command
                data_file = "tools/ft_dataset.jsonl"
                timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                logs_dir = Path("logs")
                logs_dir.mkdir(parents=True, exist_ok=True)
                log_path = logs_dir / f"training_{timestamp}.log"

                if train_backend == "HF-full":
                    script = Path("tools/train_hf_finetune.py")
                    if not script.exists():
                        st.error("HF training script not found.")
                    cmd = f"poetry run python {shlex.quote(str(script))} --data {shlex.quote(data_file)} --model {shlex.quote(model_name)} --output {shlex.quote(output_dir)} --epochs {int(epochs)} --batch_size {int(batch_size)}"
                else:
                    script = Path("tools/train_peft_lora.py")
                    if not script.exists():
                        st.error("PEFT/LoRA script not found.")
                    cmd = f"poetry run python {shlex.quote(str(script))} --data {shlex.quote(data_file)} --model {shlex.quote(model_name)} --output {shlex.quote(output_dir)} --epochs {int(epochs)} --batch_size {int(batch_size)}"
                    if use_8bit:
                        cmd += " --use_8bit"

                st.write("Command:")
                st.code(cmd)

                if background:
                    # Launch detached process and redirect output to log file
                    try:
                        # open log file for append
                        lf = open(log_path, "a", encoding="utf-8")
                        proc = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=lf, env=os.environ.copy())
                        st.success(f"Training started in background (pid={proc.pid}). Logs: {log_path}")
                        st.info("You can tail the log file to monitor progress: e.g. 'tail -f {log_path}'")
                    except Exception as e:
                        st.error(f"Failed to start background training: {e}")
                else:
                    # Run in foreground and capture output
                    with st.spinner("Running training (foreground)... this will block the UI until finished"):
                        try:
                            proc = subprocess.run(cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
                            out = proc.stdout
                            st.text_area("Training output", out, height=400)
                            if proc.returncode == 0:
                                st.success("Training finished (exit 0)")
                            else:
                                st.error(f"Training finished with exit code {proc.returncode}")
                        except Exception as e:
                            st.error(f"Training run failed: {e}")

        # show existing logs
        st.markdown("---")
        st.markdown("Recent training logs:")
        try:
            logs = sorted(Path("logs").glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            for p in logs[:5]:
                st.write(f"- {p} — {p.stat().st_size} bytes")
        except Exception:
            st.write("(no logs yet)")

    with st.sidebar.form("params"):
        query = st.text_area("Query", value="", placeholder="Ask a question about the data", height=120)
        topk = st.slider("Top K", min_value=1, max_value=50, value=8)
        threshold = st.slider("Hybrid fuzzy threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        # Try to discover available Qdrant collections and default to the largest (most points)
        try:
            import urllib.request
            import urllib.parse

            base = "http://127.0.0.1:6333"
            collection_names = ["llm_engineering_chunks"]

            try:
                with urllib.request.urlopen(f"{base}/collections", timeout=2) as resp:
                    data = json.load(resp)
                    collections = [c.get("name") for c in data.get("result", {}).get("collections", []) if c.get("name")]
            except Exception:
                collections = []

            stats = []
            for name in collections:
                try:
                    qname = urllib.parse.quote(name, safe="")
                    with urllib.request.urlopen(f"{base}/collections/{qname}", timeout=2) as r2:
                        info = json.load(r2)
                        pc = info.get("result", {}).get("points_count", 0)
                except Exception:
                    pc = 0
                stats.append((name, int(pc)))

            if stats:
                stats.sort(key=lambda x: x[1], reverse=True)  # largest first
                collection_names = [s[0] for s in stats]

            # show a selectbox with discovered collections, default to the largest
            default_index = 0
            collection = st.selectbox("Qdrant collection", options=collection_names, index=default_index)
        except Exception:
            # fallback: simple text input
            collection = st.text_input("Qdrant collection", value="llm_engineering_chunks")
        run_etl = st.checkbox("Run ETL if missing", value=True, help="If checked the UI will run the project's ETL to generate raw_documents.json when it's missing")
        run_hybrid = st.checkbox("Generate hybrid if missing", value=True, help="If checked the UI will generate hybrid_summaries.jsonl when it's missing")
        out_json = st.text_input("Output JSON path", value="data/artifacts/semantic_results_ui.json")
        out_csv = st.text_input("Output CSV path", value="data/artifacts/semantic_results_ui.csv")
        run = st.form_submit_button("Run retrieval")

    if run:
        with st.spinner("Running retrieval..."):
            rows = run_retrieval(query=query, topk=topk, threshold=threshold, collection=collection, no_etl=not run_etl, no_hybrid=not run_hybrid)

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

        # RAG answer button: run retrieval+RAG to get a provenance-backed answer
        if st.button("Answer with RAG"):
            with st.spinner("Running RAG and generating answer..."):
                try:
                    # lazy import the rag helper
                    from tools.rag_tools import generate_answer

                    rag_res = generate_answer(query=query, collection=collection, topk=topk, include_summaries=True)
                except Exception as e:
                    st.error(f"RAG failed: {e}")
                    rag_res = None

            if rag_res:
                st.markdown("### RAG Answer")
                st.write(rag_res.get("answer"))

                st.markdown("### Sources (provenance)")
                for s in rag_res.get("sources", []):
                    link = s.get("link") or ""
                    summary = s.get("summary") or ""
                    # show link as a clickable markdown and the summary truncated
                    if link:
                        st.markdown(f"- [{link}]({link}) — {summary[:500]}")
                    else:
                        st.markdown(f"- {summary[:500]}")

                # allow saving the RAG result
                if st.button("Save RAG result as JSON"):
                    save_json(Path(f"data/artifacts/rag_result_{query[:40].strip().replace(' ', '_')}.json"), rag_res)
                    st.success("Saved RAG result")

                # allow appending the curated Q/A to annotations for continuous training
                if st.button("Add answer to training annotations (append)"):
                    try:
                        ann_path = Path("tools/annotations.jsonl")
                        ann_path.parent.mkdir(parents=True, exist_ok=True)
                        # Build a compact annotation record
                        sources = []
                        for s in rag_res.get("sources", []):
                            link = s.get("link") or None
                            summary = s.get("summary") or None
                            if link:
                                sources.append(str(link))
                            elif summary:
                                # fallback to short summary text if no link
                                sources.append(summary[:300])

                        record = {
                            "query": query,
                            "answer": rag_res.get("answer"),
                            "sources": sources,
                        }
                        # append as a JSONL line
                        with open(ann_path, "a", encoding="utf-8") as af:
                            af.write(json.dumps(record, ensure_ascii=False) + "\n")
                        st.success(f"Appended annotation to {ann_path}")
                        st.write(record)
                    except Exception as e:
                        st.error(f"Failed to append annotation: {e}")

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
