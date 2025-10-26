# Workflow helpers and ZenML step wrappers

This folder contains workflow helper scripts that produce on-disk artifacts (JSON/JSONL) and lightweight ZenML "step" wrappers that make it easy to record those artifact-producing operations in ZenML runs.

Overview
- Helper functions (backwards-compatible):
  - `generate_raw_documents(...)` — produces `data/artifacts/raw_documents.json`.
  - `main()` in `merge_and_clean_summaries.py` — produces `output/all_cleaned_summaries.json`.
  - `generate_cleaned_documents(...)` — produces `data/artifacts/cleaned_documents.json`.
  - `generate_hybrid_dataset(...)` — decorated as a ZenML `@step`; produces `data/artifacts/hybrid_summaries.jsonl` and a report JSON.

- ZenML step wrappers (new, prefer for provenance):
  - `generate_raw_documents_step(...)` — wrapper around `generate_raw_documents` that is decorated with `@step` and returns `(out_path, metadata)`.
  - `merge_and_clean_summaries_step(...)` — wrapper around `main()` that is decorated with `@step` and returns `(output_file, metadata)`.
  - `generate_cleaned_documents_step(...)` — wrapper around `generate_cleaned_documents` that is decorated with `@step` and returns `(out_path, metadata)`.

Why use the step wrappers?
- Calling the `*_step` functions runs them under ZenML (they become recorded runs with artifact metadata and reproducible inputs). This captures provenance for dataset construction.
- The original helper functions remain available and behave as before (fast, importable, no ZenML run).
- Each step wrapper returns a small, predictable contract: `(artifact_path, metadata_dict)` which is convenient for downstream wiring and reporting.

Quick examples

1) Run a step *and record* the run in ZenML (recommended when you want lineage):

```bash
poetry run python - <<'PY'
from tools.workflow.run_s3_etl import generate_raw_documents_step

# Calling the step will execute it under ZenML and create a run.
out_path, meta = generate_raw_documents_step(out_dir='data/artifacts', configs_path='configs/s3_etl.yaml', env_file='.env.s3')
print('written:', out_path)
print('meta:', meta)
PY
```

Notes:
- When you call a function decorated with `@step` (e.g. `generate_raw_documents_step`) ZenML will create a run (you'll see a local run recorded in your ZenML stack). This is the preferred path for reproducible artifact generation.

2) Run the helper *without* recording (fast, useful for dev/debug):

```bash
poetry run python - <<'PY'
from tools.workflow.run_s3_etl import generate_raw_documents

# This calls the plain helper and writes the artifact but does not record a ZenML run.
docs = generate_raw_documents(out_dir='data/artifacts', configs_path='configs/s3_etl.yaml', env_file='.env.s3')
print('docs_count:', len(docs))
PY
```

3) Run the hybrid generation step (already a `@step`):

```bash
poetry run python - <<'PY'
from tools.workflow.create_hybrid_dataset import generate_hybrid_dataset

# This will run under ZenML and return (report, path)
report, jsonl_path = generate_hybrid_dataset(
    summaries_path='output/all_cleaned_summaries.json',
    cleaned_docs_path='data/artifacts/cleaned_documents.json',
    raw_docs_path='data/artifacts/raw_documents.json',
)
print(report)
print('jsonl:', jsonl_path)
PY
```

Expected output contract
- Step wrappers: (artifact_path, metadata) — metadata typically contains a `count` integer.
- `generate_hybrid_dataset`: returns `(report_dict, out_jsonl_path)` (report contains paired/unmatched counts and output file path).

Integration notes
- The generator/orchestration script (`tools/workflow/generate_artifacts.py`) can be updated to prefer calling the `*_step` wrappers when you want ZenML-recorded runs, while keeping the helper calls as a fallback for fast local dev.
- The Streamlit UI can expose a toggle to choose "recorded (ZenML)" vs "fast (no ZenML)" and call the appropriate function accordingly.

If you want, I can:
- Update `generate_artifacts.py` to call the new `*_step` wrappers by default (with a --no-zenml flag for the helper fallback), or
- Wire these step calls into the Streamlit UI so users can opt-in to recorded runs from the web UI.

---
Small contact: this README is intentionally concise — tell me if you'd like more examples (e.g., showing how to retrieve ZenML run IDs or artifact metadata) or if you want me to update the generator/UI next.
