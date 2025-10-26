import json


def test_generate_hybrid_dataset_matches(tmp_path):
    """Create small synthetic summaries and raw docs and assert pairing behavior."""
    from tools.workflow.create_hybrid_dataset import generate_hybrid_dataset

    # Synthetic summaries: first contains a long sentence that is present in doc1 -> should match
    long_sent = (
        "This document explains how to train large language models using curated datasets and careful tuning."
    )

    summaries = [
        {"summary": long_sent + " Extra notes.", "source_file": "s1", "index": 0},
        {"summary": "An unrelated short summary about nothing in particular.", "source_file": "s2", "index": 1},
    ]

    raw_docs = [
        {"id": "doc1", "content": long_sent + " Additional content about training."},
        {"id": "doc2", "content": "This is a small doc about pizza toppings and recipes."},
    ]

    summaries_path = tmp_path / "summaries.json"
    raw_path = tmp_path / "raw.json"
    cleaned_path = tmp_path / "cleaned.json"
    out_jsonl = tmp_path / "hybrid.jsonl"
    report_path = tmp_path / "report.json"

    summaries_path.write_text(json.dumps(summaries, ensure_ascii=False))
    raw_path.write_text(json.dumps(raw_docs, ensure_ascii=False))
    # Provide cleaned_docs equal to raw to avoid invoking other helpers during the test
    cleaned_path.write_text(json.dumps(raw_docs, ensure_ascii=False))

    report, out_path = generate_hybrid_dataset(
        summaries_path=str(summaries_path),
        cleaned_docs_path=str(cleaned_path),
        raw_docs_path=str(raw_path),
        out_jsonl=str(out_jsonl),
        report_path=str(report_path),
    )

    # Basic assertions on the returned report
    assert report["summaries_total"] == 2
    assert report["docs_total"] >= 2
    assert report["paired"] == 1
    assert report["unmatched"] == 1

    # File-level assertions
    assert out_jsonl.exists()
    lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    recs = [json.loads(l) for l in lines]

    # First record should be matched to doc1, second should be unmatched
    assert recs[0]["matched_raw_doc_id"] == "doc1"
    assert recs[1]["matched_raw_doc_id"] is None
