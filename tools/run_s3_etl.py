
import yaml
import os
from dotenv import load_dotenv
from pipelines.s3_etl import s3_etl_pipeline
from steps.etl.crawl_s3_bucket import crawl_s3_bucket
import json
import traceback
import uuid
import io
import mimetypes
try:
    import PyPDF2
except Exception:
    PyPDF2 = None
try:
    import docx
except Exception:
    docx = None
import boto3


def _is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


if __name__ == "__main__":
    # Load AWS credentials for S3 ETL from .env.s3 only
    load_dotenv(".env.s3")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    # Load bucket_name and prefix from configs/s3_etl.yaml
    with open("configs/s3_etl.yaml", "r") as f:
        config = yaml.safe_load(f)
    bucket_name = config.get("bucket_name", "")
    prefix = config.get("prefix", "")

    out_dir = os.path.join("data", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw_documents.json")

    documents = None

    # If AWS creds are available, prefer structured S3 fetch so we can attach source URLs/keys
    if aws_access_key_id and aws_secret_access_key:
        try:
            print("Attempting structured fetch from S3 to capture source URLs...")
            s3 = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            paginator = s3.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            s3_docs = []
            for page in page_iterator:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    try:
                        if key.endswith(".txt"):
                            response = s3.get_object(Bucket=bucket_name, Key=key)
                            content = response["Body"].read().decode("utf-8")
                            s3_docs.append({"id": str(uuid.uuid4()), "content": content, "link": f"s3://{bucket_name}/{key}"})
                        elif key.endswith(".pdf") and PyPDF2 is not None:
                            response = s3.get_object(Bucket=bucket_name, Key=key)
                            pdf_bytes = response["Body"].read()
                            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                            pdf_text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
                            s3_docs.append({"id": str(uuid.uuid4()), "content": pdf_text, "link": f"s3://{bucket_name}/{key}"})
                        elif key.endswith(".docx") and docx is not None:
                            response = s3.get_object(Bucket=bucket_name, Key=key)
                            docx_bytes = response["Body"].read()
                            docx_file = io.BytesIO(docx_bytes)
                            doc = docx.Document(docx_file)
                            docx_text = "\n".join([para.text for para in doc.paragraphs])
                            s3_docs.append({"id": str(uuid.uuid4()), "content": docx_text, "link": f"s3://{bucket_name}/{key}"})
                        else:
                            # skip other file types
                            continue
                    except Exception:
                        # continue on per-file read errors
                        traceback.print_exc()
                        continue

            if s3_docs:
                documents = s3_docs
                print(f"Fetched {len(s3_docs)} documents from S3 with source links.")
        except Exception:
            print("Structured S3 fetch failed; will fall back to step/pipeline call.")
            traceback.print_exc()

    # Preferred: call the step function directly (returns a plain list of strings)
    try:
        print("Calling crawl_s3_bucket directly (preferred)...")
        # If we already populated `documents` from structured S3 fetch, skip this call
        if documents is None:
            docs_from_step = crawl_s3_bucket(
                bucket_name=bucket_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                prefix=prefix,
            )
            # crawl_s3_bucket historically returns a list of strings; normalize to dicts
            if isinstance(docs_from_step, list) and docs_from_step and isinstance(docs_from_step[0], str):
                documents = [{"id": str(uuid.uuid4()), "content": c, "link": None} for c in docs_from_step]
            else:
                documents = docs_from_step
        print("Direct call succeeded. Persisting result...")
    except Exception as e:
        print("Direct step call failed, will try running the ZenML pipeline as a fallback.")
        traceback.print_exc()

    # Fallback: run the pipeline (may return a ZenML run object). Try to extract serializable outputs.
    if documents is None:
        try:
            print("Running s3_etl_pipeline (fallback)...")
            result = s3_etl_pipeline(
                bucket_name=bucket_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                prefix=prefix,
            )

            # If the pipeline returned a plain object (list/dict/str) use it.
            if isinstance(result, (list, dict, str)):
                documents = result
            else:
                # Try to read an `outputs` attr (ZenML run responses sometimes expose this)
                outputs = getattr(result, "outputs", None)
                if outputs and isinstance(outputs, dict):
                    # Pick the first serializable output value we find
                    for k, v in outputs.items():
                        if _is_json_serializable(v):
                            documents = v
                            break
                        # Some ZenML artifacts may be small wrapper objects; try their string repr
                        try:
                            documents = str(v)
                            break
                        except Exception:
                            continue

            if documents is None:
                raise RuntimeError(
                    "Could not extract serializable outputs from pipeline run.\n"
                    "If you see this, run the ETL step directly or inspect the ZenML run object."
                )

        except Exception:
            print("Pipeline fallback failed — writing debug info to file and aborting.")
            traceback.print_exc()
            # Write the exception info to the out_path for debugging
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"error": "pipeline_run_failed", "trace": traceback.format_exc()}, f, indent=2)
            raise

    # Normalize documents into a JSON-serializable form and persist
    if isinstance(documents, str):
        # single string -> wrap in list
        documents_to_write = [documents]
    else:
        documents_to_write = documents

    # Final sanity check
    if not _is_json_serializable(documents_to_write):
        print("Result is not JSON serializable. Writing string representation for debugging.")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(str(documents_to_write))
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(documents_to_write, f, ensure_ascii=False, indent=2)
        print(f"Saved ETL output to {out_path}")
