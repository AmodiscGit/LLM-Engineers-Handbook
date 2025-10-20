import os
from dotenv import load_dotenv
import yaml
from pipelines.s3_summarization_etl_pipeline import s3_summarization_etl_pipeline

if __name__ == "__main__":
    # Load AWS credentials
    load_dotenv(".env.s3")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    # Load bucket_name and prefix from configs/s3_etl.yaml
    with open("configs/s3_etl.yaml", "r") as f:
        config = yaml.safe_load(f)
        bucket_name = config.get("bucket_name", "")
        prefix = config.get("prefix", "")

  
    summaries = s3_summarization_etl_pipeline(
        bucket_name=bucket_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        prefix=prefix,
    )
    print("\n--- Summaries ---")
    for i, summary in enumerate(summaries):
        print(f"Document {i+1} summary:\n{summary}\n{'-'*40}")