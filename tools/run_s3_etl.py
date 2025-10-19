
import yaml
import os
from dotenv import load_dotenv
from pipelines.s3_etl import s3_etl_pipeline

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

    # Run pipeline
    s3_etl_pipeline(
        bucket_name=bucket_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        prefix=prefix,
    )
