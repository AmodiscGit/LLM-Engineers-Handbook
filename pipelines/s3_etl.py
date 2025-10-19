from zenml import pipeline
from steps.etl.crawl_s3_bucket import crawl_s3_bucket

@pipeline
def s3_etl_pipeline(bucket_name: str, aws_access_key_id: str, aws_secret_access_key: str, prefix: str = ""):
    scraped_documents = crawl_s3_bucket(
        bucket_name=bucket_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        prefix=prefix,
    )
    return scraped_documents
