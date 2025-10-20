from zenml import pipeline
from steps.etl.crawl_s3_bucket import crawl_s3_bucket
from steps.summarization.summarize_documents import summarize_documents

@pipeline
def s3_summarization_etl_pipeline(
    bucket_name: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    prefix: str = ""
):
    docs = crawl_s3_bucket(
        bucket_name=bucket_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        prefix=prefix,
    )
    summaries = summarize_documents(docs)
    return summaries