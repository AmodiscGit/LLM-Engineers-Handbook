import boto3
from typing import List
from zenml import step
import io
import mimetypes
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    import docx
except ImportError:
    docx = None

@step
def crawl_s3_bucket(bucket_name: str, aws_access_key_id: str, aws_secret_access_key: str, prefix: str = "") -> List[str]:
    """
    Scrape all .txt, .pdf, and .docx files from the specified S3 bucket and return their contents as a list of strings.
    Args:
        bucket_name (str): Name of the S3 bucket.
        aws_access_key_id (str): AWS access key ID.
        aws_secret_access_key (str): AWS secret access key.
        prefix (str, optional): Prefix for filtering files. Defaults to "".
    Returns:
        List[str]: List of contents from .txt files in the bucket.
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    documents = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".txt"):
                response = s3.get_object(Bucket=bucket_name, Key=key)
                content = response["Body"].read().decode("utf-8")
                documents.append(content)
            elif key.endswith(".pdf") and PyPDF2 is not None:
                response = s3.get_object(Bucket=bucket_name, Key=key)
                pdf_bytes = response["Body"].read()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                pdf_text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
                documents.append(pdf_text)
            elif key.endswith(".docx") and docx is not None:
                response = s3.get_object(Bucket=bucket_name, Key=key)
                docx_bytes = response["Body"].read()
                docx_file = io.BytesIO(docx_bytes)
                doc = docx.Document(docx_file)
                docx_text = "\n".join([para.text for para in doc.paragraphs])
                documents.append(docx_text)
            # Optionally, add more file types here
    return documents
