# ruff: noqa: I001
from zenml import pipeline
from steps.etl.crawl_web import crawl_web, crawl_web_dynamic


@pipeline
def web_etl_pipeline(url: str):
    scraped_documents = crawl_web(url=url)
    return scraped_documents


def crawl_web_dynamic_etl_pipeline(url: str):
    scraped_documents = crawl_web_dynamic(url=url)
    return scraped_documents


from zenml import pipeline
from steps.etl.crawl_web import crawl_web, crawl_web_dynamic


@pipeline
def web_etl_pipeline(url: str):
    scraped_documents = crawl_web(url=url)
    # You can add more steps here (e.g., cleaning, saving to DB)
    return scraped_documents


def crawl_web_dynamic_etl_pipeline(url: str):
    scraped_documents = crawl_web_dynamic(url=url)
    return scraped_documents
