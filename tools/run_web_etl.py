# ruff: noqa: I001, F401, PTH123
import yaml
from pipelines.web_etl import crawl_web_dynamic, crawl_web_dynamic_etl_pipeline, web_etl_pipeline

if __name__ == "__main__":
    # Load config
    with open("configs/web_etl.yaml", "r") as f:
        config = yaml.safe_load(f)
    url = config["url"]

    # Run web crawling pipeline
    # web_etl_pipeline(url=url)

    # Run dynamic crawling pipeline
    crawl_web_dynamic_etl_pipeline(url=url)
