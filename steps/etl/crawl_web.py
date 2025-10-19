# ruff: noqa: I001, T201, F841
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing_extensions import Annotated
from zenml import step
from playwright.sync_api import sync_playwright


@step
def crawl_web(url: Annotated[str, "url"]) -> Annotated[list, "scraped_documents"]:
    """
    Recursively crawl all internal links from the given URL and scrape text from each page.
    """
    visited = set()
    to_visit = [url]
    domain = urlparse(url).netloc
    scraped = []

    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue
        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch {current_url}: {e}")
            visited.add(current_url)
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        texts = soup.stripped_strings
        content = " ".join(texts)
        scraped.append({"url": current_url, "content": content})
        visited.add(current_url)

        # Find all internal links
        for link in soup.find_all("a", href=True):
            href = link["href"]
            joined = urljoin(current_url, href)
            parsed = urlparse(joined)
            if parsed.netloc == domain and joined not in visited and joined not in to_visit:
                # Only add http(s) links, skip mailto, tel, etc.
                if parsed.scheme in ("http", "https"):
                    to_visit.append(joined)

    print(f"Scraped {len(scraped)} pages from {url}")
    for doc in scraped:
        print(f"URL: {doc['url']}")
        print(f"Content (first 500 chars): {doc['content'][:500]}")
        print("-" * 40)
    return scraped


def crawl_web_dynamic(url: Annotated[str, "url"]) -> Annotated[list, "scraped_documents"]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")
        content = page.content()
        text = page.inner_text("body")
        browser.close()
    print(f"Scraped text (first 500 chars): {text[:500]}")
    return [{"url": url, "content": text}]
