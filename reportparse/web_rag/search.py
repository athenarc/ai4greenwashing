import time
import re
from duckduckgo_search import DDGS
from googleapiclient.discovery import build
import os
from urllib.parse import urlparse
from keybert import KeyBERT

kw_model = KeyBERT()

from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keywords_smart(text, max_keywords=10, min_len=1, max_len=3, return_as_string=True):
    # Extract a larger set to allow for deduplication
    keywords_with_scores = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(min_len, max_len),
        stop_words='english',
        top_n=20
    )

    seen = set()
    keywords_cleaned = []

    for phrase, _ in keywords_with_scores:
        normalized = phrase.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            keywords_cleaned.append(phrase.strip())
            if len(keywords_cleaned) >= max_keywords:
                break

    if return_as_string:
        return " ".join(keywords_cleaned)
    else:
        return keywords_cleaned


doc_extensions = ["doc", "docx", "php", "pdf", "txt", "theFile", "file", "xls"]

pattern = r"[./=]([a-zA-Z0-9]+)$"


def google_search(query, web_sources, metadata, retries=5, backoff_time=2):

    urls = []
    sites_source = []
    # Removed exclude_terms and inurl_filters logic
    search_query = f"{metadata[0]} {extract_keywords_smart(query)}"
    print(f"SEARCH QUERY: {search_query}")

    for attempt in range(retries):
        try:
            results = DDGS().text(
                keywords=search_query,
                safesearch="off",
                max_results=web_sources,
            )

            for dict in results:
                url_domain = urlparse(dict["href"]).netloc
                match = re.search(pattern, dict["href"][-6:])
                if match:
                    file_extension = match.group(1)

                else:
                    file_extension = None

                if (
                    file_extension not in doc_extensions
                    and not any(site in url_domain for site in sites_source)
                    and not "/document/" in dict["href"]
                ):
                    urls.append(dict["href"])

            return urls

        except Exception as e:
            print(f"Error: {e}, attempt {attempt + 1}/{retries}")
            if attempt < retries - 1:
                wait_time = backoff_time * (2**attempt)  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Trying with the google_api...")
                urls = google_search_backup(query, web_sources)
                return urls


def google_search_backup(query, web_sources):
    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_KEY"))

    res = (
        service.cse()
        .list(q=query, cx=os.getenv("GOOGLE_CX_KEY"), num=web_sources)
        .execute()
    )
    urls = [item["link"] for item in res.get("items", [])]
    return urls
