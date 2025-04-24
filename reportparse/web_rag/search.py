import time
import re
from duckduckgo_search import DDGS
from googleapiclient.discovery import build
import os
from urllib.parse import urlparse


doc_extensions = ["doc", "docx", "php", "pdf", "txt", "theFile", "file", "xls"]

pattern = r"[./=]([a-zA-Z0-9]+)$"


def is_valid_result(url):
    return "google.com/search" not in url


def filter_urls(url_list, metadata):
    filtered_urls = []
    for url in url_list:
        url_domain = urlparse(url).netloc
        if str(metadata).lower() not in url_domain.lower():
            filtered_urls.append(url)
    return [r for r in filtered_urls if is_valid_result(filtered_urls)]


def google_search(query, web_sources, metadata, retries=5, backoff_time=2):
    urls = []
    sites_source = []
    # Removed exclude_terms and inurl_filters logic
    search_query = f"{query+' '+str(metadata[0])}"
    print("DDG QUERY IS:", search_query)

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

            return filter_urls(urls, str(metadata[0]))

        except Exception as e:
            print(f"Error: {e}, attempt {attempt + 1}/{retries}")
            if attempt < retries - 1:
                wait_time = backoff_time  # * (2**attempt)  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Trying with the google_api...")
                urls = google_search_backup(query, web_sources, str(metadata[0]))
                return filter_urls(urls, str(metadata[0]))


def google_search_backup(query, web_sources, metadata):
    print("GOOGLE SEARCH QUERY IS:", query + " " + str(metadata[0]))
    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_KEY"))

    res = (
        service.cse()
        .list(
            q=query + " " + str(metadata[0]),
            cx=os.getenv("GOOGLE_CX_KEY"),
            num=web_sources,
        )
        .execute()
    )
    urls = [item["link"] for item in res.get("items", [])]
    return filter_urls(urls, str(metadata[0]))
