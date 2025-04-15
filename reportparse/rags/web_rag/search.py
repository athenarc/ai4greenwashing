import time
import re
from duckduckgo_search import DDGS
from urllib.parse import urlparse

doc_extensions = ["doc", "docx", "php", "pdf", "txt", "theFile", "file", "xls"]

pattern = r"[./=]([a-zA-Z0-9]+)$"


def google_search(query, web_sources, metadata, retries=5, backoff_time=2):

    urls = []
    sites_source = []
    # Removed exclude_terms and inurl_filters logic
    search_query = f"{metadata} {query}"
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
                print("Max retries reached. Please try again later.")
                return []
