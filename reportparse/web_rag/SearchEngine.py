import pprint
from langchain_community.utilities import SearxSearchWrapper
import os
from dotenv import load_dotenv

load_dotenv()


class SearchEngine:

    def __init__(self, num_of_results):
        self.wrapper = SearxSearchWrapper(searx_host=os.getenv("SEARXNG_HOST"))
        self.num_of_results = num_of_results
        self.blacklist = ["goodonyou.eco", "theedgemalaysia.com"]

    def call_engine(self, query):
        urls = []
        query = str(query).replace('"', "")
        print("SearxSearchWrapper STARTING")
        results = self.wrapper.results(query, num_results=self.num_of_results)
        urls_ = [res.get("link") for res in results if "link" in res]
        urls_ = [
            url
            for url in urls_
            if not any(blacklist in url for blacklist in self.blacklist)
        ]
        urls.extend(urls_)
        return urls
