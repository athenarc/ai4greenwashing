from duckduckgo_search import DDGS
from urllib.parse import urlparse
import re
doc_extensions = ["doc", "docx", 'php', 'pdf', 'txt', 'theFile', 'file', 'xls']
pattern = r'[./=]([a-zA-Z0-9]+)$'

def google_search(query, web_sources):
    urls = []
    sites_source = []
    #more to be added if it is essential
    exclude_terms = ["sustainability" , "sustainability-report"]  
    inurl_filters = " ".join([f"-inurl:{term}" for term in exclude_terms])
    search_query = f"{query} {inurl_filters} -intitle:sustainability"

    results = DDGS().text(
        keywords=search_query,
        safesearch='off',
        max_results=web_sources,

    )
    for dict in results:
        url_domain = urlparse(dict['href']).netloc
        match = re.search(pattern, dict['href'][-6:])
        if match:
            file_extension = match.group(1)
            print(file_extension)
        else:
            file_extension = None
        if file_extension not in doc_extensions and not any(site in url_domain for site in sites_source) and not '/document/' in dict['href']:
            urls.append(dict['href'])
        #print(url_domain)
    return urls

