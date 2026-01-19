import yaml
from pathlib import Path
import streamlit as st

def load_companies():
    config_path = Path(__file__).parent / "companies.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    companies = [(c['display'], c['key']) for c in data['companies']]
    # Already sorted in YAML, but ensure alphabetical order
    companies.sort(key=lambda x: x[0].lower())
    
    return companies


def get_company_key(display_name, companies_list):
    for display, key in companies_list:
        if display == display_name:
            return key
    return None

import chromadb
import streamlit as st


@st.cache_resource
def get_chromadb_client(db_path: str = "chromadb"):
    return chromadb.PersistentClient(path=db_path)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_company_years_map(db_path: str = "chromadb", collection_name: str = "company_reports"):
    client = get_chromadb_client(db_path)
    collection = client.get_collection(collection_name)
    
    results = collection.get(include=["metadatas"])
    metadatas = results["metadatas"]
    
    company_years = {}
    for m in metadatas:
        company = m["company"]
        year = m["year"]
        
        if company not in company_years:
            company_years[company] = set()
        company_years[company].add(year)
    
    # Convert sets to sorted lists (descending - newest first)
    for company in company_years:
        company_years[company] = sorted(company_years[company], reverse=True)
    
    return company_years