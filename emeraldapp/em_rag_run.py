from model_loader import call_llm, extract_label, extract_justification
from prompts import zero_shot, few_shot
from utils.company_loader import load_companies
from utils.vectordb import ReportParser
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
COMPANIES = load_companies()
company_keys = [key for _, key in COMPANIES]

parser = ReportParser(
    reports_folder="",
    db_path=os.getenv("STREAMLIT_CHROMA_DB_PATH"),
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)

def approximate_token_count(text: str) -> int:
    """Estimate token count for Gemma-3 using a 4-character/token heuristic."""
    return len(text) // 4



# def normalize_company_name(name: str) -> str:
#     embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#     if not company_keys:
#         return name  
#     input_emb = embedding_model.encode([name])[0]
#     company_embs = embedding_model.encode(list(company_keys))
#     sims = np.dot(company_embs, input_emb) / (np.linalg.norm(company_embs, axis=1) * np.linalg.norm(input_emb))
#     best_idx = np.argmax(sims)
#     return list(company_keys)[best_idx]


def run(claim, company_name, year, few_shot_flag):
    
    results = parser.query(claim, n_results=8, company_filter=company_name, year_filter=year)
    context = [f"---Snippet {i+1}---\n{doc}" for i, doc in enumerate(results["documents"][0])]
    llm_guidelines = few_shot if few_shot_flag else zero_shot
    llm_prompt_base = llm_guidelines + "\n\n<CLAIM>\n" + claim + "\n\n</CLAIM>\n" + "\n\n<CONTEXT>\n"
    remaining_snippets = context.copy()

    while remaining_snippets:
        context_text = "\n\n".join(remaining_snippets)
        full_prompt = llm_prompt_base + context_text
        if approximate_token_count(full_prompt) <= 120_000:
            break
        remaining_snippets.pop()
    else:
        return {
            "label": None,
            "justification": None,
            "subgraph": None, 
            "passages": None
        }

    if not remaining_snippets:
        return {
            "label": None,
            "justification": None,
            "subgraph": None, 
            "passages": None
        }

    llm_response = call_llm(full_prompt)

    passages_json = [
        {"snippet_id": i + 1, "content": doc}
        for i, doc in enumerate(results["documents"][0][:len(remaining_snippets)])
    ]
        
    return {
        "label": extract_label(llm_response),
        "justification": extract_justification(llm_response),
        "subgraph": None,
        "passages": passages_json
    }