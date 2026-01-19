from model_loader import call_llm, extract_label, extract_justification
from prompts import zero_shot, few_shot
from utils.company_loader import load_companies
from utils.vectordb import ReportParser
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from utils.claim_to_kg import process_claim
from utils.kg_utils import get_company_id, retrieve_evidence, format_entities_as_json
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
    kg_data = process_claim(company_name, claim)
    company_id = get_company_id(company_name)
    evidence = retrieve_evidence(company_id, claim, kg_data['nodes'])
    context = format_entities_as_json(evidence, company_name)
    
    llm_guidelines = few_shot if few_shot_flag else zero_shot
    llm_prompt_base = llm_guidelines + "\n\n<CLAIM>\n" + claim + "\n\n</CLAIM>\n" + "\n\n<CONTEXT>\n"
    full_prompt = llm_prompt_base + context

    if not context:
        llm_response = None
        return {
        "label": None,
        "justification": None,
        "subgraph": None,
        "passages": None
    }
    llm_response= call_llm(full_prompt)
    return {
        "label": extract_label(llm_response),
        "justification": extract_justification(llm_response),
        "subgraph": context,
        "passages": None
    }
