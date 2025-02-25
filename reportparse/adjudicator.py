from logging import getLogger
import traceback
import argparse
import os
import pandas as pd
import ollama
import chromadb
import requests
from reportparse.structure.document import Document

parser = argparse.ArgumentParser(description="Verify greenwashing claims using LLMs")
parser.add_argument("--use_groq", action="store_true", help="Use Groq LLM instead of Ollama")
args = parser.parse_args()
logger = getLogger(__name__)
if args.use_groq and os.getenv("USE_GROQ_API") == "True":
    use_groq = True
else:
    use_groq = False

CHROMA_DB_PATH = "reportparse/database_data/chroma_db"

try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection("parsed_pages")
except Exception as e:
    logger.error(f"Error loading ChromaDB collection: {e}")
    traceback.print_exc()
    raise SystemExit("Failed to load ChromaDB. Ensure it exists and is accessible.")


document = Document.from_json_file('results/Apple_Environmental_Progress_Report_2023-8-16.pdf.json')
df = document.to_dataframe(level='page')
print(df)


def retrieve_context(page_text, page_num, k=3):
    """Retrieve relevant context from ChromaDB using similarity search."""
    try:
        results = collection.query(
            query_texts=[page_text],
            n_results=k,
            where={"page_number": {"$ne": page_num}},  # Exclude current page
        )
        return "\n\n".join(results["documents"][0]) if results and results["documents"] else ""
    except Exception as e:
        logger.error(f"Error retrieving context from ChromaDB: {e}")
        return ""


def verify_claim_with_context(claim, justification, page_text, context, use_groq=False):
    """Use an LLM (Ollama or Groq) to verify if the claim is actually greenwashing based on document context."""
    prompt = f"""
    A report page flagged a potential greenwashing claim:

    **Claim:** {claim}
    **Justification:** {justification}

    Below is the full text of the page where the claim was found:

    {page_text}

    Additionally, below is relevant context from other pages in the document:

    {context}

    Based on the full document context, is this claim actual greenwashing, or does the report substantiate it? Provide a reasoned response.
    """
    
    try:
        if use_groq:
            response = requests.post(
                "https://api.groq.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY_1')}", "Content-Type": "application/json"},
                json={"model": "mixtral-8x7b-32768", "messages": [{"role": "user", "content": prompt}]}
            )
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error: No response from Groq LLM")
        else:
            response = ollama.chat(model=os.getenv("OLLAMA_MODEL"), messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"]
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return "Error: Could not generate a response."


results = []
if use_groq:
    flag = "llm"
else:
    flag = "ollama_llm"
print(flag)

for idx, row in df.iterrows():
    if flag not in row or row[flag] in ["No greenwashing claims found", None] or pd.isna(row[flag]):
        continue  

    annotations = row[flag].split("Potential greenwashing claim: ")[1:]  # Extract individual claims
    for annotation in annotations:
        parts = annotation.split("Justification:")
        if len(parts) < 2:
            continue 

        claim, justification = parts[0].strip(), parts[1].strip()

        context = retrieve_context(row["page_text"], idx)

        verdict = verify_claim_with_context(claim, justification, row["page_text"], context, use_groq=False)

        results.append({"page": idx, "claim": claim, "verdict": verdict})

verified_df = pd.DataFrame(results)
print(verified_df)

