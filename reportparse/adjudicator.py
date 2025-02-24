from logging import getLogger
import traceback
import argparse
import os
import pandas as pd
import ollama
import chromadb
from reportparse.structure.document import Document

logger = getLogger(__name__)

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
        # results = collection.query(query_texts=[page_text], n_results=k)
        results = collection.query(
            query_texts=[page_text],
            n_results=k,
            where={"page_number": {"$ne": page_num}},  # Exclude current page
        )
        return "\n\n".join(results["documents"][0]) if results and results["documents"] else ""
    except Exception as e:
        logger.error(f"Error retrieving context from ChromaDB: {e}")
        return ""

# def verify_claim_with_context(claim, justification, context):
#     """Use an LLM to verify if the claim is actually greenwashing based on document context."""
#     prompt = f"""
#     A report page flagged a potential greenwashing claim:

#     **Claim:** {claim}
#     **Justification:** {justification}

#     Below is additional context from the rest of the document:

#     {context}

#     Based on the full document context, is this claim actual greenwashing, or does the report substantiate it? Provide a reasoned response.
#     """

#     try:
#         response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
#         return response["message"]["content"]
#     except Exception as e:
#         logger.error(f"Error calling Ollama LLM: {e}")
#         return "Error: Could not generate a response."


def verify_claim_with_context(claim, justification, page_text, context):
    """Use an LLM to verify if the claim is actually greenwashing based on document context."""
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


results = []
for idx, row in df.iterrows():
    if "ollama_llm" not in row or row["ollama_llm"] in ["No greenwashing claims found", None] or pd.isna(row["ollama_llm"]):
        continue  # Skip pages without annotations

    annotations = row["ollama_llm"].split("Potential greenwashing claim: ")[1:]  # Extract individual claims
    for annotation in annotations:
        parts = annotation.split("Justification:")
        if len(parts) < 2:
            continue 

        claim, justification = parts[0].strip(), parts[1].strip()

        context = retrieve_context(row["page_text"], idx)

        verdict = verify_claim_with_context(claim, justification, row["page_text"], context)

        results.append({"page": idx, "claim": claim, "verdict": verdict})

verified_df = pd.DataFrame(results)
print(verified_df)
