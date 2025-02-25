from logging import getLogger
import traceback
import argparse
import os
import pandas as pd
import ollama
import chromadb
import requests
from reportparse.structure.document import Document
import re
from reportparse.annotator.store_pages import ChromaDBHandler
from langchain_groq import ChatGroq

logger = getLogger(__name__)

# parser = argparse.ArgumentParser(description="Verify greenwashing claims using LLMs")
# parser.add_argument("--use_groq", action="store_true", help="Use Groq LLM instead of Ollama")
# args = parser.parse_args()
# if args.use_groq and os.getenv("USE_GROQ_API") == "True":

print(os.getenv("USE_GROQ_API"))
if os.getenv("USE_GROQ_API") == "True":
    use_groq = True
else:
    use_groq = False
print(use_groq)

use_justification = False
use_chunks = False

# parser.add_argument("--use_justification", action="store_true", help="Use justification of LLM claim")
# args = parser.parse_args()
# logger = getLogger(__name__)
# if args.use_justification:
#     use_justification = True
# else:
#     use_justification = False

# parser.add_argument("--use_chunks", action="store_true", help="Use chunk db")
# args = parser.parse_args()
# logger = getLogger(__name__)
# if args.use_chunks:
#     use_chunks = True
# else:
#     use_chunks = False

model_name = os.getenv("OLLAMA_MODEL") if not use_groq else os.getenv("GROQ_LLM_MODEL_1")
print(model_name)

CHROMA_DB_PATH = "reportparse/database_data/chroma_db_2"

try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection("parsed_pages")
except Exception as e:
    logger.error(f"Error loading ChromaDB collection: {e}")
    traceback.print_exc()
    raise SystemExit("Failed to load ChromaDB. Ensure it exists and is accessible.")


document = Document.from_json_file('results/Apple_Environmental_Progress_Report_2023.pdf.json')
df = document.to_dataframe(level='page')
print(df)


def retrieve_context(claim, page_num, db, k=3, use_chunks=False):
    try:
        collection = db.chunk_collection if use_chunks else db.page_collection
        results = collection.query(query_texts=[claim], n_results=k)  # Perform the query

        relevant_texts = "\n".join([result[0] for result in results["documents"]])  # Access the documents properly

        # Remove the current page from the context
        filtered_texts = "\n".join(
            line for line in relevant_texts.split("\n") if f"Page {page_num} " not in line
        )

        return filtered_texts.strip() if filtered_texts.strip() else ""

    except Exception as e:
        logger.error(f"Error retrieving context from ChromaDB: {e}")
        return ""





def verify_claim_with_context(claim, justification, page_text, context, use_groq=use_groq, use_justification=use_justification):
    """Use an LLM (Ollama or Groq) to verify if the claim is actually greenwashing based on document context."""
    if use_justification:
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
    else:
        prompt = f"""
        A report page flagged a potential greenwashing claim:

        **Claim:** {claim}

        Below is the full text of the page where the claim was found:

        {page_text}

        Additionally, below is relevant context from other pages in the document:

        {context}

        Based on the full document context, is this claim actual greenwashing, or does the report substantiate it? Provide a reasoned response.
        """

    try:
        if use_groq:
            try:
                # Initialize the ChatGroq object with the appropriate parameters
                llm = ChatGroq(
                    model=os.getenv("GROQ_LLM_MODEL_1"),
                    temperature=0,          
                    max_tokens=None,        
                    timeout=None,           
                    max_retries=1,          
                    groq_api_key=os.getenv("GROQ_API_KEY"), 
                )
                
                response = llm.invoke(prompt)

                # Return the response content from Groq
                return response.content
            
            except Exception as e:
                logger.error(f"Error calling Groq API: {e}")
                return "Error: Could not generate a response from Groq."

        else:
            print("here")
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"]
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return "Error: Could not generate a response."



results = []
flag = "llm"
db_handler = ChromaDBHandler()

for use_chunks in [False, True]:
    for use_justification in [False, True]:
        for idx, row in df.iterrows():
            if flag not in row or pd.isna(row[flag]):
                continue  
            
            row[flag] = re.sub(r"No greenwashing claims found\s*$", "", row[flag])
            claim_pattern = r"Potential greenwashing claim:\s*(.*?)\s*Justification:\s*(.*?)(?=\nPotential greenwashing claim:|\Z)"

            matches = re.findall(claim_pattern, row[flag], re.DOTALL)

            for claim, justification in matches:
                claim = claim.strip()
                justification = justification.strip()

                context = retrieve_context(claim, idx, db_handler, k=6, use_chunks=use_chunks)

                verdict = verify_claim_with_context(claim, justification, row["page_text"], context, use_groq=use_groq)

                results.append({"page": idx, "claim": claim, "verdict": verdict})


        filename = f"verified_{model_name}_use_justification_{use_justification}_use_chunks_{use_chunks}.json"

        verified_df = pd.DataFrame(results)
        print(verified_df)

        verified_df.to_json(filename, orient="records", indent=4)
        verified_df.to_csv(filename.replace(".json", ".csv"), index=False)
        print(f"Saved results to {filename}")

