import chromadb
import subprocess
from pydantic_settings import BaseSettings
import requests

from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Get repo root dynamically
repo_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()

class ChromaDBHandler:
    def __init__(self, db_path=repo_root + "/reportparse/database_data/chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)

        # TODO: find out why this doesn't work
        # self.embedding_model = OllamaEmbeddingFunction(
        #     url="http://localhost:11434",
        #     model_name="mxbai-embed-large",
        # )

        self.collection = self.client.get_or_create_collection(
            name="parsed_pages",
        )

    def store_page(self, doc_name: str, page_number: int, text: str):
        try:
            self.collection.add(
                ids=[f"{doc_name}_page_{page_number}"],
                documents=[text],
                metadatas=[{"doc_name": doc_name, "page_number": page_number, "type": "esg-report"}],
            )
        except Exception as e:
            print(f"Error storing page: {e}")
            raise


    def retrieve_relevant_pages(self, query: str, top_k: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k  
        )

        retrieved_pages = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_pages.append(f"Page {meta['page_number']} (Doc {meta['doc_name']}):\n{doc}\n")

        return "\n".join(retrieved_pages)  
