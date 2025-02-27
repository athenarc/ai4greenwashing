import chromadb
import subprocess
from pydantic_settings import BaseSettings
import requests

from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

repo_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()

class ChromaDBHandler:
    def __init__(self, db_path=repo_root + "/reportparse/database_data/chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)

        self.page_collection = self.client.get_or_create_collection(name="parsed_pages")
        self.chunk_collection = self.client.get_or_create_collection(name="chunked_pages")

    def chunk_text(self, text, min_chunk_size=200, max_chunk_size=500, num_chunks=3, overlap_ratio=0.25):
        """
        Dynamically chunks text:
        - If text is short, keep it as a single chunk.
        - If text is long, split it into `num_chunks` with `overlap_ratio` overlap.
        """
        words = text.split()
        total_length = len(words)

        if total_length < min_chunk_size:
            return [text]  # Keep as a single chunk

        chunk_size = max(min_chunk_size, min(max_chunk_size, total_length // num_chunks))
        overlap = int(chunk_size * overlap_ratio)

        chunks = []
        start = 0
        while start < total_length:
            end = min(start + chunk_size, total_length)
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap  # Move forward but keep overlap

        return chunks

    def store_page(self, doc_name: str, page_number: int, text: str):
        """
        Stores both full pages and chunked versions dynamically while handling ID conflicts.
        """
        try:
            page_id = f"{doc_name}_page_{page_number}"
            existing_page = self.page_collection.get(ids=[page_id])

            # Remove existing page if needed
            if existing_page and len(existing_page["documents"]) > 0:
                self.page_collection.delete(ids=[page_id])

            # Store full page
            self.page_collection.add(
                ids=[page_id],
                documents=[text],
                metadatas=[{"doc_name": doc_name, "page_number": page_number, "type": "esg-report"}],
            )

            # Store dynamically chunked text
            chunks = self.chunk_text(text)
            chunk_ids = [f"{doc_name}_page_{page_number}_chunk_{i}" for i in range(len(chunks))]

            # Remove existing chunks if needed
            existing_chunks = self.chunk_collection.get(ids=chunk_ids)
            if existing_chunks and len(existing_chunks["documents"]) > 0:
                self.chunk_collection.delete(ids=chunk_ids)

            chunk_metadata = [{"doc_name": doc_name, "page_number": page_number, "chunk_number": i} for i in range(len(chunks))]

            self.chunk_collection.add(
                ids=chunk_ids,
                documents=chunks,
                metadatas=chunk_metadata,
            )

        except Exception as e:
            print(f"Error storing page: {e}")
            raise

    # def retrieve_relevant_pages(self, query: str, top_k: int = 6, use_chunks=False):
    #     """
    #     Retrieves relevant pages or chunks based on the `use_chunks` flag.
    #     """
    #     collection = self.chunk_collection if use_chunks else self.page_collection
    #     results = collection.query(query_texts=[query], n_results=top_k)

    #     retrieved_texts = []
    #     for doc, meta in zip(results["documents"], results["metadatas"]):
    #         retrieved_texts.append(f"Page {meta['page_number']} (Doc {meta['doc_name']}):\n{doc}\n")

    #     return "\n".join(retrieved_texts)
