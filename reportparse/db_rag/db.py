import chromadb
import subprocess


repo_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()

class ChromaDBHandler:
    def __init__(self, db_path=repo_root + "/reportparse/database_data/chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name="parsed_pages",
            metadata={"hnsw:space": "cosine"},  # Ensure cosine similarity
        )

        # self.chunk_collection = self.client.get_or_create_collection(
        #     name="chunked_pages",
        #     metadata={"hnsw:space": "cosine"},  # Ensure cosine similarity
        # )
        
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
            base_id = f"{doc_name}_page_{page_number}"

            # Prepare full page entry
            page_id = f"{base_id}_full"
            page_meta = {
                "doc_name": doc_name,
                "page_number": page_number,
                "type": "page"
            }

            # Remove existing full page
            self.collection.delete(ids=[page_id])
            self.collection.add(
                ids=[page_id],
                documents=[text],
                metadatas=[page_meta],
            )
            
            # Prepare chunked entries
            chunks = self.chunk_text(text)
            chunk_ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metadata = [
                {
                    "doc_name": doc_name,
                    "page_number": page_number,
                    "chunk_number": i,
                    "type": "chunk"
                }
                for i in range(len(chunks))
            ]

            self.collection.delete(ids=chunk_ids)
            self.collection.add(
                ids=chunk_ids,
                documents=chunks,
                metadatas=chunk_metadata,
            )

        except Exception as e:
            print(f"Error storing page: {e}")
            raise