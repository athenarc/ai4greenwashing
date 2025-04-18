
import chromadb
import subprocess
from chromadb.config import Settings

repo_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()


class RedditChromaHandler:
    def __init__(self, db_path=repo_root + "/reportparse/database_data/reddit_chroma"):
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))

        self.collection = self.client.get_or_create_collection(
            name="reddit_posts",
            metadata={"hnsw:space": "cosine"},
        )

    def chunk_text(self, text, min_chunk_size=200, max_chunk_size=500, num_chunks=3, overlap_ratio=0.25):
        words = text.split()
        total_length = len(words)

        if total_length < min_chunk_size:
            return [text]

        chunk_size = max(min_chunk_size, min(max_chunk_size, total_length // num_chunks))
        overlap = int(chunk_size * overlap_ratio)

        chunks = []
        start = 0
        while start < total_length:
            end = min(start + chunk_size, total_length)
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap

        return chunks

    def store_post(self, post_id: str, text: str, metadata: dict):
        """
        Stores full Reddit post and optionally chunked versions.
        Metadata should contain:
        - 'url', 'is_relevant', 'company', 'techniques', etc.
        """
        try:
            # Full post
            doc_id = f"reddit_post_{post_id}"
            existing = self.collection.get(ids=[doc_id])
            if existing and len(existing["documents"]) > 0:
                self.collection.delete(ids=[doc_id])

            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
            )

            # Chunked version
            chunks = self.chunk_text(text)
            chunk_ids = [f"reddit_post_{post_id}_chunk_{i}" for i in range(len(chunks))]

            existing_chunks = self.collection.get(ids=chunk_ids)
            if existing_chunks and len(existing_chunks["documents"]) > 0:
                self.collection.delete(ids=chunk_ids)

            chunk_metadatas = [metadata | {"chunk_number": i} for i in range(len(chunks))]

            self.collection.add(
                ids=chunk_ids,
                documents=chunks,
                metadatas=chunk_metadatas,
            )

        except Exception as e:
            print(f"Error storing Reddit post: {e}")
            raise