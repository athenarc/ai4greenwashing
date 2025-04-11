import chromadb
import subprocess
from chromadb.config import Settings
from reportparse.db_rag.db import ChromaDBHandler
import pandas as pd
from sentence_transformers import SentenceTransformer
import ast


class NewsChromaHandler(ChromaDBHandler):
    def __init__(self):
        super().__init__(extra_path="news_db")
        self.df = pd.read_csv("reportparse/news_db/news_df.csv")
        self.df = self.df.dropna(subset=["content"])
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name="news_db",
            metadata={"hnsw:space": "cosine"},
        )

    def chunk_text(
        self,
        text,
        min_chunk_size=200,
        max_chunk_size=500,
        num_chunks=3,
        overlap_ratio=0.25,
    ):
        words = text.split()
        total_length = len(words)

        if total_length < min_chunk_size:
            return [text]

        chunk_size = max(
            min_chunk_size, min(max_chunk_size, total_length // num_chunks)
        )
        overlap = int(chunk_size * overlap_ratio)

        chunks = []
        start = 0
        while start < total_length:
            end = min(start + chunk_size, total_length)
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap

        return chunks

    # store to the database the articles and their metadata
    def store_articles(self, batch_size=100):
        batch_num = 1
        for start in range(0, len(self.df), batch_size):
            end = start + batch_size
            batch_df = self.df.iloc[start:end]

            batch_docs = []
            batch_ids = []
            batch_metadata = []
            batch_embeddings = []

            for i, row in batch_df.iterrows():
                content_chunks = self.chunk_text(row["content"])

                chunk_embeddings = self.model.encode(content_chunks)

                for idx, (chunk, embedding) in enumerate(
                    zip(content_chunks, chunk_embeddings)
                ):
                    metadata = {
                        "id": str(row["id"]),
                        "chunk_id": f"{row['id']}_chunk_{idx}",
                        "title": row["title"],
                        "author": row["author"],
                        "news_site": row["news_site"],
                        "url": row["url"],
                        "publish_date": str(row["publish_date"]),
                        "harvest_date": str(row["harvest_date"]),
                        "type_of_news": row["type_of_news"],
                    }

                    if isinstance(row["Organization"], list):
                        for org in row["Organization"]:
                            if isinstance(org, str):
                                org = org.lower()
                        metadata["organization"] = ", ".join(row["Organization"])
                    elif pd.notna(row["Organization"]):
                        metadata["organization"] = str(row["Organization"].lower())
                    else:
                        metadata["organization"] = ""

                    batch_docs.append(chunk)
                    batch_ids.append(str(metadata["chunk_id"]))
                    batch_metadata.append(metadata)
                    batch_embeddings.append(embedding)

            self.collection.add(
                documents=batch_docs,
                embeddings=batch_embeddings,
                ids=batch_ids,
                metadatas=batch_metadata,
            )
            print(
                f"Batch {batch_num} stored successfully with {len(batch_docs)} chunks."
            )
            batch_num += 1

    def retrieve_by_organization(self, organization_names: list, claim, threshold=0.4):
        claim_embedding = self.model.encode(claim)
        if isinstance(organization_names, str):
            organization_names = [organization_names]
        all_results = {"documents": [], "metadatas": [], "distances": []}
        result = self.collection.query(
            query_embeddings=claim_embedding,
            n_results=100,
            include=["metadatas", "documents", "distances"],
        )
        if result["metadatas"]:
            for doc, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            ):
                if distance < threshold:
                    org_meta = meta.get("organization", "").lower()
                    if any(org.lower() in org_meta for org in organization_names):
                        all_results["documents"].append(
                            doc
                            + f"\n published on: {meta.get('publish_date', 'Date Unknown')} \n"
                        )
                        all_results["metadatas"].append(meta)
                        all_results["distances"].append(
                            distance
                        )  # Store the similarity score
                        if len(all_results["documents"]) >= 10:
                            break
        else:
            return [], []

        # organizations = []
        # for entry in all_results["metadatas"]:
        #     try:
        #         org_list = ast.literal_eval(entry["organization"])
        #         organizations.append(org_list)
        #     except Exception as e:
        #         print(f"Error processing organization: {e}")

        # print("RETURNIGN THOSE RESULTS: ", all_results["documents"])
        # print("RETURNING ORGANIZATIONS: ", organizations)

        print("DATA RETURNED: ")
        print()
        for a, b, c in zip(
            all_results["documents"],
            all_results["metadatas"],
            all_results["distances"],
        ):
            print("DOC: ", a)
            # print("META: ", b.get("publish_date", "Date Unknown"))
            print("DISTANCE: ", c)

        return all_results["documents"], all_results["metadatas"]

    # retrieve relevant articles from the database
    def retrieve_article(self, text):
        query_embedding = self.model.encode([text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["metadatas", "documents"],
        )
        return results

    # delete articles from the database
    def delete_articles(self):
        pass

    def chunk_document(self, doc: str, chunk_size=500):
        chunks = [doc[i : i + chunk_size] for i in range(0, len(doc), chunk_size)]
        return chunks

    def calculate_similarity(self, embedding1, embedding2):
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity([embedding1], [embedding2])[0][0]
