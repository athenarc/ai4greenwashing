# import chromadb

# class chroma_db:

#     def __init__(self):
        
#         self.client = chromadb.PersistentClient(path="./reportparse/database_data/chroma_db")
#         self.collection_name = "esg_pages"
#         self.collection = self.client.get_or_create_collection(self.collection_name)
#         self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#         return
        
#     def store(self, text):
        
#         # Encode text
#         embedding = self.embedding_model.encode(text).tolist()
        
#         # Store text in ChromaDB
#         self.collection.add(
#             documents=[text],
#             metadatas=[{"source": "document", "type": "esg-report"}],
#             ids=["text_entry"],
#             embeddings=[embedding]
#         )
        
#         print(f"Stored text in ChromaDB collection '{self.collection_name}'.")


#     def delete(self, text):  
#         pass