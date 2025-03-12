import os
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_annotations"]  # Database name
collection = db["annotations"]  # Collection name

print("Connected to MongoDB.")

# Drop the collectio
collection.drop()
print("Collection dropped.")
