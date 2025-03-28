import os
import argparse
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_annotations"]
collection = db["annotations"]

print("Connected to MongoDB.")


# Drop a specific document by name
def drop_document(doc_name):
    result = collection.delete_one({"name": doc_name})
    if result.deleted_count > 0:
        print(f"Document '{doc_name}' was dropped from the collection.")
    else:
        print(f"No document found with name '{doc_name}'.")


# Drop all documents in the collection
def drop_all_documents():
    collection.drop()
    print("All documents were dropped from the collection.")


# CLI Argument Parsing
def main():
    parser = argparse.ArgumentParser(description="Drop documents from the MongoDB annotation collection.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc", type=str, help="Name of the document to drop")
    group.add_argument("--all", action="store_true", help="Drop all documents in the collection")

    args = parser.parse_args()

    if args.doc:
        drop_document(args.doc)
    elif args.all:
        drop_all_documents()


if __name__ == "__main__":
    main()
