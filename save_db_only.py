import os
import json
import argparse
from pymongo import MongoClient
import pandas as pd

# TODO CHECK ARG LOGIC AND REMOVE MOST OF THESE
parser = argparse.ArgumentParser(description="Annotate PDF with custom settings.")
parser.add_argument(
    "--input_path", type=str, required=False, help="Path to the input PDF file."
)
parser.add_argument(
    "--output_path",
    type=str,
    required=False,
    help="Path to save the annotated document.",
)

args = parser.parse_args()

input_path = args.input_path if args.input_path else "./reportparse/asset/example.pdf"
output_path = args.output_path if args.output_path else "./results/example.pdf.json"

################### MAIN ###################

if not os.path.exists("./results"):
    os.makedirs("./results")

with open(output_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_annotations"]
collection = db["annotations"]
print("Connected to MongoDB.")

pdf_id = data["name"]
new_pages = data["pages"]

existing_doc = collection.find_one({"name": pdf_id})

if existing_doc:
    existing_pages = {page["num"] for page in existing_doc["pages"]}

    # **Filter new pages that are not in the database AND have annotations**
    new_pages_to_insert = [
        page
        for page in new_pages
        if page["num"] not in existing_pages
        and "annotations" in page
        and page["annotations"]
    ]

    if new_pages_to_insert:
        # Push only the new pages with annotations
        collection.update_one(
            {"name": pdf_id}, {"$push": {"pages": {"$each": new_pages_to_insert}}}
        )
        print(
            f"Inserted {len(new_pages_to_insert)} new pages with annotations into {pdf_id}."
        )
    else:
        print(f"No new annotated pages to insert for {pdf_id}.")
else:
    # If the document doesn't exist, insert only pages that have annotations
    annotated_pages = [
        page for page in new_pages if "annotations" in page and page["annotations"]
    ]

    if annotated_pages:
        data["pages"] = annotated_pages
        collection.insert_one(data)
        print(
            f"Inserted new document with {len(annotated_pages)} annotated pages from {pdf_id}."
        )
    else:
        print(f"No annotated pages found. Document {pdf_id} was not inserted.")

# Verify the final document
result = collection.find_one({"name": pdf_id})
if result:
    print(f"Final document contains {len(result['pages'])} annotated pages.")
else:
    print("Document was not inserted.")
