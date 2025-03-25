import os
import json
import argparse
from pymongo import MongoClient
from reportparse.reader.base import BaseReader
from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
import pandas as pd
import subprocess

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

parser.add_argument(
    "--clean",
    type=bool,
    default=False,
    action='store_true',
    help="Clean the db if true, otherwise append.",
)

parser.add_argument(
    "--max_pages", type=int, default=120, help="Maximum number of pages to process."
)
parser.add_argument(
    "--pages_to_gw",
    type=int,
    default=120,
    help="Number of pages to process for greenwashing analysis.",
)

parser.add_argument(
    "--web_rag_annotator_name",
    type=str,
    default="llm-test",
    help="Name for the web RAG annotator.",
)
parser.add_argument(
    "--web_rag_text_level",
    type=str,
    choices=["page", "sentence", "block"],
    default="page",
)
parser.add_argument(
    "--web_rag_target_layouts",
    type=str,
    nargs="+",
    default=["text", "list", "cell"],
    choices=LAYOUT_NAMES,
)
parser.add_argument("--web_rag", type=str, default="yes")
parser.add_argument("--chroma_annotator_name", type=str, default="chroma")
parser.add_argument(
    "--chroma_text_level",
    type=str,
    choices=["page", "sentence", "block"],
    default="page",
)
parser.add_argument(
    "--chroma_target_layouts",
    type=str,
    nargs="+",
    default=["text", "list", "cell"],
    choices=LAYOUT_NAMES,
)
parser.add_argument("--use_chroma", action="store_true", help="Enable ChromaDB usage")
parser.add_argument(
    "--use_chunks", action="store_true", help="Use chunks instead of pages"
)
parser.add_argument(
    "--start_page",
    type=int,
    help=f"Choose starting page number (0-indexed)",
    default=0,
)

parser.add_argument(
    "--reddit_text_level",
    type=str,
    choices=["page", "sentence", "block"],
    default="page",
)

parser.add_argument(
    "--reddit_target_layouts",
    type=str,
    nargs="+",
    default=["text", "list", "cell"],
    choices=LAYOUT_NAMES,
)

parser.add_argument("--use_reddit", action="store_true", help="Enable reddit usage")

parser.add_argument(
    "--reddit_pages_to_gw",
    type=int,
    help=f"Choose between 1 and esg-report max page number",
    default=1,
)

parser.add_argument(
    "--reddit_start_page",
    type=int,
    help=f"Choose starting page number (0-indexed)",
    default=0,
)

group = parser.add_mutually_exclusive_group()
group.add_argument("--clean_doc", type=str, help="Name of a document to delete from MongoDB.")
group.add_argument("--clean_all", action="store_true", help="Drop all documents from MongoDB.")

args = parser.parse_args()

input_path = args.input_path if args.input_path else "./reportparse/asset/example.pdf"
base_name = os.path.basename(input_path)
output_dir = args.output_path if args.output_path else "./results"
outfile = f"./{output_dir}/{base_name}.json"

if args.clean_doc:
    subprocess.run(["python", "clean_mongo.py", "--doc", args.clean_doc])
elif args.clean_all:
    subprocess.run(["python", "clean_mongo.py", "--all"])
    
################### MAIN ###################

# reader and args
reader = BaseReader.by_name("pymupdf")()

document = reader.read(input_path=input_path, max_pages=args.max_pages)

# annotators
llm_agg = BaseAnnotator.by_name("llm_agg")()
climate_annotator = BaseAnnotator.by_name("climate")()
climate_commitment_annotator = BaseAnnotator.by_name("climate_commitment")()
climate_specificity_annotator = BaseAnnotator.by_name("climate_specificity")()
climate_sentiment_annotator = BaseAnnotator.by_name("climate_sentiment")()

document = llm_agg.annotate(document=document, args=args)
document = climate_annotator.annotate(document=document)
document = climate_commitment_annotator.annotate(document=document)
document = climate_specificity_annotator.annotate(document=document)
document = climate_sentiment_annotator.annotate(document=document)


if not os.path.exists("./results"):
    os.makedirs("./results")

document.save(outfile)

with open(outfile, "r", encoding="utf-8") as file:
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

df = document.to_dataframe_ext(level="page")

print(df.columns)
print()
