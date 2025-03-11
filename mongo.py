import os
import json
import argparse
from pymongo import MongoClient
from reportparse.reader.base import BaseReader
from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES

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
    "--pages_to_gw",
    type=int,
    default=120,
    help="Number of pages to process for greenwashing analysis.",
)
parser.add_argument(
    "--max_pages", type=int, default=120, help="Maximum number of pages to process."
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
parser.add_argument("--use_chroma",action="store_true", help="Enable ChromaDB usage")
parser.add_argument(
    "--use_chunks", action="store_true", help="Use chunks instead of pages"
)
args = parser.parse_args()


reader = BaseReader.by_name("pymupdf")()

input_path = args.input_path if args.input_path else "./reportparse/asset/example.pdf"
document = reader.read(input_path=input_path)

llm_agg = BaseAnnotator.by_name("llm_agg")()

document = llm_agg.annotate(document=document, args=args)

# document = BaseAnnotator.by_name("climate")().annotate(document=document)
# document = BaseAnnotator.by_name("climate_commitment")().annotate(document=document)
# document = BaseAnnotator.by_name("climate_sentiment")().annotate(document=document)
# document = BaseAnnotator.by_name("climate_specificity")().annotate(document=document)


if not os.path.exists("./results"):
    os.makedirs("./results")

# create json
document.save("./results/example.pdf.json")

df = document.to_dataframe(level="block")
df_2 = document.to_dataframe(level="page")

# print(df)

# # eda
# print(df.describe())
# print(df.info())
# print(df.head())
# print(df.tail())
# print(df.columns)
# print(df.index)

# # eda for df_2
# print(df_2.describe())
# print(df_2.info())
# print(df_2.head())
# print(df_2.tail())
# print(df_2.columns)
# print(df_2.index)

with open("./results/example.pdf.json", "r", encoding="utf-8") as file:
    data = json.load(file)

print("Connecting to MongoDB...")
client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_annotations"]  # Database name
collection = db["annotations"]  # Collection name
print("Connected to MongoDB.")

pdf_id = data["name"]
new_pages = data["pages"]

# Check if the document already exists in the database
existing_doc = collection.find_one({"name": pdf_id})

if existing_doc:
    # Extract existing page numbers
    existing_pages = {page["num"] for page in existing_doc["pages"]}

    # **Filter new pages that are not in the database AND have annotations**
    new_pages_to_insert = [
        page
        for page in new_pages
        if page["num"] not in existing_pages and "annotations" in page and page["annotations"]
    ]

    if new_pages_to_insert:
        # Push only the new pages with annotations
        collection.update_one(
            {"name": pdf_id}, {"$push": {"pages": {"$each": new_pages_to_insert}}}
        )
        print(f"Inserted {len(new_pages_to_insert)} new pages with annotations into {pdf_id}.")
    else:
        print(f"No new annotated pages to insert for {pdf_id}.")
else:
    # If the document doesn't exist, insert only pages that have annotations
    annotated_pages = [page for page in new_pages if "annotations" in page and page["annotations"]]

    if annotated_pages:
        data["pages"] = annotated_pages
        collection.insert_one(data)
        print(f"Inserted new document with {len(annotated_pages)} annotated pages from {pdf_id}.")
    else:
        print(f"No annotated pages found. Document {pdf_id} was not inserted.")

# Verify the final document
result = collection.find_one({"name": pdf_id})
if result:
    print(f"Final document contains {len(result['pages'])} annotated pages.")
else:
    print("Document was not inserted.")