import pandas as pd
from reportparse.reader.base import BaseReader
from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.annotator.web_rag import WEB_RAG_Annotator
from reportparse.annotator.chroma_annotator import ChromaAnnotator
from reportparse.annotator.reddit_annotator import RedditAnnotator
from reportparse.annotator.chroma_esg_annotator import ChromaESGAnnotator
from reportparse.annotator.news_annotator import NewsAnnotator
from reportparse.annotator.llm_aggregator import LLMAggregator
from reportparse.annotator.crawler_annotator import WebCrawlerAnnotator
from reportparse.llm_prompts import LLM_AGGREGATOR_PROMPT
from reportparse.llm_prompts import LLM_AGGREGATOR_PROMPT_2
from reportparse.llm_prompts import LLM_AGGREGATOR_PROMPT_FINAL
from reportparse.climate_cti import cti_classification
from reportparse.llm_evaluation import llm_evaluation
import argparse
import re
from pymongo import MongoClient
import json
from dotenv import load_dotenv
from logging import getLogger
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import torch
import gc
import torch

logger = getLogger(__name__)


# annotator objects
load_dotenv()
web = WEB_RAG_Annotator()
web_crawler = WebCrawlerAnnotator()
chroma = ChromaAnnotator()
reddit = RedditAnnotator()
chroma_esg = ChromaESGAnnotator()
news_annotator = NewsAnnotator()
llm_agg = LLMAggregator()

# final aggregator prompt
agg_prompt_final = LLM_AGGREGATOR_PROMPT_FINAL
# flags for annotator invocation
news_flag = True
chroma_esg_flag = False
web_rag_flag = True
web_crawler_flag = False
chroma_db_flag = True
reddit_flag = False
aggregator_flag = False
first_pass_flag = False
reddit_flag = True
final_aggregator_flag = True
# flags for cti metrics
eval = llm_evaluation()


# define reader
reader = BaseReader.by_name("pymupdf")()

# define parser parameters. They are not utilized but needed for the pymupdf invocation
parser = argparse.ArgumentParser(
    description="Process and analyze CSV containing Greenwashing claims."
)

parser.add_argument(
    "--output",
    type=str,
    default="./csv_results",
    help="Directory to save output files.",
)

parser.add_argument(
    "--max_pages",
    type=int,
    default=400,
    help="Directory to save output files.",
)


parser.add_argument(
    "--skip_pages",
    type=list,
    default=None,
    help="Pages to skip.",
)

parser.add_argument(
    "--skip_load_image",
    type=bool,
    default=0,
    help="Skip load image or not. 1 for yes, 0 for no.",
)


args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)


df = pd.read_csv("Greenwashing_claims_esg_reports.csv")
df = df.dropna(subset=["Company"])
df["Year"] = df["Year"].apply(int)
df = df[df["Year"] >= 2021]
df = df[["Company", "Year", "Claim"]]
result_df_path = "gw_result_exp1.csv"
if os.path.exists(result_df_path):
    result_df = pd.read_csv(result_df_path)
else:
    result_df = pd.DataFrame()


i = 0

# blacklisted_reports = ["adidas_2021", "lululemon_2022"]
blacklisted_reports = [
    "adidas_2021",
    "lululemon_2022",
    "h&m_2021",
    "dbs_2023",
    # "petronas_2023",
    "equinor_2022",
    "henniez_2023",
    "jbs_2021",
    "greenavocadomattress_2023",
    "giantgroup_2022",
]


for idx, row in df.iterrows():

    company = str(row["Company"]).lower().strip()
    year = str(row["Year"]).lower().strip()
    esg_report = company + "_" + year
    claim = row["Claim"]
    print(f"Examining claim with id {idx}")
    print(f"Claim: {claim}")
    print()
    input_path = f"Greenwashing_claims_esg_reports/{esg_report}.pdf"
    # saving the document
    json_output_path = os.path.join(args.output, f"{esg_report}_{idx}" + ".json")
    if os.path.exists(json_output_path):
        print(f"✅ JSON found: {input_path}")
        print("Skipping procedure....")
        continue
    else:
        print(f"❌ JSON missing: {json_output_path}")
        print("Starting the procedure...")

    if esg_report in blacklisted_reports:
        print("Cannot run this report due to size. Skipping it....")
        continue

    try:
        document = reader.read(input_path=input_path, args=args)
        # if len(document.pages) >= 401:
        #     print(f"File {esg_report} is too big. Skipping reading proceduce.....")
        #     continue
    except Exception as e:
        print(f"ERROR: {e}")
        continue
    # document.save(json_output_path)
    # logger.info(f'Saved the full output file to: "{json_output_path}".')
    # calling aggregators

    if chroma_db_flag:
        print("Starting storing in Chroma")
        for page in document.pages:
            page_number = page.num
            text = page.get_text_by_target_layouts(
                target_layouts=["text", "list", "cell"]
            )
            chroma.chroma_db.store_page(
                doc_name=document.name, page_number=page_number, text=text
            )
            print(f"Stored page {page_number}")
        print("Successfully stored all pages in chroma")

    if chroma_db_flag:
        logger.info("Chroma annotator starting")
        chroma_result, chroma_retrieved_pages, chroma_context = chroma.call_chroma(
            claim,
            document.name,
            "",
            page_number,
            chroma.chroma_db,
            k=6,
            use_chunks=False,
        )

    if web_rag_flag:

        logger.info("Web rag crawler starting")
        # add web_rag aggregation
        web_rag_result, url_list, web_info = web_crawler.web_crawler_rag(
            claim, 5, company
        )

    if news_flag:
        logger.info("News Annotator starting")
        news_result, news_retrieved_sources, news_context = news_annotator.call_news_db(
            claim, list(company), news_annotator.news_db, k=6
        )

    if reddit_flag:
        logger.info("Reddit starting")
        reddit_result, reddit_retrieved_posts, reddit_context = reddit.call_reddit(
            claim,
            company,
            reddit.reddit_db,
            k=6,
        )

    if final_aggregator_flag:
        logger.info("Final Aggregator starting")
        aggregator_result = llm_agg.call_aggregator_final(
            claim,
            chroma_result,
            web_rag_result,
            reddit_result,
            "",
            news_result,
        )

    logger.info(f"Adding a new dataframe row....")
    new_data = {
        "id": idx,
        "Claim": claim,
        "Year": year,
        "Company": company,
        "ESG_Report": document.name,
        "Chroma_retrieved_pages": [int(page) + 1 for page in chroma_retrieved_pages],
        "Chroma_context": chroma_context,
        "Chroma_label": chroma.extract_label(chroma_result),
        "Chroma_justification": chroma.extract_justification(chroma_result),
        "Web_rag_context": web_info,
        "Web_rag_urls": url_list,
        "Web_rag_label": web.extract_label(web_rag_result),
        "Web_rag_justification": web.extract_justification(web_rag_result),
        "News_context": news_context,
        "News_retrieved_pages": news_retrieved_sources,
        "News_label": news_annotator.extract_label(news_result),
        "News_justification": news_annotator.extract_justification(news_result),
        "Reddit_context": reddit_context,
        "Reddit_retrieved_pages": reddit_retrieved_posts,
        "Reddit_label": reddit.extract_label(reddit_result),
        "Reddit_justification": reddit.extract_justification(reddit_result),
        "Aggregator result": aggregator_result,
        "Aggregator label": chroma.extract_label(aggregator_result),
        "Aggregator justification": chroma.extract_justification(aggregator_result),
    }

    new_data_df = pd.DataFrame([new_data])
    result_df = pd.concat([result_df, new_data_df], ignore_index=True)
    result_df.to_csv(result_df_path, index=False)

    logger.info(f"Row added successfully")

    document.save(json_output_path)
    logger.info(f'Saved the full output file to: "{json_output_path}".')
    # break
