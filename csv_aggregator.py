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
from news_knowledge_graph.neo4j_handler import neo4j_handler
from news_knowledge_graph.traverse_graph import traverse_graph
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
import time
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
neo4j = neo4j_handler()
news_kg_annotator = traverse_graph()
# final aggregator prompt
agg_prompt_final = LLM_AGGREGATOR_PROMPT_FINAL
# flags for annotator invocation
news_flag = True
news_kg_flag = True
# chroma_esg_flag = False
web_rag_flag = False
chroma_db_flag = False
chroma_db_flag_store = False
reddit_flag = False
final_aggregator_flag = False
first_pass_flag = False
parse_flag = False
parse_flag_store = False
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
df_meta = pd.read_csv("experiment1_with_def.csv")
df = df.dropna(subset=["Company"])
df["Year"] = df["Year"].apply(int)
df = df[df["Year"] >= 2021]
df = df[["Company", "Year", "Claim"]]
result_df_path = "news_db_comparison_better_claim.csv"  # name it as you like, change aggregation prompt
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
    "pepsi_2022",
    "h&m_2022",
    "nestle_2023",
]

# on Greenwashing_claims_esg_reports, please rename ethiad to etihad and levissima into nestle before parsing

for idx, row in df_meta.iterrows():
    df_meta_id = row["id"]
    if "id" in result_df.columns and df_meta_id in result_df["id"].values:
        print(f"⚠️ ID {idx} already in result_df — skipping.")
        continue
    # chroma_db_flag_store = True
    # try:
    #     # df_meta_id = df_meta.loc[df_meta["Claim"] == row["Claim"], "id"].values[0]
    #     df_meta_id = row["id"]
    # except Exception as e:
    #     continue
    company = str(row["Company"]).lower().strip()
    year = str(row["Year"]).lower().strip()
    esg_report = company + "_" + year
    document_name = esg_report + ".pdf"
    claim = row["web_rag_claim_query"]
    ground_truth = row["ground_truth"]
    print(f"Examining claim with id {row['id']}")
    print(f"Claim: {claim}")
    print()

    # saving the document
    if True:
        input_path = f"Greenwashing_claims_esg_reports/{esg_report}.pdf"
        json_output_path = os.path.join(
            args.output, f"{esg_report}_{df_meta_id}" + ".json"
        )
        if os.path.exists(json_output_path):
            print(f"✅ JSON found: {input_path}")
            print("Skipping parsing....")
            parse_flag = False
        else:
            print(f"❌ JSON missing: {json_output_path}")
            print("Starting parsing...")
            parse_flag = True

    if esg_report in blacklisted_reports:
        print("Cannot run this report due to size. Skipping it....")
        continue

    if parse_flag:
        try:
            document = reader.read(input_path=input_path, args=args)
            print(f"Document name: {document.name}")
            # if len(document.pages) >= 401:
            #     print(f"File {esg_report} is too big. Skipping reading proceduce.....")
            #     continue
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    # document.save(json_output_path)
    # logger.info(f'Saved the full output file to: "{json_output_path}".')
    # calling aggregators

    if chroma_db_flag_store:
        print("Starting storing in Chroma")
        for page in document.pages:
            page_number = page.num
            text = page.get_text_by_target_layouts(
                target_layouts=["text", "list", "cell"]
            )
            chroma.chroma_db.store_page(
                doc_name=document_name, page_number=page_number, text=text
            )
            print(f"Stored page {page_number}")
        print("Successfully stored all pages in chroma")

    if chroma_db_flag:
        logger.info("Chroma annotator starting")
        # chroma_result, chroma_retrieved_pages, chroma_context = chroma.call_chroma(
        #     claim,
        #     document_name,
        #     "",
        #     None,
        #     chroma.chroma_db,
        #     k=6,
        #     use_chunks=False,
        # )

        chroma_result = f""" 
                Greenwashing claim: { df_meta.loc[df_meta['id'] == df_meta_id, 'Claim'].values[0]}
                Label: {df_meta.loc[df_meta['id'] == df_meta_id, 'chroma_label'].values[0]}
                regression_score: {df_meta.loc[df_meta['id'] == df_meta_id, 'chroma_score'].values[0]}
                Justification: {df_meta.loc[df_meta['id'] == df_meta_id, 'chroma_justification'].values[0]}
                """
        chroma_retrieved_pages = df_meta.loc[
            df_meta["id"] == df_meta_id, "chroma_retrieved_pages"
        ].values[0]
        chroma_context = df_meta.loc[
            df_meta["id"] == df_meta_id, "chroma_context"
        ].values[0]

    if not chroma_db_flag:
        chroma_result, chroma_retrieved_pages, chroma_context = "", [], ""

    if web_rag_flag:

        # logger.info("Web rag crawler starting")

        # llm_claim_result = web_crawler.call_llm(claim, False, company_name=company)
        # claim_query = web_crawler.extract_claim(llm_claim_result)
        # # add web_rag aggregation
        # web_rag_result, url_list, web_info = web_crawler.web_crawler_rag(
        #     claim_query, 5, company
        # )

        web_rag_result = f""" 
        Greenwashing claim: {df_meta.loc[df_meta['id'] == df_meta_id, 'Claim'].values[0]}
        Label: {df_meta.loc[df_meta['id'] == df_meta_id, 'web_rag_label'].values[0]}
        regression_score: {df_meta.loc[df_meta['id'] == df_meta_id, 'web_rag_score'].values[0]}
        Justification: {df_meta.loc[df_meta['id'] == df_meta_id, 'web_rag_justification'].values[0]}
        """

        claim_query = df_meta.loc[
            df_meta["id"] == df_meta_id, "web_rag_claim_query"
        ].values[0]
        web_info = df_meta.loc[df_meta["id"] == df_meta_id, "web_rag_context"].values[0]
        url_list = df_meta.loc[df_meta["id"] == df_meta_id, "web_rag_urls"].values[0]

    if not web_rag_flag:
        web_rag_result, url_list, web_info, claim_query = "", [], "", ""

    if news_flag:
        # logger.info("News Annotator starting")
        # news_result, news_retrieved_sources, news_context = news_annotator.call_news_db(
        #     claim, list(company), news_annotator.news_db, k=6
        # )
        news_result = f""" 
        Greenwashing claim: {df_meta.loc[df_meta['id'] == df_meta_id, 'Claim'].values[0]}
        Label: {df_meta.loc[df_meta['id'] == df_meta_id, 'news_label'].values[0]}
        regression_score: {df_meta.loc[df_meta['id'] == df_meta_id, 'news_score'].values[0]}
        Justification: {df_meta.loc[df_meta['id'] == df_meta_id, 'news_justification'].values[0]}
        """

        news_retrieved_sources = df_meta.loc[
            df_meta["id"] == df_meta_id, "news_retrieved_pages"
        ].values[0]
        news_context = df_meta.loc[df_meta["id"] == df_meta_id, "news_context"].values[
            0
        ]

    if not news_flag:
        news_result, news_retrieved_sources, news_context = "", [], ""

    if news_kg_flag:
        graph = traverse_graph()
        graph_result = graph.verify_claim_with_kg(neo4j, claim)
        news_kg_retrieved_sources = None
        news_kg_retrieved_texts = (
            graph_result["chunk_texts"] if graph_result["chunk_texts"] else None
        )
        news_kg_retrieved_ids = (
            graph_result["chunk_ids"] if graph_result["chunk_ids"] else None
        )
        news_kg_result = (
            graph_result["llm_answer"] if graph_result["llm_answer"] else None
        )

        if graph_result["chunk_ids"] and graph_result["chunk_texts"]:
            news_kg_retrieved_sources = [
                {"chunk_id": cid, "chunk_text": ctext}
                for cid, ctext in zip(
                    graph_result["chunk_ids"], graph_result["chunk_texts"]
                )
            ]

    if not news_kg_flag:
        (
            news_kg_result,
            news_kg_retrieved_sources,
            news_kg_retrieved_texts,
            news_kg_retrieved_ids,
        ) = ("", [], "", [])

    if reddit_flag:
        logger.info("Reddit starting")
        reddit_result, reddit_retrieved_posts, reddit_context = reddit.call_reddit(
            claim,
            company,
            reddit.reddit_db,
            k=6,
        )

    if not reddit_flag:
        reddit_result, reddit_retrieved_posts, reddit_context = "", [], ""

    if final_aggregator_flag:
        logger.info("Final Aggregator starting")
        aggregator_result = llm_agg.call_aggregator_final_2(
            claim,
            chroma_result,
            web_rag_result,
            news_result,
        )
    time.sleep(10)

    if not final_aggregator_flag:
        aggregator_result = ""

    if first_pass_flag:
        first_pass_result = web_crawler.call_llm(claim)
        print(f"First pass result: {first_pass_result}")
        first_pass_result = str(first_pass_result)
    if not first_pass_flag:
        first_pass_result = ""

    logger.info(f"Adding a new dataframe row....")
    new_data = {
        "id": df_meta_id,
        "Claim": claim,
        "Year": year,
        "Company": company,
        # "ESG_Report": document.name,
        "chroma_retrieved_pages": chroma_retrieved_pages,
        "chroma_context": chroma_context,
        "chroma_label": chroma.extract_label(chroma_result),
        "chroma_score": web_crawler.extract_regression_score(chroma_result),
        "chroma_justification": chroma.extract_justification(chroma_result),
        "web_rag_context": web_info,
        "web_rag_urls": url_list,
        "web_rag_label": chroma.extract_label(web_rag_result),
        "web_rag_score": web_crawler.extract_regression_score(web_rag_result),
        "web_rag_justification": web.extract_justification(web_rag_result),
        "web_rag_claim_query": claim_query,
        "news_context": news_context,
        "news_retrieved_pages": news_retrieved_sources,
        "news_label": chroma.extract_label(news_result),
        "news_score": web_crawler.extract_regression_score(news_result),
        "news_justification": news_annotator.extract_justification(news_result),
        # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        "news_kg_context": news_kg_retrieved_texts,
        "news_kg_retrieved_pages": news_kg_retrieved_sources,
        "news_kg_label": chroma.extract_label(news_kg_result),
        "news_kg_score": web_crawler.extract_regression_score(news_kg_result),
        "news_kg_justification": news_annotator.extract_justification(news_kg_result),
        "reddit_context": reddit_context,
        "reddit_retrieved_pages": reddit_retrieved_posts,
        "reddit_label": chroma.extract_label(reddit_result),
        "reddit_score": web_crawler.extract_regression_score(reddit_result),
        "reddit_justification": reddit.extract_justification(reddit_result),
        "aggregator_result": aggregator_result,
        "aggregator_label": chroma.extract_label(aggregator_result),
        "aggregator_score": web_crawler.extract_regression_score(aggregator_result),
        "aggregator_justification": chroma.extract_justification(aggregator_result),
        "first_pass_label": web_crawler.extract_label(first_pass_result),
        "first_pass_score": web_crawler.extract_regression_score(first_pass_result),
        "first_pass_justification": web_crawler.extract_justification(
            first_pass_result
        ),
        "ground_truth": ground_truth,
    }

    new_data_df = pd.DataFrame([new_data])
    result_df = pd.concat([result_df, new_data_df], ignore_index=True)
    result_df.to_csv(result_df_path, index=False)

    logger.info(f"Row added successfully")

    if parse_flag_store:
        document.save(json_output_path)
        logger.info(f'Saved the full output file to: "{json_output_path}".')

    # i += 1
    # if i >= 5:
    #     break


# result_df_path = "final_aggregation_without_intermediate_steps.csv"  # name it as you like, change aggregation prompt
# if os.path.exists(result_df_path):
#     result_df = pd.read_csv(result_df_path)
# else:
#     result_df = pd.DataFrame()


# df = pd.read_csv("experiment1_with_def.csv")
# for idx, row in df.iterrows():

#     if "id" in result_df.columns and idx in result_df["id"].values:
#         print(f"⚠️ ID {idx} already in result_df — skipping.")
#         continue

#     company = str(row["Company"]).lower().strip()
#     year = str(row["Year"]).lower().strip()
#     esg_report = company + "_" + year
#     document_name = esg_report + ".pdf"
#     claim = row["Claim"]
#     web_rag_context = row["web_rag_context"] if row["web_rag_context"] else ""
#     news_context = row["news_context"] if row["web_rag_context"] else ""
#     chroma_context = row["chroma_context"] if row["web_rag_context"] else ""

#     # add llm_aggregator funciton
#     aggregator_result = llm_agg.call_aggregator_final_2(
#         claim, chroma_context, web_rag_context, news_context
#     )
#     logger.info(f"Adding a new dataframe row....")
#     new_data = {
#         "id": idx,
#         "Claim": claim,
#         "Year": year,
#         "Company": company,
#         "chroma_context": chroma_context,
#         "web_rag_context": web_rag_context,
#         "news_context": news_context,
#         "aggregator_result": aggregator_result,
#         "aggregator_label": chroma.extract_label(aggregator_result),
#         "aggregator_score": web_crawler.extract_regression_score(aggregator_result),
#         "aggregator_justification": chroma.extract_justification(aggregator_result),
#     }
#     new_data_df = pd.DataFrame([new_data])
#     result_df = pd.concat([result_df, new_data_df], ignore_index=True)
#     result_df.to_csv(result_df_path, index=False)

#     logger.info(f"Row {idx} added successfully")
