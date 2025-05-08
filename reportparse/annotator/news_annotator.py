import os
import re
import argparse
from dotenv import load_dotenv
from logging import getLogger
from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES
from reportparse.news_db.news_chroma_handler import NewsChromaHandler
from reportparse.llm_prompts import FIRST_PASS_PROMPT, NEWS_PROMPT
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

import json

logger = getLogger(__name__)


@BaseAnnotator.register("news")
class NewsAnnotator(BaseAnnotator):

    def __init__(self):
        load_dotenv()
        self.news_db = NewsChromaHandler()
        self.first_pass_prompt = FIRST_PASS_PROMPT
        self.news_prompt = NEWS_PROMPT
        if os.getenv("USE_GROQ_API") == "True":
            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL"),
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=1,
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )

            self.llm_2 = ChatGroq(
                model=os.getenv("GROQ_LLM_MODEL_1"),
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=1,
                groq_api_key=os.getenv("GROQ_API_KEY_1"),
            )
        else:
            self.llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)
            self.llm_2 = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)
        return

    def call_llm(self, text):
        messages = [
            (
                "system",
                self.first_pass_prompt,
            ),
            ("human", text),
        ]

        try:
            ai_msg = self.llm.invoke(messages)
            print("AI message 1: ", ai_msg.content)
            return ai_msg.content
        except Exception as e:
            print(e)
            try:
                ai_msg = self.llm_2.invoke(messages)
                return ai_msg.content
            except Exception as e:
                print("llm invokation failed. Returning none...")
                print(e)
                return None

    def retrieve_context(self, claim, company_name, db, k=6, distance=0.6):
        try:
            result, metadata = self.news_db.retrieve_by_organization(
                company_name, claim
            )

            return result, metadata
        # try:
        #     logger.info("Retrieving context from NewsDB...")

        #     # Query only relevant posts
        #     results = db.collection.query(
        #         query_texts=[claim],
        #         n_results=k,
        #     )

        #     if results is None:
        #         return "", []
        #     relevant_texts = []
        #     retrieved_sources = []
        #     print(f"COMPANY_NAME: {company_name}")

        #     for i, (doc, score) in enumerate(
        #         zip(results["documents"], results["distances"])
        #     ):
        #         print("distance: ", score[0])
        #         if score[0] > distance:
        #             continue

        #         metadata = results["metadatas"][i][0] if results["metadatas"][i] else {}

        #         # Handle single or multiple target companies
        #         if isinstance(company_name, str):
        #             target_companies = [company_name.lower()]
        #             print("TARGET COMPANY IS: ", target_companies)
        #         elif isinstance(company_name, list):
        #             target_companies = [c.lower() for c in company_name]
        #             print("TARGET COMPANY LIST IS: ", target_companies)
        #         else:
        #             target_companies = []
        #             print("TARGET COMPANY LIST 2 IS: ", target_companies)
        #         company_words = set(
        #             re.split(
        #                 r"[\s,]+",
        #                 str(metadata.get("organization", ""))
        #                 .replace("[", "")
        #                 .replace("]", "")
        #                 .lower(),
        #             )
        #         )
        #         print("COMPANY WORDS IS: ", company_words)
        #         target_companies = [c.lower() for c in company_name]
        #         print("TARGET COMPANY 2 IS: ", target_companies)
        #         if not any(tc in company_words for tc in target_companies):
        #             continue

        #         # Append if passed
        #         relevant_texts.append(f"From News database: :\n{doc[0]}")

        #     return "\n\n".join(relevant_texts).strip(), retrieved_sources

        except Exception as e:
            logger.error(f"Error retrieving context from News Database: {e}")
            return "", []

    def verify_claim_with_context(self, claim, context):
        if context:
            messages = [
                (
                    "system",
                    self.news_prompt,
                ),
                (
                    "human",
                    f""" Statement: {claim}
                    Context from news: {context}
                    """,
                ),
            ]
            try:
                logger.info("Calling LLM to verify claim with context")
                ai_msg = self.llm.invoke(messages)
                print("AI message: ", ai_msg.content)
                return ai_msg.content
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                return "Error: Could not generate a response."
        else:
            return "No content in News database."

    def call_news_db(self, claim, company_name, news_db, k=6):
        context, retrieved_sources = self.retrieve_context(
            claim=claim, company_name=company_name, db=news_db, k=k
        )

        result = self.verify_claim_with_context(
            claim=claim,
            context=context,
        )
        return result, retrieved_sources, context

    def extract_label(self, text):
        try:
            match = re.search(
                r"Result of the statement:(.*?)Justification:", text, re.DOTALL
            )
            return match.group(1).strip() if match else ""
        except Exception as e:
            print(f"Error during label extraction: {e}")
            return None

    def extract_justification(self, text):
        try:
            match = re.search(r"Justification:\s*(.*)", text, re.DOTALL)
            return match.group(1).strip() if match else ""
        except Exception as e:
            print(f"Error during justification extraction: {e}")
            return None

    def add_argument(self, parser: argparse.ArgumentParser):

        parser.add_argument("--news_annotator_name", type=str, default="news")

        parser.add_argument(
            "--news_text_level",
            type=str,
            choices=["page", "sentence", "block"],
            default="page",
        )

        parser.add_argument(
            "--news_target_layouts",
            type=str,
            nargs="+",
            default=["text", "list", "cell"],
            choices=LAYOUT_NAMES,
        )

        parser.add_argument(
            "--use_news", action="store_true", help="Enable news_db usage"
        )

        parser.add_argument(
            "--news_pages_to_gw",
            type=int,
            help=f"Choose between 1 and esg-report max page number",
            default=1,
        )
