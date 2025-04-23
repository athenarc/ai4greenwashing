from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.annotator.web_rag import WEB_RAG_Annotator
from reportparse.annotator.chroma_annotator import ChromaAnnotator
from reportparse.annotator.reddit_annotator import RedditAnnotator
from reportparse.annotator.chroma_esg_annotator import ChromaESGAnnotator
from reportparse.annotator.news_annotator import NewsAnnotator
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


@BaseAnnotator.register("llm_agg")
class LLMAggregator(BaseAnnotator):

    def __init__(self):
        load_dotenv()
        self.web = WEB_RAG_Annotator()
        self.chroma = ChromaAnnotator()
        self.reddit = RedditAnnotator()
        self.chroma_esg = ChromaESGAnnotator()
        self.news_annotator = NewsAnnotator()
        self.mongo_client = MongoClient("mongodb://localhost:27017/")
        self.mongo_db = self.mongo_client["pdf_annotations"]  # Database name
        self.mongo_collection = self.mongo_db["annotations"]  # Collection name
        self.agg_prompt_final = LLM_AGGREGATOR_PROMPT_FINAL
        self.agg_prompt = LLM_AGGREGATOR_PROMPT
        self.agg_prompt_2 = LLM_AGGREGATOR_PROMPT_2
        self.chroma_result = ""
        self.web_rag_result = ""
        self.reddit_result = ""
        self.chroma_esg_result = ""
        self.news_result = ""
        self.web_info = ""
        self.news_flag = True
        self.chroma_esg_flag = True
        self.web_rag_flag = False
        self.chroma_db_flag = True
        self.reddit_flag = True
        self.aggregator_flag = True
        self.first_pass_flag = True
        self.eval = llm_evaluation()

        if os.getenv("USE_GROQ_API") == "True":

            self.llm_2 = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL"),
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=1,
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )

            self.llm = ChatGroq(
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

    def call_aggregator(self, claim, chroma_result, web_rag_result):
        messages = [
            (
                "system",
                self.agg_prompt,
            ),
            (
                "human",
                f"""Statement: {claim}  
                Database Verdict: {chroma_result}  
                Web Verdict: {web_rag_result}  
                """,
            ),
        ]
        try:
            logger.info("Calling LLM aggregator to verify claim with context")
            try:

                ai_msg = self.llm.invoke(messages)
                print("AI message: ", ai_msg.content)
                return ai_msg.content
            except Exception as e:
                print(f"Invokation error: {e}. Invoking with the second llm....")
                ai_msg = self.llm_2.invoke(messages)
                print("AI message: ", ai_msg.content)
                return ai_msg.content
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "Error: Could not generate a response."

    def call_aggregator_2(self, claim, chroma_result, web_rag_result):
        messages = [
            (
                "system",
                self.agg_prompt,
            ),
            (
                "human",
                f"""Statement: {claim}  
                Web Verdict: {web_rag_result}  
                Database Verdict: {chroma_result}  
                """,
            ),
        ]
        try:
            logger.info("Calling LLM to verify claim with context")
            try:
                ai_msg = self.llm.invoke(messages)
                print("AI message: ", ai_msg.content)
                return ai_msg.content
            except Exception as e:
                print(f"Invokation error: {e}. Invoking with the second llm....")
                ai_msg = self.llm_2.invoke(messages)
                print("AI message: ", ai_msg.content)
                return ai_msg.content
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "Error: Could not generate a response."

    def call_aggregator_final(
        self,
        claim,
        chroma_result,
        web_rag_result,
        reddit_result,
        chroma_esg_result,
        news_result,
    ):

        messages = [
            (
                "system",
                self.agg_prompt_final,
            ),
            (
                "human",
                f"""Statement: {claim}  
                Chroma Verdict: {chroma_result}
                Web Verdict: {web_rag_result }
                Reddit Verdict: {reddit_result}
                Chroma-ESG Verdict: {chroma_esg_result}
                News Verdict: {news_result} 
                """,
            ),
        ]
        try:
            logger.info("Calling LLM aggregator to verify claim with context")
            try:
                ai_msg = self.llm.invoke(messages)
                print("AI message: ", ai_msg.content)
                return ai_msg.content
            except Exception as e:
                print(f"Invokation error: {e}. Invoking with the second llm....")
                ai_msg = self.llm_2.invoke(messages)
                print("AI message: ", ai_msg.content)
                return ai_msg.content
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "Error: Could not generate a response."

    def annotate(
        self,
        document: Document,
        args=None,
        level="page",
        target_layouts=("text", "list", "cell"),
        annotator_name="llm-test",
    ) -> Document:
        annotator_name = (
            args.web_rag_annotator_name if args is not None else annotator_name
        )
        level = args.web_rag_text_level if args is not None else level
        target_layouts = (
            args.web_rag_target_layouts if args is not None else list(target_layouts)
        )
        gw_pages = (
            args.pages_to_gw if args.pages_to_gw is not None else len(document.pages)
        )
        use_chunks = args.use_chunks if args.use_chunks is not None else False
        start_page = args.start_page if args.start_page is not None else 0
        if start_page > len(document.pages):
            print("Start page is greater than the number of pages in the document")
            start_page = 0
            print("Starting from page 0")

        def _annotate(
            _annotate_obj: AnnotatableLevel, _text: str, annotator_name: str, metadata
        ):
            _annotate_obj.add_annotation(
                annotation=Annotation(
                    parent_object=_annotate_obj,
                    annotator=annotator_name,
                    value=_text,
                    meta=json.loads(metadata),
                )
            )

        # add pages to chroma_db. The total number of stored pages is defined by the --max_pages parameter
        print("Starting storing in Chroma")
        for page in document.pages:
            if level == "page":
                page_number = page.num
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                self.chroma.chroma_db.store_page(
                    doc_name=document.name, page_number=page_number, text=text
                )
                print(f"Stored page {page_number}")
            else:
                print("Page level is not specified")
        print("Successfully stored all pages in chroma")

        # temporary block of code to run for the past Hitachi esg reports
        # print("Starting storing in Chroma ESG")
        # for page in document.pages:
        #     if level == "page":
        #         page_number = page.num
        #         text = page.get_text_by_target_layouts(target_layouts=target_layouts)
        #         self.chroma_esg.chroma_esg_db.store_page(
        #             doc_name=document.name, page_number=page_number, text=text
        #         )
        #         print(f"Stored page {page_number}")
        #     else:
        #         print("Page level is not specified")
        # print("Successfully stored all pages in chroma")

        gw_index = 0
        print(f"Checking the first {gw_pages} pages for greenwashing")
        company_name = None
        for page in document.pages:
            if page.num < start_page:
                continue
            if gw_index >= gw_pages:
                break
            pdf_name = document.name
            page_number = page.num

            # check if doc exists
            existing_doc = self.mongo_collection.find_one({"name": pdf_name})

            if existing_doc:
                # Get the stored page with the same number
                existing_page = next(
                    (p for p in existing_doc["pages"] if p["num"] == page_number), None
                )

                if (
                    existing_page
                    and "annotations" in existing_page
                    and existing_page["annotations"]
                ):
                    print(
                        f"Skipping page {page_number} of {pdf_name} (already annotated)."
                    )
                    gw_index += 1
                    continue  # Skip pages that already have annotations
            # If the page has no existing annotations or doesn't exist, process it
            print(f"Processing page {page_number} of {pdf_name} (annotations missing).")

            ##News annotator
            # obj = NewsAnnotator()
            # obj.news_db.store_articles()

            if level == "page":
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)

                if self.first_pass_flag:
                    # call the first llm from chroma, that finds all potential greenwashing claims
                    result = self.chroma.call_llm(text)
                    result = str(result)

                    # add initial greenwashing detection without any annotators
                    _annotate(
                        _annotate_obj=page,
                        _text=result,
                        annotator_name="first_pass",
                        metadata=json.dumps({"info": "Simple greenwashing detection"}),
                    )

                    page_number = page.num
                    # claims = re.findall(r"(?i)(?:\b\w*\s*)*claim:\s*(.*?)(?:\n|$)", result)
                    claims = re.findall(
                        r"(?i)(?:another\s+)?potential greenwashing claim:\s*(.*)",
                        result,
                    )
                    # company_name = re.findall(
                    #     r"(?i)(?:\b\w*\s*)*Company Name:\s*(.*?)(?:\n|$)", result
                    # )
                    company_name = re.findall(r"(?i)Company Name:\s*(.*)", result)
                    claims = [c.strip() for c in claims]
                    claim_index = 0
                else:
                    claims = []
                for c in claims:

                    if self.chroma_esg_flag:
                        logger.info("Chroma_ESG starting")
                        chroma_esg_result, retrieved_sources, context = (
                            self.chroma_esg.call_chroma_esg(
                                c, self.chroma_esg.chroma_esg_db, k=6
                            )
                        )
                        print("llm result: ", chroma_esg_result)
                        if context:
                            self.chroma_esg_result = chroma_esg_result
                            normalized_chroma_esg_result = (
                                self.eval.normalize_to_string(chroma_esg_result)
                            )
                            chroma_esg_chunks = self.eval.chunk_text(
                                self.eval.normalize_to_string(context)
                            )

                            faith_eval = self.eval.faith_eval(
                                answer=normalized_chroma_esg_result,
                                retrieved_docs=context,
                                precomputed_chunks=chroma_esg_chunks,
                            )
                            groundedness_eval = self.eval.groundedness_eval(
                                answer=normalized_chroma_esg_result,
                                retrieved_docs=context,
                                precomputed_chunks=chroma_esg_chunks,
                            )
                            readability_eval = self.eval.readability_eval(
                                normalized_chroma_esg_result
                            )
                            redundancy_eval = self.eval.redundancy_eval(
                                normalized_chroma_esg_result,
                                precomputed_chunks=chroma_esg_chunks,
                            )
                            specificity_eval = self.eval.specificity_eval(
                                normalized_chroma_esg_result
                            )
                            compression_ratio_eval = self.eval.compression_ratio_eval(
                                normalized_chroma_esg_result, context
                            )
                            lexical_diversity_eval = self.eval.lexical_diversity_eval(
                                normalized_chroma_esg_result
                            )
                            noun_to_verb_ratio_eval = self.eval.noun_to_verb_ratio_eval(
                                normalized_chroma_esg_result
                            )
                        else:
                            faith_eval = groundedness_eval = readability_eval = (
                                redundancy_eval
                            ) = None
                            specificity_eval = compression_ratio_eval = (
                                lexical_diversity_eval
                            ) = noun_to_verb_ratio_eval = None

                        claim_dict = {
                            "claim": c,
                            "retrieved_esg_report_chunk": retrieved_sources,
                            "label": self.chroma.extract_label(chroma_esg_result),
                            "justification": self.chroma.extract_justification(
                                chroma_esg_result
                            ),
                            "context": context,
                            "faith_eval": faith_eval,
                            "groundedness_eval": groundedness_eval,
                            "readability_eval": readability_eval,
                            "redundancy_eval": redundancy_eval,
                            "specificity_eval": specificity_eval,
                            "compression_ratio_eval": compression_ratio_eval,
                            "lexical_diversity_eval": lexical_diversity_eval,
                            "noun_to_verb_ratio_eval": noun_to_verb_ratio_eval,
                        }

                        json_output = json.dumps(claim_dict)

                        _annotate(
                            _annotate_obj=page,
                            _text=chroma_esg_result,
                            annotator_name=f"chroma_esg_result_claim_{claim_index}",
                            metadata=json_output,
                        )

                    logger.info("News Annotator starting")
                    if self.news_flag:
                        news_result, retrieved_sources, context = (
                            self.news_annotator.call_news_db(
                                c, list(company_name), self.news_annotator.news_db, k=6
                            )
                        )
                        print("Second llm result: ", news_result)
                        if context:
                            normalized_news_result = self.eval.normalize_to_string(
                                news_result
                            )
                            self.news_result = news_result
                            news_chunks = self.eval.chunk_text(
                                self.eval.normalize_to_string(context)
                            )

                            faith_eval = self.eval.faith_eval(
                                answer=normalized_news_result,
                                retrieved_docs=context,
                                precomputed_chunks=news_chunks,
                            )
                            groundedness_eval = self.eval.groundedness_eval(
                                answer=normalized_news_result,
                                retrieved_docs=context,
                                precomputed_chunks=news_chunks,
                            )
                            readability_eval = self.eval.readability_eval(
                                normalized_news_result
                            )
                            redundancy_eval = self.eval.redundancy_eval(
                                normalized_news_result,
                                precomputed_chunks=news_chunks,
                            )
                            specificity_eval = self.eval.specificity_eval(
                                normalized_news_result
                            )
                            compression_ratio_eval = self.eval.compression_ratio_eval(
                                normalized_news_result, context
                            )
                            lexical_diversity_eval = self.eval.lexical_diversity_eval(
                                normalized_news_result
                            )
                            noun_to_verb_ratio_eval = self.eval.noun_to_verb_ratio_eval(
                                normalized_news_result
                            )
                        else:
                            faith_eval = groundedness_eval = readability_eval = (
                                redundancy_eval
                            ) = None
                            specificity_eval = compression_ratio_eval = (
                                lexical_diversity_eval
                            ) = noun_to_verb_ratio_eval = None

                        claim_dict = {
                            "claim": c,
                            "retrieved_esg_report_chunk": retrieved_sources,
                            "label": self.chroma.extract_label(news_result),
                            "justification": self.chroma.extract_justification(
                                news_result
                            ),
                            "context": context,
                            "faith_eval": faith_eval,
                            "groundedness_eval": groundedness_eval,
                            "readability_eval": readability_eval,
                            "redundancy_eval": redundancy_eval,
                            "specificity_eval": specificity_eval,
                            "compression_ratio_eval": compression_ratio_eval,
                            "lexical_diversity_eval": lexical_diversity_eval,
                            "noun_to_verb_ratio_eval": noun_to_verb_ratio_eval,
                        }

                        json_output = json.dumps(claim_dict)

                        _annotate(
                            _annotate_obj=page,
                            _text=news_result,
                            annotator_name=f"news_db_result_claim_{claim_index}",
                            metadata=json_output,
                        )

                    # add aggregation with chroma db
                    if self.chroma_db_flag:
                        logger.info("Chroma annotator starting")
                        chroma_result, retrieved_pages, context = (
                            self.chroma.call_chroma(
                                c,
                                document.name,
                                text,
                                page_number,
                                self.chroma.chroma_db,
                                k=6,
                                use_chunks=use_chunks,
                            )
                        )
                        print("llm result: ", chroma_result)
                        if context:
                            normalized_chroma_result = self.eval.normalize_to_string(
                                chroma_result
                            )
                            self.chroma_result = chroma_result
                            chroma_chunks = self.eval.chunk_text(
                                self.eval.normalize_to_string(context)
                            )

                            faith_eval = self.eval.faith_eval(
                                answer=normalized_chroma_result,
                                retrieved_docs=context,
                                precomputed_chunks=chroma_chunks,
                            )
                            groundedness_eval = self.eval.groundedness_eval(
                                answer=normalized_chroma_result,
                                retrieved_docs=context,
                                precomputed_chunks=chroma_chunks,
                            )
                            readability_eval = self.eval.readability_eval(
                                normalized_chroma_result
                            )
                            redundancy_eval = self.eval.redundancy_eval(
                                normalized_chroma_result,
                                precomputed_chunks=chroma_chunks,
                            )
                            specificity_eval = self.eval.specificity_eval(
                                normalized_chroma_result
                            )
                            compression_ratio_eval = self.eval.compression_ratio_eval(
                                normalized_chroma_result, context
                            )
                            lexical_diversity_eval = self.eval.lexical_diversity_eval(
                                normalized_chroma_result
                            )
                            noun_to_verb_ratio_eval = self.eval.noun_to_verb_ratio_eval(
                                normalized_chroma_result
                            )
                        else:
                            faith_eval = groundedness_eval = readability_eval = (
                                redundancy_eval
                            ) = None
                            specificity_eval = compression_ratio_eval = (
                                lexical_diversity_eval
                            ) = noun_to_verb_ratio_eval = None

                        claim_dict_chroma = {
                            "claim": c,
                            "retrieved_pages": retrieved_pages,
                            "label": self.chroma.extract_label(chroma_result),
                            "justification": self.chroma.extract_justification(
                                chroma_result
                            ),
                            "context": context,
                            "faith_eval": faith_eval,
                            "groundedness_eval": groundedness_eval,
                            "readability_eval": readability_eval,
                            "redundancy_eval": redundancy_eval,
                            "specificity_eval": specificity_eval,
                            "compression_ratio_eval": compression_ratio_eval,
                            "lexical_diversity_eval": lexical_diversity_eval,
                            "noun_to_verb_ratio_eval": noun_to_verb_ratio_eval,
                        }

                        json_output = json.dumps(claim_dict_chroma)

                        _annotate(
                            _annotate_obj=page,
                            _text=chroma_result,
                            annotator_name=f"chroma_result_claim_{claim_index}",
                            metadata=json_output,
                        )

                    if self.reddit_flag:
                        logger.info("Reddit starting")
                        reddit_result, retrieved_posts, context = (
                            self.reddit.call_reddit(
                                c,
                                company_name,
                                self.reddit.reddit_db,
                                k=6,
                            )
                        )

                        print("llm result: ", reddit_result)
                        if context:
                            normalized_reddit_result = self.eval.normalize_to_string(
                                reddit_result
                            )
                            self.reddit_result = reddit_result
                            reddit_chunks = self.eval.chunk_text(
                                self.eval.normalize_to_string(context)
                            )

                            faith_eval = self.eval.faith_eval(
                                answer=normalized_reddit_result,
                                retrieved_docs=context,
                                precomputed_chunks=reddit_chunks,
                            )
                            groundedness_eval = self.eval.groundedness_eval(
                                answer=normalized_reddit_result,
                                retrieved_docs=context,
                                precomputed_chunks=reddit_chunks,
                            )
                            readability_eval = self.eval.readability_eval(
                                normalized_reddit_result
                            )
                            redundancy_eval = self.eval.redundancy_eval(
                                normalized_reddit_result,
                                precomputed_chunks=reddit_chunks,
                            )
                            specificity_eval = self.eval.specificity_eval(
                                normalized_reddit_result
                            )
                            compression_ratio_eval = self.eval.compression_ratio_eval(
                                normalized_reddit_result, context
                            )
                            lexical_diversity_eval = self.eval.lexical_diversity_eval(
                                normalized_reddit_result
                            )
                            noun_to_verb_ratio_eval = self.eval.noun_to_verb_ratio_eval(
                                normalized_reddit_result
                            )
                        else:
                            faith_eval = groundedness_eval = readability_eval = (
                                redundancy_eval
                            ) = None
                            specificity_eval = compression_ratio_eval = (
                                lexical_diversity_eval
                            ) = noun_to_verb_ratio_eval = None

                        claim_dict_reddit = {
                            "claim": c,
                            "retrieved_posts": retrieved_posts,
                            "label": self.chroma.extract_label(reddit_result),
                            "justification": self.chroma.extract_justification(
                                reddit_result
                            ),
                            "context": context,
                            "faith_eval": faith_eval,
                            "groundedness_eval": groundedness_eval,
                            "readability_eval": readability_eval,
                            "redundancy_eval": redundancy_eval,
                            "specificity_eval": specificity_eval,
                            "compression_ratio_eval": compression_ratio_eval,
                            "lexical_diversity_eval": lexical_diversity_eval,
                            "noun_to_verb_ratio_eval": noun_to_verb_ratio_eval,
                        }

                        json_output = json.dumps(claim_dict_reddit)

                        _annotate(
                            _annotate_obj=page,
                            _text=reddit_result,
                            annotator_name=f"reddit_result_claim_{claim_index}",
                            metadata=json_output,
                        )

                    if self.web_rag_flag:
                        logger.info("Web rag starting")
                        # add web_rag aggregation
                        print(f"SEARCHING FOR CLAIM {c}")
                        web_rag_result, url_list, web_info = self.web.web_rag(
                            c, 1, company_name
                        )
                        self.web_rag_result = web_rag_result
                        self.web_info = web_info
                        if web_info:
                            normalized_web_rag_result = self.eval.normalize_to_string(
                                web_rag_result
                            )
                            web_chunks = self.eval.chunk_text(
                                self.eval.normalize_to_string(web_info)
                            )

                            faith_eval = self.eval.faith_eval(
                                answer=normalized_web_rag_result,
                                retrieved_docs=web_info,
                                precomputed_chunks=web_chunks,
                            )
                            groundedness_eval = self.eval.groundedness_eval(
                                answer=normalized_web_rag_result,
                                retrieved_docs=web_info,
                                precomputed_chunks=web_chunks,
                            )
                            readability_eval = self.eval.readability_eval(
                                normalized_web_rag_result
                            )
                            redundancy_eval = self.eval.redundancy_eval(
                                normalized_web_rag_result, precomputed_chunks=web_chunks
                            )
                            specificity_eval = self.eval.specificity_eval(
                                normalized_web_rag_result
                            )
                            compression_ratio_eval = self.eval.compression_ratio_eval(
                                normalized_web_rag_result, web_info
                            )
                            lexical_diversity_eval = self.eval.lexical_diversity_eval(
                                normalized_web_rag_result
                            )
                            noun_to_verb_ratio_eval = self.eval.noun_to_verb_ratio_eval(
                                normalized_web_rag_result
                            )
                        else:
                            faith_eval = groundedness_eval = readability_eval = (
                                redundancy_eval
                            ) = None
                            specificity_eval = compression_ratio_eval = (
                                lexical_diversity_eval
                            ) = noun_to_verb_ratio_eval = None

                        claim_dict_webrag = {
                            "claim": c,
                            "urls": url_list,
                            "label": self.web.extract_label(web_rag_result),
                            "justification": self.web.extract_justification(
                                web_rag_result
                            ),
                            "web_info": web_info,
                            "faith_eval": faith_eval,
                            "groundedness_eval": groundedness_eval,
                            "readability_eval": readability_eval,
                            "redundancy_eval": redundancy_eval,
                            "specificity_eval": specificity_eval,
                            "compression_ratio_eval": compression_ratio_eval,
                            "lexical_diversity_eval": lexical_diversity_eval,
                            "noun_to_verb_ratio_eval": noun_to_verb_ratio_eval,
                        }

                        json_output = json.dumps(claim_dict_webrag)

                        # annotate for web rag
                        _annotate(
                            _annotate_obj=page,
                            _text=web_rag_result,
                            annotator_name=f"web_rag_result_claim_{claim_index}",
                            metadata=json_output,
                        )

                    logger.info("CTI starting")
                    cti_results = cti_classification(c)
                    _annotate(
                        _annotate_obj=page,
                        _text=c,
                        annotator_name=f"cti_results_{claim_index}",
                        metadata=json.dumps(
                            {
                                "claim": c,
                                "cti_metrics_climate": cti_results["climate"],
                                "cti_metrics_commitment": cti_results["commitment"],
                                "cti_metrics_sentiment": cti_results["sentiment"],
                                "cti_metrics_specificity": cti_results["specificity"],
                            }
                        ),
                    )

                    if self.aggregator_flag:
                        logger.info("Final Aggregator starting")
                        aggregator_result = self.call_aggregator_final(
                            c,
                            self.chroma_result,
                            self.web_rag_result,
                            self.reddit_result,
                            self.chroma_esg_result,
                            self.news_result,
                        )
                        print("Aggregator result: ", aggregator_result)
                        if aggregator_result:
                            normalized_agg_rag_result = self.eval.normalize_to_string(
                                aggregator_result
                            )
                            result_list = [
                                self.web_rag_result,
                                self.chroma_result,
                                self.reddit_result,
                                self.chroma_esg_result,
                                self.news_result,
                            ]

                            normalized_result_list = self.eval.normalize_to_string(
                                result_list
                            )
                            agg_chunks = self.eval.chunk_text(normalized_result_list)

                            faith_eval = self.eval.faith_eval(
                                answer=normalized_agg_rag_result,
                                retrieved_docs=agg_chunks,
                                precomputed_chunks=agg_chunks,
                            )
                            groundedness_eval = self.eval.groundedness_eval(
                                answer=normalized_agg_rag_result,
                                retrieved_docs=agg_chunks,
                                precomputed_chunks=agg_chunks,
                            )
                            readability_eval = self.eval.readability_eval(
                                normalized_agg_rag_result
                            )
                            redundancy_eval = self.eval.redundancy_eval(
                                normalized_agg_rag_result, precomputed_chunks=agg_chunks
                            )
                            specificity_eval = self.eval.specificity_eval(
                                normalized_agg_rag_result
                            )
                            compression_ratio_eval = self.eval.compression_ratio_eval(
                                normalized_agg_rag_result, self.web_info
                            )
                            lexical_diversity_eval = self.eval.lexical_diversity_eval(
                                normalized_agg_rag_result
                            )
                            noun_to_verb_ratio_eval = self.eval.noun_to_verb_ratio_eval(
                                normalized_agg_rag_result
                            )
                        else:
                            faith_eval = groundedness_eval = readability_eval = (
                                redundancy_eval
                            ) = None
                            specificity_eval = compression_ratio_eval = (
                                lexical_diversity_eval
                            ) = noun_to_verb_ratio_eval = None

                        _annotate(
                            _annotate_obj=page,
                            _text=aggregator_result,
                            annotator_name=f"final_aggregator_result_claim_{claim_index}",
                            metadata=json.dumps(
                                {
                                    "claim": c,
                                    "chroma_result": self.chroma_result,
                                    "web_rag_result": self.web_rag_result,
                                    "reddit_result": self.reddit_result,
                                    "chroma_esg_result": self.chroma_esg_result,
                                    "news_result": self.news_result,
                                    "label": self.web.extract_label(aggregator_result),
                                    "justification": self.web.extract_justification(
                                        aggregator_result
                                    ),
                                    "faith_eval": faith_eval,
                                    "groundedness_eval": groundedness_eval,
                                    "readability_eval": readability_eval,
                                    "redundancy_eval": redundancy_eval,
                                    "specificity_eval": specificity_eval,
                                    "compression_ratio_eval": compression_ratio_eval,
                                    "lexical_diversity_eval": lexical_diversity_eval,
                                    "noun_to_verb_ratio_eval": noun_to_verb_ratio_eval,
                                }
                            ),
                        )

                    # logger.info("Aggregator starting")
                    # aggregator_result = self.call_aggregator(
                    #     c, chroma_result, web_rag_result
                    # )
                    # print("Aggregator result: ", aggregator_result)
                    # if aggregator_result:
                    #     normalized_agg_rag_result = self.eval.normalize_to_string(
                    #         aggregator_result
                    #     )
                    #     result_list = [chroma_result, web_rag_result]
                    #     normalized_result_list = self.eval.normalize_to_string(result_list)
                    #     agg_chunks = self.eval.chunk_text(normalized_result_list)

                    #     faith_eval = self.eval.faith_eval(
                    #         answer=normalized_agg_rag_result,
                    #         retrieved_docs=agg_chunks,
                    #         precomputed_chunks=agg_chunks,
                    #     )
                    #     groundedness_eval = self.eval.groundedness_eval(
                    #         answer=normalized_agg_rag_result,
                    #         retrieved_docs=agg_chunks,
                    #         precomputed_chunks=agg_chunks,
                    #     )
                    #     readability_eval = self.eval.readability_eval(
                    #         normalized_agg_rag_result
                    #     )
                    #     redundancy_eval = self.eval.redundancy_eval(
                    #         normalized_agg_rag_result, precomputed_chunks=agg_chunks
                    #     )
                    #     specificity_eval = self.eval.specificity_eval(
                    #         normalized_agg_rag_result
                    #     )
                    #     compression_ratio_eval = self.eval.compression_ratio_eval(
                    #         normalized_agg_rag_result, web_info
                    #     )
                    #     lexical_diversity_eval = self.eval.lexical_diversity_eval(
                    #         normalized_agg_rag_result
                    #     )
                    #     noun_to_verb_ratio_eval = self.eval.noun_to_verb_ratio_eval(
                    #         normalized_agg_rag_result
                    #     )
                    # else:
                    #     faith_eval = groundedness_eval = readability_eval = (
                    #         redundancy_eval
                    #     ) = None
                    #     specificity_eval = compression_ratio_eval = (
                    #         lexical_diversity_eval
                    #     ) = noun_to_verb_ratio_eval = None

                    # _annotate(
                    #     _annotate_obj=page,
                    #     _text=aggregator_result,
                    #     annotator_name=f"aggregator_result_claim_{claim_index}",
                    #     metadata=json.dumps(
                    #         {
                    #             "claim": c,
                    #             "chroma_result": chroma_result,
                    #             "web_rag_result": web_rag_result,
                    #             "label": self.web.extract_label(aggregator_result),
                    #             "justification": self.web.extract_justification(
                    #                 aggregator_result
                    #             ),
                    #             "faith_eval": faith_eval,
                    #             "groundedness_eval": groundedness_eval,
                    #             "readability_eval": readability_eval,
                    #             "redundancy_eval": redundancy_eval,
                    #             "specificity_eval": specificity_eval,
                    #             "compression_ratio_eval": compression_ratio_eval,
                    #             "lexical_diversity_eval": lexical_diversity_eval,
                    #             "noun_to_verb_ratio_eval": noun_to_verb_ratio_eval,
                    #         }
                    #     ),
                    # )

                    # logger.info("Aggregator 2 starting")
                    # aggregator_result = self.call_aggregator_2(
                    #     c, chroma_result, web_rag_result
                    # )
                    # print("Aggregator result: ", aggregator_result)
                    # if aggregator_result:
                    #     normalized_agg_rag_result = self.eval.normalize_to_string(
                    #         aggregator_result
                    #     )
                    #     result_list = [web_rag_result, chroma_result]
                    #     normalized_result_list = self.eval.normalize_to_string(result_list)
                    #     agg_chunks = self.eval.chunk_text(normalized_result_list)

                    #     faith_eval = self.eval.faith_eval(
                    #         answer=normalized_agg_rag_result,
                    #         retrieved_docs=agg_chunks,
                    #         precomputed_chunks=agg_chunks,
                    #     )
                    #     groundedness_eval = self.eval.groundedness_eval(
                    #         answer=normalized_agg_rag_result,
                    #         retrieved_docs=agg_chunks,
                    #         precomputed_chunks=agg_chunks,
                    #     )
                    #     readability_eval = self.eval.readability_eval(
                    #         normalized_agg_rag_result
                    #     )
                    #     redundancy_eval = self.eval.redundancy_eval(
                    #         normalized_agg_rag_result, precomputed_chunks=agg_chunks
                    #     )
                    #     specificity_eval = self.eval.specificity_eval(
                    #         normalized_agg_rag_result
                    #     )
                    #     compression_ratio_eval = self.eval.compression_ratio_eval(
                    #         normalized_agg_rag_result, web_info
                    #     )
                    #     lexical_diversity_eval = self.eval.lexical_diversity_eval(
                    #         normalized_agg_rag_result
                    #     )
                    #     noun_to_verb_ratio_eval = self.eval.noun_to_verb_ratio_eval(
                    #         normalized_agg_rag_result
                    #     )
                    # else:
                    #     faith_eval = groundedness_eval = readability_eval = (
                    #         redundancy_eval
                    #     ) = None
                    #     specificity_eval = compression_ratio_eval = (
                    #         lexical_diversity_eval
                    #     ) = noun_to_verb_ratio_eval = None

                    # _annotate(
                    #     _annotate_obj=page,
                    #     _text=aggregator_result,
                    #     annotator_name=f"aggregator_2_result_claim_{claim_index}",
                    #     metadata=json.dumps(
                    #         {
                    #             "claim": c,
                    #             "chroma_result": chroma_result,
                    #             "web_rag_result": web_rag_result,
                    #             "label": self.web.extract_label(aggregator_result),
                    #             "justification": self.web.extract_justification(
                    #                 aggregator_result
                    #             ),
                    #             "faith_eval": faith_eval,
                    #             "groundedness_eval": groundedness_eval,
                    #             "readability_eval": readability_eval,
                    #             "redundancy_eval": redundancy_eval,
                    #             "specificity_eval": specificity_eval,
                    #             "compression_ratio_eval": compression_ratio_eval,
                    #             "lexical_diversity_eval": lexical_diversity_eval,
                    #             "noun_to_verb_ratio_eval": noun_to_verb_ratio_eval,
                    #         }
                    #     ),
                    # )

                    claim_index += 1
                gw_index += 1
            else:
                return "Page extraction failed"

        return document

    def add_argument(self, parser: argparse.ArgumentParser):

        parser.add_argument(
            "--llm_agg_text_level",
            type=str,
            choices=["page", "sentence", "block"],
            default="page",
        )

        parser.add_argument(
            "--llm_agg_target_layouts",
            type=str,
            nargs="+",
            default=["text", "list", "cell"],
            choices=LAYOUT_NAMES,
        )

        parser.add_argument(
            "--pages_to_gw",
            type=int,
            help=f"Choose between 1 and esg-report max page number",
        )

        parser.add_argument(
            "--start_page",
            type=int,
            help=f"Choose starting page number (0-indexed)",
            default=0,
        )
