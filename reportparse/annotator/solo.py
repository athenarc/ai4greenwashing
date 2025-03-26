from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.annotator.web_rag import WEB_RAG_Annotator
from reportparse.annotator.chroma_annotator import ChromaAnnotator
from reportparse.annotator.reddit_annotator import RedditAnnotator
from reportparse.llm_prompts import SOLO_AGGREGATOR_PROMPT
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

logger = getLogger(__name__)


@BaseAnnotator.register("solo")
class LLMAggregator(BaseAnnotator):

    def __init__(self):
        load_dotenv()
        self.web = WEB_RAG_Annotator()
        self.chroma = ChromaAnnotator()
        self.reddit = RedditAnnotator()
        self.mongo_client = MongoClient("mongodb://localhost:27017/")
        self.mongo_db = self.mongo_client["pdf_annotations"]  # Database name
        self.mongo_collection = self.mongo_db["annotations"]  # Collection name
        self.agg_prompt = SOLO_AGGREGATOR_PROMPT
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

    def call_aggregator(self, claim, chroma_context, web_rag_context, reddit_context):
        messages = [
            (
                "system",
                self.agg_prompt,
            ),
            (
                "human",
                f"""Statement: {claim}  
                Document Context: {chroma_context}  
                Web Context: {web_rag_context}
                Reddit Context: {reddit_context}
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
        gw_pages = args.pages_to_gw if args.pages_to_gw  is not None else len(document.pages)
        print(f"Annotating {gw_pages} pages")
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
        # todo: differenciate between storing pages and greenwashing pages
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

        gw_index = 0
        print(f"Checking the first {gw_pages} pages for greenwashing")
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

            if level == "page":
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                # call the first llm from chroma, that finds all potential greenwashing claims
                result = self.chroma.call_llm(text)
                result = str(result)
                logger.info(f"First pass result: {result}")
                # add initial greenwashing detection without any annotators
                _annotate(
                    _annotate_obj=page,
                    _text=result,
                    annotator_name="first_pass",
                    metadata=json.dumps({"info": "Simple greenwashing detection"}),
                )
                logger.info("First pass completed")
                page_number = page.num
                claims = re.findall(r"(?i)(?:\b\w*\s*)*claim:\s*(.*?)(?:\n|$)", result)
                company_name = re.findall(
                    r"(?i)(?:\b\w*\s*)*Company Name:\s*(.*?)(?:\n|$)", result
                )

                claims = [c.strip() for c in claims]
                claim_index = 0
                logger.info("Starting for loop")
                for c in claims:
                    logger.info("Chroma starting")
                    chroma_context, retrieved_pages = self.chroma.retrieve_context(
                        c,
                        document.name,
                        page_number,
                        self.chroma.chroma_db,
                        k=6,
                        use_chunks=use_chunks,
                    )

                    logger.info("Reddit starting")
                    reddit_context, retrieved_reddit_posts = (
                        self.reddit.retrieve_context(
                            claim=c,
                            company_name=company_name,
                            db=self.reddit.reddit_db,
                            k=6,
                        )
                    )

                    logger.info("Web rag starting")
                    web_context, url_list, _ = self.web.search_ddg(
                        c, 3, company_name
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

                    logger.info("Aggregator starting")
                    aggregator_result = self.call_aggregator(
                        c, chroma_context, reddit_context, web_context
                    )
                    print("Aggregator result: ", aggregator_result)
                    if aggregator_result:
                        normalized_agg_rag_result = self.eval.normalize_to_string(
                            aggregator_result
                        )
                        result_list = [chroma_context, reddit_context, web_context]
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
                            normalized_agg_rag_result, normalized_result_list
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
                        annotator_name=f"aggregator_result_claim_{claim_index}",
                        metadata=json.dumps(
                            {
                                "claim": c,
                                "chroma_context": chroma_context,
                                "retreived_pages": retrieved_pages,
                                "reddit_context": reddit_context,
                                "retreived_reddit_posts": retrieved_reddit_posts,
                                "web_context": web_context,
                                "url_list": url_list,
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
                    claim_index += 1
                gw_index += 1
            else:
                return "Page extraction failed"

        return document

    def add_argument(self, parser: argparse.ArgumentParser):

        parser.add_argument(
            "--solo_agg_text_level",
            type=str,
            choices=["page", "sentence", "block"],
            default="page",
        )

        parser.add_argument(
            "--solo_agg_target_layouts",
            type=str,
            nargs="+",
            default=["text", "list", "cell"],
            choices=LAYOUT_NAMES,
        )
