from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.annotator.web_rag import WEB_RAG_Annotator
from reportparse.annotator.chroma_annotator import LLMAnnotator
from reportparse.llm_prompts import LLM_AGGREGATOR_PROMPT
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
        self.chroma = LLMAnnotator()
        self.mongo_client = MongoClient("mongodb://localhost:27017/")
        self.mongo_db = self.mongo_client["pdf_annotations"]  # Database name
        self.mongo_collection = self.mongo_db["annotations"]  # Collection name
        self.agg_prompt = LLM_AGGREGATOR_PROMPT
        self.eval = llm_evaluation()
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
        gw_pages = args.pages_to_gw if args is not None else 1
        use_chunks = args.use_chunks if args is not None else False
        start_page = args.start_page if args is not None else 0
        if start_page > len(document.pages):
            print("Start page is greater than the number of pages in the document")
            start_page = 0
            print("Starting from page 1")

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

                # add initial greenwashing detection without any annotators
                _annotate(
                    _annotate_obj=page,
                    _text=result,
                    annotator_name="first_pass",
                    metadata=json.dumps({"info": "Simple greenwashing detection"}),
                )

                page_number = page.num
                claims = re.findall(r"(?i)(?:\b\w*\s*)*claim:\s*(.*?)(?:\n|$)", result)
                company_name = re.findall(
                    r"(?i)(?:\b\w*\s*)*Company Name:\s*(.*?)(?:\n|$)", result
                )
                claims = [c.strip() for c in claims]
                claim_index = 0
                for c in claims:
                    # add aggregation with chroma db
                    chroma_result, retrieved_pages, context = self.chroma.call_chroma(
                        c,
                        document.name,
                        text,
                        page_number,
                        self.chroma.chroma_db,
                        k=6,
                        use_chunks=use_chunks,
                    )
                    print("Second llm result: ", chroma_result)
                    chroma_justification = self.chroma.extract_justification(
                        chroma_result
                    )
                    if context:
                        chroma_chunks = self.eval.chunk_text(chroma_result)
                        faith_eval = self.eval.faith_eval(
                            answer=chroma_justification,
                            retrieved_docs=context,
                            precomputed_chunks=chroma_chunks,
                        )
                        groundedness_eval = self.eval.groundedness_eval(
                            answer=chroma_justification,
                            retrieved_docs=context,
                            precomputed_chunks=chroma_chunks,
                        )
                        readability_eval = self.eval.readability_eval(
                            chroma_justification
                        )
                        redundancy_eval = self.eval.redundancy_eval(
                            chroma_justification, precomputed_chunks=chroma_chunks
                        )
                    else:
                        faith_eval = groundedness_eval = readability_eval = (
                            redundancy_eval
                        ) = None
                    # annotate for chroma
                    claim_dict_chroma = {
                        "claim": c,
                        "retrieved_pages": retrieved_pages,
                        "label": self.chroma.extract_label(chroma_result),
                        "justification": chroma_justification,
                        "context": context,
                        "faith_eval": faith_eval,
                        "groundedness_eval": groundedness_eval,
                        "readability_eval": readability_eval,
                        "redundancy_eval": redundancy_eval,
                    }
                    json_output = json.dumps(claim_dict_chroma, default=str)
                    _annotate(
                        _annotate_obj=page,
                        _text=chroma_result,
                        annotator_name=f"chroma_result_claim_{claim_index}",
                        metadata=json_output,
                    )

                    # add web_rag aggregation
                    print(f"SEARCHING FOR CLAIM {c}")
                    web_rag_result, url_list, web_info = self.web.web_rag(
                        c, 1, company_name
                    )
                    if web_info:
                        web_rag_justification = self.web.extract_justification(
                            web_rag_result
                        )
                        web_chunks = self.eval.chunk_text(web_rag_result)
                        web_embeddings = self.eval.embedder.encode(
                            web_chunks, convert_to_tensor=True
                        )

                        faith_eval = self.eval.faith_eval(
                            answer=web_rag_justification,
                            retrieved_docs=web_info,
                            precomputed_chunks=web_chunks,
                        )
                        groundedness_eval = self.eval.groundedness_eval(
                            answer=web_rag_justification,
                            retrieved_docs=web_info,
                            precomputed_chunks=web_chunks,
                        )
                        readability_eval = self.eval.readability_eval(
                            web_rag_justification
                        )
                        redundancy_eval = self.eval.redundancy_eval(
                            web_rag_justification, precomputed_chunks=web_chunks
                        )
                    else:
                        faith_eval = groundedness_eval = readability_eval = (
                            redundancy_eval
                        ) = None
                    claim_dict_webrag = {
                        "claim": c,
                        "urls": url_list,
                        "label": self.web.extract_label(web_rag_result),
                        "justification": web_rag_justification,
                        "web_info": web_info,
                        "faith_eval": faith_eval,
                        "groundedness_eval": groundedness_eval,
                        "readability_eval": readability_eval,
                        "redundancy_eval": redundancy_eval,
                    }
                    json_output = json.dumps(claim_dict_webrag, default=str)

                    # annotate for web rag
                    _annotate(
                        _annotate_obj=page,
                        _text=web_rag_result,
                        annotator_name=f"web_rag_result_claim_{claim_index}",
                        metadata=json_output,
                    )

                    cti_results = cti_classification(c)
                    aggregator_result = self.call_aggregator(
                        c, chroma_result, web_rag_result
                    )
                    print("Aggregator result: ", aggregator_result)
                    _annotate(
                        _annotate_obj=page,
                        _text=aggregator_result,
                        annotator_name=f"aggregator_result_claim_{claim_index}",
                        metadata=json.dumps(
                            {
                                "claim": c,
                                "chroma_result": chroma_result,
                                "web_rag_result": web_rag_result,
                                "label": self.web.extract_label(aggregator_result),
                                "justification": self.web.extract_justification(
                                    aggregator_result
                                ),
                                "cti metrics_detection": cti_results["detection"],
                                "cti metrics_commitment": cti_results["commitment"],
                                "cti metrics_sentiment": cti_results["sentiment"],
                                "cti metrics_specificity": cti_results["specificity"],
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
            default=1,
        )

        parser.add_argument(
            "--start_page",
            type=int,
            help=f"Choose starting page number (0-indexed)",
            default=0,
        )
