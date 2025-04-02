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
from reportparse.remove_thinking import remove_think_blocks
import itertools

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

        self.llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)
        self.llm_2 = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)
        return

    def call_aggregator(self, claim, context_dict):
        # Dynamically construct the message string based on the order in context_dict
        context_str = f"Statement: {claim}\n"
        for label, content in context_dict.items():
            context_str += f"{label}: {content}\n"

        messages = [
            ("system", self.agg_prompt),
            ("human", context_str),
        ]

        try:
            logger.info("Calling LLM aggregator to verify claim with context")
            try:
                ai_msg = self.llm.invoke(messages)
                print("AI message: ", ai_msg.content)
                msg = remove_think_blocks(ai_msg.content)
                return msg
            except Exception as e:
                print(f"Invocation error: {e}. Invoking with the second llm....")
                ai_msg = self.llm_2.invoke(messages)
                print("AI message: ", ai_msg.content)
                msg = remove_think_blocks(ai_msg.content)
                return msg
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "Error: Could not generate a response."

        
    def run_all_context_permutations(self, claim, chroma_context, web_rag_context, reddit_context):
        context_labels = ["Document Context", "Web Context", "Reddit Context"]
        context_values = [chroma_context, web_rag_context, reddit_context]
        results = []

        for perm in itertools.permutations(zip(context_labels, context_values)):
            context_dict = {label: value for label, value in perm}
            print("\n=== Running permutation ===")
            print(f"Order: {[label for label, _ in perm]}")
            result = self.call_aggregator(claim, context_dict)
            results.append({
                "order": [label for label, _ in perm],
                "result": result
            })

        return results


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
                claims = re.findall(r"(?i)\b(?:another )?potential greenwashing claim:\s*(.*?)(?:\n|$)", result)
                logger.info(f"Claims extracted: {claims}")
                company_match = re.search(r"(?i)\bcompany name:\s*(.*?)(?:\n|$)", result)
                company_name = company_match.group(1).strip() if company_match else "Unknown"

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
                    agg_permutation_results = self.run_all_context_permutations(
                        claim=c,
                        chroma_context=chroma_context,
                        web_rag_context=web_context,
                        reddit_context=reddit_context
                    )

                    for perm_result in agg_permutation_results:
                        permutation_order = perm_result["order"]
                        aggregator_result = perm_result["result"]
                        context_dict = {label: context for label, context in zip(permutation_order, [chroma_context, web_context, reddit_context])}

                        print(f"Aggregator result for {permutation_order}: {aggregator_result}")

                        if aggregator_result:
                            normalized_agg_rag_result = self.eval.normalize_to_string(aggregator_result)

                            # Build context list directly from the context_dict in the permutation order
                            context_list = [context_dict[label] for label in permutation_order]
                            normalized_context_list = self.eval.normalize_to_string(context_list)
                            agg_chunks = self.eval.chunk_text(normalized_context_list)

                            faith_eval = self.eval.faith_eval(
                                answer=normalized_agg_rag_result,
                                retrieved_docs=normalized_context_list,
                                precomputed_chunks=agg_chunks,
                            )
                            groundedness_eval = self.eval.groundedness_eval(
                                answer=normalized_agg_rag_result,
                                retrieved_docs=normalized_context_list,
                                precomputed_chunks=agg_chunks,
                            )
                            readability_eval = self.eval.readability_eval(normalized_agg_rag_result)
                            redundancy_eval = self.eval.redundancy_eval(
                                normalized_agg_rag_result, precomputed_chunks=agg_chunks
                            )
                            specificity_eval = self.eval.specificity_eval(normalized_agg_rag_result)
                            compression_ratio_eval = self.eval.compression_ratio_eval(
                                normalized_agg_rag_result, normalized_context_list
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

                        # Generate annotator name like: aggregator_DocumentWebReddit_result_claim_0
                        perm_name = "_".join([label.replace(" ", "_") for label in permutation_order])
                        annotator_label = f"aggregator_{perm_name}_result_claim_{claim_index}"

                        _annotate(
                            _annotate_obj=page,
                            _text=aggregator_result,
                            annotator_name=annotator_label,
                            metadata=json.dumps({
                                "claim": c,
                                "permutation_order": permutation_order,
                                "chroma_context": chroma_context,
                                "retreived_pages": retrieved_pages,
                                "reddit_context": reddit_context,
                                "retreived_reddit_posts": retrieved_reddit_posts,
                                "web_context": web_context,
                                "url_list": url_list,
                                "label": self.web.extract_label(aggregator_result),
                                "justification": self.web.extract_justification(aggregator_result),
                                "faith_eval": faith_eval,
                                "groundedness_eval": groundedness_eval,
                                "readability_eval": readability_eval,
                                "redundancy_eval": redundancy_eval,
                                "specificity_eval": specificity_eval,
                                "compression_ratio_eval": compression_ratio_eval,
                                "lexical_diversity_eval": lexical_diversity_eval,
                                "noun_to_verb_ratio_eval": noun_to_verb_ratio_eval,
                            }),
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
