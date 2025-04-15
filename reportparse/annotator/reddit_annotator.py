import os
import re
import string
import json
import argparse
from dotenv import load_dotenv
from logging import getLogger
from langchain_ollama import ChatOllama

from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.rags.reddit_db.reddit_chroma_handler import RedditChromaHandler
from reportparse.util.llm_prompts import FIRST_PASS_PROMPT, REDDIT_PROMPT
from reportparse.util.remove_thinking import remove_think_blocks
from reportparse.util.label_extraction import extract_label, extract_justification

logger = getLogger(__name__)


@BaseAnnotator.register("reddit")
class RedditAnnotator(BaseAnnotator):

    def __init__(self):
        load_dotenv()
        self.reddit_db = RedditChromaHandler()
        self.first_pass_prompt = FIRST_PASS_PROMPT
        self.reddit_prompt = REDDIT_PROMPT

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
            msg = remove_think_blocks(ai_msg.content)
            return msg
        except Exception as e:
            print(e)
            try:
                ai_msg = self.llm_2.invoke(messages)
                msg = remove_think_blocks(ai_msg.content)
                return msg
            except Exception as e:
                print("llm invokation failed. Returning none...")
                print(e)
                return None
    
    def normalize_company_name(self, name: str) -> set:
        # Remove punctuation and convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        cleaned = name.translate(translator)
        # Split on whitespace and commas, and filter out any empty strings
        return set(filter(None, re.split(r"[\s,]+", cleaned.lower())))

    def retrieve_context(self, claim, company_name, db, k=6, distance=0.9):
        try:
            logger.info("Retrieving context from RedditDB")
            
            # Query relevant posts based on claim (this part remains unchanged)
            results = db.collection.query(
                query_texts=[claim],
                n_results=k,
            )
            logger.info("Results from RedditDB retrieved")
            if results is None:
                return "", []
            relevant_texts = []
            retrieved_sources = []

            # Normalize the target company name for better matching
            target_company_words = self.normalize_company_name(company_name)

            for i, (doc, score) in enumerate(zip(results["documents"], results["distances"])):
                print("distance: ", score[0])
                if score[0] > distance:
                    continue

                metadata = results["metadatas"][i][0] if results["metadatas"][i] else {}
                url = metadata.get("post_url", "Unknown")

                # Normalize metadata company name
                metadata_company = str(metadata.get("company", ""))
                metadata_company_words = self.normalize_company_name(metadata_company)

                # If no shared words between target and metadata company, skip
                if not target_company_words & metadata_company_words:
                    logger.info(
                        f"Skipping post {url} due to no shared company names"
                    )
                    continue

                # Append if criteria are met
                relevant_texts.append(f"From Reddit Post ({url}):\n{doc[0]}")
                retrieved_sources.append(url)

            return "\n\n".join(relevant_texts).strip(), retrieved_sources

        except Exception as e:
            logger.error(f"Error retrieving context from RedditDB: {e}")
            return "", []

    def verify_claim_with_context(self, claim, context):
        if context:
            messages = [
                (
                    "system",
                    self.reddit_prompt,
                ),
                (
                    "human",
                    f""" Statement: {claim}
                    Reddit Context: {context}
                    """,
                ),
            ]
            try:
                logger.info("Calling LLM to verify claim with context")
                ai_msg = self.llm.invoke(messages)
                print("AI message: ", ai_msg.content)
                msg = remove_think_blocks(ai_msg.content)
                return msg
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                return "Error: Could not generate a response."
        else:
            return "No content in RedditDB"

    def call_reddit(self, claim, company_name, reddit_db, k=6):
        seacrch_string = claim + " " + company_name
        context, retrieved_sources = self.retrieve_context(claim=seacrch_string, company_name=company_name, db=reddit_db, k=k)
        print("Retrieved sources: ", retrieved_sources)
        result = self.verify_claim_with_context(claim=claim, context=context)
        return result, retrieved_sources, context

    def annotate(
        self,
        document: Document,
        args=None,
        level="block",
        target_layouts=("text", "list", "cell"),
        annotator_name="reddit",
    ) -> Document:
        annotator_name = (
            args.reddit_annotator_name if args is not None else annotator_name
        )
        level = args.reddit_text_level if args is not None else level
        target_layouts = (
            args.reddit_target_layouts if args is not None else list(target_layouts)
        )
        use_reddit = args.use_reddit if args is not None else False
        gw_pages = args.reddit_pages_to_gw if args is not None else len(document.pages)
        start_page = args.reddit_start_page if args is not None else 0
        if start_page > len(document.pages):
            print("Start page is greater than the number of pages in the document")
            start_page = 0
            print("Starting from page 0")
        print(
            "Model name: ",
            (
                os.getenv("GROQ_LLM_MODEL_1")
                if os.getenv("USE_GROQ_API") == "True"
                else os.getenv("OLLAMA_MODEL")
            ),
        )

        # Manual overrides to debug easily
        use_reddit = True

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

        gw_index = 0
        for page in document.pages:
            if page.num < start_page:
                continue
            if gw_index >= gw_pages:
                break
            if level == "page":
                print("Calling first llm to annotate")
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                result = self.call_llm(text)
                result = str(result)
                print("First llm result: ", result)
                _annotate(
                    _annotate_obj=page,
                    _text=result,
                    annotator_name="llm_result",
                    metadata=json.dumps({"info": "Simple greenwashing detection"}),
                )
                if use_reddit:
                    print("Calling second llm to annotate")
                    page_number = page.num
                    claims = re.findall(
                        r"(?i)(?:\b\w*\s*)*claim:\s*(.*?)(?:\n|$)", result
                    )
                    company_name = re.findall(
                        r"(?i)(?:\b\w*\s*)*Company Name:\s*(.*?)(?:\n|$)", result
                    )
                    claims = [c.strip() for c in claims]
                    for c in claims:
                        reddit_result, retrieved_sources, context = self.call_reddit(
                            c,
                            company_name,
                            self.reddit_db,
                            k=6,
                        )
                        print("Second llm result: ", reddit_result)
                        claim_dict = {
                            "claim": c,
                            "retrieved_sources": retrieved_sources,
                            "label": extract_label(reddit_result),
                            "justification": extract_justification(reddit_result),
                            "context": context,
                        }
                        json_output = json.dumps(claim_dict)
                        _annotate(
                            _annotate_obj=page,
                            _text=reddit_result,
                            annotator_name="reddit_result",
                            metadata=json_output,
                        )

            else:
                for block in page.blocks + page.table_blocks:
                    if (
                        target_layouts is not None
                        and block.layout_type not in target_layouts
                    ):
                        continue
                    if level == "block":
                        _annotate(
                            _annotate_obj=block, _text="This is block level result"
                        )
                    elif level == "sentence":
                        for sentence in block.sentences:
                            _annotate(
                                _annotate_obj=sentence,
                                _text="This is sentence level result",
                            )
                    elif level == "text":
                        for text in block.texts:
                            _annotate(
                                _annotate_obj=text,
                                _text="This is sentence level result",
                            )

        return document

    def add_argument(self, parser: argparse.ArgumentParser):

        parser.add_argument("--reddit_annotator_name", type=str, default="reddit")

        # todo: add page level block
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

        parser.add_argument(
            "--use_reddit", action="store_true", help="Enable reddit usage"
        )

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
