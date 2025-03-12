from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.annotator.web_rag import WEB_RAG_Annotator
from reportparse.annotator.chroma_annotator import LLMAnnotator
from reportparse.llm_prompts import LLM_AGGREGATOR_PROMPT
import argparse
import re
import json
from dotenv import load_dotenv
from logging import getLogger
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import os
logger = getLogger(__name__)

@BaseAnnotator.register("llm_agg")
class LLMAggregator(BaseAnnotator):

    def __init__(self):
        load_dotenv()
        self.web = WEB_RAG_Annotator()
        self.chroma = LLMAnnotator()
        self.agg_prompt = LLM_AGGREGATOR_PROMPT
        if os.getenv("USE_GROQ_API") == "True":

            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL"),
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=1,
                google_api_key=os.getenv("GEMINI_API_KEY")
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
                print(f'Invokation error: {e}. Invoking with the second llm....')
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
        gw_pages = args.pages_to_gw if args is not None else None
        use_chunks = args.use_chunks if args is not None else False

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

            if gw_index >= gw_pages:
                break
            if level == "page":
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                # call the first llm from chroma, that finds all potential greenwashing claims
                result = self.chroma.call_llm(text)
                result = str(result)

                # add initial greenwashing detection without any annotators
                _annotate(
                    _annotate_obj=page,
                    _text=result,
                    annotator_name="First pass",
                    metadata=json.dumps({"info": "Simple greenwashing detection"}),
                )

                page_number = page.num
                claims = re.findall(r"(?i)(?:\b\w*\s*)*claim:\s*(.*?)(?:\n|$)", result)
                company_name = re.findall(r"(?i)(?:\b\w*\s*)*Company Name:\s*(.*?)(?:\n|$)", result)
                claims = [c.strip() for c in claims]
                for c in claims:
                    # add aggregation with chroma db
                    chroma_result, retrieved_pages = self.chroma.call_chroma(
                        c,
                        document.name,
                        text,
                        page_number,
                        self.chroma.chroma_db,
                        k=6,
                        use_chunks=use_chunks,
                    )
                    print("Second llm result: ", chroma_result)
                    # annotate for chroma
                    claim_dict_chroma = {
                        "claim": c,
                        "retrieved_pages": retrieved_pages,
                        "Label": self.chroma.extract_label(chroma_result),
                        "Justification": self.chroma.extract_justification(
                            chroma_result
                        ),
                    }
                    json_output = json.dumps(claim_dict_chroma)
                    _annotate(
                        _annotate_obj=page,
                        _text=chroma_result,
                        annotator_name="chroma_result",
                        metadata=json_output,
                    )

                    # add web_rag aggregation
                    print(f'SEARCHING FOR CLAIM {c}')
                    web_rag_result, url_list = self.web.web_rag(c, 1, company_name)
                    claim_dict_webrag = {
                        "claim": c,
                        "urls": url_list,
                        "Label": self.web.extract_label(web_rag_result),
                        "Justification": self.web.extract_justification(web_rag_result),
                    }
                    json_output = json.dumps(claim_dict_webrag)

                    # annotate for web rag
                    _annotate(
                        _annotate_obj=page,
                        _text=web_rag_result,
                        annotator_name="web_rag_result",
                        metadata=json_output,
                    )

                    aggregator_result = self.call_aggregator(c, chroma_result, web_rag_result)
                    print("Aggregator result: ", aggregator_result)
                    _annotate(
                        _annotate_obj=page,
                        _text=aggregator_result,
                        annotator_name="aggregator_result",
                        metadata=json.dumps(
                            {
                                "claim": c,
                                "chroma_result": chroma_result,
                                "web_rag_result": web_rag_result,
                                "Label": self.web.extract_label(aggregator_result),
                                "Justification": self.web.extract_justification(
                                    aggregator_result
                                ),
                            }
                        ),
                    )
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
