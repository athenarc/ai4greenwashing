from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
import argparse
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from reportparse.annotator.store_pages import ChromaDBHandler
import os
import time

chroma_db = ChromaDBHandler()


@BaseAnnotator.register("chromadb")
class ChromadbAnnotator(BaseAnnotator):

    def __init__(self):
        load_dotenv()
        return

    def annotate(
        self,
        document: Document,
        args=None,
        level="page",
        target_layouts=("text", "list", "cell"),
        annotator_name="chromadb",
    ) -> Document:
        annotator_name = args.chromadb_annotator_name if args is not None else annotator_name
        level = args.chromadb_text_level if args is not None else level
        target_layouts = (
            args.chromadb_target_layouts if args is not None else list(target_layouts)
        )

        def _annotate(_annotate_obj: AnnotatableLevel, _text: str):
            _annotate_obj.add_annotation(
                annotation=Annotation(
                    parent_object=_annotate_obj,
                    annotator=annotator_name,
                    value=_text,
                    meta={"score": "This is the chromadb score field."},
                )
            )

        for page in document.pages:
            if level == "page":

                # Get page number
                page_number = page.num
                # Get text from page
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)

                chroma_db.store_page(doc_name=document.name, page_number=page_number, text=text)
                print(len(text))
                
            else:
                for block in page.blocks + page.table_blocks:
                    if (
                        target_layouts is not None
                        and block.layout_type not in target_layouts
                    ):
                        continue
                    if level == "block":
                        _annotate(_annotate_obj=block, _text="This is block level result")
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
                                _text="This is text level result",
                            )
        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        parser.add_argument("--chromadb_annotator_name", type=str, default="chromadb")

        parser.add_argument(
            "--chromadb_text_level",
            type=str,
            choices=["page", "sentence", "block"],
            default="page",
        )

        parser.add_argument(
            "--chromadb_target_layouts",
            type=str,
            nargs="+",
            default=["text", "list", "cell"],
            choices=LAYOUT_NAMES,
        )
