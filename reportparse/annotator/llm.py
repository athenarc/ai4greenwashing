from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
import argparse
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from reportparse.db_rag.db import ChromaDBHandler
import os
import time

chroma_db = ChromaDBHandler()


@BaseAnnotator.register("llm")
class LLMAnnotator(BaseAnnotator):

    def __init__(self):
        load_dotenv()
        if os.getenv("USE_GROQ_API") == "True":
            self.llm = ChatGroq( 
                model=os.getenv("GROQ_LLM_MODEL_1"), 
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=1,
                groq_api_key=os.getenv("GROQ_API_KEY_1"),
            )

            self.llm_2 = ChatGroq(
                model=os.getenv("GROQ_LLM_MODEL_2"),
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=1,
                groq_api_key=os.getenv("GROQ_API_KEY_2"),
            )

            print(f"Using Groq model: {os.getenv('GROQ_LLM_MODEL_1')} as annotator.")
        else:
            self.llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)
            # print model used
            print(f"Using Ollama model: {os.getenv('OLLAMA_MODEL')} as annotator.")
        return

    def call_llm(self, text, max_len=2048):
        text = text.strip()

        if len(text.split()) >= max_len:
            return self.reduce_llm_input(text)

        messages = [
            (
                "system",
                f"""You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. If no greenwashing claims are found, return this message:
                "No greenwashing claims found"
        DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT. Your answer should follow the following format repeated for one or more claims found, else "No greenwashing claims found": 

        <Start of format>

        Potential greenwashing claim: [the claim]
        Justification: [short justification]
        
        ...

        <End of format>
         """,
            ),
            ("human", text),
        ]

        try:
            print("Invoking LLM...")
            ai_msg = self.llm.invoke(messages)
            print(f"LLM response: {ai_msg.content}")  # debugging
            return ai_msg.content
        except Exception as e:
            print("LLM invocation failed:", e)
            return None

    def reduce_llm_input(self, text):
        """Splits large text into chunks and processes them separately, then combines results"""
        from langchain.prompts import PromptTemplate

        print("Splitting text for MapReduce...")

        chunk_size = len(text) // 3
        chunks = [text[i * chunk_size : (i + 1) * chunk_size] for i in range(3)]

        map_template = PromptTemplate.from_template(
            """
            You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. If no greenwashing claims are found, return this message:
            "No greenwashing claims found"
        DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT. Your answer should follow the following format repeated for one or more claims found, else "No greenwashing claims found": 
        <Start of format>

        Potential greenwashing claim: [the claim]
        Justification: [short justification]
        
        ...

        <End of format>
        Text to be examined: {docs}
        """
        )

        reduce_template = PromptTemplate.from_template(
            """
            Synthesize the following results into a single conclusion. If no greenwashing claims are found in any of the results, return:
            "No greenwashing claims found"
            DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT. Your answer should follow the following format repeated for one or more claims found. Only if no greenwashing claims are found in any of the results, return "No greenwashing claims found": 
            <Start of format>

            Potential greenwashing claim: [the claim]
            Justification: [short justification]
            
            ...

            <End of format>
            The results are listed below: {docs}
        """
        )

        map_chain = map_template | self.llm
        reduce_chain = reduce_template | self.llm

        results = []
        for chunk in chunks:
            results.append(map_chain.invoke({"docs": chunk}).content)

        combined_results = "\n".join(results)
        final_summary = reduce_chain.invoke({"docs": combined_results})

        return (
            final_summary.content
            if hasattr(final_summary, "content")
            else str(final_summary)
        )

    def annotate(
        self,
        document: Document,
        args=None,
        level="page",
        target_layouts=("text", "list", "cell"),
        annotator_name="llm",
    ) -> Document:
        annotator_name = args.llm_annotator_name if args is not None else annotator_name
        level = args.llm_text_level if args is not None else level
        target_layouts = (
            args.llm_target_layouts if args is not None else list(target_layouts)
        )

        def _annotate(_annotate_obj: AnnotatableLevel, _text: str):
            _annotate_obj.add_annotation(
                annotation=Annotation(
                    parent_object=_annotate_obj,
                    annotator=annotator_name,
                    value=_text,
                    meta={"score": "This is the llm score field."},
                )
            )

        for page in document.pages:
            if level == "page":

                # Get page number
                page_number = page.num
                # Get text from page
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)

                chroma_db.store_page(doc_name=document.name, page_number=page_number, text=text)
                # print(len(text))
                # if page_number < 20 and page_number % 2 == 0:
                _annotate(_annotate_obj=page, _text=self.call_llm(text))
            else:
                for block in page.blocks + page.table_blocks:
                    if (
                        target_layouts is not None
                        and block.layout_type not in target_layouts
                    ):
                        continue
                    if level == "block":
                        _annotate(_annotate_obj=block, _text=self.call_llm(block.text))
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
        parser.add_argument("--llm_annotator_name", type=str, default="llm")

        parser.add_argument(
            "--llm_text_level",
            type=str,
            choices=["page", "sentence", "block"],
            default="page",
        )

        parser.add_argument(
            "--llm_target_layouts",
            type=str,
            nargs="+",
            default=["text", "list", "cell"],
            choices=LAYOUT_NAMES,
        )

        parser.add_argument(
            "--use_groq",
            action="store_true",
            help="Use Groq LLM instead of Ollama",
        )
