from logging import getLogger
import time
import argparse
from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from reportparse.annotator.store_pages import ChromaDBHandler

chroma_db = ChromaDBHandler()
@BaseAnnotator.register("ollama_llm")
class OllamaLLMAnnotator(BaseAnnotator):

    def __init__(self):
        load_dotenv()
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        return

    def call_ollama_llm(self, text, max_len=2048):
        text = text.strip()

        if len(text.split()) >= max_len:
            return self.reduce_ollama_llm_input(text)

        messages = [
            (
                "system",
                f"""You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 
         Your answer should follow the following format: 

         Potential greenwashing claim: [the claim]
         Justification: [short justification]

         Another potential greenwashing claim: [another claim]
         Justification for the second claim: [another justification]
         
         If no greenwashing claims are found, return this message:
         "No greenwashing claims found"

         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
         """,
            ),
            ("human", text),
        ]

        try:
            print("Invoking Llama3.2 (Ollama)...")
            ai_msg = self.llm.invoke(messages)
            # print(f"LLM response: {ai_msg.content}")  # debugging
            return ai_msg.content
        except Exception as e:
            print("LLM invocation failed:", e)
            return None

    def reduce_ollama_llm_input(self, text):
        """Splits large text into chunks and processes them separately, then combines results"""
        from langchain.prompts import PromptTemplate

        print("Splitting text for MapReduce...")

        chunk_size = len(text) // 3
        chunks = [text[i * chunk_size : (i + 1) * chunk_size] for i in range(3)]

        map_template = PromptTemplate.from_template(
            """
            You are a fact-checker specializing in greenwashing. Fact-check the given text and find any greenwashing claims.
            Your answer should follow this format:
            
            Potential greenwashing claim: [the claim]
            Justification: [short justification]
            
            If no greenwashing claims are found, return:
            "No greenwashing claims found"
            
            DO NOT MAKE ANY COMMENTARY. JUST PROVIDE THE MENTIONED FORMAT.
            Text to be examined: {docs}
        """
        )

        reduce_template = PromptTemplate.from_template(
            """
            Synthesize the following results into a single conclusion. Follow this format:
            
            If no greenwashing claims are found, return:
            "No greenwashing claims found"
            Otherwise:
            Potential greenwashing claim: [the claim]
            Justification: [short justification]
            
            The results are listed below: {docs}
        """
        )

        map_chain = map_template | self.llm
        reduce_chain = reduce_template | self.llm

        results = []
        for chunk in chunks:
            results.append(map_chain.invoke({"docs": chunk}).content)
            time.sleep(3)

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
        annotator_name="ollama_llm",
    ) -> Document:
        annotator_name = (
            args.ollama_llm_annotator_name if args is not None else annotator_name
        )
        level = args.ollama_llm_text_level if args is not None else level
        target_layouts = (
            args.ollama_llm_target_layouts if args is not None else list(target_layouts)
        )

        def _annotate(_annotate_obj: AnnotatableLevel, _text: str):
            _annotate_obj.add_annotation(
                annotation=Annotation(
                    parent_object=_annotate_obj,
                    annotator=annotator_name,
                    value=_text,
                    meta={"score": "This is the ollama_llm score field."},
                )
            )

        for page in document.pages:
            if level == "page":

                # Get page number
                page_number = page.num
                # Get text from page
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)

                # Store page in ChromaDB
                chroma_db.store_page(doc_name=document.name, page_number=page_number, text=text)

                # print(text)

                _annotate(_annotate_obj=page, _text=self.call_ollama_llm(text))
            else:
                for block in page.blocks + page.table_blocks:
                    if (
                        target_layouts is not None
                        and block.layout_type not in target_layouts
                    ):
                        continue
                    if level == "block":
                        _annotate(
                            _annotate_obj=block, _text=self.call_ollama_llm(block.text)
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
                                _text="This is text level result",
                            )

        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--ollama_llm_annotator_name", type=str, default="ollama_llm"
        )

        parser.add_argument(
            "--ollama_llm_text_level",
            type=str,
            choices=["page", "sentence", "block"],
            default="page",
        )

        parser.add_argument(
            "--ollama_llm_target_layouts",
            type=str,
            nargs="+",
            default=["text", "list", "cell"],
            choices=LAYOUT_NAMES,
        )
