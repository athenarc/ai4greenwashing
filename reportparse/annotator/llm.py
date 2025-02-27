import os
import re
import time
import argparse
from dotenv import load_dotenv
from logging import getLogger
from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.db_rag.db import ChromaDBHandler
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

logger = getLogger(__name__)

chroma_db = ChromaDBHandler()
model_name = os.getenv("GROQ_LLM_MODEL_1") if os.getenv("USE_GROQ_API") == "True" else os.getenv("OLLAMA_MODEL")


@BaseAnnotator.register("llm_test")
class llm_testAnnotator(BaseAnnotator):

    def __init__(self):
        load_dotenv()
        return

    def call_llm_test(self, text):

        time.sleep(5)
        if os.getenv("USE_GROQ_API") == "True":
            self.llm_test = ChatGroq(
                model=os.getenv("GROQ_LLM_MODEL_1"),
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=1,
                groq_api_key=os.getenv("GROQ_API_KEY_1"),
            )

            self.llm_test_2 = ChatGroq(
                model=os.getenv("GROQ_LLM_MODEL_2"),
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=1,
                groq_api_key=os.getenv("GROQ_API_KEY_2"),
            )
        else:
            self.llm_test = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)
            self.llm_test_2 = ChatOllama(model=os.getenv("OLLAMA_MODEL"), temperature=0)

        messages = [
            (
                "system",
                f"""You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 
         Your answer should follow the following format: 

         Potential greenwashing claim: [the claim]
         Justification: [short justification]

         Another potential greenwashing claim: [another claim]
         Justification: [another justification]
         
         Find at least one greenwashing claim in the text.

         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
         State the claim like a statement.
         """,
            ),
            ("human", f"{text}"),
        ]

        try:
            print("Invoking with the first llm_test...")
            if len(text.split()) >= 2000:
                ai_msg = self.reduce_llm_test_input(text, self.llm_test)
                return ai_msg
            else:
                ai_msg = self.llm_test.invoke(messages)
                return ai_msg.content
        except Exception as e:
            print(e)
            try:
                print("Invoking with the second llm_test...")
                if len(text.split()) >= 1900:
                    ai_msg = self.reduce_llm_test_input(text, self.llm_test_2)
                    return ai_msg
                else:
                    ai_msg = self.llm_test_2.invoke(messages)
                    return ai_msg.content
            except Exception as e:
                print("llm_test invokation failed. Returning none...")
                print(e)
                return None

    # method to reduce llm_test input if it is too large.
    def reduce_llm_test_input(self, text, llm_test):
        import time

        print("Invoking map reduce function to split text")
        from langchain.prompts import PromptTemplate
        from langchain.chains.summarize import load_summarize_chain
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # Dynamically split the text into three parts
        chunk_size = len(text) // 3
        chunk_1 = text[:chunk_size]
        chunk_2 = text[chunk_size : 2 * chunk_size]
        chunk_3 = text[2 * chunk_size :]

        # Define the map template
        map_template = """You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 
         Your answer should follow the following format: 

         Potential greenwashing claim: [the claim]
         Justification: [short justification]

         Second potential greenwashing claim: [another claim]
         Justification: [another justification]

         Third potential greenwashing claim: [another claim]
         Justification: [another justification]
         
         Find at least one greenwashing claim in the text.
         
         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
         State the claim like a statement.
        Text to be examined: {docs}"""
        map_prompt = PromptTemplate.from_template(map_template)
        # Define the reduce template
        reduce_template = """Synthesize the following results, into a single conlcusion. Please follow the format that is given to you.
                            Find at least one greenwashing claim in the text.
                            If greenwashing claims were found, follow the format below:
                            Potential greenwashing claim: [the claim]
                            Justification: [short justification]

                            Second potential greenwashing claim: [another claim]
                            Justification: [another justification]

                            Third potential greenwashing claim: [another claim]
                            Justification: [another justification]

                            Do not make any commentary and don't create any titles. Just provide what you are told.
                            State the claim like a statement.
                           The result are listed below: {docs}"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        map_chain = map_prompt | llm_test  # Chain map prompt with llm_test
        reduce_chain = reduce_prompt | llm_test  # Chain reduce prompt with llm_test
        result_1 = map_chain.invoke({"docs": chunk_1})
        time.sleep(5)
        result_2 = map_chain.invoke({"docs": chunk_2})
        time.sleep(5)
        result_3 = map_chain.invoke({"docs": chunk_3})
        time.sleep(5)
        result_1_text = (
            result_1.content if hasattr(result_1, "content") else str(result_1)
        )
        result_2_text = (
            result_2.content if hasattr(result_2, "content") else str(result_2)
        )
        result_3_text = (
            result_3.content if hasattr(result_3, "content") else str(result_3)
        )
        combined_results = "\n".join([result_1_text, result_2_text, result_3_text])
        final_summary = reduce_chain.invoke({"docs": combined_results})
        result = (
            final_summary.content
            if hasattr(final_summary, "content")
            else str(final_summary)
        )
        return result


    def call_chroma(self, claim, text, page_number, chroma_db, k=6, use_chunks=False):
        def retrieve_context(claim, page_number, db, k=6, use_chunks=False):
            try:
                print("Retrieving context from ChromaDB")
                collection = db.chunk_collection if use_chunks else db.page_collection
                results = collection.query(query_texts=[claim], n_results=k)

                relevant_texts = "\n".join([result[0] for result in results["documents"]])

                filtered_texts = "\n".join(
                    line for line in relevant_texts.split("\n") if f"Page {page_number} " not in line
                )

                return filtered_texts.strip() if filtered_texts.strip() else ""

            except Exception as e:
                logger.error(f"Error retrieving context from ChromaDB: {e}")
                return ""

        def verify_claim_with_context(claim, text, context):
            """Use an llm_test (Ollama or Groq) to verify if the claim is actually greenwashing based on document context."""
            messages=[
            (
                    "system",
                f"""You have at your disposal a **potential greenwashing claim** and information from a report. Your task is to evaluate whether the claim is **TRUE, FALSE, PARTIALLY TRUE, or PARTIALLY FALSE** based on the provided context.

                ### **Before making a decision:**
                1. **Analyze the claim** carefully to understand its main points.
                2. **Compare the claim** against the full text of the page where it was flagged and any relevant context from other pages.
                3. **Use only the given information**, avoiding external assumptions or unsupported reasoning.

                ### **Your response should follow this format:**

                **Claim:** {claim}

                **Page Text:**
                {text}

                **Additional Context from Other Pages:**
                {context}

                **Evaluation:**
                Choose one of the following labels:
                - **TRUE**: If the claim is fully supported by the report.
                - **FALSE**: If the claim is clearly disproved by the report.
                - **PARTIALLY TRUE**: If the claim has some correct aspects but is not entirely accurate.
                - **PARTIALLY FALSE**: If the claim includes correct elements but also contains misleading or incorrect information.

                **Justification:**
                Provide a clear, concise explanation of why the claim was classified this way, referring explicitly to the provided report text and additional context. Avoid unnecessary details and keep your reasoning precise.
                """,
                ),
                ("human", f"{text}"),
            ]
            try:
                print("Calling llm_test to verify claim with context")
                try:
                    ai_msg = self.llm_test.invoke(messages)
                    return ai_msg.content
                    
                except Exception as e:
                    logger.error(f"Error calling the API: {e}")
                    return "Error: Could not generate a response from the llm_test."
                    
            except Exception as e:
                logger.error(f"Error calling llm_test: {e}")
                return "Error: Could not generate a response."

        context = retrieve_context(claim=claim, page_number=page_number, db=chroma_db, k=k, use_chunks=use_chunks)
        result = verify_claim_with_context(claim=claim, text=text, context=context)
        return result

    
    def annotate(
        self,
        document: Document,
        args=None,
        level="block",
        target_layouts=("text", "list", "cell"),
        annotator_name="llm_test-test",
    ) -> Document:
        annotator_name = args.llm_test_annotator_name if args is not None else annotator_name
        level = args.llm_test_text_level if args is not None else level
        target_layouts = (
            args.llm_test_target_layouts if args is not None else list(target_layouts)
        )
        use_chroma = args.use_chroma if args is not None else False
        use_chunks = args.use_chunks if args is not None else False

        # Manual overrides to debug easily
        # use_chunks = False
        use_chroma= True
        def _annotate(
            _annotate_obj: AnnotatableLevel,
            _text: str,
            annotator_name: str,
            score_value,
        ):
            _annotate_obj.add_annotation(
                annotation=Annotation(
                    parent_object=_annotate_obj,
                    annotator=annotator_name,
                    value=_text,
                    meta={"score": score_value},
                )
            )

        if use_chroma:
            print("Starting storing in Chroma")
            for page in document.pages:
                if level == "page":
                    page_number = page.num
                    text = page.get_text_by_target_layouts(
                        target_layouts=target_layouts
                    )
                    chroma_db.store_page(
                        doc_name=document.name, page_number=page_number, text=text
                    )
            print("Finished storing in Chroma")
        
        for page in document.pages:
            if level == "page":
                print("Calling first llm_test to annotate")
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                result = self.call_llm_test(text)
                result = str(result)
                print("First llm_test result: ", result)
                _annotate(
                    _annotate_obj=page,
                    _text=result,
                    annotator_name=(
                        args.llm_test_annotator_name if args is not None else annotator_name
                    ),
                    score_value="Simple greenwashing detection",
                )
                if use_chroma:
                    print("Calling second llm_test to annotate")
                    page_number = page.num
                    claims = re.findall(
                        r"(?i)(?:\b\w*\s*)*claim:\s*(.*?)(?:\n|$)", result
                    )
                    claims = [c.strip() for c in claims]
                    for c in claims:
                        chroma_result = self.call_chroma(c, text, page_number, chroma_db, k=6, use_chunks=use_chunks)
                        print("Second llm_test result: ", chroma_result)
                        _annotate(
                            _annotate_obj=page,
                            _text=chroma_result,
                            annotator_name="chroma_result",
                            score_value=f"Claim: {c}",
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

        parser.add_argument("--llm_test_annotator_name", type=str, default="llm_test")

        # todo: add page level block
        parser.add_argument(
            "--llm_test_text_level",
            type=str,
            choices=["page", "sentence", "block"],
            default="page",
        )

        parser.add_argument(
            "--llm_test_target_layouts",
            type=str,
            nargs="+",
            default=["text", "list", "cell"],
            choices=LAYOUT_NAMES,
        )

        parser.add_argument(
            "--use_chroma",
            action="store_true",
            help="Enable ChromaDB usage"
        )

        parser.add_argument(
            "--use_chunks",
            action="store_true",
            help="Use chunks instead of pages"
        )