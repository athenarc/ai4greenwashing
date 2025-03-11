from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.web_rag.pipeline import pipeline
import argparse
import re
import os 
import json
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
from langchain_ollama import ChatOllama

@BaseAnnotator.register("web_rag")
class WEB_RAG_Annotator(BaseAnnotator):
    
    def __init__(self):
       load_dotenv()
       self.first_pass_prompt = os.getenv('FIRST_PASS_PROMPT')
       self.web_rag_prompt = os.getenv('WEB_RAG_PROMPT')
       
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
       
    
    def call_llm(self, text):

        time.sleep(5)
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

        messages = [
            (
                "system",
                self.first_pass_prompt,
            ),
            ("human", text),
        ]

        try:
            print("Invoking the first llm...")
            ai_msg = self.llm.invoke(messages)
            return ai_msg.content
        except Exception as e:
            print(e)
            try:
                    ai_msg = self.llm_2.invoke(messages)
                    return ai_msg.content
            except Exception as e:
                print("llm invokation failed. Returning none...")
                print(e)
                return None

    #function to extract label value from llm
    # TODO: fix for no links
    def extract_label(self, text):
        try:
            match = re.search(r'Result of the statement:(.*?)Justification:', text, re.DOTALL)
            return match.group(1).strip() if match else ""
        except Exception as e:
            print(f"Error during label extraction: {e}")
            return None
    
    def extract_justification(self, text):
        try:
            match = re.search(r'Justification:\s*(.*)', text, re.DOTALL)
            return match.group(1).strip() if match else ""
        except Exception as e:
            print(f"Error during justification extraction: {e}")
            return None

    
    #todo: add info truncation if text is too big for llm to handle.
    def web_rag(self, claim, web_sources):
        pip = pipeline(claim, web_sources)
        try:
            result, url_list = pip.retrieve_knowledge()
            if result is None:
                print('Result is None')
                return 'No content was found from the web', []
        except Exception as e:
            print(e)
            return 'No content was found from the web', []
        try:
            info = "\n".join(result.astype(str))
            if info: 
                messages = [
                        (
                            "system",
                        self.web_rag_prompt
                        ),
                        ("human", f'''External info '{info}'
                         Statement: '{claim}' "'''),]
                try:
                    print('Invoking with the first llm...')
                    ai_msg = self.llm.invoke(messages)
                    return ai_msg.content, url_list
                except Exception as e:
                    try:
                        print('Invoking with the second llm...')
                        ai_msg = self.llm_2.invoke(messages)
                        return ai_msg.content, url_list
                    except Exception as e:
                        print(e)
                        return 'LLM invocation failed', []
            else: return 'No content was found from the web', []
        except Exception as e:
            print(e)
            return 'No content was found from the web', []


    def annotate(
        self,
        document: Document, args=None,
        level='block', target_layouts=('text', 'list', 'cell'),  annotator_name='llm-test',
    ) -> Document:
        annotator_name = args.web_rag_annotator_name if args is not None else annotator_name
        level = args.web_rag_text_level if args is not None else level
        target_layouts = args.web_rag_target_layouts if args is not None else list(target_layouts)
        web_rag = args.web_rag if args is not None else 'no'

        def _annotate(_annotate_obj: AnnotatableLevel, _text: str, annotator_name: str, metadata):
                _annotate_obj.add_annotation(
                annotation=Annotation(
                    parent_object=_annotate_obj,
                    annotator=annotator_name,
                    value= _text,
                    meta=json.loads(metadata)
                )
            )

        for page in document.pages:
            if level == 'page':
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                result = self.call_llm(text)
                result = str(result)
                _annotate(_annotate_obj=page, _text=result, annotator_name=
                          args.web_rag_annotator_name if args is not None else annotator_name,
                          metadata=json.dumps({"info": "Simple greenwashing detection"}))


                if web_rag =='yes':
                    
                    claims = re.findall(r'(?i)(?:\b\w*\s*)*claim:\s*(.*?)(?:\n|$)', result)
                    claims = [c.strip() for c in claims]
                    for c in claims:

                        web_rag_result, url_list = self.web_rag(c, web_sources=3)
                        print(f'SEARCHING FOR CLAIM: {c}')
                        claim_dict = {
                        "claim": c,
                        "urls": url_list,
                        "Label": self.extract_label(web_rag_result),
                        "Justification": self.extract_justification(web_rag_result)
                        }
                        json_output = json.dumps(claim_dict)

                        _annotate(_annotate_obj=page, _text=web_rag_result, annotator_name='web_rag_result', metadata=json_output)






            else:
                for block in page.blocks + page.table_blocks:
                    if target_layouts is not None and block.layout_type not in target_layouts:
                        continue
                    if level == 'block':
                        _annotate(_annotate_obj=block, _text='This is block level result')
                    elif level == 'sentence':
                        for sentence in block.sentences:
                            _annotate(_annotate_obj=sentence, _text='This is sentence level result')
                    elif level == 'text':
                        for text in block.texts:
                            _annotate(_annotate_obj=text, _text='This is sentence level result')

        return document
    
    def add_argument(self, parser: argparse.ArgumentParser):
        
        parser.add_argument(
            '--web_rag_annotator_name',
            type=str,
            default='web_rag'
        )

        #todo: add page level block
        parser.add_argument(
            '--web_rag_text_level',
            type=str,
            choices=['page', 'sentence', 'block'],
            default='page'
        )

        parser.add_argument(
            '--web_rag_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list', 'cell'],
            choices=LAYOUT_NAMES
        )


        parser.add_argument(
            '--web_rag',
            type=str,
            default='yes'
        )


