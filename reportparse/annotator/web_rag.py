from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.web_rag.pipeline import pipeline
import argparse
import re
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

@BaseAnnotator.register("web_rag")
class LLMAnnotator(BaseAnnotator):
    
    def __init__(self):
       load_dotenv()
       return
    
    def call_llm(self, text): 
        import os 
        import time
        
        time.sleep(5)

        self.llm = ChatGroq( 
            model=os.getenv("GROQ_LLM_MODEL_1"), 
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            groq_api_key=os.getenv("GROQ_API_KEY_1"),
        )

        self.web_rag_2 = ChatGroq(
            model=os.getenv("GROQ_LLM_MODEL_2"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            groq_api_key=os.getenv("GROQ_API_KEY_2"),
        )


        messages = [
            (
                "system",
                f'''You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 
         Your answer should follow the following format: 

         Potential greenwashing claim: [the claim]
         Justification: [short justification]

         Another potential greenwashing claim: [another claim]
         Justification: [another justification]
         
         If no greenwashing claims are found, return this message:
         "No greenwashing claims found"

         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
         State the claim like a statement.
         ''',
                
            ),
            ("human", f"{text}"),
        ]

        try:
            print('Invoking with the first llm...')
            if len(text.split()) >= 2000:
                ai_msg = self.reduce_web_rag_input(text, self.llm)
                return ai_msg
            else:
                ai_msg = self.llm.invoke(messages)
                return ai_msg.content
        except Exception as e:
            print(e)
            try:
                print('Invoking with the second llm...')
                if len(text.split()) >= 1900:
                    ai_msg = self.reduce_web_rag_input(text, self.web_rag_2)
                    return ai_msg
                else:
                    ai_msg = self.web_rag_2.invoke(messages)
                    return ai_msg.content
            except Exception as e:
                print('LLM invokation failed. Returning none...')
                print(e)
                return None

        
        
    #method to reduce llm input if it is too large.
    def reduce_web_rag_input(self, text, llm):
        import time
        print('Invoking map reduce function to split text')
        from langchain.prompts import PromptTemplate
        from langchain.chains.summarize import load_summarize_chain
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        

        #Dynamically split the text into three parts
        chunk_size = len(text) // 3  
        chunk_1 = text[:chunk_size]
        chunk_2 = text[chunk_size:2*chunk_size]
        chunk_3 = text[2*chunk_size:]  

        # Define the map template
        map_template = """You are a fact-checker, that specializes in greenwashing. Fact-check the given text, and find if there are any greenwashing claims. 
         Your answer should follow the following format: 

         Potential greenwashing claim: [the claim]
         Justification: [short justification]

         Second potential greenwashing claim: [another claim]
         Justification: [another justification]

         Third potential greenwashing claim: [another claim]
         Justification: [another justification]
         
         If no greenwashing claims are found, return nothing.
         
         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
         State the claim like a statement.
        Text to be examined: {docs}"""
        map_prompt = PromptTemplate.from_template(map_template)
       # Define the reduce template
        reduce_template = """Synthesize the following results, into a single conlcusion. Please follow the format that is given to you.
                            If no greenwashing claims are found, return this message:
                            "No greenwashing claims found"
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
        map_chain = map_prompt | llm  # Chain map prompt with LLM
        reduce_chain = reduce_prompt | llm  # Chain reduce prompt with LLM
        result_1 = map_chain.invoke({"docs": chunk_1})
        time.sleep(5)
        result_2 = map_chain.invoke({"docs": chunk_2})
        time.sleep(5)
        result_3 = map_chain.invoke({"docs": chunk_3})
        time.sleep(5)
        result_1_text = result_1.content if hasattr(result_1, 'content') else str(result_1)
        result_2_text = result_2.content if hasattr(result_2, 'content') else str(result_2)
        result_3_text = result_3.content if hasattr(result_3, 'content') else str(result_3)
        combined_results = "\n".join([result_1_text, result_2_text, result_3_text])
        final_summary = reduce_chain.invoke({"docs": combined_results})
        result = final_summary.content if hasattr(final_summary, 'content') else str(final_summary)
        return result
    
    #todo: add info truncation if text is too big for llm to handle.
    def web_rag(self, claim, web_sources):
        pip = pipeline(claim, web_sources)
        try:
            result, url_list = pip.retrieve_knowledge()
        except Exception as e:
            print(e)
            return None, []
        try:
            info = "\n".join(result.astype(str))
            if info: 
                messages = [
                        (
                            "system",
                        f'''You have at your disposal information '[Information]' and a statement: '[User Input]' whose accuracy must be evaluated. 
                            Use only the provided information in combination with your knowledge to decide whether the statement is TRUE, FALSE, PARTIALLY TRUE, or PARTIALLY FALSE.

                            Before you decide:

                            1. Analyze the statement clearly to understand its content and identify the main points that need to be evaluated.
                            2. Compare the statement with the information you have, evaluating each element of the statement separately.
                            3. Use your knowledge ONLY in combination with the provided information, avoiding reference to unverified information.

                            Result: Provide a clear answer by choosing one of the following labels:

                            - TRUE: If the statement is fully confirmed by the information and evidence you have.
                            - FALSE: If the statement is clearly disproved by the information and evidence you have.
                            - PARTIALLY TRUE: If the statement contains some correct elements, but not entirely accurate.
                            - PARTIALLY FALSE: If the statement contains some correct elements but also contains misleading or inaccurate information.

                            Finally, explain your reasoning clearly and focus on the provided data and your own knowledge. Avoid unnecessary details and try to be precise and concise in your analysis. Your answers should be in the following format:

                            Statement: '[User Input]'
                            Result of the statement:
                            Justification:'''),
                        ("human", f'''External info '{info}'
                         Statement: '{claim}' "'''),]
                try:
                    print('Invoking with the first llm...')
                    ai_msg = self.llm.invoke(messages)
                    return ai_msg.content, url_list
                except Exception as e:
                    try:
                        print('Invoking with the second llm...')
                        ai_msg = self.web_rag_2.invoke(messages)
                        return ai_msg.content, url_list
                    except Exception as e:
                        print(e)
                        return 'No web content found'
            else: return 'No content was found from the web'
        except Exception as e:
            print(e)
            return 'No content was found from the web'


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

                        web_rag_result, url_list = self.web_rag(c, web_sources=1)
                        claim_dict = {
                        "claim": c,
                        "urls": url_list
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


