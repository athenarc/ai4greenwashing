from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
import argparse
import ollama
from langchain_groq import ChatGroq
from reportparse.annotator.rag_database import chroma_db
from dotenv import load_dotenv

@BaseAnnotator.register("llm")
class LLMAnnotator(BaseAnnotator):
    
    def __init__(self):
       load_dotenv()
       self.db = chroma_db()
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

        self.llm_2 = ChatGroq(
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
         Justification for the second claim: [another justification]
         
         If no greenwashing claims are found, return this message:
         "No greenwashing claims found"

         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
         ''',
                
            ),
            ("human", f"{text}"),
        ]

        try:
            print('Invoking with the first llm...')
            if len(text.split()) >= 2000:
                ai_msg = self.reduce_llm_input(text, self.llm)
                return ai_msg
            else:
                ai_msg = self.llm.invoke(messages)
                return ai_msg.content
        except Exception as e:
            print(e)
            try:
                print('Invoking with the second llm...')
                if len(text.split()) >= 1900:
                    ai_msg = self.reduce_llm_input(text, self.llm_2)
                    return ai_msg
                else:
                    ai_msg = self.llm_2.invoke(messages)
                    return ai_msg.content
            except Exception as e:
                print('LLM invokation failed. Returning none...')
                print(e)
                return None

        
        
    #method to reduce llm input if it is too large.
    def reduce_llm_input(self, text, llm):
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
         Justification for the second claim: [another justification]

         Third potential greenwashing claim: [another claim]
         Justification for the third claim: [another justification]
         
         If no greenwashing claims are found, return nothing.
         
         DO NOT MAKE ANY COMMENTARY JUST PROVIDE THE MENTIONED FORMAT.
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
                            Justification for the second claim: [another justification]

                            Third potential greenwashing claim: [another claim]
                            Justification for the third claim: [another justification]

                            Do not make any commentary and don't create any titles. Just provide what you are told.
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

        
        #print(final_summary.content if hasattr(final_summary, 'content') else str(final_summary))
        result = final_summary.content if hasattr(final_summary, 'content') else str(final_summary)

        return result
    

    def annotate(
        self,
        document: Document, args=None,
        level='block', target_layouts=('text', 'list', 'cell'),  annotator_name='llm-test',
    ) -> Document:
        annotator_name = args.llm_annotator_name if args is not None else annotator_name
        #todo check if len is correct
        max_pages_to_gw = args.max_pages_to_gw if args.max_pages_to_gw <= len(document.pages) else len(document.pages)
        level = args.llm_text_level if args is not None else level
        target_layouts = args.llm_target_layouts if args is not None else list(target_layouts)

        print(f'THE DOCUMENT YOU PROVIDED HAS {len(document.pages)} PAGES')
        print(f'SEARCHING GREENWASHING CLAIMS FOR { max_pages_to_gw} PAGES')

        def _annotate(_annotate_obj: AnnotatableLevel, _text: str):
            _annotate_obj.add_annotation(
                annotation=Annotation(
                    parent_object=_annotate_obj,
                    annotator=annotator_name,
                    value= _text,
                    meta={'score': 'This is the llm score field.'}
                )
            )
        
        parsed_pages=0 #index to keep track of the number of pages that where checked for gw

        #db insertion works page-wise only for now.
        for page in document.pages:
            if level == 'page':
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                #store text in chromadb
                self.db.store_text_in_chromadb(text)
            else: break
        for page in document.pages:
            if level == 'page':
                if parsed_pages <= max_pages_to_gw-1:
                    #we loop through the 
                    _annotate(_annotate_obj=page, _text=self.call_llm(text))
                     #if we reach the limit inserted by the user, break the loop
                    parsed_pages+=1
                else:
                    break
              
            else:
                    for block in page.blocks + page.table_blocks:
                        if target_layouts is not None and block.layout_type not in target_layouts:
                            continue
                        if level == 'block':
                            #todo: implement for block level too.
                            _annotate(_annotate_obj=block, _text='This is block level result')
                        elif level == 'sentence':
                            #todo: implement for sentence level too.
                            for sentence in block.sentences:
                                _annotate(_annotate_obj=sentence, _text='This is sentence level result')
                        elif level == 'text':
                            #todo: implement for text level too.
                            for text in block.texts:
                                _annotate(_annotate_obj=text, _text='This is sentence level result')

        return document
    
    def add_argument(self, parser: argparse.ArgumentParser):
        
        parser.add_argument(
            '--llm_annotator_name',
            type=str,
            default='llm'
        )

        
        parser.add_argument(
            '--llm_text_level',
            type=str,
            choices=['page', 'sentence', 'block'],
            default='page'
        )

        parser.add_argument(
            '--llm_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list', 'cell'],
            choices=LAYOUT_NAMES
        )

        parser.add_argument(
            '--max_pages_to_gw',
            type=int,
            default=5,
            help=f"Choose a value between 1 and the the number of pages on the esg-report you provided."
        )



