from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
import argparse
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

@BaseAnnotator.register("llm_local")
class LLMAnnotator(BaseAnnotator):
    
    def __init__(self):
       load_dotenv()
       self.llm = ChatOllama(model="llama3.2", temperature=0)
       return
    
    def call_llm_local(self, text): 
        import time
        
        time.sleep(5)

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
            print("Invoking with Llama3.2 (Ollama)...")
            if len(text.split()) >= 2000:
                return self.reduce_llm_local_input(text)
            else:
                ai_msg = self.llm.invoke(messages)
                return ai_msg.content
        except Exception as e:
            print("LLM invocation failed. Returning None...")
            print(e)
            return None
    
    def reduce_llm_local_input(self, text):
        import time
        from langchain.prompts import PromptTemplate
        
        print("Splitting text for MapReduce...")

        chunk_size = len(text) // 3  
        chunk_1, chunk_2, chunk_3 = text[:chunk_size], text[chunk_size:2*chunk_size], text[2*chunk_size:]  

        map_template = PromptTemplate.from_template("""
            You are a fact-checker specializing in greenwashing. Fact-check the given text and find any greenwashing claims. 
            Your answer should follow this format:
            
            Potential greenwashing claim: [the claim]
            Justification: [short justification]
            
            If no greenwashing claims are found, return this message:
            "No greenwashing claims found"
            
            DO NOT MAKE ANY COMMENTARY. JUST PROVIDE THE MENTIONED FORMAT.
            Text to be examined: {docs}
        """)

        reduce_template = PromptTemplate.from_template("""
            Synthesize the following results into a single conclusion. Follow this format:
            
            If no greenwashing claims are found, return:
            "No greenwashing claims found"
            
            Otherwise:
            Potential greenwashing claim: [the claim]
            Justification: [short justification]
            
            The results are listed below: {docs}
        """)

        map_chain = map_template | self.llm  
        reduce_chain = reduce_template | self.llm  

        results = []
        for chunk in [chunk_1, chunk_2, chunk_3]:
            results.append(map_chain.invoke({"docs": chunk}).content)
            time.sleep(5)
        
        combined_results = "\n".join(results)
        final_summary = reduce_chain.invoke({"docs": combined_results})
        
        return final_summary.content if hasattr(final_summary, 'content') else str(final_summary)
    
    def annotate(
        self,
        document: Document, args=None,
        level='block', target_layouts=('text', 'list', 'cell'), annotator_name='llm_local-test',
    ) -> Document:
        annotator_name = args.llm_local_annotator_name if args else annotator_name
        level = args.llm_local_text_level if args else level
        target_layouts = args.llm_local_target_layouts if args else list(target_layouts)

        def _annotate(_annotate_obj: AnnotatableLevel, _text: str):
            _annotate_obj.add_annotation(
                annotation=Annotation(
                    parent_object=_annotate_obj,
                    annotator=annotator_name,
                    value=_text,
                    meta={'score': 'This is the llm score field.'}
                )
            )
        
        for page in document.pages:
            if level == 'page':
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                _annotate(page, self.call_llm_local(text))
            else:
                for block in page.blocks + page.table_blocks:
                    if target_layouts and block.layout_type not in target_layouts:
                        continue
                    if level == 'block':
                        _annotate(block, 'This is block level result')
                    elif level == 'sentence':
                        for sentence in block.sentences:
                            _annotate(sentence, 'This is sentence level result')
                    elif level == 'text':
                        for text in block.texts:
                            _annotate(text, 'This is sentence level result')

        return document
    
    def add_argument(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--llm_local_annotator_name',
            type=str,
            default='llm_local'
        )

        parser.add_argument(
            '--llm_local_text_level',
            type=str,
            choices=['page', 'sentence', 'block'],
            default='sentence'
        )

        parser.add_argument(
            '--llm_local_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list', 'cell'],
            choices=LAYOUT_NAMES
        )
