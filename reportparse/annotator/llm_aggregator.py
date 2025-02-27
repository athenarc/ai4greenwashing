from reportparse.annotator.base import BaseAnnotator
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.annotator.web_rag import WEB_RAG_Annotator
from reportparse.annotator.chroma_annotator import LLMAnnotator, chroma_db
import argparse
import re
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv


@BaseAnnotator.register("llm_agg")
class LLMAggregator(BaseAnnotator):

    def __init__(self):
       load_dotenv()
       self.web = WEB_RAG_Annotator()
       self.chroma = LLMAnnotator()
       return
    

    def annotate(
        self,
        document: Document, args=None,
        level='block', target_layouts=('text', 'list', 'cell'),  annotator_name='llm-test',
    ) -> Document:
        annotator_name = args.web_rag_annotator_name if args is not None else annotator_name
        level = args.web_rag_text_level if args is not None else level
        target_layouts = args.web_rag_target_layouts if args is not None else list(target_layouts)
        web_rag = args.web_rag if args is not None else 'no'
        use_chroma = args.use_chroma if args is not None else False
        use_chunks = args.use_chunks if args is not None else False
    

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
            if level=='page':
                #add pages to chroma_db
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
            else: print("Page level is not specified")

        for page in document.pages:
            if level == 'page':
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                #call the first llm from chroma, that finds all potential greenwashing claims
                result = self.chroma.call_llm(text)
                result = str(result)

                #add initial greenwashing detection without any annotators
                _annotate(_annotate_obj=page, _text=result, annotator_name=
                          args.web_rag_annotator_name if args is not None else annotator_name,
                          metadata=json.dumps({"info": "Simple greenwashing detection"}))
                


                page_number = page.num
                claims = re.findall(
                        r"(?i)(?:\b\w*\s*)*claim:\s*(.*?)(?:\n|$)", result
                    )
                claims = [c.strip() for c in claims]
                for c in claims:
                    #add aggregation with chroma db
                    chroma_result = self.chroma.call_chroma(c, text, page_number, chroma_db, k=6, use_chunks=use_chunks)
                    print("Second llm result: ", chroma_result)
                    #annotate for chroma
                    _annotate(
                        _annotate_obj=page,
                        _text=chroma_result,
                        annotator_name="chroma_result",
                        metadata=json.dumps({"Claim": c}),
                        )
                    
                    #add web_rag aggregation
                    web_rag_result, url_list = self.web.web_rag(c, web_sources=1)
                    claim_dict = {
                        "claim": c,
                        "urls": url_list,
                        "Label": self.web.extract_label(web_rag_result),
                        "Justification": self.web.extract_justification(web_rag_result)
                    }
                    json_output = json.dumps(claim_dict)

                    #annotate for web rag
                    _annotate(_annotate_obj=page, _text=web_rag_result, annotator_name='web_rag_result', metadata=json_output)

                






            else:
                return "Page extraction failed"

        return document
    
    def add_argument(self, parser: argparse.ArgumentParser):
        
        parser.add_argument(
            '--llm_agg_text_level',
            type=str,
            choices=['page', 'sentence', 'block'],
            default='page'
        )

        parser.add_argument(
            '--llm_agg_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list', 'cell'],
            choices=LAYOUT_NAMES
        )
