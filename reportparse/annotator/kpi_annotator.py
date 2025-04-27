import os, re, json, argparse
from logging import getLogger
from dotenv import load_dotenv

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document, Annotation, AnnotatableLevel
from reportparse.util.settings import LAYOUT_NAMES
from reportparse.util.my_embeddings import get_embedder
from reportparse.remove_thinking import remove_think_blocks

from sentence_transformers import SentenceTransformer
import faiss

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI


logger = getLogger(__name__)


@BaseAnnotator.register("kpi")
class KPIAnnotator(BaseAnnotator):

    def __init__(self):

        defs_path = "../kpi_definitions.json"
        # in __init__  (or wherever you load the JSON)
        with open(defs_path, "r", encoding="utf‑8") as f:
            self.all_kpi_defs = json.load(f)

        self._indices_by_sector = {}
        self.embedder = get_embedder()      
        load_dotenv()
        if os.getenv("USE_GROQ_API") == "True":

            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL"),
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=1,
                google_api_key=os.getenv("GEMINI_API_KEY"),
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

    def detect_company_and_sector(self, text: str):
        """Return (company_name, sector) using one LLM round‑trip."""
        SECTORS = [
            "Industrial Transportation",
            "Banks",
            "Nonlife Insurance",
            "Automobiles",
            "Electricity Utilities",
            "None of the listed",
        ]
        schema = '{"company name": "string", "sector": "string"}'
        sector_list = "\n".join(f"- {s}" for s in SECTORS)

        prompt = (
            "Extract the legal company name **exactly as written** and classify it.\n"
            f"Choose the sector strictly from this list:\n{sector_list}\n\n"
            f"TEXT SOURCE (may be partial):\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"Respond ONLY with a JSON object that follows this schema:\n{schema}"
        )

        for llm in (self.llm, self.llm_2):
            try:
                raw = llm.invoke(
                    [("system", "Company sector extractor"), ("human", prompt)]
                ).content
                reply = remove_think_blocks(raw).strip()
                logger.info("LLM reply: %s", reply)

                try:
                    if reply.startswith("```"):
                        reply = reply.lstrip("`").split("```")[0]
                    data = json.loads(reply)
                    company = str(data.get("company name", "")).strip()
                    sector  = str(data.get("sector", "")).strip()
                except (json.JSONDecodeError, TypeError, ValueError):
                    data = None

                if not data:
                    company_match = re.search(
                        r'"?company name"?\s*:\s*"([^"]+)"', reply, re.I
                    )
                    sector_match = re.search(
                        r'"?sector"?\s*:\s*"([^"]+)"', reply, re.I
                    )
                    company = company_match.group(1).strip() if company_match else ""
                    sector  = sector_match.group(1).strip()  if sector_match  else ""

                if not company:
                    company = "Unknown company"

                sector_norm = sector.casefold()
                sector = next(
                    (s for s in SECTORS if s.casefold() == sector_norm),
                    "None of the listed",
                )

                return company, sector

            except Exception as e:
                logger.warning("sector‑LLM failed (%s): %s", type(llm), e)
        return "Unknown company", "None of the listed"
    
    def get_sector_view(self, sector: str):
        sector_key = sector or "None of the listed"

        if sector_key in self._indices_by_sector:
            return self._indices_by_sector[sector_key]

        kpi_defs = [
            d for d in self.all_kpi_defs
            if sector_key in d["sector"]           
            or "None of the listed" in d["sector"]  
        ]
        if not kpi_defs:                          
            kpi_defs = self.all_kpi_defs

        sentences = [f"{d['id']} – {d['name']} – {d.get('definition','')}"
                    for d in kpi_defs]
        emb = self.embedder.encode(sentences, normalize_embeddings=True)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        self._indices_by_sector[sector_key] = (kpi_defs, index)
        return kpi_defs, index

    def closest_kpis(self, page_text: str, k: int = 8):
        emb = self.embedder.encode(page_text[:2048], normalize_embeddings=True)
        sim, idx = self.kpi_index.search(emb.reshape(1, -1), k=k)
        return [self.kpi_defs[i] for i in idx[0]]
    

    def page_llm_extract(self, page_text: str, kpi_subset: list[dict]):
        schema = """
        [
        {
            "kpi_id": "string",
            "value":  number,
            "unit":   "string",
            "year":   integer | null,
            "confidence": number,
            "snippet": "string"
        }
        ]
        """.strip()

        defs = "\n".join(f"{d['id']}: {d['definition']}" for d in kpi_subset)
        prompt = (
            "You are an ESG‑KPI extractor.\n"
            "Return ONLY a JSON array with **one object per KPI you can spot**.\n"
            f"Schema:\n{schema}\n\n"
            f"# KPI_DEFINITIONS (subset)\n{defs}\n\n"
            f"# PAGE_TEXT\n\"\"\"\n{page_text[:3500]}\n\"\"\""
        )

        for llm in (self.llm, self.llm_2):
            try:
                raw = llm.invoke([("system", "KPI extractor"), ("human", prompt)]).content
                raw = remove_think_blocks(raw).strip()
                if raw.startswith("```"):
                    raw = raw.lstrip("`").split("```")[0]
                return json.loads(raw)
            except Exception:
                continue
        return []

    def annotate(
        self,
        document: Document,
        args=None,
        level="page",
        target_layouts=("text", "list", "cell"),
        annotator_name="kpi",
    ) -> Document:
        annotator_name = args.kpi_annotator_name if args is not None else annotator_name
        level = args.kpi_text_level if args is not None else level
        target_layouts = (
            args.kpi_target_layouts if args is not None else list(target_layouts)
        )
        thresh_sim = args.kpi_sim_threshold if args else 0.35

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
        company_name = None
        sector = None

        first_page_text = document.pages[0].get_text_by_target_layouts(target_layouts)
        company, sector = self.detect_company_and_sector(first_page_text)
        logger.info("Company=%s  |  Sector=%s", company, sector)
        self.kpi_defs, self.kpi_index = self.get_sector_view(sector)

        for page in document.pages:
            if level == "page":
                text = page.get_text_by_target_layouts(target_layouts)
                if not text.strip():
                    continue
                kpi_subset = self.closest_kpis(text, k=8)
                hits = self.page_llm_extract(text, kpi_subset)
                for hit in hits:
                    hit.update({
                        "page": page.num,
                        "doc_name": document.name,
                        "sector": sector,
                        "company": company,
                    })
                    page.add_annotation(
                        Annotation(
                            parent_object=page,
                            annotator=annotator_name,
                            value=json.dumps(hit, ensure_ascii=False),
                            meta=hit,
                        )
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
        parser.add_argument(
            "--kpi_text_level", type=str, default="page", choices=["page"]
        )
        parser.add_argument(
            "--kpi_target_layouts",
            nargs="+",
            default=["text", "list", "cell"],
            choices=LAYOUT_NAMES,
        )
        parser.add_argument(
            "--kpi_sim_threshold",
            type=float,
            default=0.35,
            help="cosine‑sim threshold to accept a KPI hit",
        )
        parser.add_argument("--kpi_annotator_name", type=str, default="kpi")
