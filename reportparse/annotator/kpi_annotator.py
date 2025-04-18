import os, re, json, argparse
from logging import getLogger
from dotenv import load_dotenv

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document, Annotation, AnnotatableLevel
from reportparse.util.settings import LAYOUT_NAMES

from sentence_transformers import SentenceTransformer
import faiss
from reportparse.remove_thinking import remove_think_blocks

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

logger = getLogger(__name__)


@BaseAnnotator.register("kpi")
class KPIAnnotator(BaseAnnotator):

    def __init__(self):

        defs_path = "/home/geoka/Desktop/greenwashing/ai4greenwashing/reportparse/kpi_definitions.json"
        with open(defs_path, "r", encoding="utf‑8") as f:
            self.kpi_defs = json.load(f)
        self.build_kpi_index()

        units = [r"kWh", r"MWh", r"tCO2e?", r"mtCO2e?", r"%"]
        number = r"[\d][\d\s,.\']*"
        self.candidate_regex = re.compile(
            rf"({number})\s*({'|'.join(units)})", re.IGNORECASE
        )

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

    def build_kpi_index(self):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        kpi_sentences = [
            f"{d['id']} – {d['name']} – {d.get('definition','')}" for d in self.kpi_defs
        ]
        self.kpi_emb = self.embedder.encode(kpi_sentences, normalize_embeddings=True)
        dim = self.kpi_emb.shape[1]
        self.kpi_index = faiss.IndexFlatIP(dim)
        self.kpi_index.add(self.kpi_emb)

    def _closest_kpis(self, page_text, k=8):
        emb = self.embedder.encode(page_text[:2048], normalize_embeddings=True)
        sim, idx = self.kpi_index.search(emb.reshape(1, -1), k=k)
        return [self.kpi_defs[i] for i in idx[0]]

    def _page_llm(self, page_text: str, kpi_subset: list[dict]):
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
        prompt = (f"You are an ESG‑KPI extractor. "
                f"Return ONLY a JSON array that follows the schema.\n\n"
                f"# KPI_DEFINITIONS\n{defs}\n\n"
                f"# PAGE_TEXT\n\"\"\"\n{page_text}\n\"\"\"")

        for llm in (self.llm, self.llm_2):
            try:
                res = llm.invoke([("system", "Extract KPIs"), ("human", prompt)])
                return self.safe_json(res.content) or []
            except Exception as e:
                logger.warning(f"page‑LLM failed ({type(llm)}): {e}")
        return []

    def call_llm_extract(self, snippet: str, best_kpi: dict):
        """Prompt small context –> strict JSON."""
        schema = """
        {
          "kpi_id": "string",
          "value":  number,
          "unit":   "string",
          "year":   integer | null,
          "confidence": number,
          "snippet": "string"
        }
        """.strip()

        system = ("You are an ESG‑KPI extractor. "
                  "Return EXACTLY one JSON object conforming to the schema.")
        user   = (f"# KPI_DEFINITION\n{best_kpi}\n\n"
                  f"# PAGE_SNIPPET\n\"\"\"\n{snippet}\n\"\"\"\n\n"
                  f"# JSON_SCHEMA\n{schema}")

        for llm in (self.llm, self.llm_2):
            try:
                res = llm.invoke([("system", system), ("human", user)])
                return self.safe_json(res.content)
            except Exception as e:
                logger.warning(f"KPI LLM failed ({type(llm)}): {e}")
        return None

    @staticmethod
    def safe_json(text):
        try:
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            return json.loads(text[first_brace : last_brace + 1])
        except Exception:
            return None

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

        for page in document.pages:
            if level == "page":
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                if not text.strip():
                    continue
                kpi_subset = self._closest_kpis(text)          # ≤ TOP_K defs
                hits = self._page_llm(text, kpi_subset) 
                for h in hits:
                    h.update({"page": page.num, "doc_name": document.name})
                _annotate(page, h)

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
