import os, re, json, argparse
from logging import getLogger
from dotenv import load_dotenv

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document, Annotation, AnnotatableLevel
from reportparse.util.settings import LAYOUT_NAMES
from reportparse.util.my_embeddings import get_embedder
from reportparse.util.remove_thinking import remove_think_blocks

from sentence_transformers import SentenceTransformer
import faiss

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

import json, re, textwrap
from typing import List, Dict, Any

logger = getLogger(__name__)

CUE_REGEX = re.compile(
    r"\b("
    r"achiev\w*|reached|delivered|actual(?:ly)?|reduc\w*|increase\w*|cut|"
    r"goal|target|aim|pledge|commit\w*|baseline|base\s+year|since|from"
    r")\b",
    flags=re.I,
)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")  # naive sentence splitter


def _clean_json(raw: str) -> Any:
    """
    Attempts to load a JSON string while forgiving the three most common
    LLM quirks: Markdown fences, trailing commas, and stray ```json blocks.
    """
    # strip markdown fences
    if raw.lstrip().startswith("```"):
        raw = raw.lstrip("`").split("```", 1)[0]

    # drop trailing commas inside lists / objects
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)

    return json.loads(raw)


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
            f'TEXT SOURCE (may be partial):\n"""\n{text}\n"""\n\n'
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
                    sector = str(data.get("sector", "")).strip()
                except (json.JSONDecodeError, TypeError, ValueError):
                    data = None

                if not data:
                    company_match = re.search(
                        r'"?company name"?\s*:\s*"([^"]+)"', reply, re.I
                    )
                    sector_match = re.search(r'"?sector"?\s*:\s*"([^"]+)"', reply, re.I)
                    company = company_match.group(1).strip() if company_match else ""
                    sector = sector_match.group(1).strip() if sector_match else ""

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
            d
            for d in self.all_kpi_defs
            if sector_key in d["sector"] or "None of the listed" in d["sector"]
        ]
        if not kpi_defs:
            kpi_defs = self.all_kpi_defs

        sentences = [
            f"{d['id']} – {d['name']} – {d.get('definition','')}" for d in kpi_defs
        ]
        emb = self.embedder.encode(sentences, normalize_embeddings=True)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        self._indices_by_sector[sector_key] = (kpi_defs, index)
        return kpi_defs, index

    def closest_kpis(self, page_text: str, k: int = 8):
        emb = self.embedder.encode(page_text[:2048], normalize_embeddings=True)
        sim, idx = self.kpi_index.search(emb.reshape(1, -1), k=k)
        return [self.kpi_defs[i] for i in idx[0]]

    # def page_llm_extract(self, page_text: str, kpi_subset: list[dict]):
    #     schema = """
    #     [
    #     {
    #         "kpi_id": "string",
    #         "value":  number,
    #         "unit":   "string",
    #         "year":   integer | null,
    #         "confidence": number,
    #         "snippet": "string"
    #     }
    #     ]
    #     """.strip()

    #     defs = "\n".join(f"{d['id']}: {d['definition']}" for d in kpi_subset)
    #     prompt = (
    #         "You are an ESG‑KPI extractor.\n"
    #         "Return ONLY a JSON array with **one object per KPI you can spot**.\n"
    #         f"Schema:\n{schema}\n\n"
    #         f"**KPI_DEFINITIONS (subset)**\n{defs}\n\n"
    #         f"**TEXT SOURCE**\n\"\"\"\n{page_text}\n\"\"\""
    #     )

    #     for llm in (self.llm, self.llm_2):
    #         try:
    #             raw = llm.invoke([("system", "KPI extractor"), ("human", prompt)]).content
    #             raw = remove_think_blocks(raw).strip()
    #             if raw.startswith("```"):
    #                 raw = raw.lstrip("`").split("```")[0]
    #             return json.loads(raw)
    #         except Exception:
    #             continue
    #     return []

    def page_llm_extract(self, page_text: str, figure_desc: str, kpi_subset: List[Dict]) -> List[Dict]:
        """
        Extract KPI observations from a single page using an LLM with a
        strongly-typed prompt and defensive post-processing.
        """
        # ------------------------------------------------------------------ #
        # 1) Heuristic pre-filter: keep only sentences that contain cue words
        # ------------------------------------------------------------------ #
        # sentences = SENT_SPLIT.split(page_text)
        # cand_sents = [s for s in sentences if CUE_REGEX.search(s)]
        # # If nothing hit, fall back to first 3 500 chars (still bound the prompt)
        # extract_text = " ".join(cand_sents) if cand_sents else page_text[:3500]
        extract_text = page_text
        # ------------------------------------------------------------------ #
        # 2) Build schema + KPI definition block
        # ------------------------------------------------------------------ #
        schema = textwrap.dedent(
            """
            [
            {
                "kpi_id"   : "string",
                "metric"   : "string",
                "observations": [
                {
                    "kind"          : "baseline | target | achieved | projection",
                    "value"         : number,
                    "unit"          : "string",
                    "direction"     : "absolute | reduction | increase",
                    "year"          : integer | null,
                    "target_year"   : integer | null,
                    "baseline_year" : integer | null,
                    "confidence"    : number,
                    "snippet"       : "string (≤160 chars)"
                }
                ],
                "page"     : integer,
                "doc_name" : "string",
                "company"  : "string",
                "sector"   : "string"
            }
            ]
            """
        ).strip()

        defs = "\n".join(f"{d['id']}: {d['definition']}" for d in kpi_subset)

        prompt = textwrap.dedent(
            f"""
            <system>
            You are ESG-KPI-EXTRACTOR-V2. Produce only JSON conforming exactly
            to the schema below. If no KPI can be unambiguously extracted, output [].
            Note: Text source is enhanced with figure captions and descriptions from an improved extractor.
            If there is a discrepancy between the text and the figure descriptions, prefer the figure descriptions.
            </system>

            <human>
            **Schema**  
            {schema}

            **Classification rules**
            • baseline  – historic reference (keywords: since, baseline, base year)  
            • target    – ambition or commitment (keywords: goal, target, aim)  
            • achieved  – result already met (keywords: achieved, delivered, reduced)  
            • projection – future estimate not yet committed

            If a sentence contains several numbers for the *same* KPI, create
            separate observation objects.

            Do **not** wrap the JSON in Markdown fences or code tags.  
            Return [] rather than guess.

            **KPI_DEFINITIONS (subset)**  
            {defs}

            **TEXT SOURCE**  
            \"\"\"{extract_text}\"\"\"

            **FIGURE DESCRIPTIONS**
            \"\"\"{figure_desc}\"\"\"

            </human>
            """
        ).strip()

        # ------------------------------------------------------------------ #
        # 3) Call the LLM(s) with retries
        # ------------------------------------------------------------------ #
        for llm in (self.llm, self.llm_2):
            try:
                raw = llm.invoke(
                    [("system", "KPI extractor"), ("human", prompt)]
                ).content
                raw = remove_think_blocks(raw).strip()
                return _clean_json(raw)
            except Exception as err:
                logger.warning(
                    "page_llm_extract LLM-failure (%s): %s", type(llm), err
                )

        return []  # both LLMs failed

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
            print(page.num)
            if level == "page":
                text = page.get_text_by_target_layouts(target_layouts)
                page_index = page.num + 1
                print(document.name, page_index)
                # document name = base name without .pdf
                name = document.name.split(".pdf")[0]

                if not text.strip():
                    continue

                figure_desc_dir = args.figure_description_dir
                figure_desc_filename = f"{name}_page_{page_index}.txt"
                figure_desc_path = os.path.join(figure_desc_dir, figure_desc_filename)

                with open(figure_desc_path, "r", encoding="utf-8") as f:
                    figure_desc = f.read()


                kpi_subset = self.closest_kpis(text, k=8)
                hits = self.page_llm_extract(text, figure_desc, kpi_subset)
                for hit in hits:
                    hit.update(
                        {
                            "page": page.num,
                            "doc_name": document.name,
                            "sector": sector,
                            "company": company,
                        }
                    )
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

        parser.add_argument(
            "--figure_description_dir",
            type=str,
            required=True,
            help="Directory containing figure description .txt files (from image analysis)."
        )