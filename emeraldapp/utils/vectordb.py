import os
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import math
from transformers import AutoTokenizer
from model_loader import load_encoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportParser:
    def __init__(
        self,
        reports_folder: str,
        db_path: str,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        tokenizer_model: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size_tokens: int = 250,
        overlap_tokens: int = 50
    ):
        self.reports_folder = Path(reports_folder)
        self.db_path = db_path
        self.model = load_encoder()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.create_collection(
                name="company_reports",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            self.collection = self.client.get_collection("company_reports")
            logger.info("Using existing collection 'company_reports'")

    def parse_filename(self, filename: str) -> Tuple[str, str]:
        base_name = filename.replace('.pdf', '')
        parts = base_name.split('_')
        if len(parts) >= 2:
            year = parts[-1]
            company = '_'.join(parts[:-1])
            return company, year
        else:
            raise ValueError(f"Cannot parse filename: {filename}")

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[int, str]:
        page_texts = {}
        try:
            pdf_document = fitz.open(str(pdf_path))
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text = page.get_text().strip()
                page_texts[page_num + 1] = text
            pdf_document.close()
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
        return page_texts

    def chunk_text_tokenizer(self, text: str) -> List[str]:
        """Split text into token-based chunks using the tokenizer."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
            start += (self.chunk_size_tokens - self.overlap_tokens)
        return chunks

    def process_reports(self):
        pdf_files = list(self.reports_folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in the reports folder")
            return

        documents, metadatas, ids, embeddings = [], [], [], []

        for pdf_file in pdf_files:
            try:
                company, year = self.parse_filename(pdf_file.name)
                logger.info(f"Processing {company} - {year}")
                page_texts = self.extract_text_from_pdf(pdf_file)

                for page_num, text in page_texts.items():
                    if not text.strip():
                        continue
                    chunks = self.chunk_text_tokenizer(text)
                    for chunk_num, chunk in enumerate(chunks, start=1):
                        doc_id = f"{company}_{year}_page_{page_num}_chunk_{chunk_num}"
                        documents.append(chunk)
                        metadatas.append({
                            "company": company,
                            "year": year,
                            "page_number": page_num,
                            "chunk_number": chunk_num,
                            "source_file": pdf_file.name
                        })
                        ids.append(doc_id)
                        emb = self.model.encode(chunk, normalize_embeddings=True)
                        embeddings.append(emb.tolist())

                logger.info(f"Processed {len(page_texts)} pages for {company} - {year}")

            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue

        if documents:
            batch_size = 100
            logger.info(f"Adding {len(documents)} documents to ChromaDB")
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metadata = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]

                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )
                logger.info(f"Added batch {i//batch_size + 1}/{math.ceil(len(documents)/batch_size)}")
            logger.info("Successfully added all documents to ChromaDB")
        else:
            logger.warning("No documents to add to ChromaDB")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        company_filter: str = None,
        year_filter: str = None
    ) -> Dict:
        where_filter = None
        
        # Build filter conditions
        conditions = []
        if company_filter:
            conditions.append({"company": company_filter})
        if year_filter:
            conditions.append({"year": year_filter})
        
        # ChromaDB requires $and for multiple conditions
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        # Encode query using the same transformer used for storing embeddings
        query_emb = self.model.encode([query_text], normalize_embeddings=True).tolist()

        # Pass embedding directly to Chroma
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=n_results,
            where=where_filter
        )
        return results
    def get_collection_info(self) -> Dict:
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection.name
        }
    

    def get_snippets_by_page(
        self,
        company: str,
        year: str,
        page_number: int
    ) -> Dict:

        where_filter = {
            "company": company,
            "year": year,
            "page_number": page_number
        }

        results = self.collection.query(
            n_results=10000,  # large number to ensure all chunks are returned
            where=where_filter
        )
        return results
    
    def get_parsed_pdf_files(self) -> List[str]:
        results = self.collection.get(include=["metadatas"])
        metadatas = results["metadatas"]  # list of dicts
        pdf_files = {m["source_file"] for m in metadatas}
        return list(pdf_files)
    
    def get_pdf_page_counts(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping each PDF file to the number of pages
        that were actually stored in the ChromaDB.
        """
        results = self.collection.get(include=["metadatas"])
        metadatas = results["metadatas"]  # list of dicts
        page_counts = {}
        for m in metadatas:
            pdf_file = m["source_file"]
            page_number = m["page_number"]
            if pdf_file not in page_counts:
                page_counts[pdf_file] = set()
            page_counts[pdf_file].add(page_number)
        
        # Convert sets to counts
        page_counts = {pdf: len(pages) for pdf, pages in page_counts.items()}
        return page_counts



if __name__ == "__main__":
    pass
    # parser = ReportParser(
    #     reports_folder="../../Greenwashing_claims_esg_reports",
    #     db_path="chromadb_updated"
    # )
    # parser.process_reports()
    # info = parser.get_collection_info()
    # print(f"Collection created with {info['total_documents']} documents")


