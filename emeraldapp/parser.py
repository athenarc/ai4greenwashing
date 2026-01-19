import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import logging
import fitz  # pymupdf
import chromadb
from chromadb.config import Settings
import math
from transformers import AutoTokenizer
import json
import csv
from model_loader import load_encoder
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenericParser:
    """
    A generic document parser that supports multiple file types and flexible metadata extraction.
    """
    
    def __init__(
        self,
        documents_folder: str,
        db_path: str,
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        tokenizer_model: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size_tokens: int = 250,
        overlap_tokens: int = 50,
        file_extensions: List[str] = None,
        metadata_extractor: Optional[Callable] = None
    ):

        self.documents_folder = Path(documents_folder)
        self.db_path = db_path
        self.collection_name = collection_name
        self.model = load_encoder()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens
        self.file_extensions = file_extensions or ['.pdf', '.txt', '.json', '.csv']
        self.metadata_extractor = metadata_extractor or self._default_metadata_extractor

        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection '{collection_name}'")
        except Exception as e:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Using existing collection '{collection_name}'")

    def _default_metadata_extractor(self, filename: str) -> Dict:
        return {
            "filename": filename,
            "source": filename
        }

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[int, str]:
        page_texts = {}
        try:
            pdf_document = fitz.open(str(pdf_path))
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text = page.get_text().strip()
                if text:
                    page_texts[page_num + 1] = text
            pdf_document.close()
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
        return page_texts

    def extract_text_from_txt(self, txt_path: Path) -> str:
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading TXT {txt_path}: {e}")
            return ""

    def extract_text_from_json(self, json_path: Path) -> str:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._flatten_json_to_text(data)
        except Exception as e:
            logger.error(f"Error reading JSON {json_path}: {e}")
            return ""

    def _flatten_json_to_text(self, data) -> str:
        if isinstance(data, dict):
            return " ".join(str(v) for v in data.values() if v)
        elif isinstance(data, list):
            return " ".join(str(item) for item in data if item)
        else:
            return str(data)

    def extract_text_from_csv(self, csv_path: Path) -> str:
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = [" ".join(row) for row in reader]
                return "\n".join(rows)
        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            return ""

    def extract_text(self, file_path: Path) -> Dict[str, any]:
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return {
                'content': self.extract_text_from_pdf(file_path),
                'type': 'pdf',
                'has_pages': True
            }
        elif extension == '.txt':
            return {
                'content': self.extract_text_from_txt(file_path),
                'type': 'text',
                'has_pages': False
            }
        elif extension == '.json':
            return {
                'content': self.extract_text_from_json(file_path),
                'type': 'json',
                'has_pages': False
            }
        elif extension == '.csv':
            return {
                'content': self.extract_text_from_csv(file_path),
                'type': 'csv',
                'has_pages': False
            }
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return {
                'content': "",
                'type': 'unknown',
                'has_pages': False
            }

    def chunk_text_tokenizer(self, text: str) -> List[str]:
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

    def process_documents(self):
        # Get all files with specified extensions
        all_files = []
        for ext in self.file_extensions:
            all_files.extend(list(self.documents_folder.glob(f"*{ext}")))
        
        if not all_files:
            logger.warning(f"No files found with extensions {self.file_extensions}")
            return

        logger.info(f"Found {len(all_files)} files to process")

        documents, metadatas, ids, embeddings = [], [], [], []

        for file_path in all_files:
            try:
                logger.info(f"Processing {file_path.name}")
                
                # Extract metadata using custom or default extractor
                base_metadata = self.metadata_extractor(file_path.name)
                
                # Extract text
                extraction_result = self.extract_text(file_path)
                content = extraction_result['content']
                file_type = extraction_result['type']
                has_pages = extraction_result['has_pages']

                if has_pages:
                    # PDF with pages
                    for page_num, text in content.items():
                        if not text.strip():
                            continue
                        chunks = self.chunk_text_tokenizer(text)
                        for chunk_num, chunk in enumerate(chunks, start=1):
                            doc_id = f"{file_path.stem}_page_{page_num}_chunk_{chunk_num}"
                            documents.append(chunk)
                            
                            chunk_metadata = base_metadata.copy()
                            chunk_metadata.update({
                                "page_number": page_num,
                                "chunk_number": chunk_num,
                                "file_type": file_type,
                                "source_file": file_path.name
                            })
                            metadatas.append(chunk_metadata)
                            ids.append(doc_id)
                            
                            emb = self.model.encode(chunk, normalize_embeddings=True)
                            embeddings.append(emb.tolist())
                else:
                    # Text-based file without pages
                    if content.strip():
                        chunks = self.chunk_text_tokenizer(content)
                        for chunk_num, chunk in enumerate(chunks, start=1):
                            doc_id = f"{file_path.stem}_chunk_{chunk_num}"
                            documents.append(chunk)
                            
                            chunk_metadata = base_metadata.copy()
                            chunk_metadata.update({
                                "chunk_number": chunk_num,
                                "file_type": file_type,
                                "source_file": file_path.name
                            })
                            metadatas.append(chunk_metadata)
                            ids.append(doc_id)
                            
                            emb = self.model.encode(chunk, normalize_embeddings=True)
                            embeddings.append(emb.tolist())

                logger.info(f"Processed {file_path.name}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        # Add to ChromaDB in batches
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

    def add_single_file(self, file_path: str) -> List[str]:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        logger.info(f"Adding single file: {file_path.name}")
        
        documents, metadatas, ids, embeddings = [], [], [], []
        added_ids = []
        
        try:
            # Extract metadata using custom or default extractor
            base_metadata = self.metadata_extractor(file_path.name)
            
            # Check if file already exists in database
            existing_files = self.get_all_files()
            if file_path.name in existing_files:
                logger.warning(f"File {file_path.name} already exists in database. Skipping...")
                return []
            
            # Extract text
            extraction_result = self.extract_text(file_path)
            content = extraction_result['content']
            file_type = extraction_result['type']
            has_pages = extraction_result['has_pages']

            if has_pages:
                # PDF with pages
                for page_num, text in content.items():
                    if not text.strip():
                        continue
                    chunks = self.chunk_text_tokenizer(text)
                    for chunk_num, chunk in enumerate(chunks, start=1):
                        doc_id = f"{file_path.stem}_page_{page_num}_chunk_{chunk_num}"
                        documents.append(chunk)
                        
                        chunk_metadata = base_metadata.copy()
                        chunk_metadata.update({
                            "page_number": page_num,
                            "chunk_number": chunk_num,
                            "file_type": file_type,
                            "source_file": file_path.name
                        })
                        metadatas.append(chunk_metadata)
                        ids.append(doc_id)
                        added_ids.append(doc_id)
                        
                        emb = self.model.encode(chunk, normalize_embeddings=True)
                        embeddings.append(emb.tolist())
            else:
                # Text-based file without pages
                if content.strip():
                    chunks = self.chunk_text_tokenizer(content)
                    for chunk_num, chunk in enumerate(chunks, start=1):
                        doc_id = f"{file_path.stem}_chunk_{chunk_num}"
                        documents.append(chunk)
                        
                        chunk_metadata = base_metadata.copy()
                        chunk_metadata.update({
                            "chunk_number": chunk_num,
                            "file_type": file_type,
                            "source_file": file_path.name
                        })
                        metadatas.append(chunk_metadata)
                        ids.append(doc_id)
                        added_ids.append(doc_id)
                        
                        emb = self.model.encode(chunk, normalize_embeddings=True)
                        embeddings.append(emb.tolist())

            # Add to ChromaDB in batches
            if documents:
                batch_size = 100
                logger.info(f"Adding {len(documents)} chunks from {file_path.name} to ChromaDB")
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
                logger.info(f"Successfully added {file_path.name} to database")
            else:
                logger.warning(f"No content extracted from {file_path.name}")

        except Exception as e:
            logger.error(f"Error adding file {file_path}: {e}")
            return []
        
        return added_ids

    def remove_file(self, filename: str) -> bool:
        try:
            # Get all document IDs for this file
            results = self.collection.get(
                where={"source_file": filename},
                include=["metadatas"]
            )
            
            ids_to_delete = results['ids']
            
            if not ids_to_delete:
                logger.warning(f"No documents found for file: {filename}")
                return False
            
            # Delete all chunks
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Successfully removed {len(ids_to_delete)} chunks from file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing file {filename}: {e}")
            return False

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where_filter: Optional[Dict] = None
    ) -> Dict:
        """
        Query the collection.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            where_filter: ChromaDB where filter (e.g., {"file_type": "pdf"})
        """
        query_emb = self.model.encode([query_text], normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=n_results,
            where=where_filter
        )
        return results

    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection.name
        }

    def get_all_files(self) -> List[str]:
        """Get list of all files that have been processed."""
        results = self.collection.get(include=["metadatas"])
        metadatas = results["metadatas"]
        files = {m["source_file"] for m in metadatas}
        return sorted(list(files))

    def get_file_stats(self) -> Dict[str, Dict]:
        """Get statistics for each processed file."""
        results = self.collection.get(include=["metadatas"])
        metadatas = results["metadatas"]
        
        stats = {}
        for m in metadatas:
            filename = m["source_file"]
            if filename not in stats:
                stats[filename] = {
                    "chunk_count": 0,
                    "file_type": m.get("file_type", "unknown"),
                    "pages": set() if "page_number" in m else None
                }
            
            stats[filename]["chunk_count"] += 1
            if "page_number" in m:
                stats[filename]["pages"].add(m["page_number"])
        
        # Convert page sets to counts
        for filename in stats:
            if stats[filename]["pages"] is not None:
                stats[filename]["page_count"] = len(stats[filename]["pages"])
                del stats[filename]["pages"]
        
        return stats


if __name__ == "__main__":
    pass