from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import numpy as np
from rank_bm25 import BM25Okapi
import spacy
import textstat
import torch
import gc
class llm_evaluation:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nli_model = pipeline("text-classification", model="facebook/bart-large-mnli", device=0 if device == "cuda" else -1)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        self.nlp = spacy.load("en_core_web_sm")

    #recursive character text splitter
    def chunk_text(self, text):
        return self.text_splitter.split_text(text)
    
    def clear_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    # measures the contradiction or entailment between llm's answer and the retrieved information, along with a confidence score (logit differences)
    def faith_eval(self, answer, retrieved_docs, precomputed_chunks=None):
        if not retrieved_docs:
            return {
                "avg_faithfulness": 0,
                "general_stance": "NEUTRAL",
                "avg_logit_diff": 0,
                "stance_counts": {},
                "entailment_scores": [],
                "logit_differences": []
            }

        context = "".join(retrieved_docs)
        chunks = precomputed_chunks or self.chunk_text(context)
        stance_counts = Counter()
        entailment_scores, logit_differences = [], []

        for chunk in chunks:
            with torch.no_grad():
                result = self.nli_model([{"text": chunk, "text_pair": answer}], return_all_scores=True)[0]

            label_scores = {res["label"]: res["score"] for res in result}
            label_logits = {res["label"]: np.log(res["score"]) for res in result}

            stance_counts[max(label_scores, key=label_scores.get)] += 1
            entailment_scores.append(label_scores.get("entailment", 0))
            logit_differences.append(label_logits.get("entailment", 0) - label_logits.get("contradiction", 0))

        general_stance = stance_counts.most_common(1)[0][0] if stance_counts else "NEUTRAL"
        return {
            "answer": answer,
            "retrieved_docs": context,
            "avg_faithfulness": float(np.mean(entailment_scores)) if entailment_scores else 0,
            "general_stance": general_stance,
            "avg_logit_diff": float(np.mean(logit_differences)) if logit_differences else 0,
            "stance_counts": dict(stance_counts),
            "entailment_scores": [float(score) for score in entailment_scores],
            "logit_differences": [float(diff) for diff in logit_differences]
        }


    # measures relevance between the user's query and the retrieved information
    def query_info_relevance_eval(self, query, retrieved_docs, precomputed_chunks=None):
        if not retrieved_docs:
            return {"query_info_relevance_score": 0}

        doc_chunks = precomputed_chunks or self.chunk_text(" ".join(retrieved_docs))
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        doc_embeddings = self.embedder.encode(doc_chunks, convert_to_tensor=True)

        similarities = util.cos_sim(query_embedding, doc_embeddings).cpu().numpy().flatten()
        self.clear_memory()
        final_similarity = float(np.mean(similarities)) if similarities.size > 0 else 0
        return {
            "query": query,
            "retrieved_docs": doc_chunks,
            "query_info_relevance_score": final_similarity}

    # measures hallucination risk, by examining the llm's answer and the retrieved information
    def groundedness_eval(self, answer, retrieved_docs, precomputed_chunks=None):
        if not retrieved_docs:
            return { "groundedness_score": 0 }

        doc_chunks = precomputed_chunks or self.chunk_text(" ".join(retrieved_docs))
        answer_chunks = self.chunk_text(answer)

        tokenized_chunks = [chunk.split() for chunk in doc_chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        answer_tokens = [chunk.split() for chunk in answer_chunks]
        bm25_scores = [np.mean(bm25.get_scores(tokens)) for tokens in answer_tokens if tokens]

        max_possible_bm25 = max(bm25_scores) if bm25_scores else 1
        normalized_bm25_scores = [score / max_possible_bm25 for score in bm25_scores]

        answer_embedding = self.embedder.encode(answer_chunks, convert_to_tensor=True)
        doc_embeddings = self.embedder.encode(doc_chunks, convert_to_tensor=True)

        cos_sim_scores = util.cos_sim(answer_embedding, doc_embeddings).cpu().numpy().flatten()
        self.clear_memory()

        avg_bm25_score = float(np.mean(normalized_bm25_scores)) if normalized_bm25_scores else 0
        avg_cos_sim = float(np.mean(cos_sim_scores)) if cos_sim_scores.size > 0 else 0
        final_groundedness_score = (avg_bm25_score + avg_cos_sim) / 2
        return { 
            "answer": answer,
            "retrieved_docs": doc_chunks,
            "avg_bm25_score": avg_bm25_score,
            "avg_cos_sim": avg_cos_sim,
            "groundedness_score": final_groundedness_score}
    
    #measures the specificity of an LLM-generated answer by calculating the proportion of named entities
    def specificity_eval(self, answer):
        doc = self.nlp(answer)
        token_count = len(doc)
        specificity_score = float(len(doc.ents) / token_count) if token_count > 0 else 0
        return { 
            "answer":answer,
            "token_count": token_count,
            "specificity_score": specificity_score}

    #measures redundancy in an LLM-generated answer
    def redundancy_eval(self, answer, precomputed_chunks=None):
        answer_chunks = precomputed_chunks or self.chunk_text(answer)
        if len(answer_chunks) < 2:
            return { 
                "answer":answer,
                "redundancy_score": 0 }

        embeddings = self.embedder.encode(answer_chunks, convert_to_tensor=True)
        redundancy_scores = util.cos_sim(embeddings[:-1], embeddings[1:]).cpu().numpy().flatten()
        self.clear_memory()
        avg_redundancy = float(np.mean(redundancy_scores)) if redundancy_scores.size > 0 else 0
        return {"answer":answer,
                "redundancy_scores":redundancy_scores,
                 "redundancy_score": avg_redundancy}

    #measures readability in an LLM-generated answer
    def readability_eval(self, answer):
        readability_score = textstat.flesch_reading_ease(answer)
        return {
            "answer":answer,
            "readability_score": readability_score
        }