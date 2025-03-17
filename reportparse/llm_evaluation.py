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
        self.nli_model = pipeline(
            "text-classification", model="facebook/bart-large-mnli"
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def chunk_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=128
        )

        chunks = text_splitter.split_text(text)
        return chunks

    # measures the contradiction or entailment between llm's answer and the retrieved information, along with a confidence score (logit differences)
    def faith_eval(self, answer, retrieved_docs):
        context = " ".join(retrieved_docs)
        chunks = self.chunk_text(context)
        stance_counts = Counter()

        if not chunks:
            return 0, "", 0

        entailment_scores = []
        logit_differences = []

        for chunk in chunks:
            premise = chunk
            hypothesis = answer

            
            with torch.no_grad():
                result = self.nli_model(f"{premise} [SEP] {hypothesis}", return_all_scores=True)[0]

            label_scores = {res["label"]: res["score"] for res in result}
            label_logits = {res["label"]: np.log(res["score"]) for res in result}

            stance_counts[max(label_scores, key=label_scores.get)] += 1

            entailment_scores.append(label_scores.get("entailment", 0))
            logit_diff = label_logits.get("entailment", 0) - label_logits.get(
                "contradiction", 0
            )
            logit_differences.append(logit_diff)

        general_stance = (
            stance_counts.most_common(1)[0][0] if stance_counts else "NEUTRAL"
        )
        avg_faithfulness = (
            sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0
        )
        avg_logit_diff = (
            sum(logit_differences) / len(logit_differences) if logit_differences else 0
        )

        torch.cuda.empty_cache()
        gc.collect()
        return avg_faithfulness, general_stance, avg_logit_diff

    # measures relevance between the user's query and the retrieved information
    def query_info_relevance_eval(self, query, retrieved_docs):

        similarities = []
        docs = " ".join(retrieved_docs)
        doc_chunks = self.chunk_text(docs)
        if not doc_chunks:
            return 0

        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        doc_chunks_embeddings = [
            self.embedder.encode(chunk, convert_to_tensor=True) for chunk in doc_chunks
        ]

        for emb in doc_chunks_embeddings:
            similarities.append(util.cos_sim(query_embedding, emb)[0].item())
            print(util.cos_sim(query_embedding, emb)[0])

        final_similarity = np.mean(similarities)
        torch.cuda.empty_cache()
        gc.collect()
        return final_similarity

    # measures hallucination risk, by examining the llm's answer and the retrieved information
    def groundedness_eval(self, answer, retrieved_docs):
        if not retrieved_docs:
            return 0

        retrieved_text = " ".join(retrieved_docs)
        doc_chunks = self.chunk_text(retrieved_text)

        answer_chunks = self.chunk_text(answer)

        tokenized_chunks = [chunk.split() for chunk in doc_chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        bm25_scores = []
        for answer_chunk in answer_chunks:
            answer_tokens = answer_chunk.split()
            score = np.mean(bm25.get_scores(answer_tokens)) if doc_chunks else 0
            bm25_scores.append(score)

        max_possible_bm25 = max(bm25_scores) if bm25_scores else 1
        normalized_bm25_scores = [score / max_possible_bm25 for score in bm25_scores]
        avg_bm25_score = (
            np.mean(normalized_bm25_scores) if normalized_bm25_scores else 0
        )

        model = SentenceTransformer("all-MiniLM-L6-v2")
        answer_embedding = model.encode(answer, convert_to_tensor=True)
        doc_embeddings = model.encode(doc_chunks, convert_to_tensor=True)

        cos_sim_scores = [
            util.cos_sim(answer_embedding, doc_emb)[0].item()
            for doc_emb in doc_embeddings
        ]
        avg_cos_sim = np.mean(cos_sim_scores) if cos_sim_scores else 0

        final_groundedness_score = (avg_bm25_score + avg_cos_sim) / 2
        torch.cuda.empty_cache()
        gc.collect()
        return final_groundedness_score

    # measures how specific the llm response is regarding NER
    def specificity_eval(self, answer):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(answer)

        named_entities = len(doc.ents)
        token_count = len(doc)

        specificity_score = named_entities / token_count if token_count > 0 else 0
        torch.cuda.empty_cache()
        gc.collect()
        return specificity_score

    # measures whether the llm repeats itself on its answer
    def redundancy_eval(self, answer):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        answer_chunks = self.chunk_text(answer)

        if len(answer_chunks) < 2:
            return 0.0

        embeddings = model.encode(answer_chunks, convert_to_tensor=True)

        redundancy_scores = []
        for i in range(len(embeddings) - 1):
            sim = util.cos_sim(embeddings[i], embeddings[i + 1])[0].item()
            redundancy_scores.append(sim)

        avg_redundancy = np.mean(redundancy_scores) if redundancy_scores else 0
        torch.cuda.empty_cache()
        gc.collect()
        return avg_redundancy

    # measures the answer and readability
    def readability_eval(self, answer):
        score = textstat.flesch_reading_ease(answer)
        torch.cuda.empty_cache()
        gc.collect()
        return score
