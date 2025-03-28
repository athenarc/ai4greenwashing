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
    def __init__(self, clear_mem=True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.clear_mem = clear_mem

        self.nli_model = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=0 if device == "cuda" else -1
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_cache = {}

    def normalize_to_string(self, content):
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return " ".join(map(str, content))
        elif isinstance(content, dict):
            return " ".join(f"{k}: {v}" for k, v in content.items())
        else:
            return str(content)


    #recursive character text splitter
    def chunk_text(self, text):
        return self.text_splitter.split_text(text)
    
    def clear_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def encode_cached(self, text_list):
        uncached = [t for t in text_list if t not in self.embedding_cache]
        if uncached:
            new_embeddings = self.embedder.encode(uncached, convert_to_tensor=True)
            for text, emb in zip(uncached, new_embeddings):
                self.embedding_cache[text] = emb
        return torch.stack([self.embedding_cache[t] for t in text_list])

    # measures the contradiction or entailment between llm's answer and the retrieved information, along with a confidence score (logit differences)
    def faith_eval(self, answer, retrieved_docs, precomputed_chunks=None):
        if not retrieved_docs:
            return {
                "avg_faithfulness": 0,
                "general_stance": "NEUTRAL",
            }

        context = " ".join(retrieved_docs)
        chunks = precomputed_chunks or self.chunk_text(context)

        pairs = [{"text": chunk, "text_pair": answer} for chunk in chunks]
        results = self.nli_model(pairs, return_all_scores=True)

        stance_counts = Counter()
        entailment_scores = []
        logit_differences = []

        for result in results:
            label_scores = {res["label"]: res["score"] for res in result}
            label_logits = {label: np.log(score) for label, score in label_scores.items()}

            top_label = max(label_scores, key=label_scores.get)
            stance_counts[top_label] += 1

            entailment_scores.append(label_scores.get("entailment", 0))
            logit_differences.append(
                label_logits.get("entailment", 0) - label_logits.get("contradiction", 0)
            )

        if self.clear_mem:
            self.clear_memory()

        general_stance = stance_counts.most_common(1)[0][0] if stance_counts else "NEUTRAL"

        return {
            "avg_faithfulness": np.mean(entailment_scores),
            "general_stance": general_stance
        }

    # measures relevance between the user's query and the retrieved information
    def query_info_relevance_eval(self, query, retrieved_docs, precomputed_chunks=None):
        if not retrieved_docs:
            return {"query_info_relevance_score": 0}

        doc_chunks = precomputed_chunks or self.chunk_text(" ".join(retrieved_docs))
        embeddings = self.encode_cached([query] + doc_chunks)
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]

        similarities = util.cos_sim(query_embedding, doc_embeddings).cpu().numpy().flatten()
        if self.clear_mem:
            self.clear_memory()

        return {"query_info_relevance_score": float(np.mean(similarities))}

    # measures hallucination risk, by examining the llm's answer and the retrieved information
    def groundedness_eval(self, answer, retrieved_docs, precomputed_chunks=None):
        if not retrieved_docs:
            return {"groundedness_score": 0}

        doc_chunks = precomputed_chunks or self.chunk_text(" ".join(retrieved_docs))
        answer_chunks = self.chunk_text(answer)

        tokenized_chunks = [chunk.split() for chunk in doc_chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        answer_tokens = [chunk.split() for chunk in answer_chunks if chunk.strip()]
        bm25_scores = [np.mean(bm25.get_scores(tokens)) for tokens in answer_tokens if tokens]
        max_bm25 = max(bm25_scores) if bm25_scores else 1
        normalized_bm25_scores = [score / max_bm25 for score in bm25_scores]

        embeddings = self.encode_cached(answer_chunks + doc_chunks)
        answer_embeddings = embeddings[:len(answer_chunks)]
        doc_embeddings = embeddings[len(answer_chunks):]

        cos_sim_scores = util.cos_sim(answer_embeddings, doc_embeddings).cpu().numpy().flatten()

        if self.clear_mem:
            self.clear_memory()

        avg_bm25 = np.mean(normalized_bm25_scores) if normalized_bm25_scores else 0
        avg_cos_sim = np.mean(cos_sim_scores) if cos_sim_scores.size > 0 else 0
        return {"groundedness_score": (avg_bm25 + avg_cos_sim) / 2}
    
    #measures the specificity of an LLM-generated answer by calculating the proportion of named entities
    def specificity_eval(self, answer):
        doc = self.nlp(answer)
        token_count = len(doc)
        specificity_score = len(doc.ents) / token_count if token_count > 0 else 0
        return {"specificity_score": specificity_score}

    #measures redundancy in an LLM-generated answer
    def redundancy_eval(self, answer, precomputed_chunks=None):
        answer_chunks = precomputed_chunks or self.chunk_text(answer)
        if len(answer_chunks) < 2:
            return {"redundancy_score": 0}

        embeddings = self.encode_cached(answer_chunks)
        redundancy_scores = util.cos_sim(embeddings[:-1], embeddings[1:]).cpu().numpy().flatten()

        if self.clear_mem:
            self.clear_memory()

        return {"redundancy_score": float(np.mean(redundancy_scores))}

    #measures readability in an LLM-generated answer
    def readability_eval(self, answer):
        return {"readability_score": textstat.flesch_reading_ease(answer)}
    
    def compression_ratio_eval(self, answer, retrieved_docs):
        if isinstance(retrieved_docs, str):
            retrieved_text = retrieved_docs
        elif isinstance(retrieved_docs, list):
            retrieved_text = " ".join(map(str, retrieved_docs))
        elif isinstance(retrieved_docs, dict):
            retrieved_text = " ".join(f"{k}: {v}" for k, v in retrieved_docs.items())
        else:
            retrieved_text = str(retrieved_docs)

        ratio = len(answer) / len(retrieved_text) if len(retrieved_text) > 0 else 0
        return {"compression_ratio": ratio}

    def lexical_diversity_eval(self, answer):
        if not isinstance(answer, str):
            answer = str(answer)
        tokens = answer.split()
        unique_tokens = set(tokens)
        diversity = len(unique_tokens) / len(tokens) if tokens else 0
        return {"lexical_diversity": diversity}

    def noun_to_verb_ratio_eval(self, answer):
        if not isinstance(answer, str):
            answer = str(answer)
        doc = self.nlp(answer)
        noun_count = len([token for token in doc if token.pos_ == "NOUN"])
        verb_count = len([token for token in doc if token.pos_ == "VERB"])
        ratio = noun_count / verb_count if verb_count > 0 else 0
        return {"noun_to_verb_ratio": ratio}