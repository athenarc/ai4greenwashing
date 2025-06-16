import pandas as pd
from news_knowledge_graph.neo4j_handler import neo4j_handler
from news_knowledge_graph.triplet_extractor import triplet_extractor
from sentence_transformers import SentenceTransformer
import numpy as np
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from reportparse.llm_prompts import NEWS_PROMPT
from reportparse.annotator.news_annotator import NewsAnnotator
from logging import getLogger

logger = getLogger(__name__)


class traverse_graph:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.neo4j = neo4j_handler()
        self.news_prompt = NEWS_PROMPT
        self.news_annotator = NewsAnnotator()
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    def find_article_chunks(self, neo4j, claim):
        chunk_ids = []
        chunk_texts = []
        published_dates = []
        claim_embedding = self.model.encode(claim).tolist()
        most_similar_chunk_id, similarity_score = neo4j.find_most_similar_chunk(
            claim_embedding
        )
        subgraph_nodes = neo4j.get_subgraph(most_similar_chunk_id, claim_embedding)
        if subgraph_nodes == []:
            return [], [], []
        for node in subgraph_nodes:
            chunk_ids.append(node["chunk_id"])
            chunk_texts.append(node["chunk_text"])
            published_dates.append(node["publish_date"])

        return chunk_ids, chunk_texts, published_dates

    def call_llm(self, context, claim):

        if context:
            messages = [
                (
                    "system",
                    self.news_prompt,
                ),
                (
                    "human",
                    f""" Statement: {claim}
                    Context from news: {context}
                    """,
                ),
            ]
            try:
                logger.info("Calling LLM to verify claim with context")
                ai_msg = self.llm.invoke(messages)
                print("AI message: ", ai_msg.content)
                return ai_msg.content
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                return "Error: Could not generate a response."
        else:
            return "No content in News database."

    def verify_claim_with_kg(self, neo4j, claim):
        chunk_ids, chunk_texts, dates = self.find_article_chunks(neo4j, claim)
        context = "\n\n".join(
            a + f"\nPublished date of the snippet above: {b}"
            for a, b in zip(chunk_texts, dates)
        )

        llm_answer = self.call_llm(context, claim)

        return {
            "chunk_ids": chunk_ids,
            "chunk_texts": chunk_texts,
            "llm_answer": llm_answer,
        }
