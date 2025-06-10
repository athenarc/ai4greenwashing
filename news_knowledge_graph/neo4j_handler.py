import pandas as pd
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from triplet_extractor import triplet_extractor

load_dotenv()


class neo4j_handler:

    def __init__(self, claim):
        self.claim = claim
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASS")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_triplet_with_article(self, tx, article_id, subject, predicate, obj):
        safe_predicate = "".join(c if c.isalnum() else "_" for c in predicate.upper())
        query = f"""
        MERGE (article:Article {{id: $article_id}})
        MERGE (s:Entity {{name: $subject}})
        MERGE (o:Entity {{name: $object}})
        MERGE (s)-[:{safe_predicate}]->(o)
        MERGE (article)-[:CONTEXT_OF]->(s)
        MERGE (article)-[:CONTEXT_OF]->(o)
        """
        tx.run(
            query,
            article_id=article_id,
            subject=subject,
            predicate=predicate,
            object=obj,
        )

    def insert_triplets(self, triplets_by_article):
        with self.driver.session() as session:
            for article_id, triplets in triplets_by_article.items():
                for subj, pred, obj in triplets:
                    session.execute_write(
                        self.create_triplet_with_article, article_id, subj, pred, obj
                    )

    def generate_query(self, claim):
        llm_extractor = triplet_extractor()
        llm_result = llm_extractor.call_triplet_llm(claim)
        entities = llm_extractor.extract_entities(llm_result)
        return entities  # todo -> generate neo4j query from entities

    def retrieve_article_ids(self, query):
        pass

    def retrieve_articles(self, ids):
        pass
