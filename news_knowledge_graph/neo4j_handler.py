import pandas as pd
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from triplet_extractor import triplet_extractor

load_dotenv()


class neo4j_handler:

    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASS")
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=3600,
            connection_timeout=30,
            max_connection_pool_size=50,
        )

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def insert_triplets_transaction(self, tx, article_id, subject, predicate, obj):
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

    # generate entities from a claim
    def generate_entities(self, claim):
        try:
            llm_extractor = triplet_extractor()
            llm_result = llm_extractor.call_triplet_llm(claim)
            entities = llm_extractor.extract_entities(llm_result)
            return entities
        except Exception as e:
            print(f"Error in entity generation: {e}")
            return []

    # this function traverses the graph knowledge base to find all articles based on an entity
    def retrieve_article_ids_transaction(self, tx, entity_names):
        query = """
        UNWIND $entity_names AS entity_name 
        MATCH (start:Entity {name: entity_name}) 
        CALL apoc.path.expandConfig(start, {
            labelFilter: "+Article|+Entity",
            maxLevel: 3,  
            bfs: true
        }) YIELD path
        WITH path, last(nodes(path)) AS end_node
        WHERE "Article" IN labels(end_node)
        RETURN DISTINCT end_node.id AS reachable_article_id
        """
        result = tx.run(query, entity_names=entity_names)
        return [record["reachable_article_id"] for record in result]

    def retrieve_article_ids(self, claim):
        entity_names = [
            "dishwashing liquid",
            "automatic dish detergents",
            "renewable, plant-based ingredients",
        ]
        print(entity_names)

        if not isinstance(entity_names, list):
            raise TypeError("entity_names must be a list of strings.")
        if not entity_names:
            return [], []

        try:
            with self.driver.session() as session:
                # pass the transaction function and entity_names as parameter
                article_ids = session.execute_read(
                    self.retrieve_article_ids_transaction, entity_names
                )
            return list(set(article_ids)), entity_names
        except Exception as e:
            print(f"An error occurred while fetching articles for entities: {e}")
            return [], entity_names

    def retrieve_articles(self, ids):
        pass
