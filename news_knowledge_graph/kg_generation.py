import ast
import re
import pandas as pd
from dotenv import load_dotenv
import os
from neo4j import GraphDatabase

load_dotenv()

df = pd.read_csv("kg_results_10_per_cent.csv")
df_kg = df[df["article_graph"].apply(lambda x: x != "[]")]

pattern = re.compile(r"\[\d+\]")


def clean_triplets(triplets):
    try:
        triplets = ast.literal_eval(triplets)
    except Exception as e:
        print(f"Error parsing string: {e}")
        return []
    seen = set()
    cleaned = []
    for triplet in triplets:
        if len(triplet) != 3:
            continue
        subj, pred, obj = triplet
        if triplet in seen:
            continue
        seen.add(triplet)
        if subj == obj:
            continue
        if pattern.search(subj) or pattern.search(pred) or pattern.search(obj):
            continue
        cleaned.append(triplet)
    return cleaned


# Clean the triplets in dataframe
df_kg["article_graph"] = df_kg["article_graph"].apply(clean_triplets)

# Build the dictionary with cleaned triplets
triplets_by_article = dict(zip(df_kg["id"], df_kg["article_graph"]))

# Neo4j Aura connection info from environment
uri = os.getenv("NEO4J_URI")  # Should be like neo4j+s://abc123.databases.neo4j.io
user = os.getenv("NEO4J_USER")  # Usually "neo4j"
password = os.getenv("NEO4J_PASS")  # Your Aura password

driver = GraphDatabase.driver(uri, auth=(user, password))


def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def create_triplet_with_article(tx, article_id, subject, predicate, obj):
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
        query, article_id=article_id, subject=subject, predicate=predicate, object=obj
    )


def insert_triplets(triplets_by_article, batch_size=100):
    with driver.session() as session:
        batch = []
        for article_id, triplets in triplets_by_article.items():
            for subj, pred, obj in triplets:
                batch.append((article_id, subj, pred, obj))
                if len(batch) >= batch_size:
                    session.execute_write(batch_insert, batch)
                    batch.clear()
        # Insert any remaining
        if batch:
            session.execute_write(batch_insert, batch)


def batch_insert(tx, batch):
    for article_id, subj, pred, obj in batch:
        create_triplet_with_article(tx, article_id, subj, pred, obj)


clear_database()
