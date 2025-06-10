# Step 1: Remove redudant triplets that are duplicates or self-loop

import ast
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


df = pd.read_csv(
    "kg_results_10_per_cent.csv"
)  # the file where my triplets are stored at
df_kg = df[df["article_graph"].apply(lambda x: x != "[]")]


pattern = re.compile(r"\[\d+\]")


def clean_triplets(triplets):
    try:
        # Convert string to list of tuples
        triplets = ast.literal_eval(triplets)
    except Exception as e:
        print(f"Error parsing string: {e}")
        return []
    seen = set()
    cleaned = []
    for triplet in triplets:
        if len(triplet) != 3:
            continue  # ignore malformed triplets
        subj, pred, obj = triplet

        # Skip duplicates
        if triplet in seen:
            continue
        seen.add(triplet)

        # Skip self-loops
        if subj == obj:
            continue

        # Skip if any part contains [number]
        if pattern.search(subj) or pattern.search(pred) or pattern.search(obj):
            continue

        cleaned.append(triplet)
    return cleaned


# Step 2: Gather all triplets in a list
triplets_collection = []
for index, row in df_kg.iterrows():
    triplets = row["article_graph"]
    for triplet in triplets:
        triplets_collection.append(triplet)


# Step 5: create a {article_id1: [(subj1, pred1, obj1), (subj2, pred2, obj2)], article_id2: [...], ...} format

df_kg["article_graph"] = df_kg["article_graph"].apply(clean_triplets)
# df_kg["neo4js_graph"] = df_kg["article_graph"].apply(ast.literal_eval)

# Build the dictionary
triplets_by_article = dict(zip(df_kg["id"], df_kg["article_graph"]))


# # Step 6: Store the graph in a neo4j database
from neo4j import GraphDatabase
import os

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASS")

driver = GraphDatabase.driver(uri, auth=(user, password))


# Delete all existing nodes and relationships
def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


# Create and connect triplets to articles
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


def insert_triplets(triplets_by_article):
    with driver.session() as session:
        for article_id, triplets in triplets_by_article.items():
            for subj, pred, obj in triplets:
                session.execute_write(
                    create_triplet_with_article, article_id, subj, pred, obj
                )


# Clear existing data
# clear_database()

# Build triplets_by_article dictionary (assumes you have this)
# Format: {article_id1: [(subj1, pred1, obj1), (subj2, pred2, obj2)], article_id2: [...], ...}
insert_triplets(triplets_by_article)
