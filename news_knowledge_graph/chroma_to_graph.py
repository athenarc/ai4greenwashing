import sqlite3
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
cursor = None
conn = None
driver = None


# flags for the script
retrieve_sql = False
generate_nodes = False
similar_to_relationship = True


if retrieve_sql:
    # --- SQLite part ---
    db_path = "../reportparse/database_data/news_db/chroma.sqlite3"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM collections WHERE name = ?", ("news_db",))
    collection = cursor.fetchone()

    if not collection:
        print("❌ Collection 'news_db' not found.")
        exit()

    collection_name = collection[0]
    print(f"✅ Found collection 'news_db' with ID: {collection_name}")

    query = """
    SELECT
        e.id as embedding_id,
        MAX(CASE WHEN em.key = 'title' THEN em.string_value END) as title,
        MAX(CASE WHEN em.key = 'author' THEN em.string_value END) as author,
        MAX(CASE WHEN em.key = 'news_site' THEN em.string_value END) as news_site,
        MAX(CASE WHEN em.key = 'url' THEN em.string_value END) as url,
        MAX(CASE WHEN em.key = 'publish_date' THEN em.string_value END) as publish_date,
        MAX(CASE WHEN em.key = 'harvest_date' THEN em.string_value END) as harvest_date,
        MAX(CASE WHEN em.key = 'type_of_news' THEN em.string_value END) as type_of_news,
        MAX(CASE WHEN em.key = 'chunk_id' THEN em.string_value END) as chunk_id,
        s.type as segment_type,
        s.scope as segment_scope,
        s.collection,
        MAX(CASE WHEN em_doc.key = 'chroma:document' THEN em_doc.string_value END) as chunk_text
    FROM embeddings e
    JOIN segments s ON e.segment_id = s.id
    JOIN embedding_metadata em ON em.id = e.id
    LEFT JOIN embedding_metadata em_doc ON em_doc.id = e.id AND em_doc.key = 'chroma:document'
    WHERE s.collection = ?
    GROUP BY e.id, s.type, s.scope, s.collection
    """

    cursor.execute(query, (collection_name,))
    all_rows = cursor.fetchall()

    rows = all_rows

    columns = [
        "embedding_id",
        "title",
        "author",
        "news_site",
        "url",
        "publish_date",
        "harvest_date",
        "type_of_news",
        "chunk_id",
        "segment_type",
        "segment_scope",
        "collection",
        "chunk_text",
    ]

# --- Neo4j part ---

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASS")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Load the sentence transformer model once
model = SentenceTransformer("all-MiniLM-L6-v2")

BATCH_SIZE = 500


def batch_create_nodes(tx, batch):
    query = """
    UNWIND $nodes AS node
    MERGE (c:Chunk {chunk_id: node.chunk_id})
    ON CREATE SET c += node
    """
    tx.run(query, nodes=batch)


def delete_relationships_by_type(rel_type: str, batch_size=10000):
    with driver.session() as session:
        while True:
            result = session.run(
                f"""
                MATCH ()-[r:{rel_type}]->()
                WITH r LIMIT {batch_size}
                DELETE r
                RETURN count(r) AS deleted_count
            """
            )
            deleted = result.single()["deleted_count"]
            print(f"🧹 Deleted {deleted} :{rel_type} relationships...")
            if deleted < batch_size:
                break
    print(f"✅ All :{rel_type} relationships deleted.")


# generate nodes
if generate_nodes:
    with driver.session() as session:
        total_inserted = 0

        for i in range(0, len(rows), BATCH_SIZE):
            batch = []
            for row in rows[i : i + BATCH_SIZE]:
                row_dict = dict(zip(columns, row))
                if not row_dict.get("chunk_id") or not row_dict.get("chunk_text"):
                    continue

                # Compute embedding for chunk_text and convert to list
                embedding_vector = model.encode(row_dict["chunk_text"]).tolist()
                row_dict["embedding"] = embedding_vector

                batch.append(row_dict)

            if batch:
                session.execute_write(batch_create_nodes, batch)
                total_inserted += len(batch)
                print(f"Inserted batch {i // BATCH_SIZE + 1} with {len(batch)} nodes")

    print(f"✅ Created {total_inserted} additional chunk nodes in Neo4j.")


if similar_to_relationship:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from tqdm import tqdm

    def create_similarity_relationships_from_neo4j(top_n=3, batch_size=100):
        with driver.session() as session:
            # Step 1: Retrieve chunk_ids and stored embeddings
            result = session.run(
                """
                MATCH (c:Chunk)
                RETURN c.chunk_id AS chunk_id, c.embedding AS embedding
                """
            )
            data = result.data()

        chunk_ids = [record["chunk_id"] for record in data]
        embeddings = np.array(
            [record["embedding"] for record in data], dtype=np.float64
        )

        # Step 2: Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Step 3: Collect top-N relationships enforcing source_id < target_id
        pairs = []
        for i, source_id in enumerate(chunk_ids):
            sims = list(enumerate(similarity_matrix[i]))
            sims = sorted(sims, key=lambda x: x[1], reverse=True)

            for j, score in sims[1 : top_n + 1]:  # skip self
                target_id = chunk_ids[j]
                # Only create relationship if source_id < target_id
                if source_id < target_id:
                    pairs.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "score": float(score),
                        }
                    )

        # Step 4: Write relationships in batches
        def batch_create_relationships(tx, batch):
            tx.run(
                """
                UNWIND $batch AS rel
                MATCH (a:Chunk {chunk_id: rel.source_id})
                MATCH (b:Chunk {chunk_id: rel.target_id})
                MERGE (a)-[r:SIMILAR_TO]->(b)
                SET r.score = rel.score
                """,
                batch=batch,
            )

        with driver.session() as session:
            for i in tqdm(range(0, len(pairs), batch_size)):
                batch = pairs[i : i + batch_size]
                session.execute_write(batch_create_relationships, batch)
                print(
                    f"✅ Created batch {i // batch_size + 1} of SIMILAR_TO relationships."
                )

    create_similarity_relationships_from_neo4j()


cursor.close() if cursor is not None else print("Cursor was not initialized")
conn.close() if conn is not None else print("Connection was not initialized")
driver.close() if driver is not None else print("Driver was not initialized")
