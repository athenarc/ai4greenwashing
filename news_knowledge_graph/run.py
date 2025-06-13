# script to run the news graph dataset retrieval technique on the benchmark dataset
# the results are stored on a results.csv dataframe
import pandas as pd
from neo4j_handler import neo4j_handler
from triplet_extractor import triplet_extractor


df = pd.read_csv("../Greenwashing_claims_esg_reports.csv")
df_res = pd.DataFrame(
    columns=["id", "claim", "entity_names", "relevant_articles", "company", "year"]
)
neo4j = neo4j_handler()
triplet_extractor = triplet_extractor()


for _, row in df.iterrows():
    claim = row["Claim"]
    idx = row["id"] if "id" in row else _
    try:
        claim = "We make dishwashing liquid and automatic dish detergents powered by renewable, plant-based ingredients that fight grease with ease and make dish time less of a chore."
        article_ids, entity_names = neo4j.retrieve_article_ids(claim)
        print(
            f"""Article id: {article_ids}
                    entity_names: {entity_names}"""
        )
        # entity_names = neo4j.generate_entities(claim)

        print(claim)

        # new_row = {
        #     "id": idx,
        #     "claim": claim,
        #     "entity_names": entity_names,
        #     "relevant_articles": None,
        #     "company": row["Company"],
        #     "year": row["Year"],
        # }

        # print(new_row)
        # df_res = pd.concat([df_res, pd.DataFrame([new_row])], ignore_index=True)
        # df_res.to_csv("results.csv")
    except Exception as e:
        print(e)
        continue
    break
# Append to df_res


# # df_res.to_csv('results.csv')


# from neo4j import GraphDatabase

# uri = "bolt://localhost:7687"
# driver = GraphDatabase.driver(uri, auth=("neo4j", "johnkouf1999"))

# with driver.session() as session:
#     result = session.run("RETURN 1 AS test")
#     print(result.single()["test"])


# import os
# from neo4j import GraphDatabase
# from dotenv import load_dotenv

# load_dotenv()

# uri = os.getenv("NEO4J_URI")
# user = os.getenv("NEO4J_USER")
# password = os.getenv("NEO4J_PASS")

# driver = GraphDatabase.driver(uri, auth=(user, password))


# def test_query():
#     query = """
#     MATCH (article:Article)-[:CONTEXT_OF]->(entity:Entity)
#     RETURN article.id AS article_id, collect(entity.name) AS entities
#     LIMIT 5
#     """
#     with driver.session() as session:
#         result = session.run(query)
#         for record in result:
#             print(f"Article ID: {record['article_id']}")
#             print(f"Entities: {record['entities']}")
#             print("-" * 30)


# if __name__ == "__main__":
#     print("Testing Neo4j connection and query...")
#     test_query()
#     print("Test completed.")
