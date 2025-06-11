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

        article_ids, entity_names = neo4j.retrieve_article_ids(claim)

        print(claim)

        new_row = {
            "id": idx,
            "claim": claim,
            "entity_names": entity_names,
            "relevant_articles": article_ids,
            "company": row["Company"],
            "year": row["Year"],
        }

        print(new_row)
        df_res = pd.concat([df_res, pd.DataFrame([new_row])], ignore_index=True)
        df_res.to_csv("results.csv")
    except Exception as e:
        print(e)
        continue
    # Append to df_res


# df_res.to_csv('results.csv')
