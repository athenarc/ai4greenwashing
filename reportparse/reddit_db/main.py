import pandas as pd
from tqdm import tqdm
from relevance_classifier import is_relevant_llm
from ner_extractor import extract_company
from technique_tagger import extract_techniques
from reddit_chroma_handler import RedditChromaHandler
from transformers import pipeline

ner_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Load data
df = pd.read_csv("reportparse/reddit_db/data/reddit_greenwashing_posts.csv")
df["Article_Content"] = df["Article_Content"].fillna("")

# Label posts
tqdm.pandas()
df["is_relevant"] = df["Article_Content"].progress_apply(is_relevant_llm)
print("Relevant posts:", df["is_relevant"].sum())
texts = df["Article_Content"].tolist()
batched_results = ner_pipe(texts, batch_size=16)
df["company_name"] = [extract_company(r) for r in batched_results]

print("Company names extracted.")
df["techniques"] = df["Article_Content"].progress_apply(extract_techniques)
print("Techniques extracted.")
print(df)

handler = RedditChromaHandler()

# Store all posts (or filter as needed)
for _, row in tqdm(df.iterrows(), total=len(df)):
    handler.store_post(
        post_id=str(row["Post_id"]),
        text=row["Article_Content"],
        metadata={
            "url": row["Article_Url"],
            "is_relevant": row["is_relevant"],
            "company": ", ".join(row["company_name"]) if isinstance(row["company_name"], list) else str(row["company_name"]),
            "techniques": ", ".join(row["techniques"]) if isinstance(row["techniques"], list) else str(row["techniques"]),
        },
    )
print("Posts stored.")