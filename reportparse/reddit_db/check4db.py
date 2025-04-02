# Store all posts (or filter as needed)
import pandas as pd
from tqdm import tqdm
from relevance_classifier import is_relevant_llm
from ner_extractor import extract_company
from technique_tagger import extract_techniques
from reddit_chroma_handler import RedditChromaHandler
from transformers import pipeline

df = pd.read_csv("data/checkpoint4.csv")

handler = RedditChromaHandler()

for _, row in tqdm(df.iterrows(), total=len(df)):
    handler.store_post(
        post_id=str(row["post_id"]),
        text=row["text_summary"],
        metadata={
            "content_url": row["content_url"],
            "post_url": row["post_url"],
            "subreddit": row["label"],
            "company": ", ".join(row["company_name"]) if isinstance(row["company_name"], list) else str(row["company_name"]),
        },
    )
print("Posts stored.")