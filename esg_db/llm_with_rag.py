# --This script queries the ESG database for our experiments with Gemma-3-- #
import sys
import os
import re
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from vectordb import ReportParser
from utils.rag_prompts import zero_shot_prompt, few_shot_prompt
import google.generativeai as genai
import tiktoken
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
import transformers

transformers.utils.logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore")


load_dotenv()


MODEL = "27b"
MODEL_ID = f"google/gemma-3-{MODEL}-it"

# Load Gemma-3 model and processor once
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.bfloat16
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
parser = ReportParser(
    reports_folder="../../Greenwashing_claims_esg_reports",
    db_path="chromadb_updated",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
)


def approximate_token_count(text: str) -> int:
    return len(text) // 4


def call_llm(prompt: str):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(**inputs, temperature=1e-5, max_new_tokens=512)

    generated = output_ids[0][input_len:]
    response_text = processor.decode(generated, skip_special_tokens=True)
    return response_text.strip(), MODEL_ID


current_key_index = 0







def extract_label(text: str) -> str:
    match = re.search(
        r"Label:\s*(greenwashing|not_greenwashing|abstain)", text, re.IGNORECASE
    )
    return match.group(1).strip().lower() if match else None


def extract_type(text: str) -> str:
    match = re.search(r"Type:\s*(Type\s*[1-4]|N/?A)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_justification(text: str) -> str:
    match = re.search(
        r"Justification:\s*(.+?)(?:\n\s*\n|$)", text, re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else None


# This function takes the company names from reports stored in the ESG reports folder
# you can change the path to point to your own folder with reports, or modify the function to use a predefined list of company names
def normalize_company_name(name: str) -> str:
    reports_folder = Path("../../Greenwashing_claims_esg_reports")
    companies = {
        "_".join(f.stem.split("_")[:-1])
        for f in reports_folder.glob("*.pdf")
        if len(f.stem.split("_")) >= 2
    }
    if not companies:
        return name

    input_emb = embedding_model.encode([name])[0]
    company_embs = embedding_model.encode(list(companies))
    sims = np.dot(company_embs, input_emb) / (
        np.linalg.norm(company_embs, axis=1) * np.linalg.norm(input_emb)
    )
    best_idx = np.argmax(sims)
    return list(companies)[best_idx]


EXPERIMENTS = [
    ("rag_small", zero_shot_prompt, "zero_shot"),
    ("rag_small", few_shot_prompt, "few_shot"),
    ("rag_mixed", zero_shot_prompt, "zero_shot"),
    ("rag_mixed", few_shot_prompt, "few_shot"),
]

TYPE_INCLUDED = True
DATASETS = {
    "small": "small_dataset",
    "mixed": "mixed_dataset",
}


for prefix, llm_guidelines, mode in EXPERIMENTS:
    dataset_key = "small" if "small" in prefix else "mixed"

    dataset_name = DATASETS[dataset_key]

    save_df = f"rag_{dataset_key}_{mode}.csv"
    SAVE_PATH = "your save path"

    df = pd.read_csv(f"your_data_path.csv")

    if os.path.exists(SAVE_PATH):
        df = pd.read_csv(SAVE_PATH)
        print(f"Resuming from {SAVE_PATH} ({len(df)} rows).")
    else:
        df["llm_model_pred"] = None
        df["llm_response"] = None
        df["predicted_label"] = None
        df["predicted_justification"] = None
        df["full_prompt"] = None
        df["snippets_used"] = None
        if TYPE_INCLUDED:
            df["predicted_type"] = None

    for index, row in df.iterrows():
        if pd.notna(row.get("predicted_label")):
            print(f"Skipping claim {index+1}, already processed.")
            continue

        claim = row["claim"]
        company = row["company"]

        normalized_company = normalize_company_name(company)

        print(
            f"\nProcessing claim {index + 1}/{len(df)}: {claim} (Company: {normalized_company})"
        )

        results = parser.query(claim, n_results=8, company_filter=normalized_company)
        context = [
            f"---Snippet {i+1}---\n{doc}"
            for i, doc in enumerate(results["documents"][0])
        ]

        llm_prompt_base = (
            llm_guidelines
            + "\n\n<CLAIM>\n"
            + claim
            + "\n\n</CLAIM>\n"
            + "\n\n<CONTEXT>\n"
        )
        remaining_snippets = context.copy()

        while remaining_snippets:
            context_text = "\n\n".join(remaining_snippets)
            full_prompt = llm_prompt_base + context_text
            if approximate_token_count(full_prompt) <= 120_000:
                break
            remaining_snippets.pop()

        if not remaining_snippets:
            llm_response = None
            final_context_used = 0
        else:
            context_text = "\n".join(remaining_snippets) + "\n\n</CONTEXT>\n"
            full_prompt = llm_prompt_base + context_text
            llm_response, model_used = call_llm(full_prompt)
            final_context_used = len(remaining_snippets)

        if llm_response:
            df.at[index, "llm_model_pred"] = model_used
            df.at[index, "llm_response"] = llm_response
            df.at[index, "predicted_label"] = extract_label(llm_response)
            df.at[index, "predicted_justification"] = extract_justification(
                llm_response
            )
            df.at[index, "full_prompt"] = full_prompt
            df.at[index, "snippets_used"] = final_context_used
            if TYPE_INCLUDED:
                df.at[index, "predicted_type"] = extract_type(llm_response)
        else:
            for col in [
                "llm_model_pred",
                "llm_response",
                "predicted_label",
                "predicted_justification",
                "full_prompt",
                "snippets_used",
            ]:
                df.at[index, col] = None
            if TYPE_INCLUDED:
                df.at[index, "predicted_type"] = None

        df.to_csv(SAVE_PATH, index=False, escapechar="\\")
        print(f"Progress saved to {SAVE_PATH}")

    print(f"\nFinished experiment: {save_df}")
