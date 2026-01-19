import torch
import re
import streamlit as st
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_model():
    """Load model once and cache it for the entire session."""
    MODEL_ID = "google/gemma-3-27b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")

    return model, processor


@st.cache_resource
def load_prom():
    # -------------------------------
    # Load Prometheus 7B
    # -------------------------------
    MODEL_ID = "prometheus-eval/prometheus-7b-v2.0"

    print("Loading Prometheus 7B model...", flush=True)
    prom_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if prom_tokenizer.pad_token is None:
        prom_tokenizer.pad_token = prom_tokenizer.eos_token
    prom_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float16
    )
    return prom_model, prom_tokenizer


@st.cache_resource
def call_prom(prompt):
    prom_model, prom_tokenizer = load_prom()
    inputs = prom_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(prom_model.device)
    attention_mask = inputs["attention_mask"].to(prom_model.device)

    with torch.no_grad():
        outputs = prom_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            temperature=0.001,
            do_sample=0.001,
            top_p=1.0,
            top_k=0,
            pad_token_id=prom_tokenizer.eos_token_id,
        )

    # Decode only new tokens
    out_ids = outputs[0][len(input_ids[0]) :]
    response = prom_tokenizer.decode(out_ids, skip_special_tokens=True)

    # Try to extract only the response part (after the prompt)
    if "### Your Response (Output format only, no other text):" in response:
        response = response.split(
            "### Your Response (Output format only, no other text):"
        )[-1].strip()

    return response


@st.cache_resource
def load_encoder():
    """Load Sentence Transformer encoder once and cache it."""
    ENCODER_ID = "sentence-transformers/all-mpnet-base-v2"
    encoder = SentenceTransformer(ENCODER_ID)
    return encoder


def call_llm(prompt):
    """Call the cached LLM with a prompt."""
    model, processor = load_model()

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
        output_ids = model.generate(
            **inputs,
            temperature=1e-5,
            max_new_tokens=512,
        )

    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def extract_label(text):
    match = re.search(r"Label:\s*(greenwashing|not_greenwashing|abstain)", text, re.I)
    return match.group(1).lower() if match else None


def extract_best_label(text):
    match = re.search(
        r"Best Label:\s*(greenwashing|not_greenwashing|abstain)", text, re.I
    )
    return match.group(1).lower() if match else None


def extract_justification(text):
    match = re.search(r"Justification:\s*(.+?)(?:\n\s*\n|$)", text, re.I | re.S)
    return match.group(1).strip().replace("</OUTPUT>", "") if match else None
