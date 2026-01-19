#!/usr/bin/env python3
"""
Extract knowledge graph nodes from a single ESG claim and generate embeddings
using local Gemma 3 (27B) and SentenceTransformers.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib
import torch
from dotenv import load_dotenv
from model_loader import load_model, load_encoder
load_dotenv()
MODEL_NAME = "google/gemma-3-27b-it" 
model, processor=load_model()
encoder=load_encoder()
SCHEMA_PATH = Path("schema.json")

def load_schema(schema_path: Path) -> str:
    try:
        schema_dict = json.loads(schema_path.read_text(encoding="utf-8"))
        return json.dumps(schema_dict, separators=(",", ":"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema file not found at {schema_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading schema: {e}")


def build_prompt(company: str, claim: str, schema_str: str) -> str:
    """Build the LLM prompt for KG extraction."""
    return f"""You are an ESG knowledge-graph assistant.

Given a company and a green claim, extract all possible nodes and relationships that strictly adhere to the following schema.

Schema:
{schema_str}

Input:
company : {company}
claim   : "{claim}"

Do not include any explanatory text or markdown – only the JSON.

Instructions:
- Only use the keys and data types defined in the schema.
- Do not include prose, explanation, or formatting outside the JSON.
- Return valid JSON. If there are multiple nodes, return a JSON array.

#### POSITIVE EXAMPLE ####

nodes:
[
    {{
        "class": "Organization",
        "properties": {{
            "name": "Apple",
            "industry": "electronics",
            "valid_from": "2020-01-01",
            "valid_to": null,
            "is_current": true
        }}
    }},
    {{
        "class": "Product",
        "properties": {{
            "name": "iPhone 12",
            "description": "smartphone with improved energy efficiency",
            "valid_from": "2020-10-23",
            "valid_to": null,
            "is_current": true
        }}
    }}
]

edges:
[
    {{
        "label": "producedBy",
        "source": 1,
        "target": 0,
        "temporal_properties": {{
            "valid_from": "2020-10-23",
            "valid_to": null,
            "recorded_at": "2020-10-23"
        }}
    }}
]

Now generate the JSON for the input above.
"""


def extract_kg_from_claim(
    company: str,
    claim: str,
    schema_str: str
) -> Dict[str, Any]:
    """
    Extract knowledge graph nodes and edges from a claim using local Gemma 3 model.
    """
    user_prompt = build_prompt(company, claim, schema_str)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Return *only* valid JSON – no prose."},
        {"role": "user", "content": user_prompt}
    ]

    try:
        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = processor(
            text=prompt_text,
            return_tensors="pt"
        ).to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=2000,
                do_sample=False,
                temperature=0.0,     
            )
        generated_tokens = generation[0][input_len:]
        json_text = processor.decode(generated_tokens, skip_special_tokens=True)
        
    except Exception as err:
        raise RuntimeError(f"LLM inference failed: {err}")
    
    # Strip Markdown code blocks if present
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0].strip()
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(json_text)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from LLM: {e}\n\nRaw response:\n{json_text}")
    if 'nodes' not in parsed:
        parsed['nodes'] = []
    if 'edges' not in parsed:
        parsed['edges'] = []
    
    return parsed


def node_to_text(node: Dict[str, Any]) -> str:
    class_name = node.get('class', 'Unknown')
    properties = node.get('properties', {})
    text_parts = [f"{class_name}:"]
    for key, value in properties.items():
        if value is None or key in ['valid_from', 'valid_to', 'is_current']:
            continue
        text_parts.append(f"{key}: {value}")
    
    return " | ".join(text_parts)


def add_embeddings_to_nodes(
    kg_data: Dict[str, Any],
) -> Dict[str, Any]:
    nodes = kg_data.get('nodes', [])
    
    for node in nodes:
        node_text = node_to_text(node)
        embedding = encoder.encode(node_text, normalize_embeddings=True)
        node['embedding'] = embedding.tolist()
    
    return kg_data

def get_claim_hash(claim: str) -> str:
    return hashlib.sha256(claim.encode('utf-8')).hexdigest()

def process_claim(
    company: str,
    claim: str,
    schema_path: Path = SCHEMA_PATH,
    output_dir: Optional[Path] = None 
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = Path("kg_claims")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_hash = get_claim_hash(claim)
    output_file = output_dir / f"{claim_hash}.json"
    if output_file.exists():
        try:
            return json.loads(output_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error loading cache file {output_file}: {e}. Rerunning extraction.")
    schema_str = load_schema(schema_path)
    kg_data = extract_kg_from_claim(company, claim, schema_str)
    print(f"Extracted {len(kg_data['nodes'])} nodes and {len(kg_data['edges'])} edges")
    kg_data = add_embeddings_to_nodes(kg_data)
    output_file.write_text(
        json.dumps(kg_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return kg_data