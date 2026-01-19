#!/usr/bin/env python3
"""
Demo script for end-to-end claim classification.
Extracts KG nodes from a claim, then classifies it using the extracted knowledge graph.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv

from utils.classification_utils import (
    get_company_id,
    retrieve_evidence,
    retrieve_evidence_with_paths,
    classify_few,
    classify_zero,
    classify_few_python,
    classify_zero_python,
    retrieve_evidence_db,
    classify_zero_db,
    classify_few_db,
    rank_justification,
)

load_dotenv()

# Configuration
RATE_LIMIT_DELAY = 1
SCHEMA_PATH = Path("schema.json")
DEFAULT_EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
POS_LABEL = "greenwashing"
NEG_LABEL = "not_greenwashing"
ABSTAIN_LABEL = "abstain"


def load_schema(schema_path: Path) -> str:
    """Load and minify the KG schema JSON."""
    try:
        schema_dict = json.loads(schema_path.read_text())
        return json.dumps(schema_dict, separators=(",", ":"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema file not found at {schema_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading schema: {e}")


def build_extraction_prompt(company: str, claim: str, schema_str: str) -> str:
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


def extract_llm_json(response) -> str:
    """Extract JSON text from Gemini API response."""
    if hasattr(response, "text") and response.text:
        return response.text
    try:
        return response.candidates[0].content.parts[0].text
    except Exception:
        return str(response)


def extract_kg_from_claim(
    company: str,
    claim: str,
    schema_str: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract knowledge graph nodes and edges from a claim using Gemini API.
    
    Args:
        company: Company name
        claim: ESG claim text
        schema_str: Minified JSON schema string
        api_key: Gemini API key (defaults to GEMINI_API_KEY_1 from env)
    
    Returns:
        Dictionary with 'nodes' and 'edges' keys
    
    Raises:
        RuntimeError: If LLM call fails or returns invalid JSON
    """
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY_1")
        if not api_key:
            raise ValueError("No API key provided and GEMINI_API_KEY_1 not found in environment")
    
    client = genai.Client(api_key=api_key)
    
    cfg_json = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        system_instruction="Return *only* valid JSON – no prose.",
    )
    
    prompt = build_extraction_prompt(company, claim, schema_str)
    
    try:
        time.sleep(RATE_LIMIT_DELAY)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=cfg_json,
        )
    except Exception as err:
        raise RuntimeError(f"LLM call failed: {err}")
    
    json_text = extract_llm_json(resp)
    
    try:
        parsed = json.loads(json_text)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from LLM: {e}\n\nRaw response:\n{json_text}")
    
    # Ensure the response has the expected structure
    if 'nodes' not in parsed:
        parsed = {'nodes': [], 'edges': []}
    if 'edges' not in parsed:
        parsed['edges'] = []
    
    return parsed


def node_to_text(node: Dict[str, Any]) -> str:
    """Convert a node to text representation for embedding."""
    class_name = node.get('class', 'Unknown')
    properties = node.get('properties', {})
    text_parts = [f"{class_name}:"]
    
    for key, value in properties.items():
        # Skip temporal metadata for embedding
        if value is None or key in ['valid_from', 'valid_to', 'is_current']:
            continue
        text_parts.append(f"{key}: {value}")
    
    return " | ".join(text_parts)


def add_embeddings_to_nodes(
    kg_data: Dict[str, Any],
    encoder: SentenceTransformer
) -> Dict[str, Any]:
    """
    Add embeddings to all nodes in the KG data.
    
    Args:
        kg_data: Dictionary with 'nodes' and 'edges' keys
        encoder: SentenceTransformer model
    
    Returns:
        Updated kg_data with embeddings added to nodes
    """
    nodes = kg_data.get('nodes', [])
    
    for node in nodes:
        node_text = node_to_text(node)
        embedding = encoder.encode(node_text, normalize_embeddings=True)
        node['embedding'] = embedding.tolist()
    
    return kg_data


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def retrieve_similar_nodes_from_extracted(
    driver,
    company_id: Optional[int],
    extracted_nodes: List[Dict[str, Any]],
    top_k_per_class: int = 10,
    threshold: float = 0.3,
) -> Tuple[str, List[int]]:
    """
    Retrieve similar nodes from Neo4j by comparing with extracted node embeddings.
    Uses company_id as starting point for graph traversal.
    
    Args:
        driver: Neo4j driver
        company_id: Company ID (required for proper graph traversal)
        extracted_nodes: List of extracted nodes with embeddings
        top_k_per_class: Number of top similar nodes to retrieve
        threshold: Minimum similarity threshold
    
    Returns:
        Tuple of (context_string, cited_node_ids)
    """
    if not extracted_nodes:
        return "[]", []
    
    if not company_id:
        print("Warning: No company_id provided, cannot retrieve from graph")
        return "[]", []
    
    # Get embeddings from extracted nodes
    extracted_embeddings = []
    for node in extracted_nodes:
        if 'embedding' in node and node['embedding']:
            extracted_embeddings.append(np.array(node['embedding']))
    
    if not extracted_embeddings:
        return "[]", []
    
    # Average the embeddings to get a single query vector
    query_embedding = np.mean(extracted_embeddings, axis=0)
    
    # Retrieve nodes from Neo4j starting from the company
    # This matches the approach in classify_single_claim from 7-classify.py
    with driver.session() as session:
        # Get nodes connected to the company within 1-2 hops
        query = """
        MATCH (c:Organization)
        WHERE elementId(c) = $company_id
        MATCH (c)-[*1..2]-(n)
        WHERE n.embedding IS NOT NULL
        RETURN DISTINCT elementId(n) as node_id, n, labels(n) as labels
        """
        result = session.run(query, company_id=company_id)
        
        # Calculate similarities
        candidates = []
        for record in result:
            node = record["n"]
            node_id = record["node_id"]
            labels = record["labels"]
            
            if hasattr(node, 'get') and node.get('embedding'):
                node_embedding = np.array(node['embedding'])
                similarity = compute_cosine_similarity(query_embedding.tolist(), node_embedding.tolist())
                
                if similarity >= threshold:
                    candidates.append({
                        'node_id': node_id,
                        'similarity': similarity,
                        'node': dict(node),
                        'labels': labels
                    })
        
        # Sort by similarity and take top-k
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        top_candidates = candidates[:top_k_per_class]
        
        if not top_candidates:
            return "[]", []
        
        # Format as context (similar to the original classification script)
        context_list = []
        cited_ids = []
        
        for candidate in top_candidates:
            node_dict = candidate['node']
            labels = candidate['labels']
            node_id = candidate['node_id']
            
            # Format node properties
            node_str = f"{labels[0] if labels else 'Node'}: "
            props = []
            for key, value in node_dict.items():
                if key != 'embedding' and value is not None:
                    props.append(f"{key}={value}")
            node_str += ", ".join(props)
            
            context_list.append(node_str)
            cited_ids.append(node_id)
        
        context_str = json.dumps(context_list, ensure_ascii=False)
        return context_str, cited_ids


def classify_claim_with_extraction(
    claim: str,
    company: str,
    year: str = "",
    prompt: str = "few",
    evidence: str = "json",
    hops: str = "one",
    rag_type: str = "kg",
    top_k_per_class: int = 10,
    similarity_threshold: float = 0.3,
    schema_path: Path = SCHEMA_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    save_kg: bool = False,
    kg_output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    End-to-end claim classification: extract KG nodes, then classify.
    
    Args:
        claim: The claim text to classify
        company: The company name associated with the claim
        year: Year of the claim (optional)
        prompt: Prompting strategy - 'few' or 'zero'
        evidence: Evidence format - 'json' or 'python'
        hops: Retrieval strategy - 'one' or 'multi' (only for kg and hybrid)
        rag_type: RAG type - 'kg', 'db', or 'hybrid'
        top_k_per_class: Number of most similar nodes to retrieve per class
        similarity_threshold: Minimum similarity threshold for node selection
        schema_path: Path to KG schema JSON file
        embedding_model: SentenceTransformer model name
        save_kg: Whether to save extracted KG to file
        kg_output_path: Path to save extracted KG (if save_kg=True)
    
    Returns:
        Dictionary containing classification results and extracted KG
    """
    print(f"\n{'='*60}")
    print(f"CLAIM CLASSIFICATION WITH KG EXTRACTION")
    print(f"{'='*60}")
    print(f"Company: {company}")
    print(f"Year: {year if year else 'N/A'}")
    print(f"Claim: {claim}")
    print(f"RAG type: {rag_type}")
    print(f"Prompt strategy: {prompt}")
    print(f"Evidence format: {evidence}")
    if rag_type in ["kg", "hybrid"]:
        print(f"Retrieval strategy: {hops}")
    print(f"Top-K per class: {top_k_per_class}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"{'='*60}\n")
    
    # Initialize encoder
    print(f"Loading embedding model: {embedding_model}...")
    encoder = SentenceTransformer(embedding_model)
    
    # Initialize Neo4j connection
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "emeraldmind"))
        driver.verify_connectivity()
        print("Connected to Neo4j")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Neo4j: {e}")
    
    # Step 1: Extract KG from claim
    print(f"\n{'='*60}")
    print("STEP 1: EXTRACTING KNOWLEDGE GRAPH")
    print(f"{'='*60}\n")
    
    print("Loading schema...")
    schema_str = load_schema(schema_path)
    
    print("Extracting knowledge graph from claim...")
    kg_data = extract_kg_from_claim(company, claim, schema_str)
    
    print(f"Extracted {len(kg_data['nodes'])} nodes and {len(kg_data['edges'])} edges")
    
    print("Generating embeddings for nodes...")
    kg_data = add_embeddings_to_nodes(kg_data, encoder)
    
    if save_kg and kg_output_path:
        kg_output_path.parent.mkdir(parents=True, exist_ok=True)
        kg_output_path.write_text(
            json.dumps(kg_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"Saved KG to {kg_output_path}")
    
    # Step 2: Classify using the extracted KG
    print(f"\n{'='*60}")
    print("STEP 2: CLASSIFYING CLAIM")
    print(f"{'='*60}\n")
    
    result = {
        "company": company,
        "claim": claim,
        "year": year,
        "extracted_kg": kg_data,
        "rag_type": rag_type,
    }
    
    try:
        # Get company ID - required for KG-based retrieval
        company_id = None
        if rag_type in ["kg", "hybrid"]:
            company_id = get_company_id(driver, company)
            if company_id:
                print(f"Found company in database: {company_id}")
            else:
                print(f"Warning: Company '{company}' not found in database")
                if rag_type == "kg":
                    # For pure KG mode, we need the company
                    result.update({
                        "context": "Company not found in database",
                        "llm_label": ABSTAIN_LABEL,
                        "llm_type": None,
                        "llm_raw_label": "abstain",
                        "llm_reason": f"Company '{company}' not found in knowledge graph database",
                        "cited_node_ids": [],
                        "abstained": True,
                        "llm_full_response": None,
                    })
                    return result
                # For hybrid, we'll continue with DB only
        
        # Select classification and retrieval functions
        if rag_type == "kg":
            if prompt == "few":
                classify_fn = classify_few_python if evidence == "python" else classify_few
            elif prompt == "zero":
                classify_fn = classify_zero_python if evidence == "python" else classify_zero
            else:
                raise ValueError(f"Invalid prompt type: {prompt}")
            
            if hops == "one":
                retrieve_fn = retrieve_evidence
            elif hops == "multi":
                retrieve_fn = retrieve_evidence_with_paths
            else:
                raise ValueError(f"Invalid hops type: {hops}")
        
        elif rag_type == "db":
            if prompt == "few":
                classify_fn = classify_few_db
            elif prompt == "zero":
                classify_fn = classify_zero_db
            else:
                raise ValueError(f"Invalid prompt type: {prompt}")
        
        elif rag_type == "hybrid":
            if prompt == "few":
                classify_fn_kg = classify_few_python if evidence == "python" else classify_few
                classify_fn_db = classify_few_db
            elif prompt == "zero":
                classify_fn_kg = classify_zero_python if evidence == "python" else classify_zero
                classify_fn_db = classify_zero_db
            else:
                raise ValueError(f"Invalid prompt type: {prompt}")
            
            if hops == "one":
                retrieve_fn_kg = retrieve_evidence
            elif hops == "multi":
                retrieve_fn_kg = retrieve_evidence_with_paths
            else:
                raise ValueError(f"Invalid hops type: {hops}")
        
        # Perform classification based on RAG type
        if rag_type == "kg":
            # Check if we have extracted nodes
            if not kg_data['nodes']:
                print("Warning: No nodes extracted from claim. Cannot use KG-based retrieval.")
                result.update({
                    "context": "No nodes extracted",
                    "llm_label": ABSTAIN_LABEL,
                    "llm_type": None,
                    "llm_raw_label": "abstain",
                    "llm_reason": "No knowledge graph nodes extracted from claim",
                    "cited_node_ids": [],
                    "abstained": True,
                    "llm_full_response": None,
                })
            else:
                print(f"Retrieving similar nodes from database using extracted embeddings...")
                context_str, cited_node_ids = retrieve_similar_nodes_from_extracted(
                    driver,
                    company_id,
                    kg_data['nodes'],
                    top_k_per_class=top_k_per_class,
                    threshold=similarity_threshold,
                )
            
                if not context_str or context_str == "[]":
                    print("Warning: No relevant evidence found")
                    result.update({
                        "context": "No relevant evidence found",
                        "llm_label": ABSTAIN_LABEL,
                        "llm_type": None,
                        "llm_raw_label": "abstain",
                        "llm_reason": "No relevant evidence found in knowledge graph",
                        "cited_node_ids": [],
                        "abstained": True,
                        "llm_full_response": None,
                    })
                else:
                    print(f"Retrieved {len(cited_node_ids)} relevant nodes")
                    print(f"Classifying claim using {prompt}-shot prompting...")
                    
                    llm_label, llm_type, llm_raw_label, llm_reason, raw_response = classify_fn(
                        claim, context_str
                    )
                    
                    # Normalize label
                    pred_label_normalized = None
                    abstained = False
                    
                    if llm_label:
                        llm_label_lower = str(llm_label).strip().lower()
                        if llm_label_lower == "greenwashing":
                            pred_label_normalized = POS_LABEL
                        elif llm_label_lower == "not greenwashing":
                            pred_label_normalized = NEG_LABEL
                        elif llm_label_lower == "abstain":
                            pred_label_normalized = ABSTAIN_LABEL
                            abstained = True
                        else:
                            pred_label_normalized = ABSTAIN_LABEL
                            abstained = True
                    else:
                        pred_label_normalized = ABSTAIN_LABEL
                        abstained = True
                    
                    result.update({
                        "context": context_str,
                        "llm_label": pred_label_normalized,
                        "llm_type": llm_type,
                        "llm_raw_label": llm_raw_label,
                        "llm_reason": llm_reason,
                        "cited_node_ids": cited_node_ids,
                        "abstained": abstained,
                        "llm_full_response": raw_response,
                    })
        
        elif rag_type == "db":
            print(f"Retrieving evidence from database...")
            context_str = retrieve_evidence_db(
                claim,
                company,
                year,
                top_k_per_class
            )
            
            if not context_str or context_str == "[]":
                print("Warning: No relevant evidence found")
                result.update({
                    "context": "No relevant evidence found",
                    "llm_label": ABSTAIN_LABEL,
                    "llm_type": None,
                    "llm_raw_label": "abstain",
                    "llm_reason": "No relevant evidence found in database",
                    "cited_node_ids": [],
                    "abstained": True,
                    "llm_full_response": None,
                })
            else:
                print(f"Retrieved evidence from database")
                print(f"Classifying claim using {prompt}-shot prompting with DB context...")
                
                llm_label, llm_type, llm_raw_label, llm_reason, raw_response = classify_fn(
                    claim, context_str
                )
                
                # Normalize label
                pred_label_normalized = None
                abstained = False
                
                if llm_label:
                    llm_label_lower = str(llm_label).strip().lower()
                    if llm_label_lower == "greenwashing":
                        pred_label_normalized = POS_LABEL
                    elif llm_label_lower == "not greenwashing":
                        pred_label_normalized = NEG_LABEL
                    elif llm_label_lower == "abstain":
                        pred_label_normalized = ABSTAIN_LABEL
                        abstained = True
                    else:
                        pred_label_normalized = ABSTAIN_LABEL
                        abstained = True
                else:
                    pred_label_normalized = ABSTAIN_LABEL
                    abstained = True
                
                result.update({
                    "context": context_str,
                    "llm_label": pred_label_normalized,
                    "llm_type": llm_type,
                    "llm_raw_label": llm_raw_label,
                    "llm_reason": llm_reason,
                    "cited_node_ids": [],
                    "abstained": abstained,
                    "llm_full_response": raw_response,
                })
        
        elif rag_type == "hybrid":
            print(f"Running HYBRID RAG - both KG and DB approaches...")
            
            # KG retrieval using extracted nodes (only if company_id found)
            kg_has_evidence = False
            if kg_data['nodes'] and company_id:
                print(f"\n[KG] Retrieving similar nodes from database using extracted embeddings...")
                context_str_kg, cited_node_ids_kg = retrieve_similar_nodes_from_extracted(
                    driver,
                    company_id,
                    kg_data['nodes'],
                    top_k_per_class=top_k_per_class,
                    threshold=similarity_threshold,
                )
                kg_has_evidence = context_str_kg and context_str_kg != "[]"
            else:
                if not company_id:
                    print(f"\n[KG] Skipping KG retrieval - company not found in database")
                else:
                    print(f"\n[KG] Skipping KG retrieval - no nodes extracted")
                context_str_kg = "[]"
                cited_node_ids_kg = []
            
            # DB retrieval
            print(f"[DB] Retrieving evidence from database...")
            context_str_db = retrieve_evidence_db(
                claim,
                company,
                year,
                top_k_per_class
            )
            db_has_evidence = context_str_db and context_str_db != "[]"
            
            if not kg_has_evidence and not db_has_evidence:
                print("Warning: No relevant evidence found in either KG or DB")
                result.update({
                    "context": "No relevant evidence found",
                    "llm_label": ABSTAIN_LABEL,
                    "llm_type": None,
                    "llm_raw_label": "abstain",
                    "llm_reason": "No relevant evidence found in knowledge graph or database",
                    "cited_node_ids": [],
                    "abstained": True,
                    "llm_full_response": None,
                    "kg_context": context_str_kg,
                    "db_context": context_str_db,
                })
            else:
                # Classify with KG if available
                if kg_has_evidence:
                    print(f"[KG] Classifying claim using {prompt}-shot prompting...")
                    llm_label_kg, llm_type_kg, llm_raw_label_kg, llm_reason_kg, raw_response_kg = classify_fn_kg(
                        claim, context_str_kg
                    )
                    print(f"[KG] Label: {llm_label_kg}")
                else:
                    llm_label_kg, llm_type_kg, llm_raw_label_kg, llm_reason_kg, raw_response_kg = None, None, "no_evidence", "No evidence from KG", ""
                
                # Classify with DB if available
                if db_has_evidence:
                    print(f"[DB] Classifying claim using {prompt}-shot prompting with DB context...")
                    llm_label_db, llm_type_db, llm_raw_label_db, llm_reason_db, raw_response_db = classify_fn_db(
                        claim, context_str_db
                    )
                    print(f"[DB] Label: {llm_label_db}")
                else:
                    llm_label_db, llm_type_db, llm_raw_label_db, llm_reason_db, raw_response_db = None, None, "no_evidence", "No evidence from DB", ""
                
                # Rank justifications
                print(f"\nRanking justifications...")
                best_rag, llm_reason = rank_justification(
                    kg_justification=llm_reason_kg if kg_has_evidence else None,
                    db_justification=llm_reason_db if db_has_evidence else None,
                    claim=claim,
                    kg_label=str(llm_label_kg) if kg_has_evidence else "no_evidence",
                    db_label=str(llm_label_db) if db_has_evidence else "no_evidence",
                )
                
                print(f"Best RAG approach: {best_rag}")
                
                # Use label from best RAG
                if best_rag == "kg":
                    llm_label = llm_label_kg
                    llm_type = llm_type_kg
                    llm_raw_label = llm_raw_label_kg
                    context_str = context_str_kg
                    cited_node_ids = cited_node_ids_kg
                    raw_response = raw_response_kg
                elif best_rag == "db":
                    llm_label = llm_label_db
                    llm_type = llm_type_db
                    llm_raw_label = llm_raw_label_db
                    context_str = context_str_db
                    cited_node_ids = []
                    raw_response = raw_response_db
                else:
                    llm_label = ABSTAIN_LABEL
                    llm_type = None
                    llm_raw_label = "abstain"
                    llm_reason = "Could not rank justifications"
                    context_str = f"KG: {context_str_kg}\n\nDB: {context_str_db}"
                    cited_node_ids = cited_node_ids_kg
                    raw_response = f"KG: {raw_response_kg}\n\nDB: {raw_response_db}"
                
                # Normalize label
                pred_label_normalized = None
                abstained = False
                
                if llm_label:
                    llm_label_lower = str(llm_label).strip().lower()
                    if llm_label_lower == "greenwashing":
                        pred_label_normalized = POS_LABEL
                    elif llm_label_lower == "not greenwashing":
                        pred_label_normalized = NEG_LABEL
                    elif llm_label_lower == "abstain":
                        pred_label_normalized = ABSTAIN_LABEL
                        abstained = True
                    else:
                        pred_label_normalized = ABSTAIN_LABEL
                        abstained = True
                else:
                    pred_label_normalized = ABSTAIN_LABEL
                    abstained = True
                
                result.update({
                    "context": context_str,
                    "llm_label": pred_label_normalized,
                    "llm_type": llm_type,
                    "llm_raw_label": llm_raw_label,
                    "llm_reason": llm_reason,
                    "cited_node_ids": cited_node_ids,
                    "abstained": abstained,
                    "llm_full_response": raw_response,
                    "kg_label": llm_label_kg,
                    "kg_reason": llm_reason_kg,
                    "db_label": llm_label_db,
                    "db_reason": llm_reason_db,
                    "best_rag": best_rag,
                    "kg_context": context_str_kg,
                    "db_context": context_str_db,
                })
    
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        import traceback
        traceback.print_exc()
        
        result.update({
            "error": str(e),
            "context": "(Error during classification)",
            "llm_label": None,
            "llm_type": None,
            "llm_raw_label": f"error: {str(e)}",
            "llm_reason": f"Error during classification: {str(e)}",
            "cited_node_ids": [],
            "llm_full_response": None,
        })
    
    finally:
        driver.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION RESULT")
    print(f"{'='*60}")
    print(f"Extracted nodes: {len(kg_data['nodes'])}")
    print(f"Extracted edges: {len(kg_data['edges'])}")
    print(f"Label: {result.get('llm_label', 'N/A')}")
    print(f"Type: {result.get('llm_type', 'N/A')}")
    print(f"Abstained: {result.get('abstained', False)}")
    print(f"Reason: {result.get('llm_reason', 'N/A')}")
    if rag_type == "hybrid":
        print(f"Best RAG: {result.get('best_rag', 'N/A')}")
    print(f"{'='*60}\n")
    
    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Demo: Extract KG from claim and classify it',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-c', '--company',
        type=str,
        required=True,
        help='Company name'
    )
    
    parser.add_argument(
        '-t', '--claim',
        type=str,
        required=True,
        help='ESG claim text'
    )
    
    parser.add_argument(
        '-y', '--year',
        type=str,
        default="",
        help='Year of the claim (optional)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        choices=['few', 'zero'],
        default='few',
        help='Prompting strategy: few-shot or zero-shot'
    )
    
    parser.add_argument(
        '--evidence',
        type=str,
        choices=['json', 'python'],
        default='json',
        help='Evidence format: json or python'
    )
    
    parser.add_argument(
        '--hops',
        type=str,
        choices=['one', 'multi'],
        default='one',
        help='Retrieval strategy: single-hop or multi-hop (for kg and hybrid)'
    )
    
    parser.add_argument(
        '--rag-type',
        type=str,
        choices=['kg', 'db', 'hybrid'],
        default='kg',
        help='RAG type: kg (knowledge graph), db (database), or hybrid'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of most similar nodes to retrieve per class'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Minimum similarity threshold for node selection'
    )
    
    parser.add_argument(
        '-s', '--schema',
        type=Path,
        default=SCHEMA_PATH,
        help='Path to graph schema JSON file'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help='SentenceTransformer model name'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output JSON file path (optional)'
    )
    
    parser.add_argument(
        '--save-kg',
        action='store_true',
        help='Save extracted KG to separate file'
    )
    
    parser.add_argument(
        '--kg-output',
        type=Path,
        default=None,
        help='Path to save extracted KG JSON (if --save-kg is used)'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    try:
        result = classify_claim_with_extraction(
            claim=args.claim,
            company=args.company,
            year=args.year,
            prompt=args.prompt,
            evidence=args.evidence,
            hops=args.hops,
            rag_type=args.rag_type,
            top_k_per_class=args.top_k,
            similarity_threshold=args.threshold,
            schema_path=args.schema,
            embedding_model=args.model,
            save_kg=args.save_kg,
            kg_output_path=args.kg_output,
        )
        
        # Save full result if output path specified
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"\nResult saved to {args.output}")
        else:
            # Print JSON to stdout
            print("\n" + "="*60)
            print("FULL RESULT (JSON)")
            print("="*60)
            print(json.dumps(result, indent=2, ensure_ascii=False))
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()