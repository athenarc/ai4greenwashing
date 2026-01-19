from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
import re, unicodedata
from rapidfuzz import fuzz
from numpy import dot
from numpy.linalg import norm
import numpy as np
from dotenv import load_dotenv
from google.genai import types
from model_loader import load_encoder

encoder = load_encoder()
load_dotenv()
try:
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "s3cr3tpass"))
    driver.verify_connectivity()
    gds = GraphDataScience(driver)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Neo4j: {e}")

MODEL = "google/gemma-3-27b-it"

# variable used in names
MODEL_NAME="gemma3-27b" 


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

ALWAYS_INCLUDE_CLASSES = ["KPIObservation", "Penalty"]

_CORP_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "ltd",
    "limited",
    "plc",
    "sa",
    "ag",
    "bv",
    "oy",
}


_RX_PUNCT = re.compile(r"[\\W_]+", re.U)

CFG_JSON = types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=0.0,
)



def _clean_name(n: Any) -> str:
    """
    Normalise company names for fuzzy matching.
    Returns '' for None / non‑str inputs.
    """
    if not isinstance(n, str) or not n.strip():
        return ""

    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = _RX_PUNCT.sub(" ", n.lower()).strip()
    tokens = [t for t in n.split() if t not in _CORP_SUFFIXES]
    return " ".join(tokens)


FUZZY_MIN = 65
EMBED_MIN = 0.7


def get_company_id(
    name: str, *, fuzzy_min: int = FUZZY_MIN, embed_min: float = EMBED_MIN
) -> int | None:
    if not name:
        return None

    target_clean = _clean_name(name)
    #print(f"DEBUG: Input '{name}' cleaned to '{repr(target_clean)}'")

    with driver.session() as sess:
        rows = sess.run(
            "MATCH (o:Organization) RETURN id(o) AS id, o.name AS name"
        ).data()

    orgs = []
    for row in rows:
        if row["name"]:
            cleaned_db_name = _clean_name(row["name"])
            orgs.append((row["id"], row["name"], cleaned_db_name))
            # print(
            #     f"DEBUG: DB name '{row['name']}' cleaned to '{repr(cleaned_db_name)}'"
            # )

    for oid, raw, clean in orgs:
        if clean == target_clean:
            print(f"MATCH FOUND (Exact Clean): '{name}' -> '{raw}' (ID: {oid})")
            return oid

    containment_candidates = []
    for oid, raw, clean in orgs:
        if target_clean and clean and target_clean in clean:
            containment_candidates.append((oid, raw, clean))

    if len(containment_candidates) == 1:
        oid, raw, clean = containment_candidates[0]
        print(f"MATCH FOUND (Unique Containment): '{name}' -> '{raw}' (ID: {oid})")
        return oid
    elif len(containment_candidates) > 1:
        best_candidate = None
        best_contain_score = -1

        #print(f"DEBUG: Multiple containment candidates for '{name}':")
        for oid, raw, clean in containment_candidates:
            score = fuzz.token_sort_ratio(target_clean, clean)
            print(f"  - Candidate: '{raw}' (Cleaned: '{repr(clean)}') - Score: {score}")

            if score > best_contain_score:
                best_contain_score = score
                best_candidate = (oid, raw, clean)

        if best_candidate and best_contain_score >= fuzzy_min:
            oid, raw, clean = best_candidate
            print(
                f"MATCH FOUND (Best Containment): '{name}' -> '{raw}' (ID: {oid}) with score {best_contain_score}"
            )
            return oid
        else:
            print(f"DEBUG: No strong containment candidate found, falling back.")

    best_id_fuzzy, best_score_fuzzy = None, 0
    for oid, raw, clean in orgs:
        score = fuzz.token_set_ratio(target_clean, clean)
        if score > best_score_fuzzy:
            best_id_fuzzy, best_score_fuzzy = oid, score

    # print(
    #     f"DEBUG: Best fuzzy match: ID={best_id_fuzzy}, Score={best_score_fuzzy} (threshold={fuzzy_min})"
    # )

    if best_score_fuzzy >= fuzzy_min:
        matched_name = next(raw for oid, raw, clean in orgs if oid == best_id_fuzzy)
        print(f"MATCH FOUND (Fuzzy): '{name}' -> '{matched_name}' (ID: {best_id_fuzzy})")
        return best_id_fuzzy

    target_emb = encoder.encode(name, normalize_embeddings=True)
    best_id_emb, best_sim = None, 0.0

    for oid, raw, _ in orgs:
        db_emb = encoder.encode(raw, normalize_embeddings=True)
        sim = dot(target_emb, db_emb) / (norm(target_emb) * norm(db_emb))
        if sim > best_sim:
            best_id_emb, best_sim = oid, sim

    # print(
    #     f"DEBUG: Best embedding match: ID={best_id_emb}, Sim={best_sim:.3f} (threshold={embed_min})"
    # )

    if best_sim >= embed_min:
        matched_name = next(raw for oid, raw, _ in orgs if oid == best_id_emb)
        print(f"MATCH FOUND (Embedding): '{name}' -> '{matched_name}' (ID: {best_id_emb})")
        return best_id_emb

    print(f"NO MATCH for '{name}'")
    return None

def format_entities_as_json(entities: List[Dict[str, Any]], company_name: str) -> str:
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    def get_node_display_name(node_type: str, properties: Dict[str, Any]) -> str:
        for k, v in sorted(properties.items()):
            if k not in ['embedding', 'id', 'valid_from', 'valid_to', 'is_current'] and v is not None:
                str_v = str(v)
                return f"{node_type}({str_v})"
        return node_type
    
    def format_path(path_data: Dict[str, Any]) -> Dict[str, Any]:
        path_nodes = path_data.get('path_nodes', [])
        rel_types = path_data.get('rel_types', [])
        path_length = path_data.get('path_length', 0)
        formatted_nodes = []
        for node_data in path_nodes:
            node_props = node_data.get('properties', {})
            node_labels = node_data.get('labels', [])
            filtered_props = {}
            for k, v in sorted(node_props.items()):
                if k not in ['embedding', 'id', 'valid_from', 'valid_to', 'is_current'] and v is not None:
                    filtered_props[k] = convert_to_serializable(v)
            
            formatted_nodes.append({
                "type": node_labels[0] if node_labels else "Node",
                "properties": filtered_props
            })
        path_summary_parts = []
        for i, node in enumerate(formatted_nodes):
            node_type = node["type"]
            props = node["properties"]
            display_name = get_node_display_name(node_type, props)
            path_summary_parts.append(display_name)
            
            if i < len(rel_types):
                path_summary_parts.append(f"-[{rel_types[i]}]->")
        return {
            "summary": " ".join(path_summary_parts),
            "hops": path_length,
            "relationships": rel_types,
            "nodes": formatted_nodes
        }
    
    def format_direct_connection(entity_label: str, entity_props: Dict[str, Any], 
                                 rel_type: str, company_name: str) -> Dict[str, Any]:
        filtered_entity_props = {}
        for k, v in sorted(entity_props.items()):
            if k not in ['embedding', 'id', 'valid_from', 'valid_to', 'is_current'] and v is not None:
                filtered_entity_props[k] = convert_to_serializable(v)
        entity_display = get_node_display_name(entity_label, filtered_entity_props)
        summary = f"Organization({company_name}) -[{rel_type}]-> {entity_display}"
        
        return {
            "summary": summary,
            "hops": 1,
            "relationships": [rel_type],
            "nodes": [
                {
                    "type": "Organization",
                    "properties": {"name": company_name}
                },
                {
                    "type": entity_label,
                    "properties": filtered_entity_props
                }
            ]
        }
    entities_by_type = {}
    for entity in entities:
        labels = entity.get("labels", [])
        label = labels[0] if labels else "Unknown"
        
        if label not in entities_by_type:
            entities_by_type[label] = []
        
        props = entity.get("properties", {})
        similarity = entity.get("similarity", 0.0)
        paths = entity.get("paths", [])  # May be empty or missing
        rel_type = entity.get("rel_type", None)  # For direct connections
        filtered_props = {}
        for k, v in sorted(props.items()):
            if k not in ['embedding', 'id', 'valid_from', 'valid_to', 'is_current'] and v is not None:
                filtered_props[k] = convert_to_serializable(v)
        if paths:
            shortest_path = min(paths, key=lambda p: p.get('path_length', 999), default=None)
            connection = format_path(shortest_path) if shortest_path else None
        elif rel_type is not None:
            connection = format_direct_connection(label, filtered_props, rel_type, company_name)
        else:
            connection = None
        entity_obj = {
            "properties": filtered_props,
            "connection": connection,
        }
        
        entities_by_type[label].append(entity_obj)
    for label in entities_by_type:
        entities_by_type[label].sort(key=lambda x: str(x['properties']))
    formatted_data = {
        "company": company_name,
        "evidence_summary": {
            "total_entities": len(entities),
            "entity_types": {label: len(ents) for label, ents in entities_by_type.items()}
        },
        "evidence_by_type": entities_by_type
    }
    
    return json.dumps(formatted_data, indent=2, ensure_ascii=False)

def load_claim_nodes(claim_idx: int, claims_dir: str = "big_dataset") -> Optional[List[Dict[str, Any]]]:
    claim_file = Path(claims_dir) / f"claim_{claim_idx}.json"
    if not claim_file.exists():
        print(f"Warning: Claim file not found: {claim_file}")
        return None
    try:
        with open(claim_file, 'r') as f:
            data = json.load(f)
        nodes = data.get('nodes', [])
        nodes_with_embeddings = [n for n in nodes if 'embedding' in n]
        if len(nodes_with_embeddings) < len(nodes):
            print(f"Warning: Only {len(nodes_with_embeddings)}/{len(nodes)} nodes have embeddings in {claim_file.name}")
        
        return nodes_with_embeddings
    except Exception as e:
        print(f"Error loading {claim_file}: {e}")
        return None

# def retrieve_evidence_with_paths(
#     company_id: int,
#     claim: str,
#     claim_idx: int = 0,
#     top_k_per_class: int = 10,
#     similarity_threshold: float = 0.3,
#     max_hops: int = 3,
#     embeddings_dir: str = "big_dataset"
# ) -> List[Dict[str, Any]]:
#     print(f"\n=== Retrieving evidence (k={max_hops} hops) for company_id={company_id}, claim_idx={claim_idx} ===")
#     json_nodes = load_claim_nodes(claim_idx, embeddings_dir)
    
#     if not json_nodes:
#         print(f"No nodes found in claim_{claim_idx}.json")
#         json_nodes = []
#     else:
#         print(f"Loaded {len(json_nodes)} nodes with embeddings from claim_{claim_idx}.json")
#         json_nodes = sorted(json_nodes, key=lambda n: (
#             n.get('class', 'Unknown'),
#             str(n.get('properties', {}).get('name', '')),
#             str(n.get('properties', {}).get('description', '')),
#             str(n.get('properties', {}))
#         ))
    
#     json_classes = set()
#     evidence_paths = []
    
#     with driver.session() as session:
#         for idx, json_node in enumerate(json_nodes):
#             node_class = json_node.get('class', 'Unknown')
#             if node_class in ALWAYS_INCLUDE_CLASSES:
#                 continue
#             json_classes.add(node_class)
#             json_embedding = np.array(json_node['embedding'], dtype=np.float32)
#             print(f"\n[{idx+1}/{len(json_nodes)}] Processing JSON node of class: {node_class}")
#             query_nodes = f"""
#             MATCH (company:Organization)-[*1..{max_hops}]-(node:{node_class})
#             WHERE id(company) = $company_id
#             RETURN DISTINCT
#                 id(node) as node_id,
#                 node,
#                 labels(node) as labels
#             ORDER BY node_id
#             """
#             try:
#                 result = session.run(query_nodes, company_id=company_id).data()
#                 if not result:
#                     print(f"  No {node_class} nodes found within {max_hops} hops")
#                     continue
                
#                 print(f"  Found {len(result)} unique {node_class} nodes within {max_hops} hops, calculating similarities...")
#                 node_similarities = []
#                 for record in result:
#                     node = record['node']
#                     node_id = record['node_id']
#                     node_dict = dict(node)
#                     sorted_props = sorted(node_dict.items())
#                     node_text_parts = [f"{node_class}:"]
#                     for key, value in sorted_props:
#                         if value is not None and key not in ['valid_from', 'valid_to', 'is_current', 'embedding']:
#                             node_text_parts.append(f"{key}: {value}")
#                     node_text = " | ".join(node_text_parts)
#                     graph_node_embedding = encoder.encode(node_text, normalize_embeddings=True)
#                     similarity = cosine_similarity(json_embedding, graph_node_embedding)
#                     if similarity >= similarity_threshold:
#                         node_similarities.append({
#                             'node_id': node_id,
#                             'node': node,
#                             'labels': record['labels'],
#                             'similarity': similarity
#                         })
#                 node_similarities.sort(key=lambda x: (-x['similarity'], x['node_id']))
#                 top_nodes = node_similarities[:top_k_per_class]
#                 print(f"  Selected {len(top_nodes)} most similar nodes (out of {len(node_similarities)} above threshold)")
#                 for item in top_nodes:
#                     query_paths = f"""
#                     MATCH (company:Organization)
#                     WHERE id(company) = $company_id
#                     MATCH (node:{node_class})
#                     WHERE id(node) = $node_id
#                     MATCH path = shortestPath((company)-[*1..{max_hops}]-(node))
#                     WITH path,
#                          [rel in relationships(path) | type(rel)] as rel_types,
#                          [n in nodes(path) | {{id: id(n), labels: labels(n), properties: properties(n)}}] as path_nodes
#                     RETURN 
#                         path_nodes,
#                         rel_types,
#                         length(path) as path_length
#                     LIMIT 1
#                     """
#                     path_result = session.run(query_paths, company_id=company_id, node_id=item['node_id']).data()
#                     if path_result:
#                         evidence_dict = {
#                             'node_id': item['node_id'],
#                             'labels': sorted(item['labels']),
#                             'properties': dict(item['node']),
#                             'similarity': item['similarity'],
#                             'embedding_source': f'similar to JSON node #{idx+1}',
#                             'paths': [{
#                                 'path_nodes': path_result[0]['path_nodes'],
#                                 'rel_types': path_result[0]['rel_types'],
#                                 'path_length': path_result[0]['path_length']
#                             }]
#                         }
#                         evidence_paths.append(evidence_dict)
#                         node_name = dict(item['node']).get('name', 
#                                        dict(item['node']).get('description', 
#                                        dict(item['node']).get('target', 'unnamed')))
#                         print(f"    {node_class}: {str(node_name)[:50]}... "
#                               f"(similarity={item['similarity']:.3f}, "
#                               f"path_length={path_result[0]['path_length']} hops)")
#             except Exception as e:
#                 print(f"  Error querying Neo4j for {node_class}: {e}")
#                 continue
#         missing_always_include = sorted([cls for cls in ALWAYS_INCLUDE_CLASSES 
#                                         if cls not in json_classes])
        
#         if missing_always_include:
#             print(f"\n=== Processing ALWAYS_INCLUDE_CLASSES not in JSON: {missing_always_include} ===")
#             claim_embedding = encoder.encode(claim, normalize_embeddings=True)
#             temp = top_k_per_class
#             for class_name in missing_always_include:
#                 if class_name == "KPIObservation":
#                     top_k_per_class=3
#                 else:
#                     top_k_per_class=temp
#                 print(f"\nProcessing ALWAYS_INCLUDE class: {class_name}")
#                 query_nodes = f"""
#                 MATCH (company:Organization)-[*1..1]-(node:{class_name})
#                 WHERE id(company) = $company_id
#                 RETURN DISTINCT
#                     id(node) as node_id,
#                     node,
#                     labels(node) as labels
#                 ORDER BY node_id
#                 """
#                 try:
#                     result = session.run(query_nodes, company_id=company_id).data()
#                     if not result:
#                         print(f"  No {class_name} nodes found within 1 hops")
#                         continue
                    
#                     print(f"  Found {len(result)} unique {class_name} nodes within 1 hops")
#                     node_similarities = []
#                     for record in result:
#                         node = record['node']
#                         node_id = record['node_id']
#                         node_dict = dict(node)
#                         sorted_props = sorted(node_dict.items())
#                         node_text_parts = [f"{class_name}:"]
#                         for key, value in sorted_props:
#                             if value is not None and key not in ['valid_from', 'valid_to', 'is_current', 'embedding']:
#                                 node_text_parts.append(f"{key}: {value}")
#                         node_text = " | ".join(node_text_parts)
                        
#                         graph_node_embedding = encoder.encode(node_text, normalize_embeddings=True)
#                         similarity = cosine_similarity(claim_embedding, graph_node_embedding)
#                         if similarity >= similarity_threshold:
#                             node_similarities.append({
#                                 'node_id': node_id,
#                                 'node': node,
#                                 'labels': record['labels'],
#                                 'similarity': similarity
#                             })
#                     node_similarities.sort(key=lambda x: (-x['similarity'], x['node_id']))
#                     top_nodes = node_similarities[:top_k_per_class]
#                     print(f"  Selected {len(top_nodes)} most similar to claim (out of {len(node_similarities)} above threshold)")
#                     for item in top_nodes:
#                         query_paths = f"""
#                         MATCH (company:Organization)
#                         WHERE id(company) = $company_id
#                         MATCH (node:{class_name})
#                         WHERE id(node) = $node_id
#                         MATCH path = shortestPath((company)-[*1..1]-(node))
#                         WITH path,
#                              [rel in relationships(path) | type(rel)] as rel_types,
#                              [n in nodes(path) | {{id: id(n), labels: labels(n), properties: properties(n)}}] as path_nodes
#                         RETURN 
#                             path_nodes,
#                             rel_types,
#                             length(path) as path_length
#                         LIMIT 1
#                         """
#                         path_result = session.run(query_paths, company_id=company_id, node_id=item['node_id']).data()
#                         if path_result:
#                             evidence_dict = {
#                                 'node_id': item['node_id'],
#                                 'labels': sorted(item['labels']),
#                                 'properties': dict(item['node']),
#                                 'similarity': item['similarity'],
#                                 'embedding_source': 'similar to claim (ALWAYS_INCLUDE)',
#                                 'paths': [{
#                                     'path_nodes': path_result[0]['path_nodes'],
#                                     'rel_types': path_result[0]['rel_types'],
#                                     'path_length': path_result[0]['path_length']
#                                 }]
#                             }
#                             evidence_paths.append(evidence_dict)
                            
#                             node_name = dict(item['node']).get('name',
#                                            dict(item['node']).get('description',
#                                            dict(item['node']).get('target', 'unnamed')))
#                             print(f"    {class_name}: {str(node_name)[:50]}... "
#                                   f"(similarity={item['similarity']:.3f}, "
#                                   f"path_length={path_result[0]['path_length']} hops)")
                    
#                 except Exception as e:
#                     print(f"  Error querying Neo4j for {class_name}: {e}")
#                     continue
    
#     print(f"\nTotal evidence nodes retrieved: {len(evidence_paths)}")
#     evidence_paths.sort(key=lambda x: (
#         -x['similarity'],
#         x['node_id'] if x['node_id'] is not None else 0,
#         x['labels'][0] if x['labels'] else ''
#     ))
#     top_k_per_class=temp
#     return evidence_paths

def retrieve_evidence(
    company_id: int,
    claim: str,
    generated_nodes: List[Dict[str, Any]],
    top_k_per_class: int = 3,
    similarity_threshold: float = 0.2,
    embeddings_dir: str = "big_dataset"
) -> List[Dict[str, Any]]:
    
    json_nodes = generated_nodes
 
    json_nodes = sorted(json_nodes, key=lambda n: (
            n.get('class', 'Unknown'),
            str(n.get('properties', {}).get('name', '')),
            str(n.get('properties', {}).get('description', '')),
            str(n.get('properties', {}))
        ))
    json_classes = set()
    evidence = []
    with driver.session() as session:
        for idx, json_node in enumerate(json_nodes):
            node_class = json_node.get('class', 'Unknown')
            json_classes.add(node_class)
            json_embedding = np.array(json_node['embedding'], dtype=np.float32)
            print(f"\n[{idx+1}/{len(json_nodes)}] Processing JSON node of class: {node_class}")
            query = f"""
            MATCH (company:Organization)-[r]-(node:{node_class})
            WHERE id(company) = $company_id
            RETURN 
                id(node) as node_id,
                node,
                type(r) as rel_type,
                labels(node) as labels
            ORDER BY node_id
            """
            try:
                result = session.run(query, company_id=company_id).data()
                if not result:
                    print(f"  No {node_class} nodes found in graph connected to company")
                    continue
                print(f"  Found {len(result)} {node_class} nodes in graph, calculating similarities...")
                node_similarities = []
                for graph_node in result:
                    node = graph_node['node']
                    node_id = graph_node['node_id']
                    node_dict = dict(node)
                    sorted_props = sorted(node_dict.items())
                    
                    node_text_parts = [f"{node_class}:"]
                    for key, value in sorted_props:
                        if value is not None and key not in ['valid_from', 'valid_to', 'is_current', 'embedding']:
                            node_text_parts.append(f"{key}: {value}")
                    node_text = " | ".join(node_text_parts)
                    graph_node_embedding = encoder.encode(node_text, normalize_embeddings=True)
                    similarity = cosine_similarity(json_embedding, graph_node_embedding)
                    if similarity >= similarity_threshold:
                        node_similarities.append({
                            'node_id': node_id,
                            'node': node,
                            'labels': graph_node['labels'],
                            'rel_type': graph_node['rel_type'],
                            'similarity': similarity
                        })
                node_similarities.sort(key=lambda x: (-x['similarity'], x['node_id']))
                top_nodes = node_similarities[:top_k_per_class]
                print(f"  Selected {len(top_nodes)} most similar nodes (out of {len(node_similarities)} above threshold)")
                for item in top_nodes:
                    evidence_dict = {
                        'node_id': item['node_id'],
                        'labels': sorted(item['labels']),  # Sort labels for determinism
                        'properties': dict(item['node']),
                        'rel_type': item['rel_type'],
                        'similarity': item['similarity'],
                        'embedding_source': f'similar to JSON node #{idx+1}'
                    }
                    evidence.append(evidence_dict)
                    node_name = dict(item['node']).get('name', dict(item['node']).get('description', dict(item['node']).get('target', 'unnamed')))
                    print(f"    {node_class}: {str(node_name)[:50]}... (similarity={item['similarity']:.3f})")
            except Exception as e:
                print(f"  Error querying Neo4j for {node_class}: {e}")
                continue
        missing_always_include = sorted([cls for cls in ALWAYS_INCLUDE_CLASSES if cls not in json_classes])
        if missing_always_include:
            print(f"\n=== Processing ALWAYS_INCLUDE_CLASSES not in JSON: {missing_always_include} ===")
            claim_embedding = encoder.encode(claim, normalize_embeddings=True)
            temp = top_k_per_class
            for class_name in missing_always_include:
                if class_name == "KPIObservation":
                    top_k_per_class=3
                else:
                    top_k_per_class=temp

                print(f"\nProcessing ALWAYS_INCLUDE class: {class_name}")
                
                query = f"""
                MATCH (company:Organization)-[r]-(node:{class_name})
                WHERE id(company) = $company_id
                RETURN 
                    id(node) as node_id,
                    node,
                    type(r) as rel_type,
                    labels(node) as labels
                ORDER BY node_id
                """
                try:
                    result = session.run(query, company_id=company_id).data()
                    
                    if not result:
                        print(f"  No {class_name} nodes found in graph")
                        continue
                    
                    print(f"  Found {len(result)} {class_name} nodes in graph")
                    node_similarities = []
                    for graph_node in result:
                        node = graph_node['node']
                        node_id = graph_node['node_id']
                        node_dict = dict(node)
                        sorted_props = sorted(node_dict.items())
                        node_text_parts = [f"{class_name}:"]
                        for key, value in sorted_props:
                            if value is not None and key not in ['valid_from', 'valid_to', 'is_current', 'embedding']:
                                node_text_parts.append(f"{key}: {value}")
                        node_text = " | ".join(node_text_parts)
                        graph_node_embedding = encoder.encode(node_text, normalize_embeddings=True)
                        similarity = cosine_similarity(claim_embedding, graph_node_embedding)
                        if similarity >= similarity_threshold:
                            node_similarities.append({
                                'node_id': node_id,
                                'node': node,
                                'labels': graph_node['labels'],
                                'rel_type': graph_node['rel_type'],
                                'similarity': similarity
                            })
                    node_similarities.sort(key=lambda x: (-x['similarity'], x['node_id']))
                    top_nodes = node_similarities[:top_k_per_class]
                    print(f"  Selected {len(top_nodes)} most similar to claim (out of {len(node_similarities)} above threshold)")
                    for item in top_nodes:
                        evidence_dict = {
                            'node_id': item['node_id'],
                            'labels': sorted(item['labels']),  # Sort labels for determinism
                            'properties': dict(item['node']),
                            'rel_type': item['rel_type'],
                            'similarity': item['similarity'],
                            'embedding_source': 'similar to claim (ALWAYS_INCLUDE)'
                        }
                        evidence.append(evidence_dict)
                        node_name = dict(item['node']).get('name', dict(item['node']).get('description', dict(item['node']).get('target', 'unnamed')))
                        print(f"    {class_name}: {str(node_name)[:50]}... (similarity={item['similarity']:.3f})")
                    
                except Exception as e:
                    print(f"  Error querying Neo4j for {class_name}: {e}")
                    continue
    print(f"\nTotal evidence nodes retrieved from Neo4j: {len(evidence)}")
    evidence.sort(key=lambda x: (
        -x['similarity'],
        x['node_id'] if x['node_id'] is not None else 0,
        x['labels'][0] if x['labels'] else ''
    ))
    top_k_per_class = temp
    return evidence


# Ensure you have these defined globally or imported
# ALWAYS_INCLUDE_CLASSES = ["Organization", "Product", "Plan", "KPIObservation"] 

# def retrieve_evidence(
#     company_id: int,
#     claim: str,
#     generated_nodes: List[Dict[str, Any]],  # <--- NEW ARGUMENT
#     top_k_per_class: int = 3,
#     similarity_threshold: float = 0.3,
# ) -> List[Dict[str, Any]]:
    
#     print(f"\n=== Retrieving evidence for company_id={company_id} ===")
    
#     # Use the passed nodes directly instead of loading from file
#     json_nodes = generated_nodes
    
#     if not json_nodes:
#         print("No generated nodes provided.")
#         json_nodes = []
#     else:
#         print(f"Processing {len(json_nodes)} generated nodes with embeddings.")
        
#         # Sort for deterministic processing order
#         json_nodes = sorted(json_nodes, key=lambda n: (
#             n.get('class', 'Unknown'),
#             str(n.get('properties', {}).get('name', '')),
#             str(n.get('properties', {}).get('description', '')),
#             str(n.get('properties', {}))
#         ))

#     json_classes = set()
#     evidence = []
    
#     # We assume 'driver' and 'encoder' are available globally or passed in context
#     with driver.session() as session:
#         for idx, json_node in enumerate(json_nodes):
#             node_class = json_node.get('class', 'Unknown')
#             json_classes.add(node_class)
            
#             # Ensure embedding is a numpy array
#             json_embedding = np.array(json_node['embedding'], dtype=np.float32)
            
#             print(f"\n[{idx+1}/{len(json_nodes)}] Processing JSON node of class: {node_class}")
            
#             query = f"""
#             MATCH (company:Organization)-[r]-(node:{node_class})
#             WHERE id(company) = $company_id
#             RETURN 
#                 id(node) as node_id,
#                 node,
#                 type(r) as rel_type,
#                 labels(node) as labels
#             ORDER BY node_id
#             """
            
#             try:
#                 result = session.run(query, company_id=company_id).data()
                
#                 if not result:
#                     print(f"  No {node_class} nodes found in graph connected to company")
#                     continue
                
#                 print(f"  Found {len(result)} {node_class} nodes in graph, calculating similarities...")
                
#                 node_similarities = []
#                 for graph_node in result:
#                     node = graph_node['node']
#                     node_id = graph_node['node_id']
                    
#                     # Create text representation for embedding
#                     node_dict = dict(node)
#                     sorted_props = sorted(node_dict.items())
#                     node_text_parts = [f"{node_class}:"]
#                     for key, value in sorted_props:
#                         if value is not None and key not in ['valid_from', 'valid_to', 'is_current', 'embedding']:
#                             node_text_parts.append(f"{key}: {value}")
#                     node_text = " | ".join(node_text_parts)
                    
#                     # Encode graph node
#                     graph_node_embedding = encoder.encode(node_text, normalize_embeddings=True)
                    
#                     # Calculate Cosine Similarity
#                     # (Dot product works if both are normalized)
#                     similarity = np.dot(json_embedding, graph_node_embedding)
                    
#                     if similarity >= similarity_threshold:
#                         node_similarities.append({
#                             'node_id': node_id,
#                             'node': node,
#                             'labels': graph_node['labels'],
#                             'rel_type': graph_node['rel_type'],
#                             'similarity': float(similarity)
#                         })
                
#                 node_similarities.sort(key=lambda x: (-x['similarity'], x['node_id']))
#                 top_nodes = node_similarities[:top_k_per_class]
                
#                 print(f"  Selected {len(top_nodes)} most similar nodes")
                
#                 for item in top_nodes:
#                     evidence_dict = {
#                         'node_id': item['node_id'],
#                         'labels': sorted(item['labels']),
#                         'properties': dict(item['node']),
#                         'rel_type': item['rel_type'],
#                         'similarity': item['similarity'],
#                         'embedding_source': f'similar to extracted {node_class}'
#                     }
#                     evidence.append(evidence_dict)
                    
#             except Exception as e:
#                 print(f"  Error querying Neo4j for {node_class}: {e}")
#                 continue

#         # --- ALWAYS_INCLUDE_CLASSES Handling ---
#         # Assuming ALWAYS_INCLUDE_CLASSES is defined globally
#         ALWAYS_INCLUDE_CLASSES = ["KPIObservation", "Penalty"] # Example defaults
        
#         missing_always_include = sorted([cls for cls in ALWAYS_INCLUDE_CLASSES if cls not in json_classes])
        
#         if missing_always_include:
#             print(f"\n=== Processing ALWAYS_INCLUDE_CLASSES not in generated nodes: {missing_always_include} ===")
#             claim_embedding = encoder.encode(claim, normalize_embeddings=True)
            
#             original_top_k = top_k_per_class
            
#             for class_name in missing_always_include:
#                 # Custom overrides (optional)
#                 current_top_k = 3 if class_name == "KPIObservation" else original_top_k

#                 print(f"\nProcessing ALWAYS_INCLUDE class: {class_name}")
                
#                 query = f"""
#                 MATCH (company:Organization)-[r]-(node:{class_name})
#                 WHERE id(company) = $company_id
#                 RETURN 
#                     id(node) as node_id,
#                     node,
#                     type(r) as rel_type,
#                     labels(node) as labels
#                 ORDER BY node_id
#                 """
                
#                 try:
#                     result = session.run(query, company_id=company_id).data()
                    
#                     if not result:
#                         print(f"  No {class_name} nodes found in graph")
#                         continue
                    
#                     node_similarities = []
#                     for graph_node in result:
#                         node = graph_node['node']
                        
#                         # Generate embedding text
#                         node_dict = dict(node)
#                         node_text_parts = [f"{class_name}:"]
#                         for key, value in sorted(node_dict.items()):
#                             if value is not None and key not in ['valid_from', 'valid_to', 'is_current', 'embedding']:
#                                 node_text_parts.append(f"{key}: {value}")
#                         node_text = " | ".join(node_text_parts)
                        
#                         graph_node_embedding = encoder.encode(node_text, normalize_embeddings=True)
#                         similarity = np.dot(claim_embedding, graph_node_embedding)
                        
#                         if similarity >= similarity_threshold:
#                             node_similarities.append({
#                                 'node_id': graph_node['node_id'],
#                                 'node': node,
#                                 'labels': graph_node['labels'],
#                                 'rel_type': graph_node['rel_type'],
#                                 'similarity': float(similarity)
#                             })
                            
#                     node_similarities.sort(key=lambda x: (-x['similarity'], x['node_id']))
#                     top_nodes = node_similarities[:current_top_k]
                    
#                     for item in top_nodes:
#                         evidence.append({
#                             'node_id': item['node_id'],
#                             'labels': sorted(item['labels']),
#                             'properties': dict(item['node']),
#                             'rel_type': item['rel_type'],
#                             'similarity': item['similarity'],
#                             'embedding_source': 'similar to claim (ALWAYS_INCLUDE)'
#                         })
                        
#                 except Exception as e:
#                     print(f"  Error querying Neo4j for {class_name}: {e}")
#                     continue

#     print(f"\nTotal evidence nodes retrieved from Neo4j: {len(evidence)}")
    
#     # Final sort
#     evidence.sort(key=lambda x: (-x['similarity'], x['node_id'] if x['node_id'] else 0))
    
#     return evidence

def _safe_json(text: str) -> Dict[str, Any]:
    def _clean_text(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r'^```[a-z]*\n?', '', s)
            s = re.sub(r'```$', '', s)
        s = re.sub(r'<OUTPUT>\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*</OUTPUT>', '', s, flags=re.IGNORECASE)
        return s.strip()
    def _extract_label(s: str) -> str | None:
        patterns = [
            r'Label:\s*\*{0,2}(greenwashing|not_greenwashing|abstain)\*{0,2}',
            r'Label:\s*\*\*(greenwashing|not_greenwashing|abstain)\*\*',
            r'\*\*(greenwashing|not_greenwashing|abstain)\*\*',
            r'Label:\s*(greenwashing|not_greenwashing|abstain)',
            r'Label:\s*["\']?(greenwashing|not_greenwashing|abstain)["\']?',
        ]
        for pattern in patterns:
            match = re.search(pattern, s, re.IGNORECASE | re.MULTILINE)
            if match:
                label = match.group(1).lower().strip()
                # Normalize variants
                if label in ['greenwashing', 'not_greenwashing', 'abstain']:
                    return label
        return None
    
    def _extract_type(s: str) -> str | None:
        patterns = [
            r'Type:\s*\*{0,2}(Type\s*[1-4]|N/A)\*{0,2}',
            r'Type:\s*\*\*(Type\s*[1-4]|N/A)\*\*',
            r'Type:\s*(Type\s*[1-4]|N/A)',
            r'Type:\s*["\']?(Type\s*[1-4]|N/A)["\']?',
            r'Type:\s*\*{0,2}([1-4])\*{0,2}(?!\d)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, s, re.IGNORECASE | re.MULTILINE)
            if match:
                type_val = match.group(1).strip()
                # Normalize
                if type_val.upper() == 'N/A':
                    return 'N/A'
                # If just a number, add "Type" prefix
                if type_val.isdigit():
                    return f'Type {type_val}'
                # Normalize spacing: "Type1" -> "Type 1"
                type_val = re.sub(r'Type\s*(\d)', r'Type \1', type_val, flags=re.IGNORECASE)
                return type_val
        
        return None
    
    def _extract_justification(s: str) -> str:
        match = re.search(r'Justification:\s*', s, re.IGNORECASE | re.MULTILINE)
        if not match:
            return ""
        start_pos = match.end()
        remaining_text = s[start_pos:]
        end_markers = [
            r'\n\s*Label:',
            r'\n\s*Type:',
            r'</OUTPUT>',
            r'</',  # Any closing tag
        ]
        end_pos = len(remaining_text)
        for marker in end_markers:
            marker_match = re.search(marker, remaining_text, re.IGNORECASE)
            if marker_match:
                end_pos = min(end_pos, marker_match.start())
        justification = remaining_text[:end_pos].strip()
        justification = justification.strip('"\'')
        justification = re.sub(r'[ \t]+', ' ', justification)  # Normalize spaces on same line
        justification = justification.strip()
        return justification if len(justification) > 3 else ""
    
    def _try_structured_parse(s: str) -> Dict[str, Any] | None:
        result = {}
        label = _extract_label(s)
        type_val = _extract_type(s)
        justification = _extract_justification(s)
        if label:
            result['label'] = label
            result['type'] = type_val if type_val else 'N/A'
            result['reasoning'] = justification if justification else ''
            return result
        
        return None
    
    def _try_json_fallback(s: str) -> Dict[str, Any] | None:
        try:
            s = re.sub(r'^```json\s*', '', s)
            s = re.sub(r'^```\s*', '', s)
            s = re.sub(r'```$', '', s)
            
            data = json.loads(s.strip())
            
            if isinstance(data, dict):
                result = {}
                
                # Label
                for key in ['label', 'Label', 'LABEL']:
                    if key in data:
                        result['label'] = str(data[key]).lower()
                        break
                
                # Type
                for key in ['type', 'Type', 'TYPE']:
                    if key in data:
                        result['type'] = str(data[key])
                        break
                
                # Justification
                for key in ['justification', 'Justification', 'JUSTIFICATION', 'reasoning']:
                    if key in data:
                        result['reasoning'] = str(data[key])
                        break
                
                if 'label' in result:
                    result.setdefault('type', 'N/A')
                    result.setdefault('reasoning', '')
                    return result
        
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    if not text or not isinstance(text, str):
        return {
            "label": "error_parsing",
            "type": "N/A",
            "reasoning": "Empty or invalid response text"
        }
    
    original_text = text
    text = _clean_text(text)
    
    # Strategy 1: Structured text parsing (primary)
    result = _try_structured_parse(text)
    if result:
        return result
    
    # Strategy 2: Try with original text
    result = _try_structured_parse(original_text)
    if result:
        return result
    
    # Strategy 3: JSON fallback (in case LLM went rogue)
    result = _try_json_fallback(text)
    if result:
        return result
    
    result = _try_json_fallback(original_text)
    if result:
        return result
    
    # All strategies failed
    return {
        "label": "error_parsing",
        "type": "N/A",
        "reasoning": f"Could not parse LLM response. Raw: {original_text[:300]}"
    }
