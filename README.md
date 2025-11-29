# EmeraldGraph: Knowledge Graph Construction for Greenwashing Detection

Domain-specific knowledge graph infrastructure for automated greenwashing detection, supporting the EmeraldMind framework (WWW 2026).

---


## Overview

EmeraldGraph is the domain-specific knowledge graph infrastructure powering the EmeraldMind framework for automated greenwashing detection. It transforms unstructured ESG reports into structured, queryable knowledge that captures company-specific context, temporal dependencies, and verifiable evidence chains.

**Key Features:**
- Domain-specific schema (Company, KPIObservation, Policy, Goal entities)
- Company-centered topology with temporal consistency
- Evidence traceability to source documents
- Schema-driven retrieval for claim verification

**Paper:** "EmeraldMind: A Knowledge Graph–Augmented Framework for Greenwashing Detection"

---


## Installation
```bash
git clone https://github.com/EmeraldMind/emeraldgraph.git
cd emeraldgraph
pip install -r requirements.txt

# Setup Neo4j
docker run -d --name neo4j-emerald \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/emeraldmind \
  neo4j

# Configure environment
cp .env.example .env
# Edit .env

# Extract pre-processed reports
cd data
ls *.gz |xargs -n1 tar -xzf
cd ..
mkdir reports
cp -r data/*/* reports/
```

---

## Pipeline Execution

**Quick Start:** The repository includes a pre-built knowledge graph (`graph_latest.json`). You can **skip directly to Step 5** to load it into Neo4j, then proceed with Steps 6-7 for classification.

To rebuild the graph from scratch, follow Steps 0-4 below.

---

### Step 0 (Optional): Extract Image Descriptions

Only needed if you have raw PDFs without pre-extracted data.
```bash
python 0-get_all_image_descriptions.py
```

Scans `reports/*.pdf`, extracts charts/figures using Gemini Vision, creates `{company}_{year}_images/` folders.

---

### Step 1: Extract KPI Observations
```bash
python 1-kpi-extraction.py \
  --reports-dir reports \
  --kpi_defs kpi_definitions.json
```

Reads `reports/{company}_{year}_text/` and `reports/{company}_{year}_images/`, extracts KPI values, saves to `reports/{company}_{year}_kpi/`.

---

### Step 2: Extract Entity-Relation Triples
```bash
python 2-extract-triple.py \
  --input_dir reports \
  --schema graph_schema.json \
  --output graphs_folder
```

Uses LLM to extract (entity, relation, entity) triples from text/KPI/image data.

---

### Step 3: Validate Triples
```bash
python 3-fix-invalid-triple.py \
  --input_dir graphs_folder \
  --schema graph_schema.json \
```

Validates triples against schema, fixes type mismatches, removes duplicates. Stores all valid triples in `graphs_folder/all_validated_triples.json`  and invalid in `graphs_folder/unfixable_triples.json`.

---

### Step 4: Entity Resolution
```bash
python 4-entity_resolution.py \
  --input_file graphs_folder/all_validated_triples.json \
  --output_file graph_latest.json \
  --schema graph_schema.json \
  --similarity_threshold 0.8
```

Finds and merges duplicate entities using embedding similarity + LLM validation.

---

### Step 5: Load to Neo4j

**Start here if using the pre-built graph.**
```bash
python 5-load_edgelist_graph.py \
  --graph graph_latest.json \
  -- scjhema graph_schema.json \
  --clear
```

Loads graph into Neo4j. Use `--clear` to remove existing data first.

Verify at `http://localhost:7474`:
```cypher
MATCH (n) RETURN count(n);
MATCH ()-[r]->() RETURN count(r);
```

---

### Step 6a: Parse Claims to Nodes
```bash
python 6a-parse_claims_to_nodes.py \
  -i datasets/mixed_dataset.csv \
  -o claims_mixed \
  --claim-col claim \
  --company-col company
```
```bash
python 6a-parse_claims_to_nodes.py \
  -i datasets/small_dataset.csv \
  -o claims_small \
  --claim-col claim \
  --company-col company
```

Grounds claims to graph entities, creates `claim_{i}.json` files.

**Required:** `GEMINI_API_KEY_1` through `GEMINI_API_KEY_8` in `.env` for parallel processing.

---

### Step 6b: Generate Embeddings
```bash
python 6b-generate_embeddings.py \
  --claims_dir claims_mixed \
  --model multi-qa-MiniLM-L6-cos-v1
```
```bash
python 6b-generate_embeddings.py \
  --claims_dir claims_small \
  --model multi-qa-MiniLM-L6-cos-v1
```

Adds embeddings to nodes in `claim_{i}.json` files.

Options:
- `--overwrite`: Regenerate existing embeddings
- `--verify`: Check embeddings without generating

---

### Step 7: Run Classification
```bash
python 7-classify.py \
  --dataset datasets/mixed_dataset.csv \
  --claims_dir claims_mixed \
  --output results/mixed_results.json
```
```bash
python 7-classify.py \
  --dataset datasets/small_dataset.csv \
  --claims_dir claims_small \
  --output results/small_results.json
```

Runs EmeraldMind pipeline: claim grounding → retrieval → classification.

---

## Quick Start Options

### Option A: Use Pre-built Graph (Recommended)
```bash
# Start from Step 5
python 5-load_edgelist_graph.py --graph graph_latest.json --clear
python 6a-parse_claims_to_nodes.py -i datasets/mixed_dataset.csv -o claims_mixed
python 6a-parse_claims_to_nodes.py -i datasets/small_dataset.csv -o claims_small
python 6b-generate_embeddings.py --claims_dir claims_mixed
python 6b-generate_embeddings.py --claims_dir claims_small
python 7-classify.py --dataset datasets/mixed_dataset.csv --claims_dir claims_mixed
python 7-classify.py --dataset datasets/small_dataset.csv --claims_dir claims_small
```

### Option B: Build Graph from Scratch
```bash
# Run all steps (with pre-extracted data)
python 1-kpi-extraction.py
python 2-extract-triple.py
python 3-fix-invalid-triple.py
python 3-merge-graphs.py
python 4-entity_resolution.py
python 5-load_edgelist_graph.py --clear
python 6a-parse_claims_to_nodes.py -i datasets/mixed_dataset.csv -o claims_mixed
python 6a-parse_claims_to_nodes.py -i datasets/small_dataset.csv -o claims_small
python 6b-generate_embeddings.py --claims_dir claims_mixed
python 6b-generate_embeddings.py --claims_dir claims_small
python 7-classify.py --dataset datasets/mixed_dataset.csv --claims_dir claims_mixed
python 7-classify.py --dataset datasets/small_dataset.csv --claims_dir claims_small
```

---

## Environment Variables

Required in `.env`:
```bash
# Gemini API (8 keys for parallel processing in Step 6a)
GEMINI_API_KEY_1=your_key_1
GEMINI_API_KEY_2=your_key_2
GEMINI_API_KEY_3=your_key_3
GEMINI_API_KEY_4=your_key_4
GEMINI_API_KEY_5=your_key_5
GEMINI_API_KEY_6=your_key_6
GEMINI_API_KEY_7=your_key_7
GEMINI_API_KEY_8=your_key_8
HF_TOKEN=your_hf_token
```

---

## Citation
```bibtex
@inproceedings{emeraldmind2026,
  title={EmeraldMind: A Knowledge Graph–Augmented Framework for Greenwashing Detection},
  author={Anonymous},
  booktitle={WWW 2026},
  year={2026}
}
```

---

## License

MIT License