# Emerald: AI-Powered Greenwashing Detection


## Introduction

Corporate sustainability reporting has become ubiquitous, yet distinguishing genuine environmental commitments from misleading claims—known as **greenwashing**—remains a significant challenge. The Emerald project addresses this gap by combining domain-specific knowledge graphs with retrieval-augmented generation to provide transparent, evidence-based verification of ESG (Environmental, Social, Governance) claims.

This repository contains two complementary components:

| Component | Description | Best For |
|-----------|-------------|----------|
| **[EmeraldMind](./emeraldmind/README.md)** | Research framework with full KG construction pipeline | Researchers, developers building custom pipelines |
| **[EmeraldApp](./emeraldapp/README.md)** | Interactive Streamlit application | End users, demonstrations, quick verification |

---

## Key Features

- **EmeraldGraph**: A domain-specific knowledge graph capturing ESG entities, relationships, and KPIs extracted from corporate sustainability reports
- **EmeraldDB**: A vector database of chunked ESG report information, enabling semantic retrieval
- **Multiple Verification Pipelines**: Choose from RAG, KG-RAG, Hybrid, or baseline approaches
- **Transparent Justifications**: Every classification includes evidence-backed explanations
- **Responsible Abstention**: The system abstains when evidence is insufficient rather than guessing

---

## Verification Pipelines

| Pipeline | Method | Use Case |
|----------|--------|----------|
| **EM-RAG** | Retrieves evidence from EmeraldDB (vector similarity search over report chunks) | Fast semantic matching |
| **EM-KGRAG** | Retrieves evidence from EmeraldGraph (structured knowledge graph traversal) | Precise entity and relationship queries |
| **EM-HYBRID** | Combines both retrieval methods | Maximum coverage and accuracy |
| **EM-NR** | Direct LLM classification without retrieval | Baseline comparison |

---

## Quick Start

### Option 1: Run the Application (Recommended for Users)

```bash
cd emeraldapp
pip install -r requirements.txt

# Ensure EmeraldDB and EmeraldGraph are loaded (see EmeraldMind setup)
streamlit run app.py
```

### Option 2: Use the Research Framework

```bash
cd emeraldmind
pip install -r requirements.txt

# Setup Neo4j for the knowledge graph
docker run -d --name neo4j-emerald \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/emeraldmind \
  neo4j

# Load pre-built graph and run classification
python src/EmeraldKG/5-load_edgelist_graph.py --graph graph_latest.json --clear
python src/EmeraldKG/7-classify.py --dataset datasets/emerald_data.csv
```

For complete installation and pipeline instructions, refer to the component-specific READMEs.

---

## Repository Structure

```
emerald/
├── emeraldmind/           # Research framework
│   ├── src/               # Core source code
│   │   ├── EmeraldDB/     # Vector database and RAG implementation
│   │   └── EmeraldKG/     # Knowledge graph pipeline
│   ├── datasets/          # Processed datasets
│   ├── schemas/           # Graph and classification schemas
│   └── llm_judge_scripts/ # Evaluation scripts
│
├── emeraldapp/            # Interactive application
│   ├── utils/             # Helper utilities
│   ├── app.py             # Streamlit entry point
│   └── *_run.py           # Pipeline execution scripts
│
└── README.md              # This file
```

---

## Citation

If you use Emerald in your research, please cite:

```bibtex
@inproceedings{kaoukis2026emeraldmind,
  author    = {Kaoukis, Georgios and Koufopoulos, Ioannis-Aris and Psaroudaki, Eleni 
               and Karidi, Danae Pla and Pitoura, Evaggelia and Papastefanatos, George 
               and Tsaparas, Panayiotis},
  title     = {EmeraldMind: A Knowledge Graph-Augmented Framework for Greenwashing Detection},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  location  = {Dubai, United Arab Emirates},
  publisher = {ACM},
  address   = {New York, NY, USA},
  pages     = {11}
}
```

---

## Documentation

- **[EmeraldMind Documentation](./emeraldmind/README.md)**: Full pipeline details, KG construction, evaluation scripts
- **[EmeraldApp Documentation](./emeraldapp/README.md)**: Application setup and usage guide

