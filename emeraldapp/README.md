# EmeraldMind: A Knowledge Graph–Augmented Framework for Greenwashing Detection


---

## Overview

EmeraldApp is an application designed to utilize the [EmeraldMind framework](https://github.com/athenarc/ai4greenwashing/blob/main/README.md), a knowledge graph–augmented approach for automated greenwashing detection. The EmeraldMind framework integrates a domain-specific ESG knowledge graph with retrieval-augmented generation to verify corporate sustainability claims against verifiable evidence, producing transparent, fact-based justifications and responsibly abstaining when evidence is insufficient.

EmeraldApp classifies claims as greenwashing, not greenwashing, or abstains, and generates evidence-backed explanations grounded in structured corporate ESG data and retrieval methods. The tool supports different LLM pipelines presented below:

**LLM Pipelines:**

- EM-RAG: A pipeline that retrieves evidence from EmeraldDB, a vector database of ESG(Environmental, Social, Governance) report chunks;
- EM-KGRAG: A pipeline that retrieves evidence from EmeraldGraph, a domain-specific knowledge graph;
- EM-HYBRID: A pipeline which combines the other two retrieval pipelines
- EM-NR: A simple LLM call pipeline, without retrieval capabilities

This README provides instructions for installation, environment setup, pipeline execution, and quick-start options.

---

<!--## Citation

If you use any feature presented in EmeraldApp in your research, please cite our paper as follows

```
@inproceedings{kaoukis2026emeraldmind,
  author = {Kaoukis, Georgios and Koufopoulos, Ioannis-Aris and Psaroudaki, Eleni and Karidi, Danae Pla and Pitoura, Evaggelia and Papastefanatos, George and Tsaparas, Panayiotis},
  title = {EmeraldMind: A Knowledge Graph-Augmented Framework for Greenwashing Detection},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year = {2026},
  location = {Dubai, United Arab Emirates},
  publisher = {ACM},
  address = {New York, NY, USA},
  pages = {11}
}
``` -->
## Final directory structure

```
emeraldmind/
├── .streamlit/              # Contains theme configuration settings
├── history/                 # Contains past runs in JSON format
├── utils/
│   ├── __init__.py
│   ├── claim_to_kg.py       # Transforms a natural language claim into a small kg graph
│   ├── companies.yaml       # Contains company-year list as show in the demo
│   ├── company_loader.py    # Loads and parses company data
│   ├── kg_utils.py          # Knowledge graph manipulation utilities
│   └── vectordb.py          # Vector database setup and querying
├── app.py                   # Streamlit application entry point
├── baseline_run.py          # Runs the EM-NR pipeline
├── em_rag_run.py            # Runs the EM-RAG pipeline
├── hybrid_run.py            # Runs the EM-HYBRID pipeline
├── kg_rag_run.py            # Runs the EM-KGRAG pipeline
├── model_loader.py          # Loads inference LLMs for label and justification production
├── classify_claim_demo.py   # Demo script for end-to-end claim classification.
├── parser.py                # A generic document parser for metadata extraction.
├── prompts.py               # File containing the prompts that are inserted to the LLMs
├── visualize_graph.py       # Generates an interactive PyVis HTML string for the evidence subgraph.
├── requirements.txt         # Requirements file
└── schema.json              # Contains the knowledge graph schema in json format.
```

---

## Dependencies installation

```bash
# To run the code, please install the dependencies required: 
pip install -r requirements.txt

# Run streamilt in a localhost environment
streamlit run app.py
```


---

## Installation of EmeraldDB and EmeraldGraph

To run the **EmeraldApp** successfully, you must first load the **EmeraldDB** and the **EmeraldGraph** into the repository.  
For detailed instructions, please refer to [this guide](https://github.com/athenarc/ai4greenwashing/blob/main/README.md#installation) and the steps outlined below.

## Environment Variables

Required in `.env`:

```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key
STREAMLIT_CHROMA_DB_PATH='path_to_your_emerald_db'
```

