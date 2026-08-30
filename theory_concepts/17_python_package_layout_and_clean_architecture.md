# Concept 17: Enterprise Python `src/` Layout & Clean Architecture

## 1. Why Root Directory Flat Layouts Fail at Scale
In small prototypes, placing all `.py` files in the repository root directory works fine. However, as systems grow to include multiple RAG pipelines, agents, evaluation suites, and API servers, flat root folders create:
* **Root Pollution:** Dozens of unorganized `.py` files sitting alongside configuration files (`README.md`, `.env`, `Dockerfile`).
* **Name Collisions & Circular Imports:** Unstructured imports make it hard to determine module boundaries.
* **Packaging Overhead:** Standard Python tools (PyPI, `pip install -e .`) mandate the **`src/` layout** pattern recommended by the PyPA (Python Packaging Authority).

## 2. Proposed Clean Architecture Layout

```
AcadRAG/
├── data/                    # PDF Storage
├── theory_concepts/         # Documentation Notes
├── src/                     # Python Package Core
│   ├── __init__.py
│   ├── ingestion/           # Data Parsing & Database Population
│   │   ├── __init__.py
│   │   ├── pdf_loader.py    # IBM Docling DOM Parser
│   │   ├── chunker.py       # Header-Aware Splitter (ingest.py)
│   │   └── populate.py      # Milvus & Neo4j Ingestion
│   ├── core/                # Core GraphRAG Architecture
│   │   ├── __init__.py
│   │   ├── llm_factory.py   # Provider-Agnostic LLM Gateway
│   │   ├── retriever.py     # Hybrid Search & BGE Cross-Encoder
│   │   ├── query.py         # Grounded Prompting & Citations
│   │   └── graph.py         # LangGraph State Machine (rag_graph.py)
│   ├── agents/              # Autonomous AI Agents
│   │   ├── __init__.py
│   │   └── arxiv_agent.py   # Two-Tier ArXiv Shortlister
│   └── evaluation/          # Benchmarking Suite
│       ├── __init__.py
│       ├── evaluate.py      # Local BioASQ Evaluator
│       └── langsmith_eval.py# LangSmith Cloud Evaluation
├── main.py                  # Single Entrypoint Script in Root
├── requirements.txt
├── README.md
├── LICENSE
├── SOUL.md
└── CHALLENGES.md
```

## 3. Benefits of `src/` Package Architecture
1. **Clear Module Boundaries:** Separation of concerns between `ingestion`, `core`, `agents`, and `evaluation`.
2. **Explicit Package Imports:** Replaces `from retriever import hybrid_search` with clean package imports `from src.core.retriever import hybrid_search`.
3. **Enterprise Readiness:** Matches production Python standards used by LangChain, FastAPI, and PyTorch repositories.
