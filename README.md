# AcadRAG: Enterprise Hybrid GraphRAG Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain%20%2F%20LangGraph-green.svg)](https://www.langchain.com/)
[![Milvus](https://img.shields.io/badge/VectorDB-Milvus-blue.svg)](https://milvus.io/)
[![Neo4j](https://img.shields.io/badge/GraphDB-Neo4j-red.svg)](https://neo4j.com/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-orange.svg)](https://smith.langchain.com/)

**AcadRAG** is a production-grade, layout-aware **Hybrid GraphRAG System** designed to perform high-accuracy semantic search, entity relationship retrieval, and grounded Q&A over complex academic PDF documents.

---

## 🌟 Key Features

* **Autonomous Two-Tier ArXiv Research Agent:** Stateful **LangGraph StateMachine** that searches ArXiv for research topics (e.g., *"Face Recognition on IoT Edge"*), performs **LLM-as-a-Judge** abstract semantic scoring, and auto-ingests shortlisted papers into Milvus & Neo4j.
* **3-Stage Human-in-the-Loop (HITL) Guardrails:** Interactive verification checkpoints before downloading PDFs, running layout extraction, and mutating database states.
* **DOM Layout-Aware Parsing (IBM Docling):** Parses complex academic PDFs while preserving multi-column layouts, mathematical LaTeX formulas, and HTML table structures with dynamic CPU/GPU VRAM offloading.
* **Hierarchical Header-Aware Chunking:** Employs a two-pass chunking strategy (`MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`) to preserve section context metadata (`Header 1`, `Header 2`).
* **Dual-Engine Persistence:**
  * **Milvus Vector DB:** CUDA-accelerated dense vector similarity search using `BAAI/bge-small-en-v1.5`.
  * **Neo4j Knowledge Graph:** Schema-constrained entity-relationship extraction (`LLMGraphTransformer`), backed by Cypher uniqueness constraints, batched insertions, and local MD5 chunk hashing (<30s ingestion).
* **Cross-Encoder Reranking:** Filters and re-orders combined vector and graph context using `BAAI/bge-reranker-base`.
* **Zero-Code LLM Factory (`llm_factory.py`):** Seamlessly toggle between local execution (**Ollama Llama 3.1 8B / Qwen 2.5**) and cloud LPUs/APIs (**Google Gemini**, **Groq LPU**, **OpenRouter**, **NVIDIA Nemotron**) via `.env`.
* **Anti-Hallucination Guardrails:** Enforces 100% inline source citations (`[Source X]`) and strict fallback logic (*"I cannot answer this based on the provided documents."*).
* **LangSmith Automated Evaluation:** Built-in LLM-as-a-Judge benchmark suites tracking correctness scores, latency, and execution traces live on LangSmith Cloud.

---

## 🏗️ Architecture Flow Diagram

```
                       ┌─────────────────────────┐
                       │  ArXiv Topic Search     │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Tier 1: LLM Abstract    │
                       │ Reranker (Top 2-3)      │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ 3-Stage HITL Guardrails │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ IBM Docling DOM Parser  │
                       │ (CPU/GPU VRAM Offload)  │
                       └────────────┬────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
     ┌─────────────────────┐                 ┌────────────────────┐
     │  Milvus Vector DB   │                 │  Neo4j Graph DB    │
     │  (Dense BGE Embeds) │                 │  (Entity Graph)    │
     └──────────┬──────────┘                 └─────────┬──────────┘
                │                                      │
                └───────────────────┬──────────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │ BGE Cross-Encoder Rerank│
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Grounded Citation Answer│
                       └─────────────────────────┘
```

---

## ⚡ Prerequisites & Setup

### 1. Requirements
* Linux / macOS
* Python 3.10+
* Docker & Docker Compose (for running Milvus & Neo4j containers)
* Local Ollama (if using local execution) or API Keys (Gemini / Groq / OpenRouter / LangSmith)

### 2. Environment Configuration (`.env`)
Create a `.env` file in the root directory:

```env
# Choose active provider: 'ollama' (default), 'gemini', 'groq', 'openrouter', or 'nemotron'
LLM_PROVIDER="ollama"

# Optional Cloud API Keys
GEMINI_API_KEY="your_gemini_api_key"
GROQ_API_KEY="gsk_your_groq_api_key"
OPENROUTER_API_KEY="sk-or-v1-your_openrouter_api_key"

# LangSmith Observability & Evals
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="AcadRAG-Evaluation"

# Neo4j Graph DB Credentials
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password"
```

### 3. Installation
```bash
git clone https://github.com/Allen203060/AcadRAG.git
cd AcadRAG

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🎮 Unified CLI Usage (`main.py`)

AcadRAG provides a unified CLI orchestrator script `main.py` supporting both an interactive menu and single-command flag executions.

### Interactive CLI Menu
```bash
python main.py
```

### Command-Line Flags
* **Autonomous ArXiv Agent (Research Topic Search):**
  ```bash
  python main.py --arxiv "Face Recognition on IoT Edge"
  ```
* **Ingest & Populate Databases (Milvus + Neo4j):**
  ```bash
  python main.py --populate
  ```
* **Start RAG Query Session (Terminal QA):**
  ```bash
  python main.py --query
  ```
* **Execute LangGraph Stateful Agent:**
  ```bash
  python main.py --graph
  ```
* **Run LangSmith Cloud Evaluation Suite:**
  ```bash
  python main.py --eval
  ```
* **Run Full End-to-End Pipeline:**
  ```bash
  python main.py --all
  ```

---

## 📁 Repository Structure

```
AcadRAG/
├── data/                    # Document store directory (contains .gitkeep)
│   └── .gitkeep
├── theory_concepts/         # 14 Detailed RAG & Graph Architecture Notes
├── src/                     # Enterprise Python Source Package
│   ├── ingestion/           # Document Parsing & Ingestion Engine
│   │   ├── pdf_loader.py    # IBM Docling DOM PDF parser (with VRAM memory check)
│   │   ├── chunker.py       # Header-Aware Hierarchical Text Splitter
│   │   └── populate.py      # Vector (Milvus) & Graph (Neo4j) Ingestion Pipeline
│   ├── core/                # Core RAG Architecture
│   │   ├── llm_factory.py   # Centralized LLM Gateway (Ollama/Gemini/Groq/OpenRouter)
│   │   ├── retriever.py     # Hybrid Search & BGE Reranking Engine
│   │   ├── query.py         # Grounded Citation Generation
│   │   └── graph.py         # Stateful Retrieval Agent Workflow
│   ├── agents/              # Autonomous Agents
│   │   └── arxiv_agent.py   # Two-Tier Autonomous ArXiv Agent with HITL Checkpoints
│   └── evaluation/          # Benchmarking Suite
│       ├── test_benchmark.py# Local Domain Benchmark Suite
│       └── langsmith_eval.py# LangSmith Cloud Evaluation Suite
├── main.py                  # CLI Orchestrator Entrypoint
└── requirements.txt         # Dependencies
```

---

## 📚 Theory Learning Notes Index

The `/theory_concepts/` directory contains in-depth documentation covering all foundational concepts implemented in this repository:

1. `01_environment_and_architecture.md`
2. `02_ingestion_and_semantic_chunking.md`
3. `03_embedding_and_database_population.md`
4. `04_hybrid_retrieval_and_reranking.md`
5. `05_grounded_generation_and_anti_hallucination.md`
6. `06_docling_dom_and_pdf_parsing.md`
7. `07_hierarchical_header_aware_chunking.md`
8. `08_langsmith_evaluation_and_dataset_management.md`
9. `09_graphrag_ingestion_optimization.md`
10. `10_cloud_llm_providers_and_lpu_acceleration.md`
11. `11_llm_factory_and_provider_decoupling.md`
12. `12_api_rate_limits_and_concurrency_tuning.md`
13. `13_neo4j_indexing_and_batch_ingestion.md`
14. `14_cli_orchestration_and_pipeline_entrypoints.md`

---

## 📜 License
MIT License. Built for advanced RAG research and production evaluation.
