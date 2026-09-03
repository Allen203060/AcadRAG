# AcadRAG - Project Soul

## 1. Core Architecture
This project is an entirely local Hybrid GraphRAG pipeline designed to maximize context quality using Knowledge Graphs and Vector Retrieval. The core stack includes:
*   **LLM & Embeddings:** Local execution via Ollama.
*   **Vector Database:** Milvus for semantic similarity search.
*   **Graph Database:** Neo4j for Knowledge Graph storage (entities and relationships).
*   **Reranker:** BGE Reranker (HuggingFace) to filter top results.
*   **Pipeline Orchestration:** LangChain, LangGraph, and LangSmith (strictly referenced via the MCP server).
*   **Knowledge Base:** A `/theory_concepts` directory to store educational notes on major RAG/LangChain paradigms.

## 2. Current Progress
*   [x] Designed the initial high-level system architecture.
*   [x] Defined the core operating directives (`ANTIGRAVITY.md`) prioritizing strict manual handoff and tutor mode.
*   [x] Updated `ANTIGRAVITY.md` to mandate LangChain/LangGraph usage and pre-implementation concept teaching.
*   [x] Initialized `SOUL.md`.
*   [x] **Phase 1 (Complete):** Environment Setup (Docker + Python dependencies).
*   [x] **Phase 2 (Complete):** Ingestion & Semantic Chunking.
*   [x] **Phase 3 (Complete):** Embedding & Database Population.
*   [x] **Phase 4 (Complete):** Hybrid Retrieval & Reranking.
*   [x] **Phase 5 (Complete):** Generation & Citations.
*   [x] **Phase 6 (Complete):** Testing & Evaluation (LangSmith & Local Benchmarking).
*   [x] **Phase 7 (Complete):** Advanced PDF & Layout-Aware OCR Ingestion (Unlimited-OCR GGUF + llama.cpp).
*   [x] **Phase 8 (Complete):** Docling DOM Ingestion & Header-Aware Hierarchical Chunking.
*   [x] **Phase 9 (Complete):** Pipeline Benchmarking & Retrieval Evaluation.
*   [x] **Phase 10 (Complete):** LangSmith Integration, Dataset Management & Custom Evaluation Suite.
*   [x] **Phase 11 (Complete):** Async Parallel Ingestion & Production Optimization (Vector & Knowledge Graph Speedup).
*   [x] **Phase 12 (Complete):** Cloud LLM API Providers & High-Throughput LPU Acceleration (Groq, OpenRouter, Nemotron).
*   [x] **Phase 13 (Complete):** Modular LLM Factory Pattern & Zero-Code Provider Switching (Ollama vs. Cloud APIs via `.env`).
*   [x] **Phase 16 (Complete):** Unified CLI Pipeline Orchestrator (`main.py`), `README.md`, and `LICENSE` (MIT License).
*   [x] **Phase 17 (Complete):** Enterprise `src/` Package Restructuring (Clean Package Separation of `src.ingestion`, `src.core`, `src.agents`, and `src.evaluation`).
*   [x] **Phase 18 (Complete):** Graft Repository Indexing & Test Suite Consolidation (43 symbol nodes indexed, BioASQ removed).
*   [ ] **Phase 19 (In Progress):** Autonomous Two-Tier ArXiv Agent as a Stateful LangGraph StateMachine (`src/agents/arxiv_agent.py`).

## 3. Next Logical Steps
*   **Phase 19 Implementation:** Refactor `src/agents/arxiv_agent.py` into a 4-node `StateGraph` state machine (`search_arxiv_node` -> `score_abstracts_node` -> `download_ingest_node` -> `synthesize_node`), update tests in `src/evaluation/test_arxiv_agent.py`, and run live queries.

## 4. Key Decisions
*   **Graft Codebase Graph Navigation:** Integrated Graft AST indexer ([TrailHQ/Graft](https://github.com/trailhq/Graft)). All codebase queries, caller analysis, and symbol lookups will use `graft ask`, `graft callers`, `graft skeleton`, and `graft grep` before reading full files, cutting token overhead by ~96%.
*   **Strict Manual Control:** AI is barred from autonomously writing or executing code (outside of updating this `SOUL.md` file) to ensure the developer maintains full comprehension of the RAG system's intricacies.
