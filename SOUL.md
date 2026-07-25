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
*   [ ] **Phase 7 (In Progress):** Advanced PDF & Layout-Aware OCR Ingestion (Unlimited-OCR GGUF + llama.cpp).

## 3. Next Logical Steps
*   **Phase 7: Advanced PDF & OCR Ingestion:** Download `Unlimited-OCR-Q4_K_M.gguf` and `mmproj-model-f16.gguf` from `sahilchachra/Unlimited-OCR-GGUF`. Compile `llama.cpp` with multimodal support, and update `pdf_loader.py` to render PDF pages into images and process them via GGUF Vision OCR for flawless markdown extraction.

## 4. Key Decisions
*   **Strict Manual Control:** AI is barred from autonomously writing or executing code (outside of updating this `SOUL.md` file) to ensure the developer maintains full comprehension of the RAG system's intricacies.
