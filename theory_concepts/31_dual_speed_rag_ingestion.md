# Concept 31: Dual-Speed RAG Ingestion (Vector vs. Graph Ingestion Asymmetry)

## 1. The Computational Discrepancy
- **Vector Ingestion (O(N) Forward Passes):** Dense embedding models (e.g. `bge-small-en-v1.5`) run parallel matrix multiplications over batches on GPU/CPU. Ingesting 1,000 chunks takes ~2-3 seconds.
- **Graph Ingestion (O(N * Tokens) Autoregressive Generation):** Knowledge graph extraction (`LLMGraphTransformer`) requires a multi-token generative LLM pass per chunk to extract entity nodes and Cypher relationship edges. Ingesting 1,000 chunks can take 30–60 minutes on local consumer hardware.

## 2. Decoupled Dual-Speed Architecture
- **Fast-Path Mode (`vector_only=True`):** Ingests document chunks exclusively into Milvus. Provides sub-second ingestion for rapid literature triage and standard question-answering.
- **Deep-Path Mode (`vector_only=False`):** Executes full Knowledge Graph extraction into Neo4j for multi-hop relational synthesis.
- **Graceful Retrieval Fallback:** The Reciprocal Rank Fusion (RRF) retriever seamlessly fuses `[vector_candidates, []]`, ensuring that when the knowledge graph is skipped, the Cross-Encoder reranks the vector pool with zero runtime disruption.
