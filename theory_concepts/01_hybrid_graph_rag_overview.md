# 01. Hybrid GraphRAG & LangGraph Overview

## Core Concepts
*   **Vector RAG (Milvus):** Fast, semantic similarity search. Excellent for finding specific paragraphs that match the meaning of a query, but fails at multi-hop reasoning (connecting disparate dots).
*   **GraphRAG (Neo4j):** Extracts Entities (nodes) and Relationships (edges) to build a Knowledge Graph. Excels at multi-hop reasoning and mapping complex relationships (like citations or corporate structures).
*   **Hybrid GraphRAG:** The architecture we are building. It runs a vector search AND a graph traversal in parallel, merging the results to provide the LLM with both semantic context and structural relationship data.
*   **Reranking (BGE Reranker):** A crucial step that takes the combined results from Milvus and Neo4j, scores them against the query, and filters out the noise, keeping only the top N chunks.

## Orchestration
*   **LangGraph:** A framework by LangChain for building stateful, multi-actor applications. Instead of linear chains, LangGraph allows us to build RAG pipelines as cyclical graphs (State Machines) with loops, conditionals, and routing, making the AI highly agentic and resilient to bad retrievals.
