# 04. Hybrid Retrieval & Reranking

## Hybrid Retrieval
*   **Concept:** Searching multiple distinct databases to pull both semantic and structural context.
*   **Execution:** A query triggers a similarity search in Milvus (for paragraphs of text) and a Cypher keyword search in Neo4j (for node relationships). The results are merged into an unstructured list.

## Reranking (Cross-Encoders)
*   **The Problem:** The merged list contains too much noise, which degrades LLM generation quality and wastes tokens.
*   **The Solution:** We pass every retrieved item through a Cross-Encoder (like BGE-Reranker). Unlike normal embeddings, Cross-Encoders process the query and the document simultaneously, yielding a highly accurate relevance score.
*   **Result:** We sort the merged context by this Reranker score and truncate the list to the Top N (usually 5) items. The LLM only sees the absolute highest-quality context.


A. Hybrid Retrieval (Dual-Routing) When a user asks a question like "How does RAG reduce hallucination?", a pure vector RAG system just queries Milvus. But we want a richer context.

Vector Search: We convert the query into a vector and ask Milvus for the top 10 chunks that match the meaning.
Graph Traversal: Simultaneously, we extract keywords from the query (e.g., "RAG", "hallucination") and run a Cypher query against Neo4j. We ask Neo4j to return any nodes that match those keywords, along with their immediate neighbors (edges). This provides structural context (e.g., "RAG" -[MITIGATES]-> "Hallucination").

The Merge: We dump all the retrieved text chunks and the graph relationships into a single giant list of potential context.

B. The Problem of Noise & The Cross-Encoder (Reranker) The problem with the list we just created is that it's too big and noisy. If we send all 15+ chunks/relationships to the LLM, it will dilute the LLM's attention, waste tokens, and potentially degrade the answer.

We solve this using a Cross-Encoder Reranker (specifically, BGE-Reranker).

Standard Embeddings (Bi-Encoder): Encodes the query and document separately, then measures the distance. Fast, but less accurate.
Reranker (Cross-Encoder): Feeds the query and the document into the neural network at the exact same time, allowing the model to see how the words interact with each other. It outputs an exact relevance score (0.0 to 1.0). It is computationally expensive but incredibly accurate.
We take our massive list of combined context, pass every item through the Reranker with the query, sort by the score, and keep only the Top 5.