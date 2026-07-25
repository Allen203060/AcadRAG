# 03. Embedding & Graph Extraction

## Vector Databases (Milvus)
*   **Purpose:** To store the dense vector representations of our text chunks and perform blazing-fast similarity searches.
*   **How it works:** Sentences with similar meanings are located physically closer to each other in high-dimensional vector space. We use `bge-small-en-v1.5` to calculate these coordinates.

## Knowledge Graphs (Neo4j)
*   **Purpose:** To provide structural, multi-hop reasoning capabilities that pure vector search lacks. 
*   **Extraction (LLMGraphTransformer):** We use a local LLM to read every chunk and extract:
    1.  **Nodes:** Key entities (People, Concepts, Organizations).
    2.  **Edges:** The relationships between those entities.
*   **The Graph:** By pushing these Nodes and Edges into Neo4j, overlapping entities merge. This allows the AI to traverse relationships across the entire document corpus, mapping out complex citations or conceptual lineages.
