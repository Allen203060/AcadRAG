    # 02. Document Loaders & Semantic Chunking

## Document Loaders
*   **Purpose:** Ingest unstructured data (PDFs, TXT, HTML) and convert them into standardized LangChain `Document` objects.
*   **Structure:** Each `Document` has `page_content` (the text) and `metadata` (source, page number). Metadata is the backbone of proper citation generation.

## Chunking Strategies
*   **Why Chunk?** LLMs have strict context limits, and Vector Databases (like Milvus) match search queries to text chunks. If chunks are too large, the retrieved context is noisy. If too small, it lacks meaning.
*   **Recursive Character Chunking:** Breaks text purely by length (e.g., 500 characters). Fast, but frequently severs context by splitting thoughts down the middle.
*   **Semantic Chunking:** Calculates the vector embedding of individual sentences. It groups sentences into a chunk until it detects a significant shift in meaning (a "breakpoint"), preserving entire cohesive thoughts together. This yields far superior context for the LLM during generation.
