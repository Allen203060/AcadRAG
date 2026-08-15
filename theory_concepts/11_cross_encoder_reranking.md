# Bi-Encoders vs. Cross-Encoders (Reranking)

## 1. Bi-Encoders (Vector Databases)
Standard embedding models (like `bge-small`) are **Bi-Encoders**. 
* **Mechanism:** The Query and the Document are embedded completely independently of each other into vector space.
* **Pros:** Extremely fast. We can pre-calculate millions of document vectors and use indexing to find the closest neighbors in milliseconds.
* **Cons:** Shallow understanding. Because the query and document don't interact during processing, the model only measures general semantic similarity, not logical answering capability.

## 2. Cross-Encoders (BGE-Reranker)
Reranking models (like `bge-reranker-base`) are **Cross-Encoders**.
* **Mechanism:** The Query and the Document are concatenated into a single string (`Query [SEP] Document`) and passed through the Transformer together.
* **Pros:** Deep logical reasoning. The attention mechanism calculates the relationship between the query words and document words simultaneously, resulting in a highly accurate relevance score.
* **Cons:** Extremely computationally expensive. 

## 3. The Two-Stage Retrieval Architecture
To get the best of both worlds, modern RAG pipelines use a Two-Stage system:
1. **Fetch (Bi-Encoder):** Use Milvus to rapidly filter 1,000,000 documents down to the Top 20.
2. **Rerank (Cross-Encoder):** Use the Reranker to deeply analyze those 20 documents and return the absolute Top 5 to the LLM.
