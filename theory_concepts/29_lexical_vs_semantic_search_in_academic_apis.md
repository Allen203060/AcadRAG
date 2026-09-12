# Concept 29: Lexical vs. Semantic Search in Academic Repositories

## 1. Lexical Search (PubMed / ArXiv)
- **Mechanism:** Inverted index of tokens (B-trees) scored via term frequency (TF-IDF / BM25).
- **Automatic Term Mapping (ATM):** PubMed augments keyword search with MeSH (Medical Subject Headings) to expand queries into hardcoded medical synonyms.
- **Limitation:** Fails when authors express a concept using novel vocabulary not present in the query or ontology. Multi-word queries without field filters cause term explosion across full-text documents.

## 2. Neural Semantic Search (SPECTER / Milvus)
- **Mechanism:** Text is mapped into continuous high-dimensional vector spaces where geometric distance reflects conceptual meaning.
- **Advantage:** Retrieves papers with identical scientific concepts even if they share zero identical keywords (e.g. "renal clearance" matches "kidney excretion").

## 3. The Two-Tier Academic Funnel
1. **Tier 1 (Fast Lexical Ingestion):** PubMed and ArXiv REST APIs provide high-recall discovery of candidates in milliseconds.
2. **Tier 2 (Neural Filtering & RAG):** Local LLMs score abstract relevance, and Milvus + Neo4j provide dense semantic retrieval and graph-relational reasoning.
