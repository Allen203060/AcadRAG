# Concept 30: Query Transformation for Legacy Search Engines (MeSH & Boolean Entrez)

## 1. The Natural Language vs. Boolean Divide
- Neural RAG pipelines ingest conversational natural language queries.
- Legacy search backends (NCBI Entrez, PubMed, Elasticsearch, SQL) expect structured boolean operators (`AND`, `OR`, `NOT`) and field qualifiers (`[Title/Abstract]`, `[MeSH Terms]`).
- Directly forwarding conversational strings causes term scattering and zero-relevance recall across full-text databases.

## 2. The Query Transformation Layer
Before querying legacy backends, an intermediate transformation step executes:
1. **Stopword Elimination:** Strips conversational noise words ("using", "approach", "with", "methods").
2. **Compound Noun Quoting:** Detects multi-word biomedical concepts ("gene editing", "deep learning", "cell therapy") and encloses them in quotes to enforce phrase proximity.
3. **Field Qualification:** Appends `[Title/Abstract]` to restrict lexical matching to high-density document sections rather than 30-page full-text bodies.
4. **Boolean Conjunction:** Bridges required entities with strict `AND` operators and appends open-access filters (`AND open access[filter]`).
