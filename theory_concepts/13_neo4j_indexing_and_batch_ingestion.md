# Concept 13: Neo4j Cypher Indexing, Batch Ingestion & Entity Caching

## 1. Why Neo4j Graph Ingestion Slows Down (The 8-Minute Problem)

When `LLMGraphTransformer` extracts nodes and relationships, `graph.add_graph_documents()` executes Cypher `MERGE` queries into Neo4j:
```cypher
MERGE (e:__Entity__ {id: $entity_id})
MERGE (source:Document {id: $doc_id})
MERGE (e)-[:MENTIONED_IN]->(source)
```

### The 2 Hidden Bottlenecks:
1. **$O(N^2)$ Unindexed Label Scans:** Without a Cypher Uniqueness Constraint on `__Entity__(id)` and `Document(id)`, Neo4j performs a **full database scan** for every single entity and relationship created. As the graph grows to hundreds of nodes, every new chunk insertion gets slower and slower.
2. **Unbatched Single-Document Transactions:** Writing graph documents individually incurs round-trip Bolt network latency for every Cypher transaction.

## 2. The 4 Sub-Minute Ingestion Optimizations

### Optimization 1: Pre-Creating Neo4j Uniqueness Constraints
Creating explicit schema indexes before ingestion ensures instantaneous $O(1)$ B-Tree lookups during `MERGE` statements:
```cypher
CREATE CONSTRAINT IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
```

### Optimization 2: Batched Graph Database Transactions
Passing `batch_size=25` (or `50`) to `graph.add_graph_documents(..., batch_size=25)` groups multiple graph entities into single atomic Cypher execution transactions.

### Optimization 3: Incremental Chunk MD5 Caching
By hashing each text chunk (`hashlib.md5(chunk.page_content.encode())`), we store extracted graph structures in a local `graph_cache.json`.
* **First Run:** Extracts entities via LLM (~30s - 1 min).
* **Subsequent Runs:** Loads from cache instantly (**0 seconds**).

### Optimization 4: Disabling Extraneous Node Property Extraction
Setting `node_properties=False` in `LLMGraphTransformer` prevents the LLM from generating detailed property dictionaries for every node, reducing output token generation by **>50%**.
