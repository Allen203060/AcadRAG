# Concept 09: Optimizing Knowledge Graph & Vector DB Ingestion Bottlenecks

## 1. Why Knowledge Graph Extraction Takes 2 Hours
In a naive Hybrid GraphRAG pipeline, vector embeddings take 5 seconds, while Knowledge Graph extraction takes **99.9% of the runtime**.
* **Serial LLM Invocations:** In `populate.py`, calling `llm_transformer.convert_to_graph_documents([chunk])` in a synchronous Python `for` loop executes LLM queries sequentially. For 100 chunks taking ~45 seconds each, runtime scales linearly to **~1.25 to 2 hours**.
* **Unconstrained Prompt Complexity:** Without an explicit schema (`allowed_nodes`, `allowed_relationships`), the LLM spends excessive output tokens trying to extract every vague noun and verb, generating massive JSON payloads.

## 2. The 4 Pillars of Production Ingestion Optimization

### Pillar 1: Async Concurrent Execution (`aconvert_to_graph_documents`)
Using `asyncio.gather` with `max_concurrency=4` (or `8`) allows multiple chunks to be processed concurrently, leveraging GPU batching in Ollama / vLLM.
$$\text{Runtime} \approx \frac{\text{Total Chunks} \times \text{Latency per Chunk}}{\text{Concurrency Factor}}$$

### Pillar 2: Schema Constraints (`allowed_nodes` & `allowed_relationships`)
Restricting extraction to domain-specific entities:
* `allowed_nodes = ["Concept", "Architecture", "Method", "Metric", "Formula"]`
* `allowed_relationships = ["USES", "PROPOSES", "EVALUATED_ON", "PART_OF"]`
This shrinks LLM prompt processing overhead and reduces output generation tokens by **>60%**.

### Pillar 3: GPU Device Pinning for Embeddings
Explicitly setting `model_kwargs={'device': 'cuda'}` for `HuggingFaceEmbeddings` processes hundreds of vector chunks in sub-second parallel GPU batches.

### Pillar 4: Chunk-Level Hashing & Caching
Compute MD5/SHA256 hashes of each chunk. Skip LLM graph extraction for chunks that have already been processed in prior runs.
