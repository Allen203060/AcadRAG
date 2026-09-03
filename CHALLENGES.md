# AcadRAG - Engineering Challenges & Debug Journal

This journal tracks significant technical hurdles, debugging workflows, and architectural decisions made during the development of the AcadRAG Hybrid GraphRAG pipeline. It serves as an artifact for technical interviews to demonstrate problem-solving methodologies.

---

## 1. The Cypher Injection Crash (Neo4j)

**The Bug/Challenge:**
During the Hybrid Retrieval phase, querying the Neo4j Knowledge Graph crashed with a `neo4j.exceptions.CypherSyntaxError: Invalid input 'self'`. This occurred specifically when a user queried for terms containing punctuation, such as: *"How does 'Self-Attention' compare to a 'Recurrent' layer?"*

**The Root Cause:**
The retrieval script extracted keywords and dynamically injected them into the Cypher query using a Python f-string:
```cypher
WHERE toLower(n.id) CONTAINS '{word}'
```
When the word was extracted as `'self-attention'` (including the single quotes), the f-string formatted it as `CONTAINS ''self-attention''`. Two adjacent single quotes in Cypher terminate the string literal, causing the database to interpret `self-attention` as an illegal identifier, crashing the transaction. This is a classic injection vulnerability.

**The Solution:**
Refactored the query to use **Parameterized Cypher Queries**. Instead of formatting strings, the query string was static (`CONTAINS $keyword`), and the parameter was passed via the driver wrapper `params={"keyword": word}`. Furthermore, a regex filter (`re.sub(r'[^\w-]', '', w)`) was introduced in the tokenization step to strip dangerous punctuation from the keyword extraction before hitting the database.

---

## 2. Vision OCR Data Loss on Math & Tables (Phase 7 vs. Phase 8)

**The Bug/Challenge:**
When running evaluation benchmarks against academic papers (e.g., *Attention Is All You Need*), the RAG pipeline scored 0/10 on Table Comparison and 3/10 on Complex Math Retrieval. The LLM was severely hallucinating equations and citing non-existent table metrics.

**The Root Cause:**
The ingestion pipeline was relying on a purely visual OCR model (`Unlimited-OCR` GGUF via `llama.cpp`). While good for scanned images, passing digital-born LaTeX PDFs through pixel rasterization caused two catastrophic failures:
1. Complex math formulas (like Feed-Forward networks) were completely dropped, leaving empty `$$ \[ $$` artifacts in the Markdown.
2. Tables were flattened into single, un-delimited strings without `|` pipes, rendering them completely unreadable by semantic embedders and Cross-Encoders.

**The Solution:**
Architected a shift from visual OCR to **DOM (Document Object Model) Parsing**. Replaced the `llama.cpp` vision layer with **IBM Docling**. Docling directly extracts the native binary text stream (guaranteeing 100% math symbol fidelity) and uses an internal specialized model (`TableFormer`) to analyze geometric spacing and perfectly reconstruct invisible table grids into valid Markdown. The evaluation scores subsequently skyrocketed.

---

## 3. Context Dilution via Flat Text Splitters

**The Bug/Challenge:**
When the LLM retrieved a chunk of text, it often missed the structural context. For example, it would retrieve a paragraph about "masking to prevent leftward information flow," but the LLM wouldn't know if this applied to the Encoder or Decoder.

**The Root Cause:**
The original `SemanticChunker` split the document as a flat string. It severed the parent section header (e.g., `## 3.2.2 Decoder Attention`) from its child paragraphs.

**The Solution:**
Implemented a **Two-Pass Hierarchical Chunking** architecture. 
- **Pass 1:** `MarkdownHeaderTextSplitter` scans the file as a state machine, breaking it by `#` and injecting the header lineage into the `metadata` dictionary of the chunk. 
- **Pass 2:** `RecursiveCharacterTextSplitter` sub-splits the oversized sections based on character limits, automatically propagating the parent header metadata down to every child sub-chunk. This ensures every vector in Milvus retains its structural anchor to the original paper.

---

## 4. CUDA Hardware Acceleration & Compiler Collisions

**The Bug/Challenge:**
Attempting to compile `llama.cpp` for GPU hardware acceleration on an NVIDIA RTX 3050 failed with severe `nvcc` compilation errors regarding `noexcept(true)` signature mismatches on math functions (`sinpi`, `cospi`, `rsqrt`).

**The Root Cause:**
A bleeding-edge OS environment (Fedora 43) shipped with `glibc 2.41`, which redefined core C++ math header signatures. The NVIDIA CUDA Toolkit (v12.9) headers (`math_functions.h`) were strictly expecting older standard library definitions, causing the host compiler to abort during CUDA linkage.

**The Solution:**
Manually intervened in the toolchain. Switched the host compiler via CMake to `clang-19` (`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++-19`) which handles modern C++ standards more gracefully. Used `sed` to patch the NVIDIA Toolkit header files to explicitly remove the conflicting `noexcept(true)` constraints, allowing the GPU driver to successfully compile and offload tensor operations to the RTX 3050 VRAM.

---

## 5. Vector Database Schema Collisions (Milvus vs. LangChain)

**The Bug/Challenge:**
Attempting to populate the Milvus Vector Database crashed with a `pymilvus.exceptions.MilvusException: Invalid field name: Header 2`.

**The Root Cause:**
During the chunking phase, the `MarkdownHeaderTextSplitter` ingeniously parsed the hierarchy of the academic papers and stored the section titles inside the chunk's `metadata` dictionary (e.g., `{"Header 1": "Abstract", "Header 2": "Methodology"}`). When LangChain handed these chunks to Milvus, Milvus attempted to dynamically map the metadata dictionary into its SQL-like relational schema. However, Milvus strictly prohibits spaces in field (column) names. It encountered the space in `Header 2` and threw a fatal schema error.

**The Solution:**
Refactored the upstream chunking configuration in `ingest.py`. Changed the tuple mappings from `("#", "Header 1")` to `("#", "Header_1")`. By utilizing underscores, the upstream chunker generates natively sanitized schema keys that cleanly map into Milvus without requiring a downstream sanitization loop.

---

## 6. Rigid Schemas vs. Dynamic Metadata (Milvus DataNotMatchException)

**The Bug/Challenge:**
After fixing the header spacing issue, the vector insertion crashed again halfway through processing with `pymilvus.exceptions.DataNotMatchException: Insert missed an field Header_2`.

**The Root Cause:**
When LangChain automatically initializes a Milvus collection from a list of `Document` chunks, it attempts to infer the strict SQL-style schema by reading the metadata of the *first* chunk. If the first chunk belongs to a deep sub-section (containing `Header_1`, `Header_2`, and `Header_3`), Milvus locks those three fields as required columns for the entire collection. When LangChain attempted to insert a later chunk (e.g., a top-level abstract) that only possessed `Header_1`, Milvus rejected the row because it was missing the required `Header_2` column.

**The Solution:**
Enabled dynamic schemas in the Milvus initialization parameters (`enable_dynamic_field=True`). This fundamentally shifted the database architecture from rigid columnar metadata to a dynamic JSON-blob metadata architecture, allowing documents with completely heterogeneous metadata structures (varying levels of headers) to coexist in the same vector space without crashing.

---

## 7. Scaling Bottlenecks in PDF Parse Time (Two-Tier Abstract Filtering)

**The Bug/Challenge:**
Attempting to scale the academic research assistant to search 30 candidate research papers from ArXiv introduced a severe performance bottleneck: parsing 30 full-length PDFs using Docling layout analysis and Neo4j Graph Extraction took over 25 minutes per query, creating an unusable user experience.

**The Root Cause:**
Full-text PDF layout parsing, vector embedding, and LLM Knowledge Graph entity extraction are computationally intensive $O(N \cdot \text{pages})$ operations. Ingesting unvetted candidate papers before verifying their specific domain relevance generates massive token and time waste.

**The Solution:**
Architected a **Two-Tier Hierarchical Research Funnel**:
- **Tier 1 (Fast Filter):** Query the lightweight ArXiv REST API for titles and abstracts (retrieving 30 papers in < 1s). Pass the abstracts through a fast LLM scoring prompt to rank candidate papers on a scale of 0–100.
- **Tier 2 (Deep Ingestion):** Automatically download ONLY the top 3–5 highest-scoring PDFs into `./data/` and trigger full-text Docling DOM parsing, Milvus vector embedding, and Neo4j graph extraction. This reduced total pipeline execution time from 25 minutes down to < 45 seconds while ensuring high-quality, targeted retrieval.

---

## 8. Procedural Script Coupling vs. LangGraph StateMachine Fault Tolerance

**The Bug/Challenge:**
Initial prototype implementations of the ArXiv Agent ran as a flat procedural Python function (`search_arxiv_and_shortlist`). If network glitches occurred during PDF downloads or database vector inserts failed, the script crashed completely, wiping out all previously computed LLM abstract relevance scores and forcing a expensive restart.

**The Root Cause:**
Procedural control flows lack explicit state schemas, node isolation, and transition boundaries. The agent state was stored in local function variables rather than an immutable shared graph state object.

**The Solution:**
Refactored the agent into a formal **LangGraph `StateGraph` state machine**. Modeled the pipeline into four isolated nodes (`search_arxiv_node`, `score_abstracts_node`, `download_ingest_node`, `synthesize_node`) orchestrated by a shared `ArxivAgentState` dictionary. This enables state persistence, per-node trace granularity in LangSmith, and clean unit testing of individual graph transition steps.
