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
