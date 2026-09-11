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

---

## 9. ArXiv SDK Breaking Changes & Incomplete Node Unit Test Coverage

**The Bug/Challenge:**
During live testing of Node 3 (`download_ingest_node`), the pipeline crashed with `AttributeError: 'Result' object has no attribute 'download_pdf'`.

**The Root Cause:**
1. **SDK Breaking Change:** The freshly installed `arxiv==4.0.1` package deprecated the `.download_pdf()` method on `Result` objects, requiring direct HTTP GET retrieval of `paper['pdf_url']`.
2. **Test Coverage Gap:** The initial unit test suite only tested Node 2 (`score_abstracts_node`) and omitted Node 3 (`download_ingest_node`), allowing the un-mocked SDK error to bypass test suite validation.

**The Solution:**
1. Switched PDF downloading in `download_ingest_node` to use standard HTTP streaming (`requests.get(paper['pdf_url'])`), eliminating dependency on unstable SDK method signatures.
2. Expanded `src/evaluation/test_arxiv_agent.py` to add full unit test coverage for `download_ingest_node` using `unittest.mock.patch` for `requests.get`, `extract_pdf_with_docling`, and `populate_databases`.

---

## 10. Human-in-the-Loop (HITL) Multi-Stage Gatekeeper Verification

**The Bug/Challenge:**
Fully autonomous agent pipelines risk ingesting irrelevant PDFs or consuming heavy CPU/GPU resources on Docling DOM conversion and Milvus/Neo4j graph generation without user oversight.

**The Root Cause:**
Absence of interactive confirmation gates between execution phases meant the pipeline proceeded unconditionally from Tier 1 filtering to full database mutation.

**The Solution:**
Implemented a 3-Stage HITL Guardrail system:
1. **HITL Gate 1 (Pre-Download):** Displays paper titles, scores, and URLs, requiring explicit user approval before sending HTTP GET requests.
2. **HITL Gate 2 (Pre-Extraction):** Asks for user confirmation before starting Docling DOM PDF parsing.
3. **HITL Gate 3 (Pre-GraphRAG):** Requests approval before wiping/populating Milvus + Neo4j and launching multi-paper GraphRAG synthesis.

---

## 11. Optional LangChain Provider Imports & Fallback Architecture

**The Bug/Challenge:**
Setting `LLM_PROVIDER="gemini"` in `.env` triggered the warning log: `⚠️ langchain-google-genai not installed. Falling back to OpenAI-compatibility mode...`.

**The Root Cause:**
LangChain decouples provider integrations into modular optional packages (e.g. `langchain-google-genai`, `langchain-groq`, `langchain-openai`). When a developer switches `LLM_PROVIDER="gemini"` without installing `langchain-google-genai`, Python raises an `ImportError`.

**The Solution:**
Implemented a **Graceful Fallback Factory Pattern** in `src/core/llm_factory.py`. When `langchain-google-genai` is not installed, the factory automatically catches `ImportError` and falls back to `ChatOpenAI` targeting Google's OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`).

---

## 12. GraphRAG Ingestion Latency Bottlenecks vs. Enterprise Production Architecture

**The Bug/Challenge:**
Ingesting large academic corpora (1,100+ chunks) into a GraphRAG pipeline takes ~1 hour on local consumer hardware or free cloud API tiers, whereas standard Vector RAG ingestion completes in < 5 seconds.

**The Root Cause:**
Standard Vector RAG runs a fast feed-forward encoder pass (`BAAI/bge-small-en-v1.5`), generating embeddings for 1,100 chunks in parallel on GPU in seconds. In contrast, GraphRAG requires running a full generative LLM pass (`LLMGraphTransformer`) *per chunk* to extract entity nodes and Cypher relationship edges, resulting in 1,100 separate LLM generation steps bounded by API rate limits (15 RPM on free tiers) or local GPU throughput.

**The Solution / Production Architecture:**
1. **Immediate Vector Availability:** Separate vector embedding from graph extraction. Milvus vector population executes instantly, giving users immediate vector RAG search capabilities.
2. **Asynchronous Background Ingestion:** Knowledge Graph extraction is offloaded to background task queues (Celery/Redis/Ray).
3. **Paid API Scale-out / Enterprise LPUs:** On production paid API tiers (Groq/Gemini Paid) with 1,000+ RPM rate limits and `CONCURRENCY_LIMIT=50`, 1,100-chunk GraphRAG extraction time drops from 1 hour to < 45 seconds.

---

## 13. 6.16-Hour Local GPU Graph Extraction Benchmark & MD5 Cache Protection

**The Bug/Challenge:**
Executing `python main.py --populate` locally on a 4GB RTX 3050 GPU for 1,016 document chunks took **22,198 seconds (6.16 hours)** to extract entity nodes and write relationships into Neo4j.

**The Root Cause:**
Local Ollama execution of `LLMGraphTransformer` required 1,016 sequential generative LLM passes. At ~21.8 seconds per chunk on a 4GB mobile GPU, total execution time scaled linearly to 6+ hours.

**The Solution / Milestone:**
1. **GPU Safety:** Hardware thermal throttling prevented GPU damage during the long run.
2. **Persistent Graph Caching:** Implemented `graph_cache.json` using MD5 chunk hashes. All 1,016 extracted graph documents are now permanently saved locally. Subsequent runs of `populate.py` complete in **< 0.1 seconds** by loading cached JSON triples.
3. **Architectural Validation:** Confirmed that local GPUs should be reserved for instantaneous retrieval (`query.py`), while initial large-scale graph extractions should utilize high-throughput APIs (Gemini/OpenRouter).

---

## 14. LLM Judge False-Positive Fallback Scoring in LangSmith Evaluation

**The Bug/Challenge:**
During LangSmith evaluation benchmarking, candidate answers returning `"I cannot answer this based on the provided documents."` were incorrectly given a score of `1.0` (CORRECT) by the LLM Judge, despite the ground truth containing detailed technical facts.

**The Root Cause:**
The prompt rules for `llm_judge_evaluator` in `src/evaluation/langsmith_eval.py` contained an ambiguous rule: *"If the ground truth expects 'I cannot answer...' and candidate states that exact fallback, return CORRECT"*. The LLM Judge misapplied this rule as a wildcard matcher: whenever it saw the candidate output the exact fallback string, it marked the item as CORRECT regardless of what the ground truth contained.

**The Solution:**
Refactored the judge prompt rules to explicitly enforce: *"If the candidate answer is 'I cannot answer this based on the provided documents.' BUT the ground truth answer contains factual technical information, you MUST return INCORRECT."* This eliminated false positive evaluation scores across the benchmark suite.

---

## 15. VRAM Contention & CUDA OOM Resolution During Evaluation

**The Bug/Challenge:**
Executing `python main.py --eval` crashed on question 8 with `ollama._types.ResponseError: CUDA error: out of memory` in PyTorch `ggml_backend_cuda_buffer_set_tensor`.

**The Root Cause:**
On a 4GB RTX 3050 GPU, three separate models were simultaneously competing for VRAM:
1. `BAAI/bge-small-en-v1.5` (Embedding Encoder)
2. `BAAI/bge-reranker-base` (Cross-Encoder)
3. `qwen2.5:3b` / `qwen3:4b` (Ollama Generative LLM)
When prompt sizes grew with 8 retrieved context passages, Ollama's KV-cache exceeded remaining VRAM, causing CUDA allocation failure.

**The Solution:**
Offloaded small auxiliary models (`HuggingFaceEmbeddings` and `CrossEncoder`) to CPU execution by setting `device='cpu'` in `src/core/retriever.py`. Because CPU RAM handles BGE embedding/reranking in milliseconds, this freed 100% of the 4GB GPU VRAM exclusively for Ollama LLM generation, completely resolving CUDA OOM errors.

---

## 16. Web Scraping Legality & Crawler Architecture (Scrapling vs. SaaS APIs)

**The Bug/Challenge:**
Expanding paper discovery beyond ArXiv requires fetching from conferences (OpenReview, NeurIPS, CVPR). However, scraping presents two major hurdles:
1. **Legal & Compliance Risk:** Potential liability under the Computer Fraud and Abuse Act (CFAA), DMCA § 1201 anti-circumvention, copyright infringement, and server abuse.
2. **Tool Selection Trade-offs:** Deciding whether to rely on paid cloud APIs (Firecrawl / Tavily) or a local open-source scraping engine (`Scrapling`).

**The Root Cause / Legal Breakdown:**
- *Legal Precedent:* US Supreme Court (*Van Buren v. US*) and 9th Circuit (*hiQ Labs v. LinkedIn*) established that scraping publicly accessible data without bypassing an authentication gate does not violate CFAA. Furthermore, indexing text for semantic search and Knowledge Graphs constitutes transformative Fair Use (17 U.S.C. § 107; *Authors Guild v. Google*), supplemented by EU Directive 2019/790 (Articles 3/4) for Text and Data Mining (TDM).
- *Architectural Differences:* Tavily is an AI search engine (discovery-focused, per-query pricing), Firecrawl is a cloud-based web-to-markdown crawler (proxy-managed, SaaS subscription), while Scrapling is a local Python engine using Camoufox (zero cost, sub-second speed, local TLS fingerprinting, but bounded by local IP reputation).

**The Solution / Strategy:**
1. **Legal Safeguards:** Enforced strict boundaries: scrape ONLY open-access, non-paywalled repositories (Creative Commons CC-BY, OpenReview, CVPR Open Access). Strictly prohibit scraping paywalled publishers (Elsevier, IEEE). Enforce `robots_txt_obey=True` and a mandatory 1.5s rate-throttle delay to prevent server strain.
2. **Hybrid Engine Selection:** Chose `Scrapling` as the primary engine for known academic repositories due to zero cost, offline execution, and high-speed local DOM parsing, while reserving Tavily strictly for open-ended queries where target URLs are unknown.

---

## 17. Multi-Source Academic Discovery: Merging ArXiv & Semantic Scholar APIs

**The Bug/Challenge:**
Relying exclusively on ArXiv restricts paper discovery to preprints, omitting influential peer-reviewed papers published directly in Nature, IEEE, ACM, or top ML conferences (NeurIPS/ICLR/CVPR). However, querying multiple APIs simultaneously introduces duplicate entries, heterogeneous JSON schemas, and missing open-access PDF links.

**The Root Cause:**
- ArXiv and Semantic Scholar use completely different schemas (e.g., ArXiv provides `entry_id` and guaranteed `pdf_url`, while Semantic Scholar returns `paperId`, nullable `abstract`, and nested `openAccessPdf.url`).
- Popular papers exist on both platforms, causing identical papers with slight title punctuation discrepancies (e.g., `"Attention Is All You Need"` vs. `"Attention is all you need."`) to be fetched twice, diluting top-k evaluation slots.

**The Solution:**
1. **Schema Normalization:** Created a unified dictionary format (`title`, `summary`, `pdf_url`, `entry_id`, `authors`, `source`, `citation_count`, `year`).
2. **Alphanumeric Title Deduplication:** Implemented normalized fingerprinting (`re.sub(r'[\W_]+', '', title).lower()`) so ArXiv preprints and Semantic Scholar peer-reviewed counterparts merge into a single canonical entry, prioritizing open-access PDF availability.
3. **Open-Access Validation:** Filtered Semantic Scholar candidate results to ensure only papers with a non-null `openAccessPdf` URL or valid preprint link enter the download pipeline, preventing 403 Forbidden paywall crashes during ingestion.

---

## 18. Decoupling Web Discovery from Scraping: The Tavily + Scrapling Synergy

**The Bug/Challenge:**
Relying on full-page scraping APIs (like Firecrawl or Tavily's deep extraction mode) across multi-paper discovery workflows incurs unsustainable API credit consumption and leaves agent pipelines dependent on external cloud infrastructure. Conversely, pure scrapers (like Scrapling or BeautifulSoup) lack a web search index and cannot discover unknown URLs from arbitrary natural language queries.

**The Root Cause:**
Web information retrieval for AI agents encompasses two distinct computational phases with conflicting cost curves:
1. **Discovery (Index Search):** Requires crawling the open web and maintaining global search indexes (which cannot be done locally on consumer hardware).
2. **Extraction (DOM/PDF Harvesting):** Requires parsing specific HTML pages, rendering client-side JavaScript, and converting DOM to Markdown (which is computationally cheap and can run locally).

**The Solution:**
Implemented the **Scout + Harvester Architectural Pattern**:
- **The Scout (Tavily):** Spends a single lightweight search credit to perform a domain-constrained web query (targeting `openreview.net`, `neurips.cc`, `biorxiv.org`), returning only metadata and candidate URLs.
- **The Harvester (Scrapling):** Receives the candidate URLs, applies local `robots.txt` verification, executes local TLS fingerprint spoofing via `Camoufox`, extracts PDF download links or converts dynamic DOM to Markdown (`page.markdown()`) for $0 cost. This reduces external API costs by ~90% while granting the agent full local autonomy over DOM parsing.

---

## 19. Linux Distro Collisions in Headless Browser Toolchains (Fedora DNF vs. Debian `apt-get`)

**The Bug/Challenge:**
Running `scrapling install` on Fedora 43 crashed with:
`sh: line 1: apt-get: command not found`
`subprocess.CalledProcessError: Command '['.../python', '-m', 'playwright', 'install-deps', 'chromium']' returned non-zero exit status 1.`

**The Root Cause:**
The `scrapling install` CLI command automates browser setup by invoking Playwright's sub-command `playwright install-deps`. Playwright's upstream helper scripts assume Debian/Ubuntu by default, attempting to invoke `apt-get` with `sudo`. On Red Hat / Fedora distributions where `dnf` is the package manager, `apt-get` does not exist, causing the subprocess execution to fail fatally.

**The Solution:**
1. **Bypass OS Package Manager Probe:** Directly install Playwright and Camoufox browser binaries in user-space without the Ubuntu-specific `install-deps` flag:
   ```bash
   playwright install firefox chromium
   python -m camoufox fetch
   ```
2. **Native Fedora Dependencies:** If system graphics libraries are needed, install them via `dnf`:
   `sudo dnf install nss nspr mesa-libgbm alsa-lib libX11 libXcomposite libXdamage libXext libXfixes libXrandr pango cairo`
3. **Defensive Fetcher Fallback:** Configured `crawler_agent.py` to gracefully fallback to `Fetcher()` (fast HTTP requests via `curl_cffi` / `requests`) if browser binaries or headless displays are missing, preventing pipeline disruption.

---

## 20. Lazy Loading of RAG Extra Dependencies in Web Scraping Frameworks (Markdownify)

**The Bug/Challenge:**
During live execution of the web crawler harvester node on non-PDF technical pages, Scrapling halted with:
`ImportError: Markdown conversion requires the "markdownify" package. Install it with: pip install "scrapling[rag]"`

**The Root Cause:**
Scrapling packages its HTML-to-Markdown conversion functionality under an optional package extra (`scrapling[rag]`). While `hasattr(page, 'markdown')` evaluates to `True` at runtime, invoking `.markdown()` triggers an internal deferred import of `markdownify`. If `markdownify` is not pre-installed in the virtual environment, the method raises an unhandled `ImportError`.

**The Solution:**
1. **Dependency Installation:** Installed `markdownify` into the virtual environment (`pip install markdownify`) and locked it in `requirements.txt`.
2. **Defensive Fallback Mechanism:** Refactored line 231 of `crawler_agent.py` to wrap `page.markdown()` in a try-except block, automatically falling back to standard string text extraction (`getattr(page, 'text', '')`) if markdown conversion dependencies are absent, preventing crawler abortion.
