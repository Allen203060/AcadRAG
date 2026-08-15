# Concept 07: Docling DOM Parsing & Markdown Header-Aware Hierarchical Chunking

## 1. Document Object Model (DOM) Parsing with Docling
Traditional PDF parsers treat documents as flat text streams. **Docling** (developed by IBM Research) treats documents as an interactive **Document Object Model (DOM) tree**:
* **Structural Hierarchy:** Recognizes Document $\rightarrow$ Section $\rightarrow$ Subsection $\rightarrow$ Paragraph / Table / Equation relationships.
* **TableFormer AI:** Uses specialized vision models to preserve table structures into Markdown (`| Col | Col |`) or HTML grid structures without splitting rows.
* **Native Text Stream Preservation:** Extracts native PDF text directly, eliminating OCR hallucinations on digital-born arXiv papers.

## 2. Flat Semantic Chunking vs. Header-Aware Hierarchical Chunking

### Flat Semantic Chunking (Naive)
Evaluates sentence embedding distances only.
* **Problem:** Splitting happens randomly mid-paragraph or mid-table. A chunk containing `Multi-Head Attention` loses the knowledge that it belongs under `# 3. Synthetic Architecture`.

### Header-Aware Hierarchical Chunking (LangChain `MarkdownHeaderTextSplitter` & `HybridChunker`)
Splits by Markdown headers (`#`, `##`, `###`) and injects section path into document metadata.

```
Document Header (# 3. Model Architecture)
   └── Subsection (## 3.2 Attention Mechanism)
          └── Paragraph Chunk
                 ├── Content: "Multi-head attention allows the model to jointly attend..."
                 └── Metadata: {"Header 1": "3. Model Architecture", "Header 2": "3.2 Attention Mechanism"}
```

## 3. Benefits in Hybrid GraphRAG Pipeline
1. **Milvus Metadata Injection:** When Milvus retrieves a vector chunk, the LLM receives the full section header context alongside the snippet.
2. **Neo4j Subgraph Enrichment:** Section titles serve as structural anchor nodes in the Neo4j Knowledge Graph.
