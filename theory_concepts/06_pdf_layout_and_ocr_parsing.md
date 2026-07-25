# Concept 06: Layout-Aware PDF Ingestion & OCR for Academic RAG

## 1. The Multi-Column Academic PDF Problem
Standard PDF text extractors (like naive PyPDF) read PDF objects strictly by coordinate order or stream sequence. In academic papers, this causes critical failures:
* **Multi-Column Reading Order Failure:** Paragraphs in column 1 get mixed horizontally with column 2.
* **Table & Figure Destruction:** Tables lose row/column structural boundaries and turn into meaningless text blobs.
* **Latex & Math Formula Corruptions:** Subscripts, superscripts, and mathematical symbols are misread as plain characters.

## 2. Layout-Aware Parsing & OCR Pipeline
To feed academic PDFs into our Hybrid GraphRAG pipeline, we must convert PDFs into **Structured Markdown**:
1. **Layout Detection:** Identify reading regions (Headers, Multi-column text blocks, Tables, Figure Captions, Equations).
2. **Reading Order Sorting:** Group text blocks into their true logical reading flow (Column 1 top-to-bottom, then Column 2 top-to-bottom).
3. **OCR Engine / Vision Model Execution:** Run OCR (or Vision-Language Models like INT4 quantized PaddleOCR / Unlimited-OCR / Marker) on scanned images or complex layouts.
4. **Markdown Formatting:** Output structured Markdown containing `# Headings`, `| Tables |`, and `$$ LaTeX $$` blocks.

## 3. Integration into AcadRAG Pipeline
```
[ PDF Document ] ➡️ [ Layout OCR Engine ] ➡️ [ Markdown Text ] ➡️ [ Semantic Chunker ] ➡️ [ Milvus & Neo4j ]
```
Once converted to Markdown, the text flows directly into our existing `SemanticChunker`, `Milvus` vector store, and `Neo4j` graph extractor without breaking table relationships or entity links.
