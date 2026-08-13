# Docling Architecture & Document Object Model (DOM) Parsing

## The Core Problem with PDF Ingestion for RAG
A PDF file does not inherently understand text structure. It does not possess concepts like "Paragraph," "Header," or "Table." It is essentially a visual coordinate canvas dictating where to draw glyphs (e.g., "Draw the letter 'T' at x:150, y:400"). 

Standard chunking and extraction pipelines fail because they either read straight across the page (destroying two-column layouts) or rely on visual OCR (which hallucinates complex math symbols and squashes tables).

## The Docling 5-Stage Pipeline

IBM Docling solves this by constructing a hierarchical Document Object Model (DOM) before generating any text.

### Stage 1: Native Stream Extraction
Instead of taking a picture of the page for OCR, Docling directly extracts the native binary text stream embedded by LaTeX. 
* **Benefit:** 100% spelling accuracy. Zero hallucination of complex math operators, superscripts, or subscripts.

### Stage 2: Vision & Layout Analysis (RT-DETR)
The page is rendered as an image and processed by **RT-DETR-V2**, a highly optimized object detection model trained on the `DocLayNet` dataset.
* **Function:** Draws bounding boxes and classifies regions into categories: `Title`, `Paragraph`, `List`, `Table`, `Equation`, or `Figure`.
* **Reading Order:** Uses these boxes to map out correct two-column reading paths, ensuring text flows logically rather than strictly horizontally.

### Stage 3: TableFormer
If a bounding box is flagged as a `Table`, Docling hands it to a specialized AI called **TableFormer**.
* **Function:** Analyzes geometric spacing of words to predict an invisible grid. It perfectly reconstructs rows, columns, and merged cells, even when the PDF lacks visible table borders.

### Stage 4: DOM Construction
Docling organizes the parsed elements into an in-memory Tree Structure (similar to HTML DOM).
* Elements are nested logically (e.g., a Table belongs to a specific Section Header).

### Stage 5: Serialization
Docling walks the DOM tree from top to bottom and serializes it into structured Markdown.
* `Header 1` becomes `# `
* `Header 2` becomes `## `
* `Table` elements are converted into standard Markdown grid syntax (`| Col 1 | Col 2 |`).

## Why this is critical for Hybrid GraphRAG
By outputting layout-aware Markdown, Docling perfectly sets up LangChain's `MarkdownHeaderTextSplitter`. This allows Vector Databases (Milvus) to keep tables perfectly intact and LLMs to receive chunks of text with their hierarchical section titles attached in the metadata.
