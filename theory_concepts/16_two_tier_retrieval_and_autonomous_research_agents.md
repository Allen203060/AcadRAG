# Concept 16: Two-Tier Filtering & Autonomous Academic Research Agents

## 1. The Challenge of Full-Text PDF Processing at Scale
In academic research, downloading, parsing, chunking, embedding, and constructing Knowledge Graphs for 30 full-length PDFs requires:
* **Time Overhead:** 30 papers $\times$ 15 pages = 450 pages of parsing (~20–30 minutes).
* **Token Overhead:** Millions of tokens generated across unneeded papers.

## 2. The Solution: Two-Tier Hierarchical Filtering

```
                          [ User Research Topic ]
                 (e.g., "Face Recognition in IoT Edge")
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │  Tier 1: ArXiv API Metadata Search      │
                │  Fetch 20–30 Candidate Titles & Abstracts│
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │  Abstract Reranking & Scoring Filter    │
                │  (Fast Cross-Encoder / LLM Evaluator)   │
                └────────────────────┬────────────────────┘
                                     │ (Filters to Top 3–5 Most Relevant)
                                     ▼
                ┌─────────────────────────────────────────┐
                │  Tier 2: Automatic PDF Ingestion        │
                │  Download PDFs -> Docling -> GraphRAG   │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │  Deep GraphRAG Synthesis & Synthesis    │
                │  (Milvus + Neo4j Cross-Paper QA)        │
                └─────────────────────────────────────────┘
```

## 3. Why This Architecture Wins
1. **Computational Efficiency:** Abstract filtering evaluates 30 papers in **2 seconds** before doing heavy PDF layout parsing.
2. **Autonomous Workflow:** The agent automatically sources, curates, downloads, ingests, and synthesizes research papers without manual intervention.
3. **High Portfolio Value:** Demonstrates multi-agent orchestration (LangGraph), external API integration (ArXiv API), and two-stage RAG ranking.
