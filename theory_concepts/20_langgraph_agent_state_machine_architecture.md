# Concept 20: LangGraph Agent State Machine Architecture

## 1. Why Convert Multi-Step Pipelines to LangGraph Nodes?

In procedural Python scripts, if step 3 (e.g. PDF parsing) fails, the entire script crashes and all previous computational work (e.g. ArXiv API requests & LLM abstract scoring) is lost.

By modeling the Two-Tier Autonomous Agent as a **LangGraph StateGraph**, we achieve:
1. **Explicit State Schema (`TypedDict`):** Every node reads and updates a shared `AgentState` object.
2. **Fault Tolerance & Checkpointing:** Nodes can be retried or interrupted independently.
3. **LangSmith Integration:** LangGraph natively integrates with LangSmith, automatically mapping each graph node to a parent-child span trace.

---

## 2. ArXiv Agent Graph Topology

```
             ┌─────────────────────────┐
             │       START NODE        │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │   search_arxiv_node     │ (Fetches candidate abstracts)
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │  score_abstracts_node   │ (Evaluates & ranks abstracts 0-100)
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │ download_ingest_node    │ (Docling DOM -> Milvus + Neo4j)
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │     synthesize_node     │ (Cross-paper GraphRAG answer)
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │        END NODE         │
             └─────────────────────────┘
```
