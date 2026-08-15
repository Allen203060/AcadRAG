# Concept 08: LangGraph Stateful Orchestration for RAG Pipelines

## 1. What is LangGraph?
While standard LangChain LCEL chains execute functions in a rigid linear sequence (`Prompt | LLM | Parser`), **LangGraph** models applications as **Stateful Cyclic Graphs**:
* **State (`TypedDict`):** A single shared memory schema accessible by all graph nodes.
* **Nodes (Python Functions):** Discrete processing units (e.g. `retrieve`, `rerank`, `generate`) that receive the current state and return state updates.
* **Edges (Control Flow):** Define navigation between nodes. Edges can be static (`START -> retrieve`) or conditional (`route_based_on_score`).

## 2. Why Use LangGraph in AcadRAG?
1. **State Traceability:** Every step (raw query $\rightarrow$ hybrid context $\rightarrow$ reranked chunks $\rightarrow$ LLM generation) is explicitly stored in the state object.
2. **Conditional Branching:** Enables fallback routing (e.g., if reranked context score is low, trigger query rewriting or fallback search).
3. **First-Class LangSmith Integration:** LangGraph automatically logs state transitions to LangSmith with zero boilerplate.

## 3. Graph Architecture for AcadRAG

```
[ START ] ──► (1. retrieve_node) ──► (2. rerank_node) ──► (3. generate_node) ──► [ END ]
                  │                       │                     │
            Queries Milvus           Cross-Encoder         Llama 3.1
               & Neo4j                 Scoring             Grounding
```
