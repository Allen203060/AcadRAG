# Concept 14: CLI Orchestration & Pipeline Entrypoints

## 1. What is a Pipeline Entrypoint?
In modular production architectures, individual scripts handle distinct responsibilities:
* `pdf_loader.py` & `ingest.py`: Chunking & Document Parsing.
* `populate.py`: Database Ingestion.
* `query.py` & `rag_graph.py`: Retrieval & State Machine Execution.
* `langsmth_eval.py`: Evaluation Benchmarking.

A **Unified Orchestrator (`main.py`)** acts as the single operational entrypoint for the system. It allows developers or automated CI/CD jobs to execute individual pipeline stages or the full end-to-end flow via flags or an interactive CLI menu.

## 2. Orchestration Architecture Diagram

```
                             ┌────────────────────────┐
                             │       main.py          │
                             │  (Unified Orchestrator)│
                             └───────────┬────────────┘
                                         │
       ┌──────────────────┬──────────────┼──────────────┬──────────────────┐
       ▼                  ▼              ▼              ▼                  ▼
 [ 1. Populate DBs ] [ 2. Query ] [ 3. LangGraph ] [ 4. LangSmith ] [ 5. Run Full ]
    (populate.py)    (query.py)   (rag_graph.py) (langsmth_eval)     (End-to-End)
```

## 3. Benefits of a Unified Orchestrator
1. **Single Entrypoint:** Developers don't need to remember separate file names or CLI parameters.
2. **Flag-Based & Interactive Execution:** Supports both automated scripting (`python main.py --all`) and interactive user menus.
3. **Environment Verification:** Verifies environment health (Neo4j, Milvus, `.env` provider keys) before running retrieval tasks.
