# Concept 08: LangSmith Observability, Datasets, and Evaluation

## 1. What is LangSmith?
**LangSmith** is an enterprise-grade LLM observability, tracing, and evaluation platform built by LangChain. It provides:
1. **Full-Stack Tracing:** Visualizes intermediate steps in your RAG pipeline (document loading, vector retrieval, cross-encoder scores, LLM prompt assembly, generation).
2. **Dataset Management:** Programmatically creates labeled test suites (Inputs vs Reference Outputs) for benchmark tracking over time.
3. **Automated Offline Experiments (`evaluate()`):** Runs target functions across datasets and scores outputs using custom evaluators or LLM-as-a-Judge.

## 2. LangSmith Tracing Architecture
```
[ User Query ] ──► ( @traceable Wrapper ) ──► [ LangSmith UI Dashboard ]
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                     ▼
 [ Retrieval Span: Milvus/Neo4j ]   [ Generation Span: Ollama Llama 3.1 ]
   (Latency, Top K Docs)               (Prompt tokens, Output tokens)
```

By enabling environment variables (`LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_ENDPOINT`), LangChain automatically intercepts all agent calls, token usages, and latency spans and streams them to the LangSmith cloud UI.

## 3. LangSmith SDK Evaluation Workflow (`evaluate()`)
1. **Create Dataset:** `client.create_dataset(dataset_name)` + `client.create_examples(...)`.
2. **Define Target Function:** A wrapper function `target(inputs: dict) -> dict` that executes the RAG pipeline on `inputs["question"]`.
3. **Define Evaluator:** A scoring function `evaluator(inputs, outputs, reference_outputs)` returning a score (e.g. `1.0` for pass, `0.0` for fail).
4. **Run Experiment:** `client.evaluate(target, data="Dataset Name", evaluators=[evaluator])`.
