# Concept 11: LLM Factory Pattern & Provider Decoupling

## 1. What is the Factory Pattern?
The **Factory Pattern** is a creational design pattern in software engineering that provides a centralized interface for creating objects without hardcoding the exact class of object that will be instantiated.

In an Enterprise RAG Pipeline:
* Without Factory: `populate.py`, `query.py`, and `langsmith_eval.py` all directly instantiate `ChatOllama(model="llama3.1:8b")`. Switching providers requires modifying code across multiple files.
* With Factory: Scripts request `get_llm()` from `llm_factory.py`. The factory reads configuration (e.g. `.env` variable `LLM_PROVIDER`) and dynamically returns the appropriate `BaseChatModel` instance (`ChatOllama`, `ChatGroq`, or `ChatOpenAI`).

## 2. Decoupled Architecture Diagram

```
                ┌──────────────────────────────┐
                │        .env Config           │
                │     LLM_PROVIDER=groq        │
                └──────────────┬───────────────┘
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │                   llm_factory.py                          │
 │                                                           │
 │  if provider == "groq": return ChatGroq(...)             │
 │  elif provider == "openrouter": return ChatOpenAI(...)   │
 │  else: return ChatOllama(...)                            │
 └─────────────────────────────┬─────────────────────────────┘
                               │ (Returns Unified BaseChatModel)
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
   [ populate.py ]       [ query.py ]       [ langsmith_eval.py ]
```

## 3. Benefits for Production Hybrid GraphRAG
1. **Zero-Code Switching:** Switch between offline privacy (Local Ollama) and sub-second ingestion (Groq LPU) by changing a single line in `.env`.
2. **Graceful Fallbacks:** If an API key is missing or a cloud endpoint drops, the factory automatically degrades gracefully to local `Ollama` without crashing the pipeline.
3. **Unified Interface:** Because LangChain standardizes `invoke()` and `aconvert_to_graph_documents()`, all pipeline components remain provider-agnostic.
