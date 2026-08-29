# Concept 12: Cloud API Rate Limits (RPM/TPM) and Concurrency Tuning

## 1. Local Hardware vs. Cloud API Throughput Constraints

When running locally (Ollama), performance is limited by **VRAM / Compute Latency**. When running on Cloud APIs (Groq, OpenRouter), performance is limited by **API Rate Limits**:

1. **RPM (Requests Per Minute):** The maximum number of API calls allowed per minute.
2. **TPM (Tokens Per Minute):** The maximum combined input + output tokens processed per minute.
3. **RPD (Requests Per Day):** Daily quota cap.

### Groq Free Tier Quota Limits (Llama-3.3-70b-versatile)
* **RPM:** 30 Requests / minute
* **TPM:** 6,000 Tokens / minute

## 2. Concurrency Tuning Strategy

If `populate.py` fires 50 concurrent async graph extraction tasks simultaneously:
* **Local Ollama:** Bottlenecks CPU/VRAM queue safely.
* **Groq Free Tier:** Triggers `429 RateLimitError: Limit reached for TPM/RPM` after ~10-15 requests.

### Solution: Provider-Aware Concurrency Control
```
                 ┌─────────────────────────────┐
                 │    Active Provider Check    │
                 └──────────────┬──────────────┘
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼                                             ▼
  [ Local Ollama ]                              [ Groq Cloud API ]
  CONCURRENCY_LIMIT = 4                        CONCURRENCY_LIMIT = 2
                                               + max_retries = 3
```

## 3. Why Chunking Parameters (`chunk_size`) Should NOT Change
Chunking size (`chunk_size=1000`, `chunk_overlap=200`) governs the **granularity of vector embeddings in Milvus** and **node density in Neo4j**. 
* Changing chunk size when swapping LLM models invalidates vector indices and forces full database rebuilds.
* Keep chunking parameters identical so that your vector space and graph architecture remain provider-agnostic.
