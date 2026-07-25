# 05. Generation & Citations

## Grounding the LLM
*   **Parametric Memory vs. Retrieved Context:** Left to its own devices, an LLM will guess answers using the data it was trained on (which causes hallucinations).
*   **Prompt Constraints:** By explicitly instructing the LLM to rely *solely* on the provided context, we "ground" its generation in empirical facts.

## Auditability via Citations
*   **Formatting:** Retrieved chunks are injected into the prompt with distinct labels (e.g., `[Source 1]`).
*   **Enforcement:** The LLM is instructed to append these exact labels inline immediately after stating a fact derived from that specific chunk. This allows researchers to instantly verify the AI's output against the source material.
