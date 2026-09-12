# Concept 25: Multi-Source Academic Discovery & Two-Tier Crawler Patterns

## 1. ArXiv vs. Semantic Scholar Graph API
- **ArXiv:** The leading preprint server for computer science, mathematics, and physics. REST API returns Atom XML feeds with guaranteed open-access PDF links, but omits papers published directly in peer-reviewed conferences without preprints.
- **Semantic Scholar (S2AG):** Indexes over 214 million papers across peer-reviewed conferences (NeurIPS, ICML, CVPR) and publishers (IEEE, ACM, Nature, PubMed). Returns rich citation metrics (`citationCount`), external identifiers (DOI, PubMed, ArXiv), and verified open-access PDF links (`openAccessPdf`).

## 2. Alphanumeric Title Deduplication
When querying multiple academic databases simultaneously, popular papers appear in both feeds with varying punctuation or whitespace discrepancies (e.g., *"Attention Is All You Need"* vs *"Attention is all you need."*).
Canonical title fingerprinting:
```python
clean_title = re.sub(r'[\W_]+', '', title).lower()
```
This merges preprints and conference camera-ready papers into a single canonical entry, preventing redundant LLM abstract scoring passes and saving token budget.

## 3. The "Scout + Harvester" Pattern (Tavily + Scrapling)
- **The Scout (Tavily API):** Used for open-ended queries where target URLs are unknown. Performs live internet search and returns high-confidence candidate URLs.
- **The Harvester (Scrapling):** Receives target URLs, renders dynamic JavaScript locally via Camoufox, bypasses anti-bot defenses, and converts DOM to Markdown for $0.
- **Why Firecrawl is Redundant:** Scrapling provides local headless rendering and native Markdown conversion (`page.markdown()`), while Docling handles deep PDF layout parsing. Firecrawl adds paid cloud API latency and per-page billing without additional benefit for local RAG pipelines.
