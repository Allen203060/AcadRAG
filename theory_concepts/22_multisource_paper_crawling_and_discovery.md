# Concept 22: Multi-Source Research Paper Crawling & Discovery

## 1. Beyond ArXiv: Expanding Research Discovery Sources

While ArXiv is the primary preprint server for CS and AI, restricting an autonomous agent to ArXiv creates blind spots for published conference papers (NeurIPS, ICML, CVPR, ACL), bio/med preprints (bioRxiv, medRxiv), and peer-reviewed journals.

To discover papers across the entire web, we utilize a **Multi-Source Provider Pipeline**:

| Source / Provider | Target Content / Domain | Access Mechanism | Open Access PDF Availability |
| :--- | :--- | :--- | :--- |
| **ArXiv API** | AI, ML, Physics, Math Preprints | REST API (`arxiv` SDK) | 100% Direct PDF Links |
| **Semantic Scholar API** | 200M+ Multidisciplinary Papers | REST API (`api.semanticscholar.org`) | ~70% Open-Access PDF Links |
| **OpenAlex API** | 250M+ Global Scholarly Works | Open REST API (`api.openalex.org`) | High (Unpaywall Indexing) |
| **Tavily / Firecrawl** | OpenReview, NeurIPS, CVPR, ACL | Web Crawler (Markdown Conversion) | Scrapes direct PDF anchor tags |

---

## 2. Universal Multi-Source Agent Architecture

```
                       [ User Query: "Edge AI" ]
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
[ ArXiv Search ]        [ Semantic Scholar Search ]    [ Firecrawl/Tavily Web Search ]
       │                          │                          │
       └──────────────────────────┼──────────────────────────┘
                                  ▼
                [ Candidate Deduplication (by Title/DOI) ]
                                  │
                                  ▼
                [ Tier 1: LLM Abstract Relevance Scoring ]
                                  │
                                  ▼
                [ Tier 2: Docling DOM Parsing + GraphRAG ]
```

---

## 3. Web Crawling with Tavily & Firecrawl for Non-API Sources

For paper repos without formal APIs (e.g. `openreview.net` or `thecvf.com`):
1. **Tavily Search API:** Searches Google/Bing index, returning pre-parsed clean markdown abstracts and source URLs.
2. **Firecrawl Scraping API:** Executes dynamic JavaScript rendered pages and extracts direct `.pdf` download links via DOM query selectors.
