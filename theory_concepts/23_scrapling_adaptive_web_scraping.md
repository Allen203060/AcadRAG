# Concept 23: Scrapling - Adaptive & Stealthy Web Scraping Framework

## 1. What is Scrapling?

**Scrapling** (`D4Vinci/Scrapling`) is a modern, high-performance Python web scraping framework designed to handle everything from simple HTTP requests to full stealth browser automation and large-scale crawling.

Key capabilities include:
1. **Anti-Bot & Cloudflare Bypass (`StealthyFetcher`):** Automatically bypasses Cloudflare Turnstile, DDoS-Guard, and bot protection systems using TLS fingerprint impersonation (JA3/HTTP2) and stealth browser automation.
2. **RAG-Ready Markdown (`page.markdown()`):** Converts complex HTML pages directly into clean, LLM-ready Markdown without needing external API calls (e.g. Firecrawl) or LLM preprocessing.
3. **Adaptive Element Tracking (`adaptive=True`):** Uses similarity algorithms to relocate elements (e.g., paper titles or PDF download links) even if a website redesigns its HTML layout.
4. **Zero API Cost (100% Local & Open Source):** Runs completely locally on your hardware without external API rate limits or subscriptions.

---

## 2. Scrapling vs. Other Scraping Tools for RAG

| Feature | Scrapling | Firecrawl API | Tavily API | BeautifulSoup + Requests |
| :--- | :--- | :--- | :--- | :--- |
| **Cost** | 🆓 100% Free / Open Source | 💳 Paid API | 💳 Paid API | 🆓 100% Free |
| **Anti-Bot / Cloudflare Bypass** | ✅ Built-in (`StealthyFetcher`) | ✅ Server-side | ✅ Server-side | ❌ Fails on Cloudflare |
| **JS Rendering (Dynamic DOM)** | ✅ Built-in (Playwright/Chrome) | ✅ Server-side | ❌ Limited | ❌ No JS Execution |
| **RAG Markdown Output** | ✅ `page.markdown()` | ✅ Native | ✅ Native | ❌ Manual HTML parsing |
| **Self-Healing Selectors** | ✅ Adaptive Tracking | ❌ Static | ❌ Static | ❌ Static |

---

## 3. How Scrapling Fits into AcadRAG (Phase 20)

In AcadRAG, Scrapling can act as the primary local scraper for conference platforms that lack REST APIs (such as OpenReview.net, NeurIPS, and CVPR/thecvf.com):

```
                       [ Academic Query ]
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
[ ArXiv REST API ]   [ Semantic Scholar API ]   [ Scrapling Scraper ]
(Preprints)          (200M+ Papers API)         (OpenReview / NeurIPS)
       │                       │                       │
       └───────────────────────┴───────────────────────┘
                               │
                               ▼
            [ Unified Deduplication & LLM Scoring ]
```
