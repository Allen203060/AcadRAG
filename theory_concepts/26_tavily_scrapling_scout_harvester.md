# Concept 26: The Scout + Harvester Crawler Paradigm (Tavily + Scrapling)

## 1. The Architectural Divide
- **Discovery (The Scout):** An AI search engine (`TavilyClient`) queries the live internet to discover URLs based on natural language queries. Constrained to academic domains (`include_domains=["openreview.net", "proceedings.neurips.cc", "openaccess.thecvf.com"]`).
- **Harvesting (The Harvester):** A local stealth web scraper (`Scrapling`) receives target URLs, executes locally via Camoufox, bypasses Cloudflare Turnstile, and extracts PDFs or converts DOM to Markdown (`page.markdown()`) for $0.

## 2. Why This Beats Pure SaaS Crawlers
- **Cost Efficiency:** Using Firecrawl or Tavily's raw content extraction across hundreds of pages rapidly depletes paid API credits. The Scout + Harvester pattern consumes only 1 lightweight search credit on Tavily, offloading all subsequent DOM rendering, JS hydration, and Markdown parsing to local hardware.
- **Data Privacy & Governance:** Full papers, extracted tables, and DOM trees are processed strictly on your local machine, keeping research topics and candidate papers private.
- **Resilience:** If a site lacks direct PDF links, native HTML-to-Markdown extraction acts as an immediate fallback for full text ingestion.
