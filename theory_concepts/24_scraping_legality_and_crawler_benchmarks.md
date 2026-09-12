# Concept 24: Web Scraping Legality & Crawler Benchmarks (Scrapling vs. Cloud APIs)

## 1. Statutory & Case Law Framework
1. **CFAA & Public Data Access:** Under *hiQ Labs v. LinkedIn* (9th Cir. 2022) and *Van Buren v. US* (Supreme Court 2021), automated retrieval of publicly accessible data not behind a password/login gate does not violate the Computer Fraud and Abuse Act (CFAA).
2. **Transformative Fair Use (17 U.S.C. § 107):** Indexing text for vector embeddings and Knowledge Graph synthesis is non-expressive, transformative fair use (*Authors Guild v. Google, Inc.*, 2015). It produces an algorithmic index and factual citations rather than an expressive reading substitute.
3. **EU TDM Exceptions (Directive 2019/790):** Articles 3 & 4 provide statutory exceptions permitting Text and Data Mining (TDM) for scientific research across legally accessible works.
4. **Open Access Licenses (CC-BY):** Academic venues such as ArXiv, OpenReview, NeurIPS, and CVPR publish under Creative Commons licenses (CC-BY 4.0), which explicitly grant permission to download, read, and index documents.

## 2. Compliance Checklist for Ethical Scraping
- ❌ **Never circumvent paywalls (Elsevier, IEEE):** Violates DMCA § 1201 (anti-circumvention of technological measures).
- ❌ **Never publicly redistribute raw copyrighted PDF files:** Violates copyright distribution rights (Sci-Hub style).
- ✅ **Always parse and respect `robots.txt` directives:** Follow crawl-delay and disallow paths.
- ✅ **Implement mandatory rate-limiting:** Enforce a polite 1.5s–2.0s delay between requests to eliminate server overburden ("Trespass to Chattels").
- ✅ **Provide a transparent `User-Agent` header:** Include project name and contact information.

## 3. Scrapling vs. Tavily vs. Firecrawl Matrix

| Metric | Scrapling | Firecrawl API | Tavily API |
| :--- | :--- | :--- | :--- |
| **Category** | Local Python Engine | Cloud Crawler SaaS | Cloud AI Search Engine |
| **Cost** | $0.00 (Local / Open-Source) | Paid per-page credits | Paid per-query credits |
| **Speed** | Sub-second (HTTP) / ~2s (Browser) | 3s–7s (Cloud browser) | 1.5s–3s |
| **Anti-Bot Bypass** | Local Camoufox (TLS/JA3 spoofing) | Cloud residential proxy rotation | Abstracted search engine |
| **IP Footprint** | Local IP (requires throttle) | Rotating pool of cloud IPs | Rotating cloud IPs |
| **Best Used For** | Conference paper archives & crawling | Deep domain-wide scraping | Open-ended web discovery |
