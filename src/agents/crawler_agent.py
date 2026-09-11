import os
import re
import time
import requests
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, TypedDict

from langsmith import traceable
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from scrapling.fetchers import Fetcher, StealthyFetcher

from src.core.llm_factory import get_llm
from src.ingestion.pdf_loader import extract_pdf_with_docling
from src.ingestion.populate import populate_databases
from src.core.graph import run_rag_pipeline

# ==============================================================================
# 1. State Schema Definition
# ==============================================================================
class CrawlerAgentState(TypedDict):
    query: str
    max_search_results: int
    top_k: int
    discovered_urls: List[Dict[str, Any]]
    shortlisted_urls: List[Dict[str, Any]]
    harvested_files: List[str]
    synthesis_report: str

# ==============================================================================
# 2. Node 1: Tavily Web Scout (Targeted Discovery)
# ==============================================================================
@traceable(name="Tavily Scout Node", run_type="chain")
def tavily_scout_node(state: CrawlerAgentState) -> Dict[str, Any]:
    """
    Acts as the Scout: Queries Tavily search index focused on academic 
    and open-access conference proceedings (OpenReview, NeurIPS, CVF, arXiv).
    """
    query = state["query"]
    max_results = state.get("max_search_results", 5)
    print(f"\n--- [Node 1: Tavily Scout] Searching live web for academic resources: '{query}' ---")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("⚠️ TAVILY_API_KEY not found in environment. Falling back to targeted domain query.")
        return {"discovered_urls": []}

    client = TavilyClient(api_key=api_key)
    
    # Target high-value academic and preprint repositories
    academic_domains = [
        "openreview.net",
        "proceedings.neurips.cc",
        "openaccess.thecvf.com",
        "arxiv.org",
        "biorxiv.org"
    ]

    try:
        response = client.search(
            query=f"{query} research paper",
            search_depth="advanced",
            include_domains=academic_domains,
            max_results=max_results
        )
        
        candidates = []
        for res in response.get("results", []):
            candidates.append({
                "title": res.get("title", "Untitled"),
                "url": res.get("url"),
                "snippet": res.get("content", ""),
                "score": res.get("score", 0.0)
            })
            
        print(f"✅ Tavily discovered {len(candidates)} high-authority academic URLs.")
        return {"discovered_urls": candidates}
    except Exception as e:
        print(f"⚠️ Tavily search failed: {e}")
        return {"discovered_urls": []}

# ==============================================================================
# 2. Node 2: LLM Relevance Filter
# ==============================================================================
@traceable(name="Filter URLs Node", run_type="chain")
def filter_urls_node(state: CrawlerAgentState) -> Dict[str, Any]:
    """
    Uses the configured LLM to evaluate the relevance of discovered web snippets.
    """
    query = state["query"]
    discovered = state.get("discovered_urls", [])
    top_k = state.get("top_k", 3)

    if not discovered:
        print("⚠️ No discovered URLs to filter.")
        return {"shortlisted_urls": []}

    print(f"\n--- [Node 2: Filter URLs] Scoring {len(discovered)} Candidate URLs with LLM ---")
    llm = get_llm(temperature=0.0)
    scored_items = []

    for idx, item in enumerate(discovered, 1):
        prompt = f"""You are an academic research reviewer selecting relevant papers and conference pages.
User Query: "{query}"
Page Title: {item['title']}
URL: {item['url']}
Page Snippet: {item['snippet']}

Rate the page's relevance to the user's research topic from 0 to 100.
Return strictly valid JSON with two keys: "score" (integer) and "reason" (short string).

JSON Response:"""
        try:
            res = llm.invoke(prompt)
            clean_json = re.sub(r'```json\s*|\s*```', '', res.content.strip()).strip()
            import json
            parsed = json.loads(clean_json)
            score = int(parsed.get("score", 50))
            reason = parsed.get("reason", "Relevant candidate")
        except Exception:
            score = 50
            reason = "Default fallback score"

        scored_items.append({
            **item,
            "relevance_score": score,
            "eval_reason": reason
        })
        print(f" [{idx}/{len(discovered)}] Score: {score}/100 | {item['title'][:50]}...")

    shortlist = sorted(scored_items, key=lambda x: x["relevance_score"], reverse=True)[:top_k]
    print(f"✅ Shortlisted Top {len(shortlist)} Web Candidates.")
    return {"shortlisted_urls": shortlist}

# ==============================================================================
# 3. Node 3: Scrapling Harvester (Local Anti-Bot Scraper)
# ==============================================================================
@traceable(name="Scrapling Harvester Node", run_type="chain")
def scrapling_harvester_node(state: CrawlerAgentState) -> Dict[str, Any]:
    """
    Acts as the Harvester: Takes shortlisted URLs, visits them with Scrapling,
    extracts PDF download links, or dumps clean LLM-ready Markdown for $0.
    """
    shortlist = state.get("shortlisted_urls", [])
    if not shortlist:
        print("🛑 No URLs to harvest.")
        return {"harvested_files": []}

    print("\n" + "="*70)
    print("🔒 [HITL CHECKPOINT 1] WEB CANDIDATES SHORTLISTED FOR HARVESTING")
    print("="*70)
    for i, item in enumerate(shortlist, 1):
        print(f" [{i}] Score: {item['relevance_score']}/100")
        print(f"     Title: {item['title']}")
        print(f"     URL:   {item['url']}")
        print(f"     Note:  {item['eval_reason']}\n")

    confirm = input("🌐 Proceed with harvesting these pages via Scrapling? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("🛑 Harvesting aborted by user.")
        return {"harvested_files": []}

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    harvested_files = []

    for item in shortlist:
        url = item["url"]
        safe_title = re.sub(r'[^\w\-_\. ]', '_', item['title'])[:50].strip()
        print(f"\n🌾 Harvesting: {item['title'][:50]} ({url})...")
        
        # Polite rate-limiting delay
        time.sleep(1.5)

        try:
            # Case 1: Direct PDF URL candidate
            if url.lower().endswith(".pdf") or "/pdf/" in url.lower():
                print(f"   📄 Direct PDF Link Detected. Downloading binary...")
                pdf_filename = f"{safe_title}.pdf"
                pdf_dest = os.path.join(data_dir, pdf_filename)
                
                if not os.path.exists(pdf_dest):
                    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                    r.raise_for_status()
                    with open(pdf_dest, "wb") as f:
                        f.write(r.content)
                harvested_files.append(pdf_dest)
                print(f"   ✅ Saved PDF to: {pdf_filename}")
                continue

            # Case 2: Web HTML Page (Use Scrapling)
            is_dynamic = "openreview.net" in url
            page = None

            if is_dynamic:
                try:
                    print("   ↳ Attempting StealthyFetcher (Camoufox engine)...")
                    page = StealthyFetcher.fetch(url)
                except Exception as browser_err:
                    print(f"   ⚠️ StealthyFetcher browser unavailable ({browser_err}). Falling back to Fast Fetcher...")
                    page = Fetcher.get(url)
            else:
                print("   ↳ Using Fast Fetcher (Direct HTTP)...")
                page = Fetcher.get(url)

            # Search for PDF download links in DOM
            pdf_link = None
            for a in page.css("a"):
                href = a.attrib.get("href", "")
                if href.lower().endswith(".pdf") or "/pdf/" in href.lower():
                    pdf_link = urljoin(url, href)
                    break

            if pdf_link:
                print(f"   📄 Located PDF Link in DOM: {pdf_link}")
                pdf_filename = f"{safe_title}.pdf"
                pdf_dest = os.path.join(data_dir, pdf_filename)
                
                if not os.path.exists(pdf_dest):
                    r = requests.get(pdf_link, timeout=30)
                    r.raise_for_status()
                    with open(pdf_dest, "wb") as f:
                        f.write(r.content)
                harvested_files.append(pdf_dest)
            else:
                # Convert DOM directly to Markdown
                print("   📝 No direct PDF found. Converting DOM directly to Markdown...")
                md_filename = f"{safe_title}.md"
                md_dest = os.path.join(data_dir, md_filename)
                
                try:
                    markdown_content = page.markdown()
                except Exception as md_err:
                    print(f"   ⚠️ Markdown conversion fallback ({md_err}). Using raw text...")
                    markdown_content = getattr(page, 'text', '') or str(page)

                with open(md_dest, "w", encoding="utf-8") as f:
                    f.write(f"# {item['title']}\n\nSource: {url}\n\n{markdown_content}")
                harvested_files.append(md_dest)
                print(f"   ✅ Saved clean Markdown to: {md_filename}")

        except Exception as e:
            print(f"   ⚠️ Failed to harvest {url}: {e}")

    # HITL Gate 2: Ingestion & Database Population
    if harvested_files:
        print("\n" + "="*70)
        print("🔒 [HITL CHECKPOINT 2] INGESTION & HYBRID DATABASE POPULATION")
        print("="*70)
        print(f"Discovered {len(harvested_files)} new harvested files in ./data/.")
        confirm_ingest = input("⚡ Run Docling extraction and populate Milvus + Neo4j? [Y/n]: ").strip().lower()
        if not confirm_ingest or confirm_ingest == 'y':
            # Run Docling on newly downloaded PDFs
            for fpath in harvested_files:
                if fpath.endswith(".pdf"):
                    base = os.path.splitext(fpath)[0]
                    md_target = f"{base}.md"
                    print(f"📄 Parsing {os.path.basename(fpath)} with Docling...")
                    extract_pdf_with_docling(fpath, md_target)
            
            print("⚡ Updating Milvus Vector DB & Neo4j Knowledge Graph...")
            populate_databases()

    return {"harvested_files": harvested_files}


# ==============================================================================
# 4. Node 4: Synthesis Node
# ==============================================================================
@traceable(name="Web Research Synthesis Node", run_type="chain")
def crawler_synthesize_node(state: CrawlerAgentState) -> Dict[str, Any]:
    query = state["query"]
    print(f"\n--- [Node 4: Synthesize] Synthesizing Harvested Content for: '{query}' ---")
    synthesis_prompt = f"Synthesize the key findings, methodologies, and benchmarks from the newly crawled research documents on: {query}"
    result = run_rag_pipeline(synthesis_prompt)
    report = result["answer"]
    
    print("\n" + "="*70)
    print("🎯 CRAWLER AGENT SYNTHESIS REPORT")
    print("="*70)
    print(report)
    print("="*70 + "\n")
    return {"synthesis_report": report}

# ==============================================================================
# 5. Build the LangGraph StateMachine
# ==============================================================================
def build_crawler_agent_graph():
    graph = StateGraph(CrawlerAgentState)

    graph.add_node("tavily_scout", tavily_scout_node)
    graph.add_node("filter_urls", filter_urls_node)
    graph.add_node("scrapling_harvest", scrapling_harvester_node)
    graph.add_node("synthesize", crawler_synthesize_node)

    graph.set_entry_point("tavily_scout")
    graph.add_edge("tavily_scout", "filter_urls")
    graph.add_edge("filter_urls", "scrapling_harvest")
    graph.add_edge("scrapling_harvest", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()

@traceable(name="Web Crawler Agent", run_type="chain")
def run_web_crawler_agent(query: str, max_search_results: int = 5, top_k: int = 2):
    app = build_crawler_agent_graph()
    initial_state: CrawlerAgentState = {
        "query": query,
        "max_search_results": max_search_results,
        "top_k": top_k,
        "discovered_urls": [],
        "shortlisted_urls": [],
        "harvested_files": [],
        "synthesis_report": ""
    }
    return app.invoke(initial_state)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    test_query = "Mamba-2 State Space Models NeurIPS 2024"
    run_web_crawler_agent(test_query, max_search_results=3, top_k=2)
