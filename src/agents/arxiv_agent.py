import os
import re
import json
import arxiv
import requests 
from typing import List, Dict, Any, TypedDict
from langsmith import traceable
from langgraph.graph import StateGraph, END
from src.core.llm_factory import get_llm
from src.ingestion.pdf_loader import extract_pdf_with_docling
from src.ingestion.populate import populate_databases
from src.core.graph import run_rag_pipeline

# 1. Define Agent State Schema
class ArxivAgentState(TypedDict):
    topic: str
    max_results: int
    top_k: int
    candidates: List[Dict[str, Any]]
    shortlist: List[Dict[str, Any]]
    synthesis_report: str

# ==============================================================================
# Node 1: Multi-Source Discovery (ArXiv REST API + Semantic Scholar Graph API)
# ==============================================================================

def _normalize_title(title: str) -> str:
    """Normalizes titles by stripping punctuation and whitespace for deduplication."""
    return re.sub(r'[\W_]+', '', title).lower()
def _fetch_semantic_scholar(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Queries the Semantic Scholar Academic Graph API.
    Retrieves metadata, citation metrics, and direct open-access PDF links.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,openAccessPdf,citationCount,year,url,externalIds"
    }
    headers = {
        "User-Agent": "AcadRAG-Research-Agent/1.0 (mailto:researcher@acadrag.local)"
    }
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"⚠️ Semantic Scholar API returned status {response.status_code}: {response.text[:100]}")
            return []
        
        data = response.json()
        results = []
        for paper in data.get("data", []):
            # Only include papers with an abstract and a legally accessible open-access PDF
            pdf_info = paper.get("openAccessPdf")
            pdf_url = pdf_info.get("url") if pdf_info else None
            abstract = paper.get("abstract") or ""
            if not abstract:
                continue
            # Check if paper originates from PubMed or PMC
            ext_ids = paper.get("externalIds") or {}
            pmid = ext_ids.get("PubMed")
            pmcid = ext_ids.get("PubMedCentral")

            if not pdf_url and pmcid:
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
            source_tag = "Semantic Scholar"
            if pmid:
                source_tag = f"PubMed (PMID:{pmid})"
            elif pmcid:
                source_tag = f"PubMed Central ({pmcid})"

            results.append({
                "title": paper.get("title", "Untitled"),
                "summary": abstract.replace("\n", " "),
                "pdf_url": pdf_url,
                "entry_id": paper.get("paperId", paper.get("url")),
                "authors": [a.get("name") for a in paper.get("authors", [])],
                "source": source_tag,
                "citation_count": paper.get("citationCount", 0),
                "year": paper.get("year", "N/A")
            })
        return results
    except Exception as e:
        print(f"⚠️ Semantic Scholar request failed: {e}")
        return []

def _fetch_pubmed(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Queries NCBI Entrez E-utilities API for PubMed Central (PMC) papers.
    1. esearch: Resolves query keywords to PMC IDs with [Title/Abstract] relevance constraints.
    2. esummary & efetch: Retrieves exact titles, genuine scientific abstracts, and direct PDF URLs.
    """
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    # 1. Clean query for NCBI boolean syntax and restrict to Title/Abstract
    clean_query = re.sub(r'[^\w\s-]', ' ', query).strip()
    search_term = f'("{clean_query}"[Title/Abstract] OR ({clean_query})[Title/Abstract]) AND open access[filter]'

    params_search = {
        "db": "pmc",
        "term": search_term,
        "sort": "relevance",
        "retmode": "json",
        "retmax": limit
    }
    
    try:
        # Step 1: Find high-relevance PMC IDs
        res = requests.get(esearch_url, params=params_search, timeout=15)
        if res.status_code != 200:
            return []
        
        id_list = res.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            # Fallback: broaden search if exact phrase yielded 0 results
            params_search["term"] = f'({clean_query})[Title/Abstract] AND open access[filter]'
            res = requests.get(esearch_url, params=params_search, timeout=15)
            id_list = res.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

        # Step 2: Fetch metadata summaries
        params_summary = {
            "db": "pmc",
            "id": ",".join(id_list),
            "retmode": "json"
        }
        sum_res = requests.get(esummary_url, params=params_summary, timeout=15)
        summary_data = sum_res.json().get("result", {}) if sum_res.status_code == 200 else {}

        # Step 3: Fetch real abstracts via efetch XML
        params_fetch = {
            "db": "pmc",
            "id": ",".join(id_list),
            "retmode": "xml"
        }
        fetch_res = requests.get(efetch_url, params=params_fetch, timeout=20)
        abstracts = {}
        if fetch_res.status_code == 200:
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(fetch_res.content)
                for article in root.findall(".//article"):
                    # Find PMC ID
                    pmc_elem = article.find(".//article-id[@pub-id-type='pmc']")
                    if pmc_elem is not None and pmc_elem.text:
                        pid = pmc_elem.text.replace("PMC", "")
                        # Find Abstract Text
                        abst_elem = article.find(".//abstract")
                        if abst_elem is not None:
                            abstract_text = "".join(abst_elem.itertext()).strip()
                            abstracts[pid] = abstract_text
            except Exception:
                pass

        results = []
        for pmc_id in id_list:
            item = summary_data.get(pmc_id, {})
            title = item.get("title", "Untitled")
            
            # Use real abstract if parsed, else fallback to descriptive title
            real_abstract = abstracts.get(pmc_id) or f"Biomedical paper on {title}. Published in {item.get('source', 'PMC')}."
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"
            
            results.append({
                "title": title,
                "summary": real_abstract.replace("\n", " "),
                "pdf_url": pdf_url,
                "entry_id": f"PMC{pmc_id}",
                "authors": [a.get("name") for a in item.get("authors", [])],
                "source": "PubMed Central",
                "citation_count": None,
                "year": item.get("pubdate", "").split()[0] if item.get("pubdate") else "N/A"
            })
            
        return results
    except Exception as e:
        print(f"⚠️ PubMed fetch failed: {e}")
        return []


@traceable(name="Multi-Source Paper Discovery Node", run_type="chain")
def search_arxiv_node(state: ArxivAgentState) -> Dict[str, Any]:
    topic = state["topic"]
    max_results = state.get("max_results", 15)
    print(f"\n--- [Node 1: Multi-Source Discovery] Searching ArXiv & Semantic Scholar for: '{topic}' ---")
    
    candidates: List[Dict[str, Any]] = []
    seen_titles = set()
    # 1. Fetch from ArXiv REST API
    try:
        print("🔍 Querying ArXiv REST API...")
        client = arxiv.Client()
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        for paper in client.results(search):
            norm_title = _normalize_title(paper.title)
            if norm_title not in seen_titles:
                seen_titles.add(norm_title)
                candidates.append({
                    "title": paper.title,
                    "summary": paper.summary.replace("\n", " "),
                    "pdf_url": paper.pdf_url,
                    "entry_id": paper.entry_id,
                    "authors": [a.name for a in paper.authors],
                    "source": "ArXiv",
                    "citation_count": None,
                    "year": paper.published.year if hasattr(paper, 'published') else "N/A"
                })
        print(f"   ↳ Fetched {len(candidates)} candidates from ArXiv.")
    except Exception as e:
        print(f"⚠️ ArXiv fetch failed: {e}")
    # 2. Fetch from Semantic Scholar Academic Graph API
    print("🔍 Querying Semantic Scholar Academic Graph API...")
    s2_papers = _fetch_semantic_scholar(topic, limit=max_results)
    added_s2 = 0
    for paper in s2_papers:
        norm_title = _normalize_title(paper["title"])
        if norm_title not in seen_titles:
            seen_titles.add(norm_title)
            candidates.append(paper)
            added_s2 += 1
    print(f"   ↳ Fetched {added_s2} unique candidates from Semantic Scholar.")
    
    # 3. Fetch from PubMed Central (Open Access Biomedical)
    print("🔍 Querying PubMed Central API for Open Access Biomedical papers...")
    pmc_papers = _fetch_pubmed(topic, limit=max_results)
    added_pmc = 0
    for paper in pmc_papers:
        norm_title = _normalize_title(paper["title"])
        if norm_title not in seen_titles:
            seen_titles.add(norm_title)
            candidates.append(paper)
            added_pmc += 1
    print(f"   ↳ Fetched {added_pmc} unique candidates from PubMed Central.")
    print(f"✅ Total deduplicated candidates across all three platforms: {len(candidates)}")
    
    return {"candidates": candidates}

# 3. Node 2: LLM Abstract Scoring & Shortlisting
@traceable(name="Score Abstracts Node", run_type="chain")
def score_abstracts_node(state: ArxivAgentState) -> Dict[str, Any]:
    topic = state["topic"]
    candidates = state["candidates"]
    top_k = state.get("top_k", 3)
    
    print(f"\n--- [Node 2: score_abstracts_node] LLM Scoring {len(candidates)} Candidates ---")
    llm = get_llm(temperature=0.0)
    scored_papers = []

    for idx, paper in enumerate(candidates, 1):
        prompt = f"""You are a strict research paper reviewer evaluating relevance.
                    User Target Topic: "{topic}"
                    Candidate Paper Title: {paper['title']}
                    Abstract: {paper['summary']}

                    Rate the paper's direct relevance to the user's target topic on a scale of 0 to 100.
                    Provide your response strictly as valid JSON with two keys: "score" (integer) and "reason" (short string).

                    JSON Response:"""

        try:
            res = llm.invoke(prompt)
            raw_text = res.content.strip()
            clean_json = re.sub(r'```json\s*|\s*```', '', raw_text).strip()
            eval_data = json.loads(clean_json)
            
            score = int(eval_data.get("score", 0))
            reason = eval_data.get("reason", "No reason provided")
        except Exception:
            score = 50
            reason = "Default fallback score due to parsing exception."

        scored_papers.append({
            **paper,
            "relevance_score": score,
            "eval_reason": reason
        })
        print(f" [{idx}/{len(candidates)}] Score: {score}/100 | {paper['title'][:50]}...")

    shortlist = sorted(scored_papers, key=lambda x: x["relevance_score"], reverse=True)[:top_k]
    print(f"✅ Shortlisted Top {len(shortlist)} Papers.")
    return {"shortlist": shortlist}

# 4. Node 3: PDF Download & Multi-Modal Ingestion
@traceable(name="Download & Ingest Node", run_type="chain")
def download_ingest_node(state: ArxivAgentState) -> Dict[str, Any]:
    shortlist = state["shortlist"]
    
    print("\n" + "="*70)
    print("🔒 [HITL CHECKPOINT 1] SHORTLISTED RESEARCH PAPERS FOR DOWNLOAD")
    print("="*70)
    for i, paper in enumerate(shortlist, 1):
        print(f" [{i}] Score: {paper['relevance_score']}/100")
        print(f"     Title:  {paper['title']}")
        print(f"     URL:    {paper['pdf_url']}")
        print(f"     Reason: {paper['eval_reason']}\n")

    # HITL Gate 1: Confirm PDF Download
    confirm_download = input("📥 Proceed with downloading these PDFs? [Y/n]: ").strip().lower()
    if confirm_download and confirm_download != 'y':
        print("🛑 Download aborted by user.")
        return {"shortlist": []}
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    # Download PDFs
    for paper in shortlist:
        if not paper.get('pdf_url'):
            print(f"⚠️ Skipping '{paper['title'][:40]}...' (No open-access PDF URL available).")
            continue    
        safe_title = re.sub(r'[^\w\-_\. ]', '_', paper['title'])[:50].strip()
        pdf_path = os.path.join(data_dir, f"{safe_title}.pdf")
        if not os.path.exists(pdf_path):
            print(f"📥 Downloading PDF from: {paper['pdf_url']}...")
            res = requests.get(paper['pdf_url'], timeout=30)
            res.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(res.content)
        
    print("\n" + "="*70)
    print("🔒 [HITL CHECKPOINT 2] DOCLING DOM LAYOUT EXTRACTION")
    print("="*70)
    
    # HITL Gate 2: Confirm Docling DOM Extraction
    confirm_docling = input("📄 Proceed with Docling DOM PDF-to-Markdown parsing? [Y/n]: ").strip().lower()
    if confirm_docling and confirm_docling != 'y':
        print("🛑 Docling extraction aborted by user.")
        return {}
    for paper in shortlist:
        safe_title = re.sub(r'[^\w\-_\. ]', '_', paper['title'])[:50].strip()
        pdf_path = os.path.join(data_dir, f"{safe_title}.pdf")
        md_path = os.path.join(data_dir, f"{safe_title}.md")
        print(f"📄 Running Docling DOM Conversion on {safe_title}.pdf...")
        extract_pdf_with_docling(pdf_path, md_path)

    print("\n" + "="*70)
    print("🔒 [HITL CHECKPOINT 3] HYBRID DATABASE POPULATION (MILVUS + NEO4J)")
    print("="*70)
    
    # HITL Gate 3: Confirm DB Population
    confirm_db = input("⚡ Populate Milvus Vector DB & Neo4j Knowledge Graph? [Y/n]: ").strip().lower()
    if confirm_db and confirm_db != 'y':
        print("🛑 Database population aborted by user.")
        return {}
    populate_databases()
    return {}

# 5. Node 4: GraphRAG Synthesis
@traceable(name="Synthesize Node", run_type="chain")
def synthesize_node(state: ArxivAgentState) -> Dict[str, Any]:
    topic = state["topic"]
    print(f"\n--- [Node 4: synthesize_node] Deep GraphRAG Synthesis ---")
    synthesis_query = f"Synthesize the key methodologies, findings, and architectures across these shortlisted papers regarding: {topic}"
    graph_res = run_rag_pipeline(synthesis_query)
    
    report = graph_res["answer"]
    print("\n==================== RESEARCH SYNTHESIS REPORT ====================")
    print(report)
    print("===================================================================\n")
    return {"synthesis_report": report}

# 6. Build the LangGraph StateGraph Workflow
def build_arxiv_agent_graph():
    workflow = StateGraph(ArxivAgentState)

    # Add Nodes
    workflow.add_node("search_arxiv", search_arxiv_node)
    workflow.add_node("score_abstracts", score_abstracts_node)
    workflow.add_node("download_ingest", download_ingest_node)
    workflow.add_node("synthesize", synthesize_node)

    # Add Edges
    workflow.set_entry_point("search_arxiv")
    workflow.add_edge("search_arxiv", "score_abstracts")
    workflow.add_edge("score_abstracts", "download_ingest")
    workflow.add_edge("download_ingest", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()

# Master Execution Entrypoint
@traceable(name="ArXiv Research Agent Graph", run_type="chain")
def search_arxiv_and_shortlist(topic: str, max_results: int = 15, top_k: int = 3) -> Dict[str, Any]:
    app = build_arxiv_agent_graph()
    initial_state = {
        "topic": topic,
        "max_results": max_results,
        "top_k": top_k,
        "candidates": [],
        "shortlist": [],
        "synthesis_report": ""
    }
    return app.invoke(initial_state)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    search_arxiv_and_shortlist("Face Recognition on IoT Edge", max_results=10, top_k=2)

