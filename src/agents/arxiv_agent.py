import os
import re
import json
import arxiv
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

# 2. Node 1: Search ArXiv API
@traceable(name="Search ArXiv Node", run_type="chain")
def search_arxiv_node(state: ArxivAgentState) -> Dict[str, Any]:
    topic = state["topic"]
    max_results = state.get("max_results", 15)
    print(f"\n--- [Node 1: search_arxiv_node] Searching ArXiv for: '{topic}' ---")
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    candidates = []
    for paper in client.results(search):
        candidates.append({
            "title": paper.title,
            "summary": paper.summary.replace("\n", " "),
            "pdf_url": paper.pdf_url,
            "entry_id": paper.entry_id,
            "authors": [a.name for a in paper.authors]
        })

    print(f"✅ Fetched {len(candidates)} candidate abstracts.")
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
    print(f"\n--- [Node 3: download_ingest_node] Ingesting {len(shortlist)} Shortlisted PDFs ---")
    
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    search_client = arxiv.Client()

    for paper in shortlist:
        safe_title = re.sub(r'[^\w\-_\. ]', '_', paper['title'])[:50].strip()
        pdf_path = os.path.join(data_dir, f"{safe_title}.pdf")
        md_path = os.path.join(data_dir, f"{safe_title}.md")

        if not os.path.exists(pdf_path):
            print(f"📥 Downloading PDF: {paper['title'][:40]}...")
            paper_obj = next(search_client.results(arxiv.Search(id_list=[paper['entry_id'].split('/')[-1]])))
            paper_obj.download_pdf(dirpath=data_dir, filename=f"{safe_title}.pdf")
            
        print(f"📄 Running Docling DOM Conversion on {safe_title}.pdf...")
        extract_pdf_with_docling(pdf_path, md_path)

    print("⚡ Populating Milvus Vector DB & Neo4j Knowledge Graph...")
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

