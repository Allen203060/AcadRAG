import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from sentence_transformers import CrossEncoder
from src.core.retriever import hybrid_search
from src.core.llm_factory import get_llm

# 1. Define the Shared State Schema
class GraphState(TypedDict):
    question: str
    raw_documents: List[str]
    reranked_documents: List[str]
    answer: str

# 2. Node 1: Hybrid Retrieval (Milvus + Neo4j)
def retrieve_node(state: GraphState) -> dict:
    query = state["question"]
    raw_docs = hybrid_search(query)
    return {"raw_documents": raw_docs}

# 3. Node 2: Cross-Encoder Reranking
def rerank_node(state: GraphState) -> dict:
    query = state["question"]
    raw_docs = state.get("raw_documents", [])
    
    if not raw_docs:
        return {"reranked_documents": []}
        
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    pairs = [[query, doc] for doc in raw_docs]
    scores = reranker.predict(pairs)
    
    scored_docs = sorted(zip(raw_docs, scores), key=lambda x: x[1], reverse=True)
    top_5 = [doc for doc, score in scored_docs[:5]]
    
    return {"reranked_documents": top_5}

# 4. Node 3: Grounded Answer Generation
def generate_node(state: GraphState) -> dict:
    query = state["question"]
    docs = state.get("reranked_documents", [])
    
    if not docs:
        return {"answer": "I cannot answer this based on the provided documents."}
        
    context_str = ""
    for i, doc in enumerate(docs, 1):
        context_str += f"[Source {i}]:\n{doc}\n\n"
        
    prompt = f"""You are a strict academic assistant. Answer the user's question using ONLY the provided context below. 
If the answer cannot be deduced from the context, say exactly: "I cannot answer this based on the provided documents."
For EVERY factual claim you make, you MUST cite the source inline using brackets matching the source label, for example: [Source 1].

Context:
{context_str}

Question: {query}
Answer:"""

    llm = get_llm(temperature=0)
    response = llm.invoke(prompt)
    return {"answer": response.content}

# 5. Assemble & Compile the LangGraph Workflow
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

def run_rag_pipeline(question: str) -> dict:
    """Helper entry point to execute the compiled LangGraph pipeline."""
    initial_state = {"question": question, "raw_documents": [], "reranked_documents": [], "answer": ""}
    final_state = app.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    res = run_rag_pipeline("What is Multi-Head Attention?")
    print("\n--- Answer ---")
    print(res["answer"])
