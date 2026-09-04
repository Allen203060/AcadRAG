import os
import re
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from sentence_transformers import CrossEncoder

def reciprocal_rank_fusion(results_list: list[list[str]], k: int = 60) -> list[str]:
    """
    Combines multiple ranked document lists into a single ranked list using Reciprocal Rank Fusion (RRF).
    RRF Score = sum(1 / (k + rank))
    """
    rrf_scores = {}
    for ranked_list in results_list:
        for rank, doc in enumerate(ranked_list, start=1):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0.0
            rrf_scores[doc] += 1.0 / (k + rank)
            
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc for doc, score in sorted_docs]

def hybrid_search(query: str):
    # --- SETUP CONNECTIONS ---
    print(f"\n--- Initiating Hybrid Search for Query: '{query}' ---")
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password"

    graph = Neo4jGraph()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Connect to the existing Milvus collection
    vector_db = Milvus(
        embedding_function=embeddings,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        collection_name="academic_papers"
    )

    # --- STEP 1: VECTOR SEARCH ---
    print("1. Querying Milvus Vector Database...")
    vector_results = vector_db.similarity_search(query, k=10)
    vector_context = [doc.page_content for doc in vector_results]

    # --- STEP 2: GRAPH SEARCH ---
    print("2. Traversing Neo4j Knowledge Graph...")
    raw_words = query.split()
    keywords = [re.sub(r'[^\w-]', '', w).lower() for w in raw_words]
    keywords = [w for w in keywords if len(w) > 3]
    
    graph_context = []
    cypher_query = """
    MATCH (n)-[r]->(m) 
    WHERE toLower(n.id) CONTAINS $keyword OR toLower(m.id) CONTAINS $keyword
    RETURN n.id as source, type(r) as relationship, m.id as target LIMIT 5
    """
    
    for word in set(keywords):
        results = graph.query(cypher_query, params={"keyword": word})
        for res in results:
            graph_context.append(f"{res['source']} -> {res['relationship']} -> {res['target']}")
            
    unique_graph_context = list(set(graph_context))
    print(f"Retrieved {len(vector_context)} vector chunks and {len(unique_graph_context)} graph relationships.")

    # --- STEP 3: RECIPROCAL RANK FUSION (RRF) ---
    print("3. Fusing Vector and Graph ranks via RRF (k=60)...")
    fused_candidates = reciprocal_rank_fusion([vector_context, unique_graph_context], k=60)
    print(f"Fused candidate pool size: {len(fused_candidates)}")

    # --- STEP 4: CROSS-ENCODER RERANKING ---
    print("4. Scoring fused candidates with BGE-Reranker (Cross-Encoder)...")
    reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
    
    pairs = [[query, doc] for doc in fused_candidates]
    scores = reranker.predict(pairs)
    
    scored_docs = sorted(zip(fused_candidates, scores), key=lambda x: x[1], reverse=True)
    top_5 = scored_docs[:5]
    
    print("\n--- Final Top 5 Reranked Contexts ---")
    for i, (doc, score) in enumerate(top_5, 1):
        print(f"\n[{i}] Relevance Score: {score:.2f}")
        print(f"{str(doc)[:250]}...")
        
    return [doc for doc, score in top_5]

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    test_query = "What is the Transformer architecture and Multi-Head Attention?"
    hybrid_search(test_query)
