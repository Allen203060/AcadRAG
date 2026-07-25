import os
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from sentence_transformers import CrossEncoder



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
    # Retrieve top 10 most similar semantic chunks
    vector_results = vector_db.similarity_search(query, k=10)
    vector_context = [doc.page_content for doc in vector_results]

    # --- STEP 2: GRAPH SEARCH ---
    print("2. Traversing Neo4j Knowledge Graph...")
    # A naive approach: Extract words from query (ignoring small words) and search Neo4j nodes
    keywords = [w.lower() for w in query.replace("?", "").split() if len(w) > 3]
    graph_context = []
    
    for word in keywords:
        # Cypher query: Find any node containing the keyword, and return its relationships
        cypher_query = f"""
        MATCH (n)-[r]->(m) 
        WHERE toLower(n.id) CONTAINS '{word}' OR toLower(m.id) CONTAINS '{word}'
        RETURN n.id as source, type(r) as relationship, m.id as target LIMIT 5
        """
        results = graph.query(cypher_query)
        for res in results:
            graph_context.append(f"{res['source']} -> {res['relationship']} -> {res['target']}")

    # Merge context and remove duplicates
    combined_docs = vector_context + list(set(graph_context))
    print(f"Merged {len(vector_context)} chunks and {len(set(graph_context))} graph relationships.")

    # --- STEP 3: RERANKING ---
    print("3. Scoring context with BGE-Reranker (Cross-Encoder)...")
    # Initialize the Reranker (use_fp16=True saves memory)
    reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
    
    # The reranker expects a list of pairs: [[query, doc1], [query, doc2], ...]
    pairs = [[query, doc] for doc in combined_docs]
    scores = reranker.predict(pairs)
    
    # Zip the documents and scores together, then sort by the highest score
    scored_docs = sorted(zip(combined_docs, scores), key=lambda x: x[1], reverse=True)
    
    # Keep only the absolute best 5 results
    top_5 = scored_docs[:5]
    
    print("\n--- Final Top 5 Reranked Contexts ---")
    for i, (doc, score) in enumerate(top_5, 1):
        print(f"\n[{i}] Relevance Score: {score:.2f}")
        # Print a snippet to keep the terminal clean
        print(f"{str(doc)[:250]}...")
        
    return [doc for doc, score in top_5]
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    test_query = "How does RAG architectures reduce hallucination rates?"
    hybrid_search(test_query)