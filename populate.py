import os
import time
import asyncio
from llm_factory import get_llm
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
from ingest import load_and_chunk_documents

CONCURRENCY_LIMIT = 2 # Process 2 chunks concurrently via Ollama

async def extract_graph_async(llm_transformer, chunks):
    """Extracts graph documents asynchronously with semaphore concurrency control."""
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    total_chunks = len(chunks)
    completed = 0

    async def process_chunk(chunk, index):
        nonlocal completed
        async with semaphore:
            # Uses async version of LLMGraphTransformer
            extracted = await llm_transformer.aconvert_to_graph_documents([chunk])
            completed += 1
            print(f"   ⚡ [{completed}/{total_chunks}] Finished Graph Extraction for Chunk {index}")
            return extracted

    tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks, 1)]
    results = await asyncio.gather(*tasks)
    
    # Flatten list of lists
    flattened = [doc for sublist in results for doc in sublist]
    return flattened

def populate_databases():
    start_total = time.time()
    
    # 1. Fetch chunks using Header-Aware Splitter (Phase 8)
    chunks = load_and_chunk_documents("./data")
    if not chunks:
        print("No chunks to process! Exiting.")
        return

    # 2. VECTOR DB: Optimized GPU Milvus Population
    print("\n--- 1. Vector Database Population (Milvus) ---")
    start_vector = time.time()
    
    # Explicitly pin HuggingFace embeddings to CUDA GPU
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Connecting to Milvus and inserting vectors...")
    vector_db = Milvus.from_documents(
        chunks,
        embeddings,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        collection_name="academic_papers",
        drop_old=True,
        auto_id=True,
        enable_dynamic_field=True 
    )
    print(f"✅ Milvus Population Complete in {round(time.time() - start_vector, 2)}s!")

    # 3. KNOWLEDGE GRAPH: Optimized Async Neo4j Population
    print("\n--- 2. Knowledge Graph Extraction (Neo4j) ---")
    start_graph = time.time()
    
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password"

    graph = Neo4jGraph()
    print("Clearing old Neo4j graph entities...")
    graph.query("MATCH (n) DETACH DELETE n")

    print("Initializing Constrained LLMGraphTransformer (Llama 3.1:8b)...")
    llm = get_llm(temperature=0)
    
    # Production Schema Constraints: Drastically cuts token generation overhead
    allowed_nodes = ["Concept", "Architecture", "Method", "Metric", "Formula", "Dataset"]
    allowed_rels = ["USES", "PROPOSES", "EVALUATED_ON", "PART_OF", "IMPROVES"]

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_rels
    )

    print(f"Executing Async Extraction with Concurrency={CONCURRENCY_LIMIT}...")
    graph_documents = asyncio.run(extract_graph_async(llm_transformer, chunks))

    print(f"Writing {len(graph_documents)} graph documents to Neo4j...")
    graph.add_graph_documents(
        graph_documents, 
        baseEntityLabel=True, 
        include_source=True
    )
    
    print(f"✅ Neo4j Population Complete in {round(time.time() - start_graph, 2)}s!")
    print(f"\n🎉 Total Pipeline Ingestion Time: {round(time.time() - start_total, 2)}s")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    populate_databases()
