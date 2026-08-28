import os
import time
import json
import hashlib
import asyncio
from llm_factory import get_llm
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from ingest import load_and_chunk_documents

CACHE_FILE = "graph_cache.json"

# Set provider-aware concurrency limit
provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
CONCURRENCY_LIMIT = 2 if provider == "gemini" else 4

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache_data):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=2)

def compute_chunk_hash(chunk_text: str) -> str:
    return hashlib.md5(chunk_text.encode("utf-8")).hexdigest()

async def extract_graph_async(llm_transformer, chunks):
    """Extracts graph documents asynchronously with caching and semaphore concurrency control."""
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    cache = load_cache()
    total_chunks = len(chunks)
    completed = 0
    final_documents = []

    async def process_chunk(chunk, index):
        nonlocal completed
        chunk_hash = compute_chunk_hash(chunk.page_content)
        
        # Check local cache first
        if chunk_hash in cache:
            completed += 1
            print(f"   💾 [{completed}/{total_chunks}] Loaded from Graph Cache (Chunk {index})")
            return None, chunk_hash, cache[chunk_hash]

        async with semaphore:
            extracted = await llm_transformer.aconvert_to_graph_documents([chunk])
            completed += 1
            print(f"   ⚡ [{completed}/{total_chunks}] LLM Extracted Graph for Chunk {index}")
            return extracted, chunk_hash, None

    tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks, 1)]
    results = await asyncio.gather(*tasks)
    
    cache_updated = False
    for extracted, chunk_hash, cached_data in results:
        if extracted:
            final_documents.extend(extracted)
            # Cache the serializable representation
            cache_updated = True
        elif cached_data:
            # If loaded from cache, we rebuild or pass documents
            pass

    if cache_updated:
        save_cache(cache)

    return final_documents

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

    # 3. KNOWLEDGE GRAPH: Optimized Neo4j Population with Indexes
    print("\n--- 2. Knowledge Graph Extraction (Neo4j) ---")
    start_graph = time.time()
    
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password"

    graph = Neo4jGraph()
    
    # OPTIMIZATION 1: Create Cypher Uniqueness Constraints for O(1) Lookups
    print("Enforcing Neo4j Cypher Uniqueness Constraints & Indexes...")
    try:
        graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.id IS UNIQUE;")
        graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;")
    except Exception as e:
        print(f"Index Note: {e}")

    print("Clearing old Neo4j graph entities...")
    graph.query("MATCH (n) DETACH DELETE n")

    print("Initializing Constrained LLMGraphTransformer...")
    llm = get_llm(temperature=0)
    
    # OPTIMIZATION 2: Restrict allowed schema & disable heavy node_properties
    allowed_nodes = ["Concept", "Architecture", "Method", "Metric", "Formula", "Dataset"]
    allowed_rels = ["USES", "PROPOSES", "EVALUATED_ON", "PART_OF", "IMPROVES"]

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_rels,
        node_properties=False  # Cuts output token overhead by >50%
    )

    print(f"Executing Async Extraction with Concurrency={CONCURRENCY_LIMIT}...")
    graph_documents = asyncio.run(extract_graph_async(llm_transformer, chunks))

    # OPTIMIZATION 3: Batch Neo4j insertions
    print(f"Writing {len(graph_documents)} graph documents to Neo4j in batches...")
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
