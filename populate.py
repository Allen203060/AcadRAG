import os
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
from ingest import load_and_chunk_documents

def populate_databases():
    # 1. Fetch our semantic chunks using the script we wrote in Phase 2
    chunks = load_and_chunk_documents("./data")
    if not chunks:
        print("No chunks to process! Exiting.")
        return

    # 2. VECTOR DB: Populate Milvus
    print("\n--- Starting Vector Database Population ---")
    print("Loading Embedding Model (BGE-Small)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    print("Connecting to Milvus and inserting chunks...")
    # This automatically connects to our local Docker instance of Milvus,
    # generates embeddings for all chunks, and stores them in a collection.
    vector_db = Milvus.from_documents(
        chunks,
        embeddings,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        collection_name="academic_papers"
    )
    print("Successfully populated Milvus Vector Database!")

    # 3. GRAPH DB: Populate Neo4j
    print("\n--- Starting Knowledge Graph Extraction ---")
    # Set Neo4j connection variables for the Neo4jGraph class
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password"

    print("Connecting to Neo4j...")
    graph = Neo4jGraph()

    print("Initializing Local LLM (Llama 3.1:8b) for Entity Extraction...")
    # We use temperature=0 because we want deterministic extraction, not creative writing.
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    # The LLMGraphTransformer is a LangChain utility that prompts the LLM to 
    # find Nodes and Edges in unstructured text.
    llm_transformer = LLMGraphTransformer(llm=llm)

    print("Instructing LLM to extract Graph Documents (This will take time based on your local GPU/CPU)...")
    graph_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"   -> Processing chunk {i} of {total_chunks}...")
        extracted = llm_transformer.convert_to_graph_documents([chunk])
        graph_documents.extend(extracted)


    print("Adding extracted graph data to Neo4j...")
    # baseEntityLabel groups everything under a generic 'Entity' tag, and include_source 
    # maps the node back to the original text chunk for citation purposes.
    graph.add_graph_documents(
        graph_documents, 
        baseEntityLabel=True, 
        include_source=True
    )
    
    print("Successfully populated Neo4j Knowledge Graph!")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    populate_databases()