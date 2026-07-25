import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Initialize the Local Embedding Model
# We use BGE-small, a highly efficient local embedding model via HuggingFace

print("Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def load_and_chunk_documents(data_dir="./data"):
    # Step 2: Load Documents
    # The DirectoryLoader will scan the folder for any .txt files
    print(f"Loading documents from {data_dir}...")
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("No documents found! Please add a .txt file to the data/ directory.")
        return []

    print(f"Loaded {len(documents)} document(s).")
    # Step 3: Semantic Chunking
    # We use the experimental SemanticChunker which uses our local embeddings
    # to group sentences by mathematical meaning.
    print("Performing Semantic Chunking...")
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile" # Breaks chunks when similarity drops below a percentile
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Successfully split into {len(chunks)} cohesive semantic chunks.")

    if chunks:
        print("\n--- Preview of First Chunk ---")
        print(f"Metadata: {chunks[0].metadata}")
        print(f"Content: {chunks[0].page_content[:200]}...")
        
    return chunks

if __name__ == "__main__":
    # Ensure the HuggingFace tokenizer parallelism warning is suppressed
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    load_and_chunk_documents()