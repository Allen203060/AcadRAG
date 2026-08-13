import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def load_and_chunk_documents(data_dir: str = "data"):
    # Step 1: Load Markdown Documents
    print(f"Loading Markdown documents from {data_dir}...")
    loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print(f"No .md files found in {data_dir}!")
        return []

    # Step 2: Pass 1 - Define Markdown Header Hierarchy
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False  # Keep header inline for extra context
    )

    # Step 3: Pass 2 - Size-Based Recursive Sub-Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )

    final_chunks = []

    for doc in documents:
        # Pass 1: Header-based split
        header_splits = markdown_splitter.split_text(doc.page_content)
        
        # Preserve original source metadata
        for split in header_splits:
            split.metadata["source"] = doc.metadata.get("source", "unknown")
            
        # Pass 2: Sub-split any oversized sections
        sub_splits = text_splitter.split_documents(header_splits)
        final_chunks.extend(sub_splits)

    print(f"✅ Generated {len(final_chunks)} header-aware chunks.")
    
    # Inspect first chunk to verify header metadata
    if final_chunks:
        print(f"\n--- Sample Chunk Metadata ---")
        print(f"Metadata: {final_chunks[0].metadata}")
        print(f"Content:\n{final_chunks[0].page_content[:200]}...\n")

    return final_chunks

if __name__ == "__main__":
    chunks = load_and_chunk_documents()
