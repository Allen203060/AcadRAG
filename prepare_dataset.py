import os
import shutil
from datasets import load_dataset

def prepare_bioasq_corpus():
    # 1. Clean the existing data directory so we don't mix test data
    data_dir = "./data"
    if os.path.exists(data_dir):
        print("Cleaning existing data directory...")
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    print("Downloading rag-mini-bioasq corpus from Hugging Face...")
    # This downloads the raw passages used for the biomedical dataset
    corpus_dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus", split="passages")

    # The dataset contains over 40,000 passages. 
    # Since you are running Graph Extraction locally on an 8B model, doing all 40,000 
    # would take days. We will extract just a subset (e.g., the first 50 passages) for testing.
    SUBSET_SIZE = 50
    
    print(f"Writing first {SUBSET_SIZE} passages to {data_dir}...")
    for i in range(SUBSET_SIZE):
        passage_text = corpus_dataset[i]['passage']
        
        # Save each passage as a standard text file for our DirectoryLoader
        with open(os.path.join(data_dir, f"bioasq_passage_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(passage_text)

    print("✅ Dataset prepared successfully! The /data folder is populated.")
    print("\n--- NEXT STEPS ---")
    print("1. Because we have new data, you MUST wipe the databases: docker compose down -v && sudo rm -rf volumes/milvus volumes/etcd volumes/minio && docker compose up -d")
    print("2. Run: python populate.py (to embed and extract the new biology graph)")
    print("3. Then we can write the LangSmith evaluation script!")

if __name__ == "__main__":
    prepare_bioasq_corpus()
