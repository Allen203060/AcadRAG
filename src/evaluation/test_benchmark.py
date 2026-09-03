import os
import time
import json
from src.core.graph import run_rag_pipeline

TEST_QUESTIONS = [
    {
        "id": "Q1",
        "category": "Architecture Concepts",
        "question": "What is Multi-Head Attention and how does it differ from single attention?",
        "expected_concept": "Projects queries, keys, and values into h sub-spaces to attend to information at different positions."
    },
    {
        "id": "Q2",
        "category": "Mathematical Formulation",
        "question": "Why is the Dot-Product Attention scaled by the square root of the key dimension (dk)?",
        "expected_concept": "Prevents large dot products from pushing softmax into regions with extremely small gradients."
    },
    {
        "id": "Q3",
        "category": "Positional Encoding",
        "question": "How does the Transformer model inject order information without using Recurrence or Convolutions?",
        "expected_concept": "Uses Positional Encodings (sine and cosine functions of different frequencies) added to input embeddings."
    },
    {
        "id": "Q4",
        "category": "Training & Optimization",
        "question": "What optimizer and learning rate schedule was used to train the Transformer model?",
        "expected_concept": "Adam optimizer with beta1=0.9, beta2=0.98, and a warmup learning rate schedule."
    },
    {
        "id": "Q5",
        "category": "Anti-Hallucination Guardrail Check",
        "question": "What was the exact battery capacity of the iPhone 15 Pro Max discussed in the paper?",
        "expected_concept": "Should trigger 'I cannot answer this based on the provided documents' (Out of domain test)."
    }
]

def run_benchmark():
    print("=" * 65)
    print("🚀 Starting AcadRAG LangGraph Benchmark Execution Suite")
    print("=" * 65)
    
    results = []
    
    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        q_id = test_case["id"]
        category = test_case["category"]
        question = test_case["question"]
        expected = test_case["expected_concept"]
        
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] Testing {q_id} ({category}):")
        print(f"❓ Question: {question}")
        
        start_time = time.perf_counter()
        
        graph_output = run_rag_pipeline(question)
        
        elapsed_sec = round(time.perf_counter() - start_time, 2)
        
        answer = graph_output["answer"]
        reranked_docs = graph_output.get("reranked_documents", [])
        
        print(f"⏱️  Latency: {elapsed_sec} seconds")
        print(f"🤖 Answer:\n{answer}\n")
        print("-" * 65)
        
        results.append({
            "id": q_id,
            "category": category,
            "question": question,
            "expected_concept": expected,
            "latency_seconds": elapsed_sec,
            "retrieved_sources_count": len(reranked_docs),
            "generated_answer": answer
        })
        
    output_filename = "benchmark_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Benchmark execution complete! Saved report to `{output_filename}`.")
    print("=" * 65)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_benchmark()
