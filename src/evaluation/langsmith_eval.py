import os
import time
from dotenv import load_dotenv
from langsmith import Client
from src.core.llm_factory import get_llm
from src.core.retriever import hybrid_search

load_dotenv(override=True)

import json

client = Client()
DATASET_NAME = "AcadRAG_25_Item_Golden_Benchmark"

def load_test_examples():
    json_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
            return [
                {
                    "inputs": {"question": item["question"]},
                    "outputs": {
                        "answer": item["ground_truth_answer"],
                        "ground_truth_context": item.get("ground_truth_context", ""),
                        "pipeline_stage": item.get("pipeline_stage_tested", "")
                    }
                }
                for item in raw_items
            ]
    return [
        {
            "inputs": {"question": "What are the two primary reasons Self-Attention is faster than Recurrent layers for long sequences according to Table 1?"},
            "outputs": {"answer": "Total computational complexity per layer is lower when sequence length n is smaller than representation dimension d, and maximum path length between long-range dependencies is O(1) compared to O(n) for recurrent layers."}
        }
    ]

TEST_EXAMPLES = load_test_examples()

def prepare_langsmith_dataset():
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"📊 Dataset '{DATASET_NAME}' already exists in LangSmith.")
        return DATASET_NAME
    
    print(f"🚀 Creating new LangSmith Dataset: '{DATASET_NAME}'...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Comprehensive 25-item benchmark suite covering Docling Parsing, Vector Retrieval, GraphRAG Traversal, Reranking, and Multi-Paper Synthesis."
    )
    
    client.create_examples(
        inputs=[e["inputs"] for e in TEST_EXAMPLES],
        outputs=[e["outputs"] for e in TEST_EXAMPLES],
        dataset_id=dataset.id
    )
    print(f"✅ Added {len(TEST_EXAMPLES)} benchmark test cases to LangSmith!")
    return DATASET_NAME

def rag_pipeline_target(inputs: dict) -> dict:
    question = inputs["question"]
    start_time = time.time()
    
    top_contexts = hybrid_search(question)
    
    context_str = ""
    for i, doc in enumerate(top_contexts, 1):
        context_str += f"[Source {i}]:\n{doc}\n\n"
        
    prompt = f"""You are a strict academic assistant. Answer the user's question using ONLY the provided context below. 
If the answer cannot be deduced from the context, say exactly: "I cannot answer this based on the provided documents."
For EVERY factual claim you make, you MUST cite the source inline using brackets matching the source label, for example: [Source 1].

Context:
{context_str}

Question: {question}
Answer:"""

    llm = get_llm(temperature=0)
    response = llm.invoke(prompt)
    latency = time.time() - start_time
    
    return {
        "answer": response.content.strip(),
        "latency_seconds": round(latency, 2)
    }

def llm_judge_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    question = inputs["question"]
    generated = outputs.get("answer", "")
    expected = reference_outputs.get("answer", "")  
    
    judge_prompt = f"""You are an expert academic evaluator. Grade the following candidate answer against the ground truth answer.

Question: {question}
Ground Truth Answer: {expected}
Candidate Answer: {generated}

Rules:
1. If the ground truth expects "I cannot answer this based on the provided documents." and the candidate states that exact fallback, return CORRECT.
2. If the candidate answer correctly conveys the core technical facts from the ground truth answer, return CORRECT.
3. Otherwise, return INCORRECT.

Respond with EXACTLY one word: CORRECT or INCORRECT."""

    judge_llm = get_llm(temperature=0)
    verdict = judge_llm.invoke(judge_prompt).content.strip().upper()
    score = 1.0 if "CORRECT" in verdict else 0.0
    
    return {
        "key": "correctness",
        "score": score,
        "comment": f"Judge Verdict: {verdict}"
    }

if __name__ == "__main__":
    dataset_name = prepare_langsmith_dataset()
    
    print("\n🧪 Running LangSmith Experiment...")
    experiment_results = client.evaluate(
        rag_pipeline_target,
        data=dataset_name,
        evaluators=[llm_judge_evaluator],
        experiment_prefix="AcadRAG-Docling-Hybrid-Eval",
        max_concurrency=1
    )
    
    print("\n🎉 Benchmark Evaluation Complete!")
    print(f"🔗 View detailed traces and experiment matrix on LangSmith Cloud.")
