import os
import time
from dotenv import load_dotenv
from langsmith import Client
from src.core.llm_factory import get_llm
from src.core.retriever import hybrid_search

load_dotenv(override=True)

client = Client()
DATASET_NAME = "Attention_Is_All_You_Need_Comprehensive_Benchmark"

TEST_EXAMPLES = [
    {
        "inputs": {"question": "What are the two primary reasons Self-Attention is faster than Recurrent layers for long sequences according to Table 1?"},
        "outputs": {"answer": "Total computational complexity per layer is lower when sequence length n is smaller than representation dimension d, and maximum path length between long-range dependencies is O(1) compared to O(n) for recurrent layers."}
    },
    {
        "inputs": {"question": "Why did the authors scale Dot-Product Attention by 1 / sqrt(d_k)?"},
        "outputs": {"answer": "For large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients."}
    },
    {
        "inputs": {"question": "What exact mathematical formula is used for Positional Encodings at even positions 2i?"},
        "outputs": {"answer": "PE_(pos, 2i) = sin(pos / 10000^(2i/d_model))"}
    },
    {
        "inputs": {"question": "What dropout rate and label smoothing value epsilon_ls were used during model training?"},
        "outputs": {"answer": "Residual Dropout rate of P_drop = 0.1 was applied, and Label Smoothing of epsilon_ls = 0.1 was used during training."}
    },
    {
        "inputs": {"question": "How many heads (h) and key/value dimensions (d_k, d_v) were used in the Transformer Base model?"},
        "outputs": {"answer": "h = 8 heads, d_k = 64, and d_v = 64 (such that d_k * h = d_model = 512)."}
    },
    {
        "inputs": {"question": "What is the function of the Masked Multi-Head Attention block in the Decoder?"},
        "outputs": {"answer": "It ensures that predictions for position i can depend only on the known outputs at positions less than i, preventing leftward information flow."}
    },
    {
        "inputs": {"question": "How long was the Transformer Big model trained on WMT 2014 English-to-German, and what BLEU score did it achieve?"},
        "outputs": {"answer": "Trained for 300,000 steps (3.5 days on 8 P100 GPUs) achieving a state-of-the-art BLEU score of 28.4."}
    },
    {
        "inputs": {"question": "What was the quantum compute unit architecture used to train the Transformer?"},
        "outputs": {"answer": "I cannot answer this based on the provided documents."}
    }
]

def prepare_langsmith_dataset():
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"📊 Dataset '{DATASET_NAME}' already exists in LangSmith.")
        return DATASET_NAME
    
    print(f"🚀 Creating new LangSmith Dataset: '{DATASET_NAME}'...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Comprehensive benchmark suite for Attention Is All You Need paper (Docling DOM + Hybrid RAG)."
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
