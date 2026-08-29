import os
import time
from dotenv import load_dotenv
from langsmith import Client
from langchain_ollama import ChatOllama
from llm_factory import get_llm
from retriever import hybrid_search

load_dotenv(override=True)

# 1. Initialize LangSmith Client
client = Client()
DATASET_NAME = "TribeV2_Brain_Encoding_Benchmark"

# 2. Define New Test Set Grounds
TEST_EXAMPLES = [
    {
        "inputs": {"question": "What are the three stimulus modalities used as inputs for the TRIBE v2 model?"},
        "outputs": {"answer": "The three modalities are video, audio, and language (text)."}
    },
    {
        "inputs": {"question": "How much fMRI recording data and how many subjects were used in total across the datasets for TRIBE v2?"},
        "outputs": {"answer": "The model leverages a unified dataset of over 1,000 hours of fMRI recordings across 720 subjects."}
    },
    {
        "inputs": {"question": "How does TRIBE v2 achieve zero-shot predictions of group responses for unseen subjects?"},
        "outputs": {"answer": "It implements a 'subject dropout' module during training (with probability p=0.1) that bypasses the subject block, forcing the model to make predictions without subject-specific information."}
    },
    {
        "inputs": {"question": "During the in-silico visual experiments, what specific brain area was correctly recovered for the processing of faces?"},
        "outputs": {"answer": "The fusiform face area (FFA)."}
    },
    {
        "inputs": {"question": "Why did the authors apply detrending to the fMRI timeseries during preprocessing?"},
        "outputs": {"answer": "Detrending was necessary because slow drifts could be exploited by the encoding model's long context window to spuriously increase its encoding score in hard-to-predict brain areas."}
    },
    {
        "inputs": {"question": "What exactly does the 'modality dropout' technique accomplish during training?"},
        "outputs": {"answer": "It randomly masks off each modality with probability p=0.3 to encourage the model to provide meaningful predictions in the absence of one or several modalities and avoid excessive reliance on a single modality."}
    },
    {
        "inputs": {"question": "How does the performance of TRIBE v2 compare to traditional linear encoding models?"},
        "outputs": {"answer": "TRIBE v2 significantly outperforms the traditional optimized 'Deep FIR' linear encoder, demonstrating the advantage of deep non-linear methods and delivering several-fold improvements in accuracy."}
    },
    {
        "inputs": {"question": "What specific quantum MRI scanner was used to capture the 15-Tesla resolution datasets?"},
        "outputs": {"answer": "I cannot answer this based on the provided documents."}  # Out of domain guardrail check
    }
]

def prepare_langsmith_dataset():
    """Programmatically creates or retrieves the LangSmith dataset."""
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

# 3. Target Function: Wraps our AcadRAG Pipeline
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

# 4. Custom LLM-as-a-Judge Evaluator
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

# 5. Main Execution Loop
if __name__ == "__main__":
    dataset_name = prepare_langsmith_dataset()
    
    print("\n🧪 Running LangSmith Experiment...")
    experiment_results = client.evaluate(
        rag_pipeline_target,
        data=dataset_name,
        evaluators=[llm_judge_evaluator],
        experiment_prefix="AcadRAG-Docling-Hybrid-Eval",
        max_concurrency=1  # Run sequentially to avoid local Ollama VRAM thrashing
    )
    
    print("\n🎉 Benchmark Evaluation Complete!")
    print(f"🔗 View detailed traces and experiment matrix on LangSmith Cloud.")
    