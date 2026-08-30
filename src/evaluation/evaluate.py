import os
import uuid
from dotenv import load_dotenv
from datasets import load_dataset
from langsmith import Client, evaluate
from src.core.retriever import hybrid_search
from src.core.llm_factory import get_llm

load_dotenv()

def predict_rag_answer(inputs: dict) -> dict:
    query = inputs["question"]
    top_contexts = hybrid_search(query)
    
    if not top_contexts:
        return {"answer": "I cannot answer this based on the provided documents."}
        
    context_str = ""
    for i, doc in enumerate(top_contexts, 1):
        context_str += f"[Source {i}]:\n{doc}\n\n"
        
    prompt = f"""You are a strict academic assistant. Answer the user's question using ONLY the provided context below. 
Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"""

    llm = get_llm(temperature=0)
    response = llm.invoke(prompt)
    return {"answer": response.content}

def run_evaluation():
    client = Client()
    dataset_name = f"BioASQ_Evaluation_{str(uuid.uuid4())[:6]}"
    
    print("Loading QA pairs from Hugging Face (rag-mini-bioasq)...")
    qa_dataset = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages", split="test")
    
    print(f"Uploading 5 test cases to LangSmith dataset: {dataset_name}...")
    dataset = client.create_dataset(dataset_name=dataset_name, description="BioASQ subset for Hybrid RAG evaluation")
    
    for i in range(5):
        example = qa_dataset[i]
        client.create_example(
            inputs={"question": example["question"]},
            outputs={"expected_answer": example["answer"]},
            dataset_id=dataset.id,
        )
    
    print("Initializing LLM Factory as the Evaluator Judge...")
    def custom_llm_judge(run, example):
        generated_answer = run.outputs["answer"]
        true_answer = example.outputs["expected_answer"]
        
        judge_llm = get_llm(temperature=0)
        prompt = f"""You are a strict teacher grading a student's answer.
        
        True Answer: {true_answer}
        Student's Answer: {generated_answer}
        
        Is the Student's Answer factually correct based on the True Answer?
        Respond with ONLY the word CORRECT or INCORRECT."""
        
        grade_text = judge_llm.invoke(prompt).content.strip().upper()
        score = 1.0 if "CORRECT" in grade_text else 0.0
        return {"key": "correctness", "score": score, "comment": grade_text}

    print("\nRunning Evaluation! (This will execute your retrieval pipeline 5 times)")
    experiment_results = evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=[custom_llm_judge],
        experiment_prefix="hybrid-graph-rag",
        metadata={"retriever": "hybrid_milvus_neo4j", "llm": "llm_factory"},
    )

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_evaluation()
