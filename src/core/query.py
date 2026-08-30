import os
from dotenv import load_dotenv
from src.core.retriever import hybrid_search
from src.core.llm_factory import get_llm

load_dotenv(override=True)

def generate_answer(query: str):
    top_contexts = hybrid_search(query)
    
    context_str = ""
    for i, doc in enumerate(top_contexts, 1):
        context_str += f"[Source {i}]:\n{doc}\n\n"
        
    prompt = f"""You are a strict academic assistant. Answer the user's question using ONLY the provided context below. 
If the answer cannot be deduced from the context, say exactly: "I cannot answer this based on the provided documents."
For EVERY factual claim you make, you MUST cite the source inline using brackets matching the source label, for example: [Source 1].

Context:
{context_str}

Question: {query}
Answer:"""

    print(f"\n--- Generating Grounded Response via LLM Factory ---")
    llm = get_llm(temperature=0)
    response = llm.invoke(prompt)
    
    print("\n==================== ANSWER ====================")
    print(response.content)
    print("================================================\n")
    return response.content

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sample_question = "What are the key advantages of Self-Attention over Recurrent layers?"
    generate_answer(sample_question)
