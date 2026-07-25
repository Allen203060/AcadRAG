import os
from retriever import hybrid_search
from langchain_ollama import ChatOllama

def generate_answer(query: str):
    # 1. Retrieve the highly relevant, reranked context from Phase 4
    top_contexts = hybrid_search(query)
    
    if not top_contexts:
        print("No relevant context found.")
        return

    # 2. Format the context for the LLM with strict Source tags
    context_str = ""
    for i, doc in enumerate(top_contexts, 1):
        context_str += f"[Source {i}]:\n{doc}\n\n"
    
    # 3. Construct the Grounded Prompt
    # This prompt strictly enforces anti-hallucination and citation rules.
    prompt = f"""You are a strict academic assistant. Answer the user's question using ONLY the provided context below. 
        If the answer cannot be deduced from the context, say exactly: "I cannot answer this based on the provided documents."
        For EVERY factual claim you make, you MUST cite the source inline using brackets matching the source label, for example: [Source 1].
        Context:
        {context_str}
        Question: {query}
        Answer:"""
    print("\n==================================================")
    print("🤖 Generating Grounded Answer (Llama)")

     # 4. Initialize LLM and stream the response
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    
    # Stream the output so it feels snappy and interactive
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
    print("\n\n==================================================")

    
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Welcome to AcadRAG! (Type 'exit' to quit)")
    while True:
        user_query = input("\nAsk a question about the papers: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        generate_answer(user_query)