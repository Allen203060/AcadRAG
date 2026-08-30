import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv(override=True)

def print_banner():
    provider = os.environ.get("LLM_PROVIDER", "ollama").upper()
    print("=" * 60)
    print(" 🚀 AcadRAG: Enterprise Hybrid GraphRAG Pipeline ")
    print(f" ⚙️  Active LLM Provider: [{provider}]")
    print("=" * 60)

def run_ingestion():
    print("\n--- Phase 1: Database Population & Ingestion ---")
    from src.ingestion.populate import populate_databases
    populate_databases()

def run_interactive_query():
    print("\n--- Phase 2: Interactive Hybrid RAG Terminal ---")
    from src.core.query import generate_answer
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_query = input("\nAsk a question about the papers: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        generate_answer(user_query)

def run_langgraph_agent(question: str = "What is Multi-Head Attention?"):
    print("\n--- Phase 3: LangGraph Stateful Agent Execution ---")
    from src.core.graph import run_rag_pipeline
    res = run_rag_pipeline(question)
    print("\n--- LangGraph Agent Answer ---")
    print(res["answer"])

def run_langsmith_evaluation():
    print("\n--- Phase 4: LangSmith Benchmark Evaluation Suite ---")
    from src.evaluation.langsmith_eval import prepare_langsmith_dataset, client, rag_pipeline_target, llm_judge_evaluator
    
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
    print(experiment_results)

def run_full_pipeline():
    print_banner()
    print("🔥 EXECUTING FULL END-TO-END PIPELINE 🔥\n")
    
    # 1. Ingest PDF and build Milvus Vector DB + Neo4j Graph DB
    run_ingestion()
    
    # 2. Run LangGraph agent sample query
    run_langgraph_agent("What are the primary advantages of Self-Attention over Recurrent layers?")
    
    # 3. Run automated LangSmith evaluation suite
    run_langsmith_evaluation()
    
    # 4. Start interactive query loop
    run_interactive_query()

def interactive_menu():
    print_banner()
    print("1. Populate Databases (Docling + Milvus + Neo4j Graph)")
    print("2. Start Interactive Query Terminal (Hybrid RAG + Citations)")
    print("3. Execute LangGraph Stateful Agent Pipeline")
    print("4. Run LangSmith Benchmark Evaluation Suite")
    print("5. Run Full End-to-End Pipeline (Populate -> Agent -> Eval -> Terminal)")
    print("6. Exit")
    
    choice = input("\nSelect an option (1-6): ").strip()
    
    if choice == '1':
        run_ingestion()
    elif choice == '2':
        run_interactive_query()
    elif choice == '3':
        q = input("Enter a question for LangGraph Agent (Press Enter for default): ").strip()
        if q:
            run_langgraph_agent(q)
        else:
            run_langgraph_agent()
    elif choice == '4':
        run_langsmith_evaluation()
    elif choice == '5':
        run_full_pipeline()
    elif choice == '6':
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = argparse.ArgumentParser(description="AcadRAG Orchestrator CLI")
    parser.add_argument("--populate", action="store_true", help="Run document ingestion & database population")
    parser.add_argument("--query", action="store_true", help="Start interactive RAG terminal")
    parser.add_argument("--graph", action="store_true", help="Run LangGraph stateful agent test query")
    parser.add_argument("--eval", action="store_true", help="Run LangSmith evaluation benchmark")
    parser.add_argument("--all", action="store_true", help="Execute full end-to-end pipeline")
    
    args = parser.parse_args()
    
    if args.populate:
        print_banner()
        run_ingestion()
    elif args.query:
        print_banner()
        run_interactive_query()
    elif args.graph:
        print_banner()
        run_langgraph_agent()
    elif args.eval:
        print_banner()
        run_langsmith_evaluation()
    elif args.all:
        run_full_pipeline()
    else:
        interactive_menu()
