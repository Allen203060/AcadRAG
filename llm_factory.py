import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

load_dotenv(override=True)

def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Factory function returning the configured LLM based on LLM_PROVIDER in .env.
    Supported providers: 'ollama' (default), 'groq', 'openrouter', 'nemotron'.
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    
    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("⚠️ GROQ_API_KEY missing! Falling back to local Ollama.")
            return ChatOllama(model="llama3.1:8b", temperature=temperature)
        try:
            from langchain_groq import ChatGroq
            print("🚀 Initializing Groq LPU LLM (meta-llama/llama-prompt-guard-2-22m)...")
            return ChatGroq(model_name="openai/gpt-oss-20b", temperature=temperature, max_retries=3)
        except ImportError:
            print("⚠️ langchain-groq not installed. Run `pip install langchain-groq`. Falling back to Ollama.")
            return ChatOllama(model="llama3.1:8b", temperature=temperature)

    elif provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("⚠️ OPENROUTER_API_KEY missing! Falling back to local Ollama.")
            return ChatOllama(model="llama3.1:8b", temperature=temperature)
        try:
            from langchain_openai import ChatOpenAI
            print("🌐 Initializing OpenRouter LLM (deepseek/deepseek-chat)...")
            return ChatOpenAI(
                model_name="deepseek/deepseek-chat",
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=temperature
            )
        except ImportError:
            print("⚠️ langchain-openai not installed. Falling back to Ollama.")
            return ChatOllama(model="llama3.1:8b", temperature=temperature)

    elif provider == "nemotron":
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            print("⚠️ NVIDIA_API_KEY missing! Falling back to local Ollama.")
            return ChatOllama(model="llama3.1:8b", temperature=temperature)
        try:
            from langchain_openai import ChatOpenAI
            print("🟢 Initializing NVIDIA Nemotron LLM...")
            return ChatOpenAI(
                model_name="nvidia/nemotron-4-340b-instruct",
                openai_api_key=api_key,
                openai_api_base="https://integrate.api.nvidia.com/v1",
                temperature=temperature
            )
        except ImportError:
            print("⚠️ langchain-openai not installed. Falling back to Ollama.")
            return ChatOllama(model="llama3.1:8b", temperature=temperature)

    else:
        # Default local Ollama pipeline
        print("🏠 Initializing Local Ollama LLM (llama3.1:8b)...")
        return ChatOllama(model="llama3.1:8b", temperature=temperature)
