import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

load_dotenv(override=True)

def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Factory function returning the configured LLM based on LLM_PROVIDER in .env.
    Supported providers: 'ollama' (default), 'groq', 'openrouter', 'nemotron', 'gemini'.
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    
    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("⚠️ GROQ_API_KEY missing! Falling back to local Ollama.")
            return ChatOllama(model="llama3.1:8b", temperature=temperature)
        try:
            from langchain_groq import ChatGroq
            print("🚀 Initializing Groq LPU LLM (llama-3.3-70b-versatile)...")
            return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=temperature, max_retries=3)
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
            print("🌐 Initializing OpenRouter LLM (openai/gpt-4o-mini)...")
            return ChatOpenAI(
                model_name="openai/gpt-4o-mini",
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=2000
            )
        except ImportError:
            print("⚠️ langchain-openai not installed. Falling back to Ollama.")
            return ChatOllama(model="llama3.1:8b", temperature=temperature)

    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ GEMINI_API_KEY missing! Falling back to local Ollama.")
            return ChatOllama(model="llama3.1:8b", temperature=temperature)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("✨ Initializing Google Gemini LLM...")
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=temperature
            )
        except ImportError:
            print("⚠️ langchain-google-genai not installed. Falling back to OpenAI-compatibility mode...")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name="gemini-2.5-flash",
                openai_api_key=api_key,
                openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                temperature=temperature
            )

    elif provider == "nemotron":
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            print("⚠️ NVIDIA_API_KEY missing! Falling back to local Ollama.")
            return ChatOllama(model="llama3.1:8b", temperature=temperature)
        try:
            from langchain_openai import ChatOpenAI
            print("🟢 Initializing NVIDIA Nemotron LLM...")
            return ChatOpenAI(
                model_name="poolside/laguna-xs-2.1",
                openai_api_key=api_key,
                openai_api_base="https://integrate.api.nvidia.com/v1",
                temperature=temperature,
            )   
        except ImportError:
            print("⚠️ langchain-openai not installed. Falling back to Ollama.")
            return ChatOllama(model="llama3.1:8b", temperature=temperature)

    else:
        # Default local Ollama pipeline
        print("🏠 Initializing Local Ollama LLM (gemma4:e2b, qwen3:4b)...")
        return ChatOllama(model="gemma4:e2b", temperature=temperature)
