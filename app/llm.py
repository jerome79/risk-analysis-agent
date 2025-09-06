import os
from langchain_community.llms import Ollama

def get_llm():
    model = os.getenv("OLLAMA_MODEL", "mistral")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    return Ollama(model=model, temperature=temperature)
