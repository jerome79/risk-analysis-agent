import os

from langchain_community.llms import Ollama


def get_llm() -> Ollama:
    """
    Returns an Ollama LLM instance configured with model and temperature
    from environment variables.

    Returns:
        Ollama: The configured Ollama language model instance.
    """
    model = os.getenv("OLLAMA_MODEL", "mistral")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    return Ollama(model=model, temperature=temperature)
