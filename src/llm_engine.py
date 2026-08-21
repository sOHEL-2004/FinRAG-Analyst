from langchain_openai import ChatOpenAI
from src.config import LLM_MODEL

def get_llm_engine(temperature: float = 0.0):
    """
    Returns the standardized LLM engine wrapper for the FinRAG project.
    Temperature is set to 0.0 by default to prevent financial hallucination.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature
    )