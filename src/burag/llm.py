"""LLM factory (OpenAI)."""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from burag.config import LLM_MODEL


def get_llm(model: str = LLM_MODEL, temperature: float = 0.1, max_tokens: int = 1024):
    """Create the chat model. Reads OPENAI_API_KEY from the environment or a .env file."""
    load_dotenv()
    return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
