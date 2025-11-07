from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def make_llm(
    provider: str,
    api_key: Optional[str],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 10000,
):
    """
    Create a LangChain-compatible chat model for OpenAI, Anthropic, or Gemini.

    Args:
        provider: 'openai', 'anthropic', or 'gemini'
        api_key: API key for the provider
        model: Model name (defaults differ by provider)
    """
    provider = provider.lower().strip()

    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider in {"gemini", "google", "google-genai"}:
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-pro",
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,   # Gemini uses `max_output_tokens`
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")
