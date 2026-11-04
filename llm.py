"""
llm.py
Single factory for the chat LLM. The rest of the codebase never imports a provider
directly, so the backend is swappable by configuration alone.

- ollama : local open-source models (no key)
- openai : any OpenAI-compatible endpoint (OpenRouter / Groq / LiteLLM / vLLM)
- gemini : native Google Gemini (preserves thinking-model thought signatures through
           the agent's tool loop, which the OpenAI-compat shim does not)
"""
from __future__ import annotations

from typing import Optional

import config


def make_llm(model: Optional[str] = None, temperature: float = 0.0):
    model = model or config.AGENT_MODEL

    if config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=config.OLLAMA_BASE_URL, temperature=temperature)

    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, google_api_key=config.GOOGLE_API_KEY, temperature=temperature
        )

    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY,
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={config.LLM_PROVIDER!r}. Use 'ollama', 'openai', or 'gemini'."
    )
