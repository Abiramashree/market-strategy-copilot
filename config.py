"""
config.py
Environment-driven configuration for the Market Strategy Copilot.

Override anything via environment variables (see .env.example). The same code runs on a
local open-source stack (Ollama), any OpenAI-compatible gateway, or native Gemini.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# --- Data & index locations -------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data/casestudies")
PERSIST_DIR = os.getenv("PERSIST_DIR", "faiss_store")

# --- Retrieval / chunking ---------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # open-source, local, CPU
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "6"))

# --- LLM backend ------------------------------------------------------------
# "ollama" (local open-source), "openai" (any OpenAI-compatible server), or "gemini".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "llama3.1:8b")
AGENT_MODEL = os.getenv("AGENT_MODEL", "llama3.1:8b")

# LLM_PROVIDER=openai (LiteLLM gateway, OpenRouter, Groq, vLLM, ...)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:4000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")

# LLM_PROVIDER=gemini (native Google Gemini; handles thinking-model tool loops)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))
