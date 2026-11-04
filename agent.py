"""
agent.py
Conversational, tool-using LangChain agent for the Market Strategy Copilot.

It grounds every answer in the case-study evidence via a `rag_search` tool, keeps
chat history, and replies like a market-strategy consultant. The LLM is configured via
llm.make_llm (open-source Ollama, any OpenAI-compatible endpoint, or native Gemini).
"""
from __future__ import annotations

from typing import Optional

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover
    from langchain.tools import tool

try:
    # LangChain >= 1.0 moved the classic agent API to langchain-classic.
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
except ImportError:  # pragma: no cover - LangChain < 1.0
    from langchain.agents import AgentExecutor, create_tool_calling_agent

from langchain_core.prompts import ChatPromptTemplate

import config
from llm import make_llm
from rag_engine import RAGSearch

SYSTEM_PROMPT = (
    "You are a senior market-analysis & strategy consultant chatbot. You cover three "
    "real case studies: (1) brand-campaign sentiment for Coca-Cola / Jaguar / BMW, "
    "(2) Nike stock performance, and (3) PowerCo SME customer churn.\n\n"
    "Rules:\n"
    "- ALWAYS call the `rag_search` tool first to ground your answer in the case-study "
    "evidence before responding.\n"
    "- Be concise and concrete: cite specific numbers, drivers and caveats from the "
    "evidence. Never invent figures.\n"
    "- When asked what to do, give a short diagnosis, then a clear recommendation and a "
    "prioritised action plan.\n"
    "- Be honest about weak data or unreliable models (e.g. a classifier with high "
    "accuracy but near-random recall).\n"
    "- If a question falls outside the three case studies, say so and answer only what "
    "the evidence supports.\n"
    "- Use short markdown (bold labels, bullets) so answers are easy to skim in chat."
)

_rag: Optional[RAGSearch] = None


def get_rag() -> RAGSearch:
    """Lazily initialise the shared RAGSearch (no index build / model load at import)."""
    global _rag
    if _rag is None:
        _rag = RAGSearch(
            persist_dir=config.PERSIST_DIR,
            embedding_model=config.EMBED_MODEL,
            llm_model=config.SUMMARY_MODEL,
        )
    return _rag


@tool("rag_search", return_direct=False)
def rag_search(query: str) -> str:
    """Retrieve relevant case-study evidence (brand campaigns, Nike stock, PowerCo churn)
    for a query from the FAISS-backed vector store."""
    results = get_rag().vectorstore.query(query, top_k=config.TOP_K)
    texts = [r.get("metadata", {}).get("text", "") for r in results if r.get("metadata")]
    joined = "\n\n".join(t for t in texts if t)
    return joined if joined.strip() else "[EMPTY]"


def build_agent_executor() -> AgentExecutor:
    """Build the conversational tool-calling agent (grounds answers via rag_search)."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    llm = make_llm(model=config.AGENT_MODEL, temperature=0)
    tools = [rag_search]
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
