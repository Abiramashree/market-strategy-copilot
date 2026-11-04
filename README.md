# 📊 Market Strategy Copilot

An AI analyst that turns market data into decision-ready strategy — a **visual dashboard**
of three real case studies plus a **chatbot** that answers follow-up questions, grounded
(RAG) in the same evidence and reasoned by a tool-using LangChain agent.

It unifies three portfolio projects that all share one move — *data → diagnosis →
strategic action plan*:

| Case study | Domain | Source project |
|---|---|---|
| Coca-Cola / Jaguar / BMW campaign sentiment | Marketing / brand strategy (NLP) | [Campaign_analysis](https://github.com/Abiramashree/Campaign_analysis) |
| Nike stock performance | Equity / BI | [Nike_stock_analysis](https://github.com/Abiramashree/Nike_stock_analysis) |
| PowerCo SME churn | Predictive analytics | [Customer-Churn-Analysis](https://github.com/Abiramashree/Customer-Churn-Analysis) |

## What it does

- **Dashboard** (real aggregates, no fabricated numbers): campaign sentiment shares +
  emotion heatmap; Nike signal cards; PowerCo churn KPIs, model metrics, and top drivers —
  including the honest caveat that the churn model's 0.90 accuracy hides 0.06 recall / 0.53 AUC.
- **Chatbot**: ask "should we give the 20% discount?" or "was it the CEO transition?" and
  get a concise, evidence-grounded diagnosis + recommendation. The agent calls a
  `rag_search` tool over the case studies before answering, and keeps conversation history.

## Architecture

```
Case studies (Markdown) ──► chunk ──► embed (SentenceTransformers, local) ──► FAISS
                                                                               │
Dashboard  ◄── real aggregates (data/metrics/*.json) ── Plotly                 │
Chatbot    ◄── LangChain tool-calling agent ── rag_search(query) ──────────────┘
                               │
                 LLM backend (swappable): Ollama / OpenAI-compatible / Gemini
```

## Tech stack

Streamlit · Plotly · LangChain (tool-calling agent) · FAISS · SentenceTransformers
(`all-MiniLM-L6-v2`, local embeddings) · swappable LLM backend.

## Run locally (open-source, no key)

Prereqs: Python 3.10+, and [Ollama](https://ollama.com/download).

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
python scripts/build_index.py
streamlit run app.py
```

## Configuration

All via environment variables (see [`.env.example`](.env.example)). Pick a backend with
`LLM_PROVIDER`:

- `ollama` (default) — local open-source models, no key.
- `openai` — any OpenAI-compatible endpoint (OpenRouter / Groq / LiteLLM / vLLM):
  set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `AGENT_MODEL`.
- `gemini` — native Google Gemini: set `GOOGLE_API_KEY`, `AGENT_MODEL=gemini-3.5-flash-lite`.

## Deploy on Railway (always-on)

Railway runs the container 24/7 with no cold-start sleep (new accounts get a ~$5 trial
credit, then the Hobby plan ~$5/month).

1. Push this repo to GitHub.
2. [railway.com](https://railway.com) → **New Project → Deploy from GitHub repo** → pick
   this repo. Railway uses the included `Dockerfile` and `railway.json` (healthcheck at
   `/_stcore/health`); the `Dockerfile` binds Streamlit to `$PORT`.
3. In the service's **Variables**, set your backend, e.g.:
   ```
   LLM_PROVIDER=gemini
   GOOGLE_API_KEY=AIza...
   AGENT_MODEL=gemini-3.5-flash-lite
   SUMMARY_MODEL=gemini-3.5-flash-lite
   ```
4. Deploy, then under **Settings → Networking → Generate Domain** to get a public URL.

## Project structure

```
├── app.py              # Streamlit dashboard + chat (entry point)
├── viz.py              # Plotly figures from data/metrics/*.json
├── config.py           # env-driven configuration
├── llm.py              # swappable LLM factory (ollama / openai / gemini)
├── rag_engine.py       # chunk, embed, FAISS store, retrieval
├── agent.py            # conversational tool-calling agent (rag_search)
├── data/casestudies/   # Markdown evidence packs (grounding for RAG)
├── data/metrics/       # real aggregates for the dashboard charts
├── scripts/build_index.py
├── tests/              # offline tests (no LLM/network)
├── Dockerfile, railway.json   # Railway deployment
└── requirements.txt, .env.example
```

## Tests

```bash
pytest -q   # retrieval load + FAISS round-trip; no LLM/network needed
```

## Note on the data

Dashboard figures are computed from the real case-study outputs (campaign sentiment/
emotion counts; churn model metrics and churn rate over 14,606 clients). Nike is shown as
qualitative signal cards — the underlying study is technical/fundamental, so no price
numbers are invented.
