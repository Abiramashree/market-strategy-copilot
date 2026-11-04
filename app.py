"""
app.py
Streamlit dashboard + chat for the Market Strategy Copilot.

Top: a visual dashboard of three real case studies (brand campaigns, Nike stock,
PowerCo churn) built from genuine aggregates. Bottom: a chatbot that answers
follow-up questions, grounded in the same evidence. Backend is swappable
(Ollama / OpenAI-compatible / Gemini). Run: streamlit run app.py
"""
import os

import streamlit as st

# Some hosts pass config via st.secrets; bridge into env before importing config.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

import config
import viz

st.set_page_config(page_title="Market Strategy Copilot", page_icon="📊", layout="wide")

EXAMPLES = [
    "Does price sensitivity explain PowerCo's churn? Should we give the 20% discount?",
    "Why did Nike's stock fall — was it the CEO transition? What's the action plan?",
    "What drove backlash across the campaigns, and what should Jaguar do next?",
]


def backend_ready() -> tuple[bool, str]:
    if config.LLM_PROVIDER == "gemini":
        ok = bool(config.GOOGLE_API_KEY)
        return ok, "Using native Google Gemini" if ok else "LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set."
    if config.LLM_PROVIDER != "ollama":
        ok = bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY != "not-needed")
        return ok, (f"Using OpenAI-compatible endpoint: {config.OPENAI_BASE_URL}" if ok
                    else "LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
    try:
        import urllib.request

        with urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2) as resp:
            return resp.status == 200, f"Ollama reachable at {config.OLLAMA_BASE_URL}"
    except Exception:
        return False, (f"Ollama not reachable at {config.OLLAMA_BASE_URL}. Run `ollama serve` "
                       f"and `ollama pull {config.AGENT_MODEL}`.")


@st.cache_resource(show_spinner=False)
def get_executor():
    from agent import build_agent_executor

    return build_agent_executor()


def to_history(messages):
    from langchain_core.messages import AIMessage, HumanMessage

    return [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
            for m in messages]


ready, backend_msg = backend_ready()

st.title("📊 Market Strategy Copilot")
st.caption("A visual + conversational analyst over three real case studies — brand campaigns, Nike stock, and PowerCo churn.")

with st.sidebar:
    st.header("About")
    st.markdown(
        "Explore the dashboards, then ask the copilot follow-up questions. Answers are "
        "RAG-grounded in the same case-study evidence and reasoned by a tool-using agent."
    )
    st.subheader("Backend")
    (st.success if ready else st.error)(backend_msg)
    st.caption(f"provider: `{config.LLM_PROVIDER}` · model: `{config.AGENT_MODEL}`")

tab_c, tab_n, tab_p = st.tabs(["📣 Brand campaigns", "📈 Nike stock", "⚡ PowerCo churn"])

with tab_c:
    data = viz.load_metrics("campaign")
    st.subheader("Sentiment & emotion across three campaigns")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sentiment (share of coverage)**")
        st.plotly_chart(viz.campaign_sentiment_fig(data), use_container_width=True)
    with c2:
        st.markdown("**Emotion mix (items per brand)**")
        st.plotly_chart(viz.campaign_emotion_fig(data), use_container_width=True)
    st.info(data["takeaway"])

with tab_n:
    data = viz.load_metrics("nike")
    st.subheader("Stock performance — signals")
    cols = st.columns(len(data["signals"]))
    for col, sig in zip(cols, data["signals"]):
        with col:
            st.metric(sig["label"], sig["value"])
            st.caption(sig["detail"])
    st.markdown("**Action plan**")
    for step in data["action_plan"]:
        st.markdown(f"- {step}")
    st.info(data["takeaway"])

with tab_p:
    data = viz.load_metrics("churn")
    k1, k2, k3 = st.columns(3)
    k1.metric("SME clients", f"{data['clients']:,}")
    k2.metric("Churn rate", f"{data['churn_rate']*100:.1f}%", f"{data['churn_count']:,} churned")
    k3.metric("Model", "Random Forest")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Model metrics** (recall & AUC in red — the honest caveat)")
        st.plotly_chart(viz.churn_metrics_fig(data), use_container_width=True)
    with c2:
        st.markdown("**Top churn drivers**")
        for d in data["top_drivers"]:
            st.markdown(f"- {d}")
    st.caption(data["metrics_note"])
    for s in data["signals"]:
        st.markdown(f"- {s}")
    st.info(data["takeaway"])

st.divider()
st.subheader("💬 Ask the copilot a follow-up")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["pending"] = ex

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about the campaigns, Nike stock, or PowerCo churn…")
if "pending" in st.session_state and not prompt:
    prompt = st.session_state.pop("pending")

if prompt:
    if not ready:
        st.error(backend_msg)
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and reasoning…"):
                try:
                    history = to_history(st.session_state["messages"][:-1])
                    result = get_executor().invoke({"input": prompt, "chat_history": history})
                    answer = result.get("output", "") or "(no answer)"
                except Exception as exc:  # noqa: BLE001
                    answer = f"⚠️ Error: {exc}"
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
