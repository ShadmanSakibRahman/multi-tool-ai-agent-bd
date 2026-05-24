"""
Streamlit UI for the Multi-Tool AI Agent for Bangladesh.

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    Push this repo to GitHub, then create a new app at https://share.streamlit.io
    pointed at app.py. Add GROQ_API_KEY in the app's Secrets section.
    On first run, the app downloads the 3 HF datasets and builds the SQLite DBs.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

st.set_page_config(page_title="BD Multi-Tool AI Agent", page_icon=":bangladesh:", layout="wide")

# Pull the Groq key from Streamlit secrets if it is set there, otherwise from env.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

DATA_DIR = Path(__file__).parent / "data"
DB_FILES = ["institutions.db", "hospitals.db", "restaurants.db"]


def ensure_databases():
    """Build the 3 SQLite DBs if any are missing (first cloud run)."""
    missing = [f for f in DB_FILES if not (DATA_DIR / f).exists()]
    if not missing:
        return
    with st.spinner(f"First-time setup: building {len(missing)} database(s) from HuggingFace ..."):
        import build_databases
        build_databases.main()


@st.cache_resource(show_spinner="Loading the agent ...")
def get_agent():
    ensure_databases()
    from agent import build_agent
    return build_agent(verbose=False)


class ToolTracker(BaseCallbackHandler):
    def __init__(self):
        self.calls = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.calls.append((serialized.get("name", "?"), input_str))


st.title("Multi-Tool AI Agent for Bangladesh")
st.caption(
    "Ask about Bangladeshi educational institutions, hospitals, or restaurants - "
    "or ask a general question and the agent will use web search."
)

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is not set. Add it to a local .env file or to Streamlit secrets."
    )
    st.stop()

with st.sidebar:
    st.header("Tools")
    st.write("**InstitutionsDBTool** — schools, colleges, madrasahs (34,901 rows)")
    st.write("**HospitalsDBTool** — hospitals, clinics, health offices (38,886 rows)")
    st.write("**RestaurantsDBTool** — restaurants (12,703 rows)")
    st.write("**WebSearchTool** — DuckDuckGo for general knowledge")
    st.divider()
    st.header("Try one of these")
    examples = [
        "How many government institutions are in Rajshahi?",
        "List 10 hospitals in Dhaka district.",
        "Find restaurants in Chattogram serving biryani.",
        "Which institutes are in Sylhet division?",
        "What is the role of DGHS in Bangladesh?",
        "What is the healthcare policy of Bangladesh?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["question"] = ex

question = st.text_area(
    "Your question",
    value=st.session_state.get("question", ""),
    height=100,
    placeholder="e.g. How many private hospitals are in Chattogram?",
)

if st.button("Ask the agent", type="primary"):
    if not question.strip():
        st.warning("Please type a question.")
    else:
        agent = get_agent()
        tracker = ToolTracker()
        with st.spinner("Thinking ..."):
            try:
                out = agent.invoke({"input": question}, config={"callbacks": [tracker]})
                answer = out.get("output", "")
            except Exception as e:
                answer = f"Error: {e}"

        if tracker.calls:
            tool_names = " -> ".join(c[0] for c in tracker.calls)
            st.info(f"Tool(s) used: **{tool_names}**")

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Show tool calls"):
            for name, inp in tracker.calls:
                st.code(f"{name}({inp[:300]})", language="text")
