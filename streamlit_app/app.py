"""
app.py — main Streamlit UI entrypoint.
Tab layout wiring the three panels together. Run alongside FastAPI:

    Terminal 1: uvicorn app.main:app --reload
    Terminal 2: streamlit run streamlit_app/app.py
"""
import streamlit as st

import os
import sys

# Add the parent directory of streamlit_app to sys.path so streamlit_app is discoverable as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit_app.components import ask_tab, contradict_tab, ingest_tab

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

st.title("📄 RAG Document Q&A with Citations")
st.caption("Ask questions across your documents, check for contradictions, and ingest new files.")

tab_ask, tab_contradict, tab_ingest = st.tabs(["💬 Ask", "⚖️ Contradict", "📥 Ingest"])

with tab_ask:
    ask_tab.render()

with tab_contradict:
    contradict_tab.render()

with tab_ingest:
    ingest_tab.render()
