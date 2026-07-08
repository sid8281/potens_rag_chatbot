"""
ingest_tab.py — File upload panel.
File uploader, call /ingest, show chunk count on success.
"""
import streamlit as st

from streamlit_app import utils
from streamlit_app.utils import APIError


def render():
    st.subheader("Ingest a new document")

    uploaded_file = st.file_uploader(
        "Upload a document", type=["pdf", "txt", "md"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        if st.button("Ingest document", type="primary"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    result = utils.ingest(uploaded_file.getvalue(), uploaded_file.name)
                except APIError as e:
                    st.error(f"**{e.code}**: {e}")
                    return
                except Exception as e:
                    st.error(f"Could not reach backend: {e}")
                    return

            st.success(
                f"✅ Ingested **{result['doc_id']}** — "
                f"{result['chunks_created']} chunks created."
            )

    st.markdown("---")
    st.markdown("**Current index status**")
    if st.button("Refresh status"):
        try:
            status = utils.health()
            st.metric("Documents indexed", status.get("chroma_docs", "—"))
            st.metric("Total chunks", status.get("total_chunks", "—"))
            st.caption(f"Embedding model: {status.get('embedding_model', '—')}")
        except Exception as e:
            st.error(f"Could not reach backend: {e}")
