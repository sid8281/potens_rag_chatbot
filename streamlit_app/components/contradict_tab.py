"""
contradict_tab.py — Contradiction checker panel.
Two doc ID inputs, topic field, render contradiction result with evidence
from both documents.
"""
import streamlit as st

from streamlit_app import utils
from streamlit_app.utils import APIError


def render():
    st.subheader("Check for contradictions between two documents")

    col1, col2 = st.columns(2)
    with col1:
        doc_id_1 = st.text_input("Document A", placeholder="e.g. gdpr_full_text.pdf")
    with col2:
        doc_id_2 = st.text_input("Document B", placeholder="e.g. google_privacy_policy.pdf")

    topic = st.text_input("Topic to compare", placeholder="e.g. data retention")

    if st.button("Check contradiction", type="primary"):
        if not (doc_id_1.strip() and doc_id_2.strip() and topic.strip()):
            st.warning("Please fill in both document IDs and a topic.")
            return

        with st.spinner("Retrieving passages and analyzing..."):
            try:
                result = utils.contradict(doc_id_1, doc_id_2, topic)
            except APIError as e:
                st.error(f"**{e.code}**: {e}")
                return
            except Exception as e:
                st.error(f"Could not reach backend: {e}")
                return

        st.markdown("---")

        if result["contradicts"]:
            st.error("⚡ **Contradiction detected**")
        else:
            st.success("✅ **No contradiction found**")

        st.markdown("**Reasoning**")
        st.write(result["reasoning"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Evidence — {doc_id_1}**")
            ev_a = result["evidence_a"]
            with st.container(border=True):
                st.caption(f"chunk {ev_a['chunk_index']} · relevance {ev_a['relevance_score']:.2f}")
                st.write(f"\"{ev_a['snippet']}\"")
        with col_b:
            st.markdown(f"**Evidence — {doc_id_2}**")
            ev_b = result["evidence_b"]
            with st.container(border=True):
                st.caption(f"chunk {ev_b['chunk_index']} · relevance {ev_b['relevance_score']:.2f}")
                st.write(f"\"{ev_b['snippet']}\"")
