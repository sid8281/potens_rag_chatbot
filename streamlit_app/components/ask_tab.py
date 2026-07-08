"""
ask_tab.py — Q&A panel.
Text input, submit, render answer, render citation cards, HITL warning
if the confidence score falls below threshold.
"""
import streamlit as st

from streamlit_app import utils
from streamlit_app.utils import APIError


def render():
    st.subheader("Ask a question")

    query = st.text_input(
        "Your question (any language)",
        placeholder="e.g. What does the policy say about data retention?",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    ask_clicked = st.button("Ask", type="primary", use_container_width=False)

    if ask_clicked:
        if not query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving and generating answer..."):
            try:
                result = utils.ask(query=query, top_k=int(top_k))
            except APIError as e:
                st.error(f"**{e.code}**: {e}")
                return
            except Exception as e:
                st.error(f"Could not reach backend: {e}")
                return

        st.markdown("---")

        # No-answer path — explicit, not silent
        if result.get("no_answer"):
            st.error("**No answer found.** The documents do not contain information to answer this question.")
            return

        # HITL warning banner
        if result.get("hitl_required"):
            st.warning(
                f"⚠️ Low confidence answer (score: {result['confidence']:.2f}) — "
                "human review recommended."
            )

        st.markdown(f"**Answer** _(responded in {result.get('language', 'English')})_")
        st.write(result["answer"])

        st.caption(f"Confidence: {result['confidence']:.2f}")

        citations = result.get("citations", [])
        if citations:
            st.markdown("**Sources**")
            for c in citations:
                page_str = f", page {c['page_number']}" if c.get("page_number") else ""
                with st.container(border=True):
                    st.markdown(
                        f"📄 **{c['source_file']}** — chunk {c['chunk_index']}{page_str} "
                        f"· relevance {c['relevance_score']:.2f}"
                    )
                    st.caption(f"\"{c['snippet']}\"")
