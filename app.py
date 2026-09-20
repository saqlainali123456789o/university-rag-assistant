"""
app.py
University Academic Knowledge Assistant — Streamlit RAG chat UI.
Loads a pre-built FAISS index (no raw documents shipped in this repo) and
answers student questions using Groq's openai/gpt-oss-120b, with source +
page citations for traceability.
"""

import streamlit as st
from rag_engine import generate_answer, get_vectorstore, DEFAULT_K

st.set_page_config(
    page_title="University Academic Assistant",
    page_icon="🎓",
    layout="centered",
)

# ---- Sidebar ----
with st.sidebar:
    st.title("🎓 Academic Assistant")
    st.caption("RAG-powered Q&A over your university's official documents.")

    st.markdown("---")
    k = st.slider("Chunks retrieved per question", min_value=2, max_value=8, value=DEFAULT_K)
    st.markdown("---")

    st.markdown(
        "**Model:** `openai/gpt-oss-120b` via Groq\n\n"
        "**Embeddings:** `BAAI/bge-small-en-v1.5`\n\n"
        "**Vector DB:** FAISS (pre-built, no raw docs stored in this repo)"
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- Load index once (cached) so failures show up early and clearly ----
try:
    get_vectorstore()
except Exception as e:
    st.error(f"Failed to load knowledge base: {e}")
    st.stop()

# ---- Chat state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("University Academic Knowledge Assistant")
st.caption("Ask about admissions, fees, exam rules, scholarships, and more.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- **{s['source']}** (page {s['page']}, "
                        f"relevance score {s['score']}) \n  \n  {s['preview']}"
                    )

# ---- Chat input ----
if question := st.chat_input("Ask a question about university policies..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = generate_answer(question, k=k)
            except Exception as e:
                result = {"answer": f"Something went wrong: {e}", "sources": []}

        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(
                        f"- **{s['source']}** (page {s['page']}, "
                        f"relevance score {s['score']}) \n  \n  {s['preview']}"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
