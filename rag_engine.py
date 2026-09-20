"""
rag_engine.py
Core RAG logic: load the pre-built FAISS index (embeddings only, no raw
documents are shipped in this repo), retrieve relevant chunks, and generate
a cited answer using Groq's openai/gpt-oss-120b model.
"""

import os
from typing import List, Tuple

import streamlit as st
from groq import Groq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---- Config ----
INDEX_DIR = "faiss_index"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # MUST match the model used at ingestion time
GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_K = 4

SYSTEM_PROMPT = """You are a university academic assistant. Answer the
student's question using ONLY the context excerpts provided below. Each
excerpt is labeled with its source document and page number.

Rules:
- If the answer is not contained in the context, say you don't have that
  information in the provided documents — do not make anything up.
- Always cite the source document and page number for any fact you state,
  in the form (Source: <document>, p. <page>).
- Be concise and clear; this is for students, not lawyers.
"""


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_vectorstore() -> FAISS:
    embeddings = get_embeddings()
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(
            f"'{INDEX_DIR}/' not found. Make sure the FAISS index built in "
            f"Colab (index.faiss + index.pkl) is committed to this repo."
        )
    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: it's our own trusted index
    )


@st.cache_resource(show_spinner=False)
def get_groq_client() -> Groq:
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Add it in Streamlit Cloud under "
            "Settings -> Secrets, or in .streamlit/secrets.toml locally."
        )
    return Groq(api_key=api_key)


def retrieve(query: str, k: int = DEFAULT_K) -> List[Tuple[Document, float]]:
    """Return top-k (chunk, similarity_score) pairs for the query."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(query, k=k)


def build_context_block(results: List[Tuple[Document, float]]) -> str:
    """Format retrieved chunks into a labeled context block for the prompt."""
    blocks = []
    for doc, _score in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, k: int = DEFAULT_K, temperature: float = 0.2) -> dict:
    """
    Run full RAG: retrieve -> build prompt -> call Groq -> return answer +
    the raw sources used, so the UI can render citations.
    """
    results = retrieve(query, k=k)

    if not results:
        return {
            "answer": "I couldn't find anything relevant in the knowledge base for that question.",
            "sources": [],
        }

    context = build_context_block(results)
    client = get_groq_client()

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {query}",
            },
        ],
    )

    answer = completion.choices[0].message.content

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": round(float(score), 4),
            "preview": doc.page_content[:220].strip() + "...",
        }
        for doc, score in results
    ]

    return {"answer": answer, "sources": sources}
