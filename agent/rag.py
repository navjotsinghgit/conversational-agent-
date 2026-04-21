"""
agent/rag.py
RAG pipeline: loads the local knowledge base, embeds it, and exposes a
`retrieve(query) -> str` helper for answering product / policy questions.
"""

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ─── Path to the knowledge base ──────────────────────────────────────────────
KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.md"

# ─── Module-level singleton so we build the index only once ──────────────────
_vectorstore: FAISS | None = None


def _build_vectorstore() -> FAISS:
    """Load, split, embed, and index the knowledge base."""
    # Simple plain-text loader — avoids heavy unstructured dependency
    text = KB_PATH.read_text(encoding="utf-8")
    docs = [Document(page_content=text, metadata={"source": str(KB_PATH)})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_vectorstore() -> FAISS:
    """Return (or lazily initialise) the singleton FAISS index."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    return _vectorstore


def retrieve(query: str, k: int = 4) -> str:
    """
    Retrieve the top-k most relevant chunks from the knowledge base.

    Returns:
        A single string with the concatenated chunk contents, separated by
        double newlines — ready to be injected into an LLM prompt.
    """
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)
