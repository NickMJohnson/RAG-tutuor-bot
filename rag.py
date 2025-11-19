# rag.py
"""
Simple Retrieval-Augmented Generation (RAG) helper
for the INFO 4940/5940 tutor bot.

- Loads .pdf, .txt, and .md files from rag_docs/
- Extracts text (page by page for PDFs)
- Embeds chunks with sentence-transformers
- Returns the top-k most similar chunks for a user query
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

DOCS_DIR = Path("rag_docs")

# How big each chunk can be before we split (characters)
MAX_CHARS_PER_CHUNK = 1500


def _extract_text_from_pdf(path: Path) -> str:
    """Extract all text from a PDF using PyPDF2."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(f"[SOURCE: {path.name} – page {i+1}]\n\n{text.strip()}")
    return "\n\n".join(pages)


def _extract_text_from_plain(path: Path) -> str:
    """Extract text from a .txt or .md file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    header = f"[SOURCE: {path.name}]\n\n"
    return header + text.strip()


def _chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """
    Split a long string into smaller chunks of at most max_chars characters.
    This gives the retriever more granular pieces to choose from.
    """
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _load_documents() -> list[str]:
    """
    Load all .pdf, .txt, and .md files from DOCS_DIR and return a list of chunks.
    Each PDF or text file may produce multiple chunks.
    """
    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"{DOCS_DIR} does not exist. "
            "Create it and add syllabus/assignments as .pdf, .txt, or .md files."
        )

    chunks: list[str] = []

    # PDFs first
    for path in sorted(DOCS_DIR.glob("*.pdf")):
        raw_text = _extract_text_from_pdf(path)
        chunks.extend(_chunk_text(raw_text))

    # Plain text / markdown
    for path in sorted(DOCS_DIR.glob("*.txt")) + sorted(DOCS_DIR.glob("*.md")):
        raw_text = _extract_text_from_plain(path)
        chunks.extend(_chunk_text(raw_text))

    if not chunks:
        raise RuntimeError(
            f"No .pdf, .txt, or .md files found in {DOCS_DIR}. "
            "Add your syllabus and homework instructions there."
        )

    return chunks


# --------- Embedding model + precomputed embeddings --------- #

# Small, fast embedding model (recommended style for RAG homework)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Load and embed at import time
_DOCUMENT_CHUNKS: list[str] = _load_documents()
_EMBEDDINGS: list[np.ndarray] = [
    embed_model.encode([doc])[0] for doc in _DOCUMENT_CHUNKS
]
_EMBEDDINGS_MATRIX = np.vstack(_EMBEDDINGS)


# --------- Public API --------- #

def get_top_k_similar_documents(user_query: str, top_k: int = 4) -> List[str]:
    """
    Return the top-k document chunks most similar to the user query.

    These chunks are short passages (1–2 paragraphs) including a [SOURCE: ...]
    header so the LLM can see which syllabus/assignment they came from.
    """
    query = user_query.strip()
    if not query:
        return []

    # Embed the query
    query_embedding = embed_model.encode([query])[0]

    # Cosine similarity between query and each chunk
    numerators = _EMBEDDINGS_MATRIX @ query_embedding
    denom_docs = np.linalg.norm(_EMBEDDINGS_MATRIX, axis=1)
    denom_query = np.linalg.norm(query_embedding) + 1e-9
    similarities = numerators / (denom_docs * denom_query + 1e-9)

    # Indices of top_k highest similarities
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [_DOCUMENT_CHUNKS[i] for i in top_indices]
