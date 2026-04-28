"""RAG service — FAISS index + sentence-transformer embeddings."""
import os
import pickle
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH  = DATA_DIR / "rickey_index.faiss"
META_PATH   = DATA_DIR / "rickey_meta.pkl"
TOP_K       = 5


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _load_index():
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return top-k relevant Rickey document chunks for a query."""
    model = _load_model()
    index, meta = _load_index()

    vec = model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        entry = meta[idx].copy()
        entry["score"] = float(dist)
        results.append(entry)
    return results


def context_block(query: str, k: int = TOP_K) -> str:
    """Return a formatted context string for injection into Claude prompts."""
    hits = retrieve(query, k)
    if not hits:
        return ""
    lines = ["--- Branch Rickey Reference ---"]
    for h in hits:
        lines.append(f"[{h.get('source', 'doc')}] {h.get('text', '')}")
    lines.append("--- End Reference ---")
    return "\n".join(lines)
