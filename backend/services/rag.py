"""RAG service — TF-IDF retrieval from Branch Rickey CSV (no PyTorch required)."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR  = BASE_DIR / "data"
CSV_PATH  = DATA_DIR / "branch-rickey-scouting.csv"
TOP_K     = 5


@lru_cache(maxsize=1)
def _load_index():
    """Load CSV, build TF-IDF matrix (cached after first call)."""
    df = pd.read_csv(CSV_PATH)

    # Keep rows with actual transcription text
    df = df[df["Transcription"].notna() & (df["Transcription"].str.strip() != "")]
    df = df.reset_index(drop=True)

    texts = df["Transcription"].tolist()
    sources = df["Item"].fillna("Rickey Doc").tolist()

    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix, texts, sources


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return top-k relevant Rickey document chunks for a query."""
    vectorizer, matrix, texts, sources = _load_index()

    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        if scores[idx] == 0:
            continue
        results.append({
            "source": sources[idx],
            "text": texts[idx][:600],   # cap chunk length
            "score": float(scores[idx]),
        })
    return results


def context_block(query: str, k: int = TOP_K) -> str:
    """Return a formatted context string for injection into Claude prompts."""
    hits = retrieve(query, k)
    if not hits:
        return ""
    lines = ["--- Branch Rickey Reference ---"]
    for h in hits:
        lines.append(f"[{h['source']}] {h['text']}")
    lines.append("--- End Reference ---")
    return "\n".join(lines)
