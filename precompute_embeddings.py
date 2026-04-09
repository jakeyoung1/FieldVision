"""
Run this ONCE before launching the app to precompute Branch Rickey embeddings.

    .venv/bin/python precompute_embeddings.py

Saves data/embeddings.npy — commit that file to your repo so the app
never needs to recompute embeddings at startup.

Uses sentence-transformers (runs locally, no API key, no quota limits).
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CSV_PATH        = "data/branch-rickey-scouting.csv"
OUTPUT_PATH     = "data/embeddings.npy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, lightweight, great for semantic search

print("Loading embedding model (downloads once, ~80 MB)…")
model = SentenceTransformer(EMBEDDING_MODEL)

df = pd.read_csv(CSV_PATH)
df = (
    df[df["Transcription"].notna() & (df["Transcription"].str.strip() != "")]
    .reset_index(drop=True)
)
texts = df["Transcription"].tolist()
print(f"Embedding {len(texts)} scouting reports…")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

embeddings_np = embeddings.astype(np.float32)
os.makedirs("data", exist_ok=True)
np.save(OUTPUT_PATH, embeddings_np)
print(f"\nDone. Saved to {OUTPUT_PATH}  shape: {embeddings_np.shape}")
print("Commit data/embeddings.npy to your repo — the app will load it instantly.")
