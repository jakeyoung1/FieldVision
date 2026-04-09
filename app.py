import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from PIL import Image

try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

import base64
import faiss
from fpdf import FPDF
import anthropic
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FieldVision | Baseball Scouting Intelligence",
    page_icon="⚾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLAUDE_MODEL    = "claude-haiku-4-5"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # local — no API quota
CSV_PATH        = "data/branch-rickey-scouting.csv"
EMBEDDINGS_PATH = "data/embeddings.npy"
TOP_K           = 5


# ---------------------------------------------------------------------------
# Local sentence-transformer model (cached — loaded once per app session)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embed_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------------------------
# Anthropic client (cached — created once per app session)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_client() -> anthropic.Anthropic:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    if not api_key:
        st.error(
            "ANTHROPIC_API_KEY not found. "
            "Add it to your Streamlit secrets or set it as an environment variable."
        )
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# RAG — load Branch Rickey CSV and build FAISS index (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_rag_index():
    if not os.path.exists(CSV_PATH):
        st.error(f"Knowledge base not found at `{CSV_PATH}`. See setup instructions.")
        st.stop()

    df = pd.read_csv(CSV_PATH)
    df = (
        df[df["Transcription"].notna() & (df["Transcription"].str.strip() != "")]
        .reset_index(drop=True)
    )
    # ---- Load pre-computed embeddings (run precompute_embeddings.py first) --
    if not os.path.exists(EMBEDDINGS_PATH):
        st.error(
            f"Embeddings file not found at `{EMBEDDINGS_PATH}`. "
            "Run `python precompute_embeddings.py` once to generate it."
        )
        st.stop()

    embeddings_np = np.load(EMBEDDINGS_PATH).astype(np.float32)

    # ---- Build FAISS index (cosine similarity via inner product) -----------
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)

    return index, df


# ---------------------------------------------------------------------------
# Transcription — Claude Vision OCR
# ---------------------------------------------------------------------------
def transcribe_image(client: anthropic.Anthropic, image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are transcribing a handwritten baseball scouting or match note. "
                            "Transcribe ALL text exactly as written, preserving line breaks, "
                            "abbreviations, shorthand, numbers, and player names. "
                            "If text is illegible indicate with [?]."
                        ),
                    },
                ],
            }
        ],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Retrieval — embed query and search FAISS index
# ---------------------------------------------------------------------------
def retrieve_context(
    query_text: str,
    index,
    df: pd.DataFrame,
    k: int = TOP_K,
) -> list[dict]:
    embed_model = get_embed_model()
    query_vec = embed_model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)

    hits = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        hits.append(
            {
                "score": float(score),
                "text": row["Transcription"],
                "item": row.get("Item", ""),
                "project": row.get("Project", ""),
            }
        )
    return hits


# ---------------------------------------------------------------------------
# Generation — produce structured scouting insights
# ---------------------------------------------------------------------------
def generate_insights(
    client: anthropic.Anthropic,
    transcriptions: list[str],
    historical_context: list[dict],
) -> str:
    new_notes = "\n\n---\n\n".join(transcriptions)

    historical_text = ""
    for i, ctx in enumerate(historical_context, 1):
        snippet = ctx["text"][:1000] + ("…" if len(ctx["text"]) > 1000 else "")
        historical_text += f"\n[Historical Report {i} | relevance {ctx['score']:.2f}]\n{snippet}\n"

    prompt = f"""You are an expert baseball analyst and scouting consultant.
You have been given newly digitized handwritten baseball scouting notes and a set of
relevant historical Branch Rickey scouting reports for context and benchmarking.

NEWLY UPLOADED SCOUTING NOTES:
{new_notes}

RELEVANT HISTORICAL BRANCH RICKEY SCOUTING REPORTS:
{historical_text}

Produce a structured Scouting Intelligence Report with the following sections:

## 1. Overview
Summarize who and what is being scouted (players, teams, game context).

## 2. Key Strengths
Bullet-point observable strengths from the uploaded notes.

## 3. Development Areas
Bullet-point areas that need improvement based on the notes.

## 4. Historical Benchmarking
Compare the scouted player(s) or observations to the standards and evaluation language
used in the Branch Rickey reports. Note similarities or differences.

## 5. Actionable Recommendations
3–5 specific, concrete coaching or recruitment actions grounded in the notes.

## 6. Priority Actions — Next 30 Days
What should the coaching staff or scout do first? Be specific."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------
def build_pdf(title: str, body: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(4)

    # Body — split into lines to handle wrapping
    pdf.set_font("Helvetica", size=11)
    for line in body.split("\n"):
        # Headings (lines starting with ##)
        if line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.ln(3)
            pdf.multi_cell(0, 8, line.replace("## ", ""))
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("# "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(3)
            pdf.multi_cell(0, 8, line.replace("# ", ""))
            pdf.set_font("Helvetica", size=11)
        else:
            pdf.multi_cell(0, 7, line if line.strip() else " ")

    return pdf.output()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def collect_images(uploaded_files) -> list[tuple[str, Image.Image]]:
    images = []
    for f in uploaded_files:
        raw = f.read()
        if f.name.lower().endswith(".pdf"):
            if not PDF_SUPPORT:
                st.warning(f"PDF support unavailable — skipping {f.name}. Install poppler.")
                continue
            pages = convert_from_bytes(raw)
            for i, page in enumerate(pages, 1):
                images.append((f"{f.name} (page {i})", page))
        else:
            images.append((f.name, Image.open(io.BytesIO(raw))))
    return images


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.title("⚾ FieldVision")
    st.markdown(
        "**Baseball Scouting Intelligence** — Upload handwritten scouting notes to "
        "digitize them and receive actionable insights backed by historical Branch Rickey data."
    )
    st.divider()

    with st.spinner("Initializing knowledge base…"):
        rag_index, rag_df = load_rag_index()

    client = get_client()

    st.subheader("Upload Scouting Notes")
    uploaded_files = st.file_uploader(
        "Choose images or PDFs of handwritten baseball scouting notes",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more scouting note images or PDFs above to get started.")
        return

    st.markdown(f"**{len(uploaded_files)} file(s) selected.**")

    if not st.button("Transcribe & Analyze", type="primary", use_container_width=True):
        return

    with st.spinner("Reading files…"):
        images = collect_images(uploaded_files)

    if not images:
        st.error("No readable images found in the uploaded files.")
        return

    # --- Transcribe all images silently with a single progress indicator ---
    transcriptions = []
    with st.spinner(f"Transcribing {len(images)} page(s)…"):
        for name, image in images:
            text = transcribe_image(client, image)
            transcriptions.append(text)

    if not transcriptions:
        st.error("No transcriptions generated.")
        return

    # --- Show uploaded pages as small thumbnails ---
    st.subheader("Uploaded Pages")
    thumb_cols = st.columns(min(len(images), 4))
    for i, (name, image) in enumerate(images):
        with thumb_cols[i % 4]:
            st.image(image, caption=name, use_container_width=True)

    # --- Show one combined transcription ---
    combined_transcription = "\n\n".join(transcriptions)
    with st.expander("Combined Transcription", expanded=False):
        st.text_area(
            label="combined",
            value=combined_transcription,
            height=300,
            label_visibility="collapsed",
        )

    # --- RAG retrieval and single unified analysis ---
    with st.spinner("Searching historical scouting database…"):
        context = retrieve_context(combined_transcription, rag_index, rag_df)

    st.subheader("Scouting Intelligence Report")
    with st.spinner("Generating insights…"):
        insights = generate_insights(client, [combined_transcription], context)

    st.markdown(insights)

    with st.expander("Historical Reports Used for Context", expanded=False):
        for i, ctx in enumerate(context, 1):
            st.markdown(f"**Report {i}** — relevance score: `{ctx['score']:.3f}`")
            st.caption(ctx["item"])
            snippet = ctx["text"][:600] + ("…" if len(ctx["text"]) > 600 else "")
            st.text(snippet)
            st.divider()

    st.divider()
    st.subheader("Download Reports")
    col1, col2 = st.columns(2)

    with col1:
        notes_pdf = build_pdf("Digitized Scouting Notes", combined_transcription)
        st.download_button(
            label="Download Digitized Notes (.pdf)",
            data=bytes(notes_pdf),
            file_name="scouting_notes.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with col2:
        analysis_pdf = build_pdf("Scouting Intelligence Report", insights)
        st.download_button(
            label="Download Analysis Report (.pdf)",
            data=bytes(analysis_pdf),
            file_name="scouting_analysis.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
