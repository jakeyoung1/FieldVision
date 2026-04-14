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

import uuid
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

SYSTEM_PROMPT = """You are an experienced baseball scout and analyst with decades of evaluating players at every level — high school, college, and professional. You have deep knowledge of hitting mechanics, pitching, fielding, baserunning, and long-term player development.

Your job is to read scouting notes and turn them into clear, useful intelligence for coaches and scouts.

When analyzing notes:
- Begin with a brief, direct summary of what the notes cover (2–3 sentences max)
- Follow with specific, actionable recommendations the coaching staff can act on immediately
- Be concrete — reference specific observations from the notes, not generic baseball advice
- Use standard scouting language and baseball terminology
- Draw on historical scouting standards when benchmarking a player
- If the user specifies a focus area or desired output, prioritize that above all else

When answering follow-up questions:
- Stay grounded in the scouting notes provided
- Be direct and concise
- If something isn't in the notes, say so rather than speculating
"""


# ---------------------------------------------------------------------------
# Session context builder
# ---------------------------------------------------------------------------
def build_session_context(session_items: list[dict]) -> str:
    """Combine all prior session uploads into a single context block."""
    if not session_items:
        return ""
    lines = ["=== PRIOR SESSION MATERIALS (use as background context) ==="]
    for item in session_items:
        lines.append(f"\n[{item['type'].upper()} — {item['label']}]")
        # Truncate each item to avoid token overflow
        lines.append(item["content"][:4000])
    return "\n".join(lines)


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
    transcription: str,
    historical_context: list[dict],
    notes_context: str = "",
    output_focus: str = "",
    session_context: str = "",
) -> str:
    historical_text = ""
    for i, ctx in enumerate(historical_context, 1):
        snippet = ctx["text"][:1000] + ("…" if len(ctx["text"]) > 1000 else "")
        historical_text += f"\n[Historical Report {i}]\n{snippet}\n"

    user_context_block = ""
    if notes_context:
        user_context_block += f"\nContext about these notes: {notes_context}"
    if output_focus:
        user_context_block += f"\nFocus the output on: {output_focus}"

    session_block = f"\n\n{session_context}" if session_context else ""

    prompt = f"""Here are the scouting notes to analyze:{user_context_block}{session_block}

SCOUTING NOTES:
{transcription}

RELEVANT HISTORICAL BRANCH RICKEY SCOUTING REPORTS (for benchmarking):
{historical_text}

Provide:

## Summary
A brief 2–3 sentence summary of what these notes cover.

## Actionable Recommendations
Give exactly 2 recommendations grounded directly in the scouting notes above. Be specific and concrete.

## Bonus Insight from the Branch Rickey Papers
Draw one additional insight by connecting something in these notes to the historical Branch Rickey scouting reports provided. Frame it as a historical parallel or lesson from Rickey's scouting philosophy that applies to what you see in these notes."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Follow-up chat response
# ---------------------------------------------------------------------------
def chat_response(
    client: anthropic.Anthropic,
    context: str,
    chat_history: list[dict],
    session_context: str = "",
) -> str:
    session_block = f"\n\n{session_context}" if session_context else ""
    context_message = f"""For reference, here is all the material available for this session:{session_block}

CURRENT DOCUMENT:
{context}

Answer the user's question based on this material."""

    messages = [
        {"role": "user",      "content": context_message},
        {"role": "assistant", "content": "Understood. I have all the session materials loaded. What would you like to know?"},
    ]
    messages += chat_history

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------
def sanitize_for_pdf(text: str) -> str:
    """Replace unicode characters that Helvetica can't render."""
    replacements = {
        "\u2026": "...", "\u2014": "--", "\u2013": "-",
        "\u2018": "'",  "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2022": "-",  "\u00a0": " ", "\u2012": "-", "\u2015": "--",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf(title: str, body: str) -> bytes:
    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, sanitize_for_pdf(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Body
    pdf.set_font("Helvetica", size=11)
    for line in body.split("\n"):
        clean = sanitize_for_pdf(line)
        if line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.ln(3)
            pdf.multi_cell(0, 8, clean.replace("## ", ""), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("# "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(3)
            pdf.multi_cell(0, 8, clean.replace("# ", ""), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        else:
            pdf.multi_cell(0, 7, clean if clean.strip() else " ", new_x="LMARGIN", new_y="NEXT")

    return pdf.output()


# ---------------------------------------------------------------------------
# Trackman — summarize CSV into clean text for Claude
# ---------------------------------------------------------------------------
def summarize_trackman(df: pd.DataFrame) -> str:
    lines = []

    # Game metadata
    date    = df["Date"].iloc[0]    if "Date"     in df.columns else "Unknown"
    stadium = df["Stadium"].iloc[0] if "Stadium"  in df.columns else "Unknown"
    home    = df["HomeTeam"].iloc[0] if "HomeTeam" in df.columns else "Unknown"
    away    = df["AwayTeam"].iloc[0] if "AwayTeam" in df.columns else "Unknown"
    lines.append(f"NOTE: This is post-game Trackman data. The game has already been completed.")
    lines.append(f"Game: {away} (Away) @ {home} (Home) | {date} | {stadium}")
    lines.append(f"Total pitches tracked: {len(df)}\n")

    def pitcher_block(pdf, pitcher):
        throws = pdf["PitcherThrows"].iloc[0]
        total  = len(pdf)
        block  = [f"\n  {pitcher} ({throws}H) — {total} pitches"]

        if "TaggedPitchType" in df.columns:
            mix     = pdf["TaggedPitchType"].value_counts()
            mix_str = ", ".join([f"{pt}: {cnt} ({cnt/total*100:.0f}%)" for pt, cnt in mix.items()])
            block.append(f"    Pitch mix: {mix_str}")

        for pitch_type, ptdf in pdf.groupby("TaggedPitchType"):
            stats = []
            if "RelSpeed"        in df.columns: stats.append(f"Velo: {ptdf['RelSpeed'].mean():.1f} mph")
            if "SpinRate"        in df.columns: stats.append(f"Spin: {ptdf['SpinRate'].mean():.0f} rpm")
            if "InducedVertBreak" in df.columns: stats.append(f"IVB: {ptdf['InducedVertBreak'].mean():.1f}\"")
            if "HorzBreak"       in df.columns: stats.append(f"HB: {ptdf['HorzBreak'].mean():.1f}\"")
            if stats:
                block.append(f"      {pitch_type}: {' | '.join(stats)}")

        if "PitchCall" in df.columns:
            calls  = pdf["PitchCall"].value_counts()
            strikes = sum(calls.get(k, 0) for k in ["StrikeCalled", "StrikeSwinging", "FoulBall", "InPlay"])
            balls   = calls.get("BallCalled", 0)
            block.append(f"    Strike%: {strikes/total*100:.0f}% | K: {(pdf['KorBB']=='Strikeout').sum()} | BB: {(pdf['KorBB']=='Walk').sum()}")

        return block

    def batter_block(bdf, batter):
        side  = bdf["BatterSide"].iloc[0]
        block = [f"\n  {batter} ({side}H)"]

        if "PlayResult" in df.columns:
            results    = bdf["PlayResult"].value_counts()
            result_str = ", ".join([f"{r}: {c}" for r, c in results.items() if r not in ("Undefined", "")])
            if result_str:
                block.append(f"    Results: {result_str}")

        if "KorBB" in df.columns:
            ks  = (bdf["KorBB"] == "Strikeout").sum()
            bbs = (bdf["KorBB"] == "Walk").sum()
            if ks or bbs:
                block.append(f"    K: {ks} | BB: {bbs}")

        contact = bdf[bdf["ExitSpeed"].notna() & (bdf["ExitSpeed"] > 0)] if "ExitSpeed" in df.columns else pd.DataFrame()
        if not contact.empty:
            block.append(f"    Exit Velo: avg {contact['ExitSpeed'].mean():.1f} mph | max {contact['ExitSpeed'].max():.1f} mph")
            if "Angle" in df.columns:
                block.append(f"    Launch Angle: avg {contact['Angle'].mean():.1f} deg")

        return block

    SMC_CODE = "STM_GAE"
    opp_code = away if home == SMC_CODE else home
    opp_label = opp_code

    # --- Saint Mary's ---
    lines.append("=== SAINT MARY'S (STM_GAE) ===")
    lines.append("-- Pitching --")
    for pitcher, pdf in df[df["PitcherTeam"] == SMC_CODE].groupby("Pitcher"):
        lines.extend(pitcher_block(pdf, pitcher))

    lines.append("\n-- Batting --")
    for batter, bdf in df[df["BatterTeam"] == SMC_CODE].groupby("Batter"):
        lines.extend(batter_block(bdf, batter))

    # --- Opponent ---
    lines.append(f"\n=== OPPONENT ({opp_label}) ===")
    lines.append("-- Pitching --")
    for pitcher, pdf in df[df["PitcherTeam"] != SMC_CODE].groupby("Pitcher"):
        lines.extend(pitcher_block(pdf, pitcher))

    lines.append("\n-- Batting --")
    for batter, bdf in df[df["BatterTeam"] != SMC_CODE].groupby("Batter"):
        lines.extend(batter_block(bdf, batter))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trackman — Claude analysis
# ---------------------------------------------------------------------------
def analyze_trackman(
    client: anthropic.Anthropic,
    summary: str,
    notes_context: str = "",
    output_focus: str = "",
    session_context: str = "",
) -> str:
    user_context_block = ""
    if notes_context:
        user_context_block += f"\nContext: {notes_context}"
    if output_focus:
        user_context_block += f"\nFocus on: {output_focus}"

    session_block = f"\n\n{session_context}" if session_context else ""

    prompt = f"""Here is a Trackman pitch-by-pitch data summary from a completed baseball game:{user_context_block}{session_block}

TRACKMAN DATA SUMMARY:
{summary}

Provide:

## Summary
A brief 2-3 sentence overview of what happened in this game from a pitching and hitting standpoint.

## Actionable Recommendations
2 specific recommendations grounded directly in the data above.

## Bonus Insight
One deeper observation — an interesting pattern, matchup, or trend in the data worth flagging to the coaching staff."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Shared chat renderer
# ---------------------------------------------------------------------------
def render_chat(client: anthropic.Anthropic, context: str, state_key: str, session_context: str = "", label: str = "Ask the Scout") -> None:
    st.markdown(f'<p class="fv-label">{label}</p>', unsafe_allow_html=True)
    st.markdown("""
        <div class="fv-card fv-card-navy" style="margin-bottom:1.25rem; padding: 1rem 1.25rem;">
            <span style="color:rgba(255,255,255,0.6); font-size:0.85rem;">
                The scout has full context from all session materials. Ask anything.
            </span>
        </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state[state_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask a follow-up question…", key=f"chat_input_{state_key}"):
        st.session_state[state_key].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = chat_response(client, context, st.session_state[state_key], session_context=session_context)
            st.markdown(reply)
        st.session_state[state_key].append({"role": "assistant", "content": reply})


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
# CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Saint Mary's College of California — Navy #002147 | Red #C8102E | White #FFFFFF */

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: radial-gradient(ellipse at top left, #001530 0%, #002147 40%, #00101f 100%);
}

/* Hero */
.fv-hero {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    background: radial-gradient(ellipse at center, rgba(200,16,46,0.1) 0%, transparent 70%);
    border-bottom: 1px solid rgba(200,16,46,0.2);
    margin-bottom: 2.5rem;
}
.fv-hero-ball { font-size: 3.5rem; line-height: 1; margin-bottom: 1rem; }
.fv-hero h1 {
    font-size: 3.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 30%, #C8102E 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
}
.fv-hero p {
    color: rgba(255,255,255,0.45);
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin: 0;
}

/* Section labels */
.fv-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(200,16,46,0.8);
    margin: 2rem 0 0.75rem;
}

/* Cards */
.fv-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.fv-card-red {
    background: rgba(200,16,46,0.05);
    border-color: rgba(200,16,46,0.2);
}
.fv-card-navy {
    background: rgba(0,33,71,0.4);
    border-color: rgba(255,255,255,0.1);
}

/* Report output */
.fv-report {
    background: rgba(0,21,48,0.5);
    border: 1px solid rgba(200,16,46,0.2);
    border-left: 3px solid #C8102E;
    border-radius: 0 14px 14px 0;
    padding: 2rem 2rem 1.5rem;
    margin: 0.5rem 0 1.5rem;
    line-height: 1.75;
    color: #f0f4ff;
}

/* Status badge */
.fv-badge {
    display: inline-block;
    background: rgba(200,16,46,0.12);
    border: 1px solid rgba(200,16,46,0.35);
    color: #ff4d6d;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #C8102E 0%, #9b0c23 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(200,16,46,0.5) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(200,16,46,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 25px rgba(200,16,46,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: rgba(255,255,255,0.05) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(255,255,255,0.1) !important;
    border-color: rgba(255,255,255,0.35) !important;
    box-shadow: 0 4px 15px rgba(255,255,255,0.08) !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(0,21,48,0.6) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(200,16,46,0.5) !important;
    box-shadow: 0 0 0 3px rgba(200,16,46,0.1) !important;
}
label { color: rgba(255,255,255,0.6) !important; font-size: 0.85rem !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,21,48,0.4) !important;
    border: 2px dashed rgba(200,16,46,0.3) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(200,16,46,0.6) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(0,21,48,0.3) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}

/* Chat */
[data-testid="stChatInput"] > div {
    background: rgba(0,21,48,0.6) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
}
[data-testid="stChatMessage"] {
    background: rgba(0,21,48,0.4) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 2rem 0 !important; }

/* Info / warning boxes */
[data-testid="stNotification"] {
    background: rgba(0,21,48,0.5) !important;
    border-radius: 10px !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # Session state init
    for key, default in [
        # Session
        ("session_id",    str(uuid.uuid4())[:8].upper()),
        ("session_items", []),   # {type, label, content, insights}
        ("session_chat",  []),
        # Handwritten notes tab
        ("notes_analyzed",     False),
        ("notes_transcription", ""),
        ("notes_insights",     ""),
        ("notes_rag_context",  []),
        ("notes_images",       []),
        ("notes_chat",         []),
        # Trackman tab
        ("trackman_analyzed", False),
        ("trackman_summary",  ""),
        ("trackman_insights", ""),
        ("trackman_chat",     []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Hero
    st.markdown("""
        <div class="fv-hero">
            <div class="fv-hero-ball">⚾</div>
            <h1>FieldVision</h1>
            <p>Baseball Scouting Intelligence</p>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading knowledge base…"):
        rag_index, rag_df = load_rag_index()

    client = get_client()

    # ── Session bar ───────────────────────────────────────────────────────────
    col_sid, col_reset = st.columns([4, 1])
    with col_sid:
        n = len(st.session_state.session_items)
        item_str = f"{n} item{'s' if n != 1 else ''} in session" if n else "No items in session yet"
        st.markdown(f"""
            <div class="fv-card" style="padding:0.75rem 1.25rem; display:flex; align-items:center; gap:1rem;">
                <span style="color:rgba(200,16,46,0.9); font-size:0.7rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase;">Session</span>
                <span style="color:white; font-size:0.9rem; font-weight:600; font-family:monospace;">{st.session_state.session_id}</span>
                <span style="color:rgba(255,255,255,0.4); font-size:0.8rem;">{item_str}</span>
            </div>
        """, unsafe_allow_html=True)
    with col_reset:
        if st.button("New Session", use_container_width=True):
            for key in ["session_id", "session_items", "session_chat",
                        "notes_analyzed", "notes_transcription", "notes_insights",
                        "notes_rag_context", "notes_images", "notes_chat",
                        "trackman_analyzed", "trackman_summary", "trackman_insights", "trackman_chat"]:
                del st.session_state[key]
            st.rerun()

    # Session summary
    if st.session_state.session_items:
        with st.expander(f"📁 Session Contents ({len(st.session_state.session_items)} items)", expanded=False):
            for i, item in enumerate(st.session_state.session_items, 1):
                st.markdown(f"**{i}. [{item['type'].upper()}]** {item['label']}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_notes, tab_trackman, tab_session = st.tabs(["📋 Handwritten Notes", "📊 Trackman Data", "💬 Session Chat"])

    # =========================================================================
    # TAB 1 — Handwritten Notes
    # =========================================================================
    with tab_notes:
        st.markdown('<p class="fv-label">Scouting Notes</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload images or PDFs of handwritten scouting notes",
            type=["jpg", "jpeg", "png", "pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="notes_uploader",
        )

        st.markdown('<p class="fv-label">Context</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            notes_context = st.text_input(
                "What are these notes about?",
                placeholder="e.g. Pitching eval for a high school recruit…",
                key="notes_context",
            )
        with col_b:
            notes_focus = st.text_input(
                "What would you like to focus on?",
                placeholder="e.g. Arm strength, whether to offer a scholarship…",
                key="notes_focus",
            )

        if not uploaded_files:
            st.markdown("""
                <div class="fv-card" style="text-align:center; padding:3rem; margin-top:1rem;">
                    <div style="font-size:2rem; margin-bottom:0.75rem;">📋</div>
                    <div style="color:rgba(255,255,255,0.35); font-size:0.9rem;">
                        Upload scouting note images or PDFs above to get started
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="fv-badge">✓ {len(uploaded_files)} file(s) ready</div>', unsafe_allow_html=True)

            if st.button("Transcribe & Analyze", type="primary", use_container_width=True, key="notes_btn"):
                with st.spinner("Reading files…"):
                    images = collect_images(uploaded_files)

                transcriptions = []
                with st.spinner(f"Transcribing {len(images)} page(s)…"):
                    for name, image in images:
                        transcriptions.append(transcribe_image(client, image))

                combined = "\n\n".join(transcriptions)

                with st.spinner("Searching historical scouting database…"):
                    context = retrieve_context(combined, rag_index, rag_df)

                session_ctx = build_session_context(st.session_state.session_items)

                with st.spinner("Generating insights…"):
                    insights = generate_insights(
                        client, combined, context,
                        notes_context=notes_context, output_focus=notes_focus,
                        session_context=session_ctx,
                    )

                label = notes_context or f"Scouting Notes ({len(images)} page(s))"
                st.session_state.session_items.append({
                    "type": "notes", "label": label,
                    "content": combined, "insights": insights,
                })
                st.session_state.notes_analyzed     = True
                st.session_state.notes_transcription = combined
                st.session_state.notes_insights      = insights
                st.session_state.notes_rag_context   = context
                st.session_state.notes_images        = images
                st.session_state.notes_chat          = []

        if st.session_state.notes_analyzed:
            images   = st.session_state.notes_images
            combined = st.session_state.notes_transcription
            insights = st.session_state.notes_insights
            context  = st.session_state.notes_rag_context

            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown('<p class="fv-label">Uploaded Pages</p>', unsafe_allow_html=True)
            thumb_cols = st.columns(min(len(images), 4))
            for i, (name, image) in enumerate(images):
                with thumb_cols[i % 4]:
                    st.image(image, caption=name, use_container_width=True)

            with st.expander("📄 View Combined Transcription", expanded=False):
                st.text_area("combined", value=combined, height=300, label_visibility="collapsed")

            st.markdown('<p class="fv-label">Scouting Intelligence Report</p>', unsafe_allow_html=True)
            st.markdown('<div class="fv-report">', unsafe_allow_html=True)
            st.markdown(insights)
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("📚 Historical Reports Used for Context", expanded=False):
                for i, ctx in enumerate(context, 1):
                    st.markdown(f"**Report {i}** — relevance `{ctx['score']:.3f}`")
                    st.caption(ctx["item"])
                    st.text(ctx["text"][:600] + ("…" if len(ctx["text"]) > 600 else ""))
                    st.divider()

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<p class="fv-label">Download Reports</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇ Digitized Notes (.pdf)",
                    data=bytes(build_pdf("Digitized Scouting Notes", combined)),
                    file_name="scouting_notes.pdf", mime="application/pdf",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    "⬇ Analysis Report (.pdf)",
                    data=bytes(build_pdf("Scouting Intelligence Report", insights)),
                    file_name="scouting_analysis.pdf", mime="application/pdf",
                    use_container_width=True,
                )

            st.markdown("<hr>", unsafe_allow_html=True)
            render_chat(client, combined, "notes_chat",
                        session_context=build_session_context(st.session_state.session_items))

    # =========================================================================
    # TAB 2 — Trackman Data
    # =========================================================================
    with tab_trackman:
        st.markdown('<p class="fv-label">Trackman CSV</p>', unsafe_allow_html=True)
        csv_file = st.file_uploader(
            "Upload a Trackman CSV export",
            type=["csv"],
            label_visibility="collapsed",
            key="trackman_uploader",
        )

        st.markdown('<p class="fv-label">Context</p>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            tm_context = st.text_input(
                "What are you looking to evaluate?",
                placeholder="e.g. Pitching staff performance, opponent tendencies…",
                key="tm_context",
            )
        with col_b:
            tm_focus = st.text_input(
                "What would you like to focus on?",
                placeholder="e.g. Spin rate trends, exit velocity against lefties…",
                key="tm_focus",
            )

        if not csv_file:
            st.markdown("""
                <div class="fv-card" style="text-align:center; padding:3rem; margin-top:1rem;">
                    <div style="font-size:2rem; margin-bottom:0.75rem;">📊</div>
                    <div style="color:rgba(255,255,255,0.35); font-size:0.9rem;">
                        Upload a Trackman CSV export above to get started
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="fv-badge">✓ CSV ready</div>', unsafe_allow_html=True)

            if st.button("Analyze Trackman Data", type="primary", use_container_width=True, key="tm_btn"):
                with st.spinner("Parsing CSV…"):
                    df = pd.read_csv(csv_file)
                    summary = summarize_trackman(df)

                session_ctx = build_session_context(st.session_state.session_items)

                with st.spinner("Generating insights…"):
                    insights = analyze_trackman(
                        client, summary,
                        notes_context=tm_context, output_focus=tm_focus,
                        session_context=session_ctx,
                    )

                label = tm_context or csv_file.name
                st.session_state.session_items.append({
                    "type": "trackman", "label": label,
                    "content": summary, "insights": insights,
                })
                st.session_state.trackman_analyzed = True
                st.session_state.trackman_summary  = summary
                st.session_state.trackman_insights = insights
                st.session_state.trackman_chat     = []

        if st.session_state.trackman_analyzed:
            summary  = st.session_state.trackman_summary
            insights = st.session_state.trackman_insights

            st.markdown("<hr>", unsafe_allow_html=True)

            with st.expander("📄 View Data Summary", expanded=False):
                st.text_area("summary", value=summary, height=300, label_visibility="collapsed")

            st.markdown('<p class="fv-label">Trackman Intelligence Report</p>', unsafe_allow_html=True)
            st.markdown('<div class="fv-report">', unsafe_allow_html=True)
            st.markdown(insights)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<p class="fv-label">Download Report</p>', unsafe_allow_html=True)
            st.download_button(
                "⬇ Trackman Analysis (.pdf)",
                data=bytes(build_pdf("Trackman Intelligence Report", insights)),
                file_name="trackman_analysis.pdf", mime="application/pdf",
                use_container_width=True,
            )

            st.markdown("<hr>", unsafe_allow_html=True)
            render_chat(client, summary, "trackman_chat",
                        session_context=build_session_context(st.session_state.session_items))

    # =========================================================================
    # TAB 3 — Session Chat
    # =========================================================================
    with tab_session:
        if not st.session_state.session_items:
            st.markdown("""
                <div class="fv-card" style="text-align:center; padding:3rem; margin-top:1rem;">
                    <div style="font-size:2rem; margin-bottom:0.75rem;">💬</div>
                    <div style="color:rgba(255,255,255,0.35); font-size:0.9rem;">
                        Analyze some notes or Trackman data first — then use this tab
                        to ask questions across all session materials combined.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            full_context = build_session_context(st.session_state.session_items)
            render_chat(client, full_context, "session_chat",
                        session_context="",
                        label="Session Chat — All Materials")


if __name__ == "__main__":
    main()
