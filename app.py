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
CLAUDE_MODEL    = "claude-sonnet-4-5"
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
            <span style="color:var(--text-muted); font-size:0.85rem;">
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
# ---------------------------------------------------------------------------
# Theme CSS — CSS custom-property approach so one stylesheet serves both modes
# ---------------------------------------------------------------------------
_DARK_VARS = """
:root {
    --bg:                  #080d18;
    --surface:             #0f1624;
    --surface-2:           #172033;
    --border:              rgba(255,255,255,0.09);
    --border-strong:       rgba(255,255,255,0.18);
    --text:                #f0f4ff;
    --text-muted:          rgba(240,244,255,0.65);
    --text-subtle:         rgba(240,244,255,0.42);
    --hero-title:          #ffffff;
    --label-color:         #e8485f;
    --badge-bg:            rgba(200,16,46,0.12);
    --badge-border:        rgba(200,16,46,0.28);
    --badge-text:          #f07080;
    --report-bg:           #0d1526;
    --report-text:         #e4eaff;
    --input-bg:            #111826;
    --input-bg-focus:      #16203a;
    --placeholder:         rgba(240,244,255,0.28);
    --upload-bg:           #0d1526;
    --upload-border:       rgba(255,255,255,0.14);
    --expander-bg:         #0f1624;
    --expander-text:       rgba(240,244,255,0.6);
    --chat-msg-bg:         #0f1624;
    --chat-input-bg:       #111826;
    --tab-inactive:        rgba(240,244,255,0.42);
    --tab-hover:           rgba(240,244,255,0.72);
    --dl-btn-text:         rgba(240,244,255,0.7);
    --dl-btn-border:       rgba(240,244,255,0.2);
    --dl-btn-hover-bg:     rgba(255,255,255,0.07);
    --dl-btn-hover-border: rgba(240,244,255,0.35);
    --dl-btn-hover-text:   #ffffff;
    --toggle-bg:           #172033;
    --toggle-border:       rgba(255,255,255,0.18);
    --toggle-text:         rgba(240,244,255,0.75);
    --toggle-hover-bg:     #1e2d47;
    --card-shadow:         none;
    --divider:             rgba(255,255,255,0.07);
    --notification-bg:     #0f1624;
    --scrollbar:           rgba(255,255,255,0.1);
    --scrollbar-hover:     rgba(255,255,255,0.2);
}
"""

_LIGHT_VARS = """
:root {
    --bg:                  #f0f4fa;
    --surface:             #ffffff;
    --surface-2:           #e8edf6;
    --border:              rgba(0,0,0,0.1);
    --border-strong:       rgba(0,0,0,0.2);
    --text:                #08111e;
    --text-muted:          rgba(8,17,30,0.65);
    --text-subtle:         rgba(8,17,30,0.45);
    --hero-title:          #002147;
    --label-color:         #a80d24;
    --badge-bg:            rgba(200,16,46,0.08);
    --badge-border:        rgba(200,16,46,0.25);
    --badge-text:          #8b0b1e;
    --report-bg:           #ffffff;
    --report-text:         #08111e;
    --input-bg:            #ffffff;
    --input-bg-focus:      #f8f9ff;
    --placeholder:         rgba(8,17,30,0.32);
    --upload-bg:           #f5f8fd;
    --upload-border:       rgba(0,0,0,0.15);
    --expander-bg:         #ffffff;
    --expander-text:       rgba(8,17,30,0.62);
    --chat-msg-bg:         #f5f8fd;
    --chat-input-bg:       #ffffff;
    --tab-inactive:        rgba(8,17,30,0.45);
    --tab-hover:           rgba(8,17,30,0.78);
    --dl-btn-text:         #002147;
    --dl-btn-border:       rgba(0,33,71,0.35);
    --dl-btn-hover-bg:     rgba(0,33,71,0.06);
    --dl-btn-hover-border: rgba(0,33,71,0.55);
    --dl-btn-hover-text:   #002147;
    --toggle-bg:           #ffffff;
    --toggle-border:       rgba(0,0,0,0.2);
    --toggle-text:         rgba(8,17,30,0.7);
    --toggle-hover-bg:     #edf1f8;
    --card-shadow:         0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.05);
    --divider:             rgba(0,0,0,0.09);
    --notification-bg:     #ffffff;
    --scrollbar:           rgba(0,0,0,0.12);
    --scrollbar-hover:     rgba(0,0,0,0.22);
}
"""

_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Saint Mary's — Navy #002147 | Red #C8102E | White #FFFFFF */

html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Background ─────────────────────────────────────────────────────────── */
.stApp { background-color: var(--bg); }

/* ── Utility text helpers (used in inline HTML snippets) ─────────────────── */
.fv-text        { color: var(--text)        !important; }
.fv-text-muted  { color: var(--text-muted)  !important; }
.fv-text-subtle { color: var(--text-subtle) !important; }

/* ── Hero ───────────────────────────────────────────────────────────────── */
.fv-hero {
    text-align: center;
    padding: 2.75rem 1rem 2rem;
    border-bottom: 1px solid var(--divider);
    margin-bottom: 2rem;
}
.fv-logo {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.8rem;
    margin-bottom: 0.55rem;
}
.fv-logo-mark {
    width: 38px; height: 38px;
    background: #C8102E;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    flex-shrink: 0;
}
.fv-hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--hero-title);
    margin: 0;
    line-height: 1;
}
.fv-hero p {
    color: var(--text-subtle);
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin: 0.6rem 0 0;
}

/* ── Section labels ─────────────────────────────────────────────────────── */
.fv-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--label-color);
    margin: 2rem 0 0.6rem;
}

/* ── Cards ───────────────────────────────────────────────────────────────── */
.fv-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: var(--card-shadow);
}
.fv-card-red {
    background: var(--badge-bg);
    border-color: var(--badge-border);
}
.fv-card-navy {
    background: var(--surface-2);
    border-color: var(--border);
}

/* ── Report output ───────────────────────────────────────────────────────── */
.fv-report {
    background: var(--report-bg);
    border: 1px solid var(--border);
    border-left: 2px solid #C8102E;
    border-radius: 0 10px 10px 0;
    padding: 1.75rem 2rem;
    margin: 0.5rem 0 1.5rem;
    line-height: 1.8;
    color: var(--report-text);
    font-size: 0.92rem;
    box-shadow: var(--card-shadow);
}

/* ── Badge ───────────────────────────────────────────────────────────────── */
.fv-badge {
    display: inline-block;
    background: var(--badge-bg);
    border: 1px solid var(--badge-border);
    color: var(--badge-text);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.28rem 0.65rem;
    border-radius: 4px;
    margin-bottom: 0.75rem;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: var(--tab-inactive) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.7rem 1.2rem !important;
    margin-bottom: -1px !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom-color: #C8102E !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--tab-hover) !important;
    background: var(--surface-2) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ── Primary CTA buttons (red, high contrast in both modes) ──────────────── */
.stButton > button {
    background: #C8102E !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.015em !important;
    padding: 0.6rem 1.3rem !important;
    box-shadow: 0 2px 8px rgba(200,16,46,0.3) !important;
    transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease !important;
}
.stButton > button:hover {
    background: #a80d26 !important;
    box-shadow: 0 4px 14px rgba(200,16,46,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Theme toggle — neutral secondary style ──────────────────────────────── */
.fv-toggle-btn .stButton > button {
    background: var(--toggle-bg) !important;
    color: var(--toggle-text) !important;
    border: 1px solid var(--toggle-border) !important;
    box-shadow: none !important;
    font-size: 1rem !important;
    padding: 0.4rem 0.6rem !important;
}
.fv-toggle-btn .stButton > button:hover {
    background: var(--toggle-hover-bg) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Download buttons ────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: transparent !important;
    color: var(--dl-btn-text) !important;
    border: 1px solid var(--dl-btn-border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    transition: all 0.15s ease !important;
}
.stDownloadButton > button:hover {
    background: var(--dl-btn-hover-bg) !important;
    border-color: var(--dl-btn-hover-border) !important;
    color: var(--dl-btn-hover-text) !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
[data-baseweb="input"],
[data-baseweb="base-input"],
[data-baseweb="textarea"],
[data-baseweb="input"] input,
[data-baseweb="base-input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="input"] > div,
[data-baseweb="base-input"] > div {
    background: var(--input-bg) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
[data-baseweb="input"]:focus-within,
[data-baseweb="base-input"]:focus-within,
[data-baseweb="textarea"]:focus-within {
    border-color: rgba(200,16,46,0.45) !important;
    box-shadow: 0 0 0 3px rgba(200,16,46,0.09) !important;
    background: var(--input-bg-focus) !important;
}
[data-baseweb="input"]:focus-within input,
[data-baseweb="base-input"]:focus-within input {
    background: var(--input-bg-focus) !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder,
[data-baseweb="input"] input::placeholder,
[data-baseweb="base-input"] input::placeholder,
[data-baseweb="textarea"] textarea::placeholder {
    color: var(--placeholder) !important;
}
label {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--upload-bg) !important;
    border: 1px dashed var(--upload-border) !important;
    border-radius: 10px !important;
    transition: border-color 0.15s ease, background 0.15s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(200,16,46,0.4) !important;
    background: var(--badge-bg) !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--expander-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.83rem !important;
    color: var(--expander-text) !important;
    font-weight: 500 !important;
}

/* ── Chat ────────────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInputTextArea"],
[data-testid="stChatInputTextArea"] textarea {
    background: var(--chat-input-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 10px !important;
}
[data-testid="stChatInputTextArea"] textarea::placeholder {
    color: var(--placeholder) !important;
}
[data-testid="stChatMessage"] {
    background: var(--chat-msg-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--divider) !important; margin: 1.75rem 0 !important; }

/* ── Notifications ───────────────────────────────────────────────────────── */
[data-testid="stNotification"] {
    background: var(--notification-bg) !important;
    border-radius: 8px !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--scrollbar); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--scrollbar-hover); }
"""


def get_css(theme: str = "dark") -> str:
    vars_block = _DARK_VARS if theme == "dark" else _LIGHT_VARS
    return f"<style>\n{vars_block}\n{_BASE_CSS}\n</style>"


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
        # Theme
        ("theme", "dark"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

    # Hero
    st.markdown("""
        <div class="fv-hero">
            <div class="fv-logo">
                <div class="fv-logo-mark">FV</div>
                <h1>FieldVision</h1>
            </div>
            <p>Baseball Scouting Intelligence &nbsp;·&nbsp; Saint Mary's College</p>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading knowledge base…"):
        rag_index, rag_df = load_rag_index()

    client = get_client()

    # ── Session bar ───────────────────────────────────────────────────────────
    col_sid, col_theme, col_reset = st.columns([4, 0.65, 1.1])
    with col_sid:
        n = len(st.session_state.session_items)
        item_str = f"{n} item{'s' if n != 1 else ''} in session" if n else "No items in session yet"
        st.markdown(f"""
            <div class="fv-card" style="padding:0.75rem 1.25rem; display:flex; align-items:center; gap:1rem;">
                <span style="color:#C8102E; font-size:0.7rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase;">Session</span>
                <span style="color:var(--text); font-size:0.9rem; font-weight:600; font-family:monospace;">{st.session_state.session_id}</span>
                <span style="color:var(--text-subtle); font-size:0.8rem;">{item_str}</span>
            </div>
        """, unsafe_allow_html=True)
    with col_theme:
        icon = "☀️" if st.session_state.theme == "dark" else "🌙"
        if st.button(icon, use_container_width=True, help="Switch to light mode" if st.session_state.theme == "dark" else "Switch to dark mode", key="theme_toggle"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
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
                    <div style="color:var(--text-subtle); font-size:0.9rem;">
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
                    <div style="color:var(--text-subtle); font-size:0.9rem;">
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
                    <div style="color:var(--text-subtle); font-size:0.9rem;">
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
