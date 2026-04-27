import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import re
from PIL import Image

try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

import uuid
import base64
from pathlib import Path
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

# ---------------------------------------------------------------------------
# Brand logo
# ---------------------------------------------------------------------------
LOGO_SVG = '''<svg class="fv-logo-svg" viewBox="0 0 680 680" role="img"
     xmlns="http://www.w3.org/2000/svg">
  <defs>
    <clipPath id="ball-clip"><circle cx="340" cy="340" r="240"/></clipPath>
  </defs>
  <circle cx="340" cy="340" r="248" fill="#0A1628" opacity="0.18"/>
  <circle cx="340" cy="340" r="240" fill="#F8F6F2"/>
  <circle cx="340" cy="340" r="240" fill="none" stroke="#0D2B6E" stroke-width="3"/>
  <path d="M200 200 Q100 340 200 480" fill="none" stroke="#CC1A1A"
        stroke-width="6" stroke-linecap="round"/>
  <path d="M480 200 Q580 340 480 480" fill="none" stroke="#CC1A1A"
        stroke-width="6" stroke-linecap="round"/>
  <g stroke="#CC1A1A" stroke-width="2.5" stroke-linecap="round">
    <line x1="194" y1="226" x2="208" y2="232"/>
    <line x1="178" y1="262" x2="193" y2="265"/>
    <line x1="168" y1="300" x2="183" y2="300"/>
    <line x1="168" y1="340" x2="183" y2="340"/>
    <line x1="168" y1="378" x2="183" y2="380"/>
    <line x1="178" y1="416" x2="193" y2="413"/>
    <line x1="194" y1="452" x2="208" y2="446"/>
  </g>
  <g stroke="#CC1A1A" stroke-width="2.5" stroke-linecap="round">
    <line x1="486" y1="226" x2="472" y2="232"/>
    <line x1="502" y1="262" x2="487" y2="265"/>
    <line x1="512" y1="300" x2="497" y2="300"/>
    <line x1="512" y1="340" x2="497" y2="340"/>
    <line x1="512" y1="378" x2="497" y2="380"/>
    <line x1="502" y1="416" x2="487" y2="413"/>
    <line x1="486" y1="452" x2="472" y2="446"/>
  </g>
  <g clip-path="url(#ball-clip)" opacity="0.78">
    <line x1="220" y1="270" x2="460" y2="270" stroke="#0D2B6E" stroke-width="1.2" opacity="0.5"/>
    <line x1="220" y1="340" x2="460" y2="340" stroke="#0D2B6E" stroke-width="1.2" opacity="0.5"/>
    <line x1="220" y1="410" x2="460" y2="410" stroke="#0D2B6E" stroke-width="1.2" opacity="0.5"/>
    <line x1="280" y1="200" x2="280" y2="480" stroke="#0D2B6E" stroke-width="1.2" opacity="0.5"/>
    <line x1="340" y1="200" x2="340" y2="480" stroke="#0D2B6E" stroke-width="1.2" opacity="0.5"/>
    <line x1="400" y1="200" x2="400" y2="480" stroke="#0D2B6E" stroke-width="1.2" opacity="0.5"/>
    <g fill="#0D2B6E" opacity="0.6">
      <rect x="276" y="266" width="8" height="8" rx="1"/>
      <rect x="336" y="266" width="8" height="8" rx="1"/>
      <rect x="396" y="266" width="8" height="8" rx="1"/>
      <rect x="276" y="336" width="8" height="8" rx="1"/>
      <rect x="396" y="336" width="8" height="8" rx="1"/>
      <rect x="276" y="406" width="8" height="8" rx="1"/>
      <rect x="336" y="406" width="8" height="8" rx="1"/>
      <rect x="396" y="406" width="8" height="8" rx="1"/>
    </g>
    <circle cx="340" cy="340" r="54" fill="#0D2B6E" opacity="0.92"/>
    <circle cx="340" cy="340" r="44" fill="#F8F6F2" opacity="1"/>
    <g stroke="#CC1A1A" stroke-width="2" fill="none" opacity="0.9">
      <line x1="340" y1="296" x2="340" y2="270"/>
      <line x1="340" y1="384" x2="340" y2="410"/>
      <line x1="296" y1="340" x2="270" y2="340"/>
      <line x1="384" y1="340" x2="410" y2="340"/>
      <line x1="312" y1="312" x2="295" y2="295"/>
      <line x1="368" y1="312" x2="385" y2="295"/>
      <line x1="312" y1="368" x2="295" y2="385"/>
      <line x1="368" y1="368" x2="385" y2="385"/>
    </g>
    <g fill="#CC1A1A" opacity="0.9">
      <circle cx="340" cy="268" r="4"/>
      <circle cx="340" cy="412" r="4"/>
      <circle cx="268" cy="340" r="4"/>
      <circle cx="412" cy="340" r="4"/>
      <circle cx="293" cy="293" r="3.5"/>
      <circle cx="387" cy="293" r="3.5"/>
      <circle cx="293" cy="387" r="3.5"/>
      <circle cx="387" cy="387" r="3.5"/>
    </g>
    <g stroke="#0D2B6E" stroke-width="1.4" fill="none" opacity="0.7">
      <path d="M340 305 L355 313 L355 330 L340 338 L325 330 L325 313 Z"/>
      <path d="M340 342 L355 350 L355 367 L340 375 L325 367 L325 350 Z"/>
    </g>
    <text x="340" y="347" text-anchor="middle" dominant-baseline="central"
      font-family="Helvetica Neue, Arial, sans-serif"
      font-size="22" font-weight="700" fill="#0D2B6E" letter-spacing="2">AI</text>
  </g>
  <circle cx="340" cy="340" r="240" fill="none" stroke="#0D2B6E" stroke-width="5"/>
  <circle cx="340" cy="340" r="232" fill="none" stroke="#CC1A1A"
          stroke-width="1.5" opacity="0.5"/>
</svg>'''


def _load_logo_data_url(path: str = "static/logo.png") -> str:
    try:
        data = Path(path).read_bytes()
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    except FileNotFoundError:
        return ""

_LOGO_DATA_URL: str = _load_logo_data_url()

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

# Grade display config
GRADE_COLORS = {
    "A": "#4ade80",   # green
    "B": "#60a5fa",   # blue
    "C": "#fbbf24",   # amber
    "D": "#f87171",   # red
    "?": "rgba(240,244,255,.42)",
}
GRADE_LABELS = {
    "A": "Elite prospect",
    "B": "Solid prospect",
    "C": "Developmental",
    "D": "Not recommended",
}


# ---------------------------------------------------------------------------
# Session context builder
# ---------------------------------------------------------------------------
def build_session_context(session_items: list[dict]) -> str:
    if not session_items:
        return ""
    lines = ["=== PRIOR SESSION MATERIALS (use as background context) ==="]
    for item in session_items:
        lines.append(f"\n[{item['type'].upper()} — {item['label']}]")
        lines.append(item["content"][:4000])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embed_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found. Add it to Streamlit secrets or set it as an env variable.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource
def load_rag_index():
    if not os.path.exists(CSV_PATH):
        st.error(f"Knowledge base not found at `{CSV_PATH}`.")
        st.stop()
    df = pd.read_csv(CSV_PATH)
    df = df[df["Transcription"].notna() & (df["Transcription"].str.strip() != "")].reset_index(drop=True)
    if not os.path.exists(EMBEDDINGS_PATH):
        st.error(f"Embeddings file not found at `{EMBEDDINGS_PATH}`. Run `python precompute_embeddings.py`.")
        st.stop()
    embeddings_np = np.load(EMBEDDINGS_PATH).astype(np.float32)
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index, df


# ---------------------------------------------------------------------------
# Transcription — Claude Vision OCR
# ---------------------------------------------------------------------------
def transcribe_image(client: anthropic.Anthropic, image: "Image.Image") -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
                {"type": "text", "text": (
                    "You are transcribing a handwritten baseball scouting or match note. "
                    "Transcribe ALL text exactly as written, preserving line breaks, "
                    "abbreviations, shorthand, numbers, and player names. "
                    "If text is illegible indicate with [?]."
                )},
            ],
        }],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Retrieval — FAISS semantic search
# ---------------------------------------------------------------------------
def retrieve_context(query_text: str, index, df: pd.DataFrame, k: int = TOP_K) -> list[dict]:
    embed_model = get_embed_model()
    query_vec = embed_model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)
    hits = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        hits.append({"score": float(score), "text": row["Transcription"],
                     "item": row.get("Item", ""), "project": row.get("Project", "")})
    return hits


# ---------------------------------------------------------------------------
# Generation — structured scouting insights
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
        model=CLAUDE_MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
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
        model=CLAUDE_MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# NEW: Player profile extraction (for talent pool + comparison)
# ---------------------------------------------------------------------------
def extract_player_profile(client: anthropic.Anthropic, label: str, insights_text: str) -> dict:
    """Use Claude to extract a structured player profile from scouting insights."""
    prompt = f"""From these baseball scouting insights, extract a player profile.
Return ONLY a valid JSON object with exactly these keys:
{{
  "name": "player name from the notes, or '{label}' if not explicitly named",
  "position": "primary position code: P, C, 1B, 2B, 3B, SS, LF, CF, RF, DH, or Unknown",
  "grade": "one letter — A (elite prospect), B (solid prospect), C (developmental/project), D (not recommended)",
  "strengths": ["short phrase 1", "short phrase 2"],
  "concerns": ["short phrase 1"],
  "summary": "one plain-English sentence about this player"
}}

INSIGHTS:
{insights_text[:1800]}

Return ONLY the JSON object — no markdown fences, no explanation."""

    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    # Strip any accidental markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        profile = json.loads(text)
        for k in ("name", "position", "grade", "strengths", "concerns", "summary"):
            if k not in profile:
                profile[k] = [] if k in ("strengths", "concerns") else "Unknown"
        return profile
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {
        "name": label, "position": "Unknown", "grade": "B",
        "strengths": [], "concerns": [], "summary": "Profile could not be extracted.",
    }


# ---------------------------------------------------------------------------
# NEW: Plain-language pitch interpretation
# ---------------------------------------------------------------------------
def interpret_pitch_metrics(client: anthropic.Anthropic, summary: str, focus: str = "") -> str:
    """Translate Trackman numbers into plain English for coaches and players."""
    focus_block = f"\nFocus especially on: {focus}" if focus else ""
    prompt = f"""A college baseball coach wants to understand what these Trackman numbers mean in plain, practical language —
what they tell us about each pitcher's stuff, how effective their approach was, and what to work on.{focus_block}

Explain each pitcher's key metrics in 2–3 sentences each. Use language a player or parent could understand — no jargon.
Then give one overall game takeaway in plain English.

PITCH DATA:
{summary}"""

    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=900,
        system="You are a college pitching coach translating advanced metrics into practical plain-English insights for your staff and players.",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------
def sanitize_for_pdf(text: str) -> str:
    replacements = {
        "…": "...", "—": "--", "–": "-",
        "‘": "'",  "’": "'", "“": '"', "”": '"',
        "•": "-",  " ": " ", "‒": "-", "―": "--",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf(title: str, body: str) -> bytes:
    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, sanitize_for_pdf(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
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
# Trackman — CSV summarizer
# ---------------------------------------------------------------------------
def summarize_trackman(df: pd.DataFrame) -> str:
    lines = []
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
            if "RelSpeed"         in df.columns: stats.append(f"Velo: {ptdf['RelSpeed'].mean():.1f} mph")
            if "SpinRate"         in df.columns: stats.append(f"Spin: {ptdf['SpinRate'].mean():.0f} rpm")
            if "InducedVertBreak" in df.columns: stats.append(f"IVB: {ptdf['InducedVertBreak'].mean():.1f}\"")
            if "HorzBreak"        in df.columns: stats.append(f"HB: {ptdf['HorzBreak'].mean():.1f}\"")
            if stats:
                block.append(f"      {pitch_type}: {' | '.join(stats)}")
        if "PitchCall" in df.columns:
            calls   = pdf["PitchCall"].value_counts()
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

    SMC_CODE  = "STM_GAE"
    opp_code  = away if home == SMC_CODE else home

    lines.append("=== SAINT MARY'S (STM_GAE) ===")
    lines.append("-- Pitching --")
    for pitcher, pdf in df[df["PitcherTeam"] == SMC_CODE].groupby("Pitcher"):
        lines.extend(pitcher_block(pdf, pitcher))
    lines.append("\n-- Batting --")
    for batter, bdf in df[df["BatterTeam"] == SMC_CODE].groupby("Batter"):
        lines.extend(batter_block(bdf, batter))

    lines.append(f"\n=== OPPONENT ({opp_code}) ===")
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
        model=CLAUDE_MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Shared chat renderer — FIX: render first, append to state after (no duplicate)
# ---------------------------------------------------------------------------
def render_chat(
    client: anthropic.Anthropic,
    context: str,
    state_key: str,
    session_context: str = "",
    label: str = "Ask the Scout",
) -> None:
    st.markdown(f'<p class="fv-label">{label}</p>', unsafe_allow_html=True)
    st.markdown("""
        <div class="fv-card fv-card-navy" style="margin-bottom:1.25rem; padding: 1rem 1.25rem;">
            <span style="color:var(--text-muted); font-size:0.85rem;">
                The scout has full context from all session materials. Ask anything.
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Render existing messages from state (already confirmed)
    for msg in st.session_state[state_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new input — render FIRST, append to state AFTER (prevents duplicate on rerun)
    if user_input := st.chat_input("Ask a follow-up question…", key=f"chat_input_{state_key}"):
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # Build history including the new user message (not yet in state)
                reply = chat_response(
                    client, context,
                    st.session_state[state_key] + [{"role": "user", "content": user_input}],
                    session_context=session_context,
                )
            st.markdown(reply)

        # Append both to state AFTER rendering (avoids duplicate display on next rerun)
        st.session_state[state_key].append({"role": "user",      "content": user_input})
        st.session_state[state_key].append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def collect_images(uploaded_files) -> list[tuple[str, "Image.Image"]]:
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
    --toggle-bg:           #dde4f0;
    --toggle-border:       rgba(180,196,218,0.6);
    --toggle-text:         #001533;
    --toggle-hover-bg:     #edf1fa;
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
    --toggle-bg:           #002147;
    --toggle-border:       rgba(0,33,71,0.0);
    --toggle-text:         #ffffff;
    --toggle-hover-bg:     #003268;
    --card-shadow:         0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.05);
    --divider:             rgba(0,0,0,0.09);
    --notification-bg:     #ffffff;
    --scrollbar:           rgba(0,0,0,0.12);
    --scrollbar-hover:     rgba(0,0,0,0.22);
}
"""

_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; font-size: 15px; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

.stApp { background-color: var(--bg); }

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
.fv-logo-svg {
    width: 76px; height: 76px; flex-shrink: 0;
    filter: drop-shadow(0 3px 10px rgba(0,33,71,0.25));
}
.fv-logo-img {
    width: 90px; height: 90px; object-fit: contain; flex-shrink: 0;
    filter: drop-shadow(0 3px 10px rgba(0,33,71,0.2));
}
.fv-hero h1 {
    font-size: 2.9rem; font-weight: 700; letter-spacing: -0.03em;
    color: var(--hero-title); margin: 0; line-height: 1;
}
.fv-hero p {
    color: var(--text-subtle); font-size: 0.82rem; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase;
    margin: 0.65rem 0 0; text-align: center;
}

/* ── Section labels ─────────────────────────────────────────────────────── */
.fv-label {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--label-color); margin: 2rem 0 0.6rem;
}

/* ── Cards ───────────────────────────────────────────────────────────────── */
.fv-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.25rem; margin-bottom: 0.75rem;
    box-shadow: var(--card-shadow);
}
.fv-card-red   { background: var(--badge-bg);  border-color: var(--badge-border); }
.fv-card-navy  { background: var(--surface-2); border-color: var(--border); }

/* ── Grade card ──────────────────────────────────────────────────────────── */
.fv-grade-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.25rem; text-align: center;
    box-shadow: var(--card-shadow);
}

/* ── Report output ───────────────────────────────────────────────────────── */
.fv-report {
    background: var(--report-bg); border: 1px solid var(--border);
    border-left: 2px solid #C8102E; border-radius: 0 10px 10px 0;
    padding: 1.75rem 2rem; margin: 0.5rem 0 1.5rem;
    line-height: 1.8; color: var(--report-text); font-size: 1rem;
    box-shadow: var(--card-shadow);
}

/* ── Badge ───────────────────────────────────────────────────────────────── */
.fv-badge {
    display: inline-block; background: var(--badge-bg); border: 1px solid var(--badge-border);
    color: var(--badge-text); font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.28rem 0.65rem; border-radius: 4px; margin-bottom: 0.75rem;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important; border-bottom: 1px solid var(--border) !important;
    gap: 0 !important; padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important; border-radius: 0 !important;
    color: var(--tab-inactive) !important; font-size: 0.9rem !important;
    font-weight: 500 !important; padding: 0.7rem 1.2rem !important;
    margin-bottom: -1px !important; transition: color 0.15s ease, border-color 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important; border-bottom-color: #C8102E !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--tab-hover) !important; background: var(--surface-2) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: #C8102E !important; color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.92rem !important;
    letter-spacing: 0.015em !important; padding: 0.6rem 1.3rem !important;
    box-shadow: 0 2px 8px rgba(200,16,46,0.3) !important;
    transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease !important;
}
.stButton > button:hover {
    background: #a80d26 !important; box-shadow: 0 4px 14px rgba(200,16,46,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.fv-toggle-btn .stButton > button {
    background: var(--toggle-bg) !important; color: var(--toggle-text) !important;
    border: 1px solid var(--toggle-border) !important; box-shadow: none !important;
    font-size: 1rem !important; padding: 0.4rem 0.6rem !important;
}
.fv-toggle-btn .stButton > button:hover {
    background: var(--toggle-hover-bg) !important; transform: none !important;
    box-shadow: none !important;
}

/* ── Download buttons ────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: transparent !important; color: var(--dl-btn-text) !important;
    border: 1px solid var(--dl-btn-border) !important; border-radius: 8px !important;
    font-weight: 500 !important; font-size: 0.82rem !important;
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
    background: var(--input-bg) !important; border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important; color: var(--text) !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.95rem !important;
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
[data-baseweb="textarea"] textarea::placeholder { color: var(--placeholder) !important; }
label { color: var(--text-muted) !important; font-size: 0.85rem !important; font-weight: 500 !important; }

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--upload-bg) !important; border: 1px dashed var(--upload-border) !important;
    border-radius: 10px !important; transition: border-color 0.15s ease, background 0.15s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(200,16,46,0.4) !important; background: var(--badge-bg) !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--expander-bg) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.83rem !important; color: var(--expander-text) !important; font-weight: 500 !important;
}

/* ── Chat ────────────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInputTextArea"],
[data-testid="stChatInputTextArea"] textarea {
    background: var(--chat-input-bg) !important; color: var(--text) !important;
    border: 1px solid var(--border-strong) !important; border-radius: 10px !important;
}
[data-testid="stChatInputTextArea"] textarea::placeholder { color: var(--placeholder) !important; }
[data-testid="stChatMessage"] {
    background: var(--chat-msg-bg) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--divider) !important; margin: 1.75rem 0 !important; }

/* ── Notifications ───────────────────────────────────────────────────────── */
[data-testid="stNotification"] { background: var(--notification-bg) !important; border-radius: 8px !important; }

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
    # ── Session state init ────────────────────────────────────────────────────
    defaults = {
        # Session
        "session_id":            str(uuid.uuid4())[:8].upper(),
        "session_items":         [],
        "session_chat":          [],
        # Notes tab
        "notes_analyzed":        False,
        "notes_transcription":   "",
        "notes_insights":        "",
        "notes_rag_context":     [],
        "notes_images":          [],
        "notes_chat":            [],
        # Batch mode
        "batch_mode":            False,
        "batch_results":         [],
        # Trackman tab
        "trackman_analyzed":     False,
        "trackman_summary":      "",
        "trackman_insights":     "",
        "trackman_chat":         [],
        "trackman_interp":       "",
        "trackman_interp_done":  False,
        # Players tab
        "talent_pool":           [],
        # Theme
        "theme":                 "dark",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    _logo_element = (
        f'<img src="{_LOGO_DATA_URL}" class="fv-logo-img" alt="FieldVision logo"/>'
        if _LOGO_DATA_URL else LOGO_SVG
    )
    st.markdown(f"""
        <div class="fv-hero">
            <div class="fv-logo">
                {_logo_element}
                <h1>FieldVision</h1>
            </div>
            <p>Baseball Scouting Intelligence</p>
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
        with st.container():
            st.markdown('<div class="fv-toggle-btn">', unsafe_allow_html=True)
            if st.button(icon, use_container_width=True, help="Toggle theme", key="theme_toggle"):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    with col_reset:
        if st.button("New Session", use_container_width=True):
            for key in list(defaults.keys()):
                if key != "theme":
                    del st.session_state[key]
            st.rerun()

    if st.session_state.session_items:
        with st.expander(f"📁 Session Contents ({len(st.session_state.session_items)} items)", expanded=False):
            for i, item in enumerate(st.session_state.session_items, 1):
                st.markdown(f"**{i}. [{item['type'].upper()}]** {item['label']}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_notes, tab_trackman, tab_players, tab_session = st.tabs([
        "📋 Scouting Notes", "📊 Trackman Data", "👥 Players", "💬 Session Chat"
    ])

    # =========================================================================
    # TAB 1 — Scouting Notes (single + batch)
    # =========================================================================
    with tab_notes:

        # Batch mode toggle
        batch_col, _ = st.columns([1.4, 3])
        with batch_col:
            new_batch_mode = st.toggle(
                "Batch mode — one report per file",
                value=st.session_state.batch_mode,
                key="batch_toggle",
                help="When ON, each uploaded file is analyzed separately as an individual player.",
            )
            if new_batch_mode != st.session_state.batch_mode:
                st.session_state.batch_mode = new_batch_mode

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

        # ── BATCH MODE ────────────────────────────────────────────────────────
        elif st.session_state.batch_mode:
            st.markdown(
                f'<div class="fv-badge">✓ {len(uploaded_files)} file(s) — batch mode active</div>',
                unsafe_allow_html=True,
            )
            if st.button("Analyze All Players", type="primary", use_container_width=True, key="batch_btn"):
                # Group images by source file
                all_images = collect_images(uploaded_files)
                file_groups: dict[str, list] = {}
                for name, img in all_images:
                    base = name.split(" (page")[0]
                    file_groups.setdefault(base, []).append((name, img))

                batch_results = []
                progress_bar = st.progress(0)
                status_msg   = st.empty()
                total_files  = len(file_groups)

                for i, (filename, file_images) in enumerate(file_groups.items()):
                    status_msg.markdown(
                        f'<p class="fv-label">Processing {filename}… ({i+1}/{total_files})</p>',
                        unsafe_allow_html=True,
                    )
                    # Transcribe pages
                    transcriptions = []
                    for _, img in file_images:
                        transcriptions.append(transcribe_image(client, img))
                    combined = "\n\n".join(transcriptions)

                    # RAG
                    rag_ctx = retrieve_context(combined, rag_index, rag_df)

                    # Insights
                    session_ctx = build_session_context(st.session_state.session_items)
                    insights = generate_insights(
                        client, combined, rag_ctx,
                        notes_context=notes_context, output_focus=notes_focus,
                        session_context=session_ctx,
                    )

                    # Extract player profile
                    profile = extract_player_profile(client, filename, insights)
                    profile["insights"] = insights
                    profile["label"]    = filename

                    batch_results.append({
                        "filename":      filename,
                        "transcription": combined,
                        "insights":      insights,
                        "context":       rag_ctx,
                        "images":        file_images,
                        "profile":       profile,
                    })

                    # Add to session and talent pool
                    st.session_state.session_items.append({
                        "type": "notes", "label": filename,
                        "content": combined, "insights": insights,
                    })
                    st.session_state.talent_pool.append(profile)

                    progress_bar.progress((i + 1) / total_files)

                st.session_state.batch_results = batch_results
                status_msg.empty()
                progress_bar.empty()
                st.success(f"✓ Analyzed {total_files} player(s). See results below and the Players tab for comparison.")

            # Show batch results
            if st.session_state.batch_results:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown(
                    f'<p class="fv-label">Batch Results — {len(st.session_state.batch_results)} Players</p>',
                    unsafe_allow_html=True,
                )
                for result in st.session_state.batch_results:
                    p = result.get("profile", {})
                    grade = p.get("grade", "?")
                    gcolor = GRADE_COLORS.get(grade, GRADE_COLORS["?"])
                    label  = f"**{p.get('name', result['filename'])}** — {p.get('position','?')} — Grade {grade}"
                    with st.expander(label, expanded=False):
                        # Thumbnails
                        if result["images"]:
                            tcols = st.columns(min(len(result["images"]), 4))
                            for j, (name, image) in enumerate(result["images"]):
                                with tcols[j % 4]:
                                    st.image(image, caption=name, use_container_width=True)
                        # Report
                        st.markdown('<div class="fv-report">', unsafe_allow_html=True)
                        st.markdown(result["insights"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        # Download
                        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", result["filename"])[:30]
                        st.download_button(
                            "⬇ Download Report (.pdf)",
                            data=bytes(build_pdf(f"Scouting Report — {result['filename']}", result["insights"])),
                            file_name=f"report_{safe_name}.pdf",
                            mime="application/pdf",
                            key=f"dl_batch_{result['filename']}",
                        )

        # ── SINGLE MODE (original behavior) ───────────────────────────────────
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
                st.session_state.notes_analyzed      = True
                st.session_state.notes_transcription = combined
                st.session_state.notes_insights      = insights
                st.session_state.notes_rag_context   = context
                st.session_state.notes_images        = images
                st.session_state.notes_chat          = []
                # Extract player profile and add to talent pool
                with st.spinner("Building player profile…"):
                    profile = extract_player_profile(client, label, insights)
                    profile["insights"] = insights
                    profile["label"]    = label
                    st.session_state.talent_pool.append(profile)

        # ── Single mode results ───────────────────────────────────────────────
        if st.session_state.notes_analyzed and not st.session_state.batch_mode:
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
    # TAB 2 — Trackman Data (+ AI pitch interpretation)
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
                st.session_state.trackman_analyzed    = True
                st.session_state.trackman_summary     = summary
                st.session_state.trackman_insights    = insights
                st.session_state.trackman_chat        = []
                st.session_state.trackman_interp      = ""
                st.session_state.trackman_interp_done = False

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

            # ── AI Pitch Interpretation ───────────────────────────────────────
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<p class="fv-label">AI Pitch Interpretation</p>', unsafe_allow_html=True)
            st.markdown("""
                <div class="fv-card fv-card-navy" style="margin-bottom:1rem; padding:1rem 1.25rem;">
                    <span style="color:var(--text-muted); font-size:0.85rem;">
                        Translate the pitch metrics into plain English — written for coaches, players, and parents.
                        No jargon, just practical meaning.
                    </span>
                </div>
            """, unsafe_allow_html=True)

            interp_col, _ = st.columns([2, 1])
            with interp_col:
                interp_focus = st.text_input(
                    "What to focus on? (optional)",
                    placeholder="e.g. Our pitchers' command, opponent fastball spin…",
                    key="interp_focus",
                )

            if st.button("Explain in Plain Language", type="primary", use_container_width=True, key="interp_btn"):
                with st.spinner("Translating metrics into plain English…"):
                    interp = interpret_pitch_metrics(client, summary, focus=interp_focus)
                st.session_state.trackman_interp      = interp
                st.session_state.trackman_interp_done = True

            if st.session_state.trackman_interp_done:
                st.markdown('<div class="fv-report">', unsafe_allow_html=True)
                st.markdown(st.session_state.trackman_interp)
                st.markdown('</div>', unsafe_allow_html=True)
                st.download_button(
                    "⬇ Plain Language Report (.pdf)",
                    data=bytes(build_pdf("Pitch Metrics — Plain Language", st.session_state.trackman_interp)),
                    file_name="pitch_interpretation.pdf", mime="application/pdf",
                    use_container_width=True,
                )

            st.markdown("<hr>", unsafe_allow_html=True)
            render_chat(client, summary, "trackman_chat",
                        session_context=build_session_context(st.session_state.session_items))

    # =========================================================================
    # TAB 3 — Players (Talent Pool Filtering + Comparison)
    # =========================================================================
    with tab_players:
        if not st.session_state.talent_pool:
            st.markdown("""
                <div class="fv-card" style="text-align:center; padding:3rem; margin-top:1rem;">
                    <div style="font-size:2rem; margin-bottom:0.75rem;">👥</div>
                    <div style="color:var(--text-subtle); font-size:0.9rem;">
                        Analyze scouting notes first — players are automatically added here
                        for comparison and filtering.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            pool = st.session_state.talent_pool

            # ── Talent Pool Filtering ─────────────────────────────────────────
            st.markdown('<p class="fv-label">Talent Pool</p>', unsafe_allow_html=True)

            fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
            with fcol1:
                grade_filter = st.multiselect(
                    "Grade", ["A", "B", "C", "D"],
                    key="grade_filter",
                    help="A=Elite, B=Solid, C=Developmental, D=Not recommended",
                )
            with fcol2:
                all_positions = sorted({p["position"] for p in pool if p.get("position") not in ("Unknown", "")})
                pos_filter = st.multiselect("Position", all_positions, key="pos_filter")
            with fcol3:
                search_q = st.text_input(
                    "Search",
                    placeholder="Player name, strength, keyword…",
                    key="player_search",
                )

            # Apply filters
            filtered = pool
            if grade_filter:
                filtered = [p for p in filtered if p.get("grade") in grade_filter]
            if pos_filter:
                filtered = [p for p in filtered if p.get("position") in pos_filter]
            if search_q:
                sq = search_q.lower()
                filtered = [
                    p for p in filtered
                    if sq in p.get("name", "").lower()
                    or sq in p.get("summary", "").lower()
                    or any(sq in s.lower() for s in p.get("strengths", []))
                    or any(sq in c.lower() for c in p.get("concerns", []))
                ]

            st.markdown(
                f'<div class="fv-badge">✓ {len(filtered)} player{"s" if len(filtered) != 1 else ""}</div>',
                unsafe_allow_html=True,
            )

            for player in filtered:
                grade  = player.get("grade", "?")
                gcolor = GRADE_COLORS.get(grade, GRADE_COLORS["?"])
                glabel = GRADE_LABELS.get(grade, "")
                header = f"**{player.get('name', player.get('label','?'))}** — {player.get('position','?')} — Grade {grade}"

                with st.expander(header, expanded=False):
                    left, right = st.columns([1, 2])
                    with left:
                        st.markdown(f"""
                            <div class="fv-grade-card" style="border-top: 3px solid {gcolor};">
                                <div style="font-size:3rem;font-weight:800;color:{gcolor};line-height:1;">{grade}</div>
                                <div style="color:var(--text-muted);font-size:0.78rem;margin-top:4px;">{glabel}</div>
                                <div style="color:var(--text-subtle);font-size:0.8rem;margin-top:8px;">{player.get('position','Unknown')}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        if player.get("strengths"):
                            st.markdown("**Strengths**")
                            for s in player["strengths"]:
                                st.markdown(f"✓ {s}")
                        if player.get("concerns"):
                            st.markdown("**Concerns**")
                            for c in player["concerns"]:
                                st.markdown(f"⚠ {c}")
                    with right:
                        st.markdown(f"*{player.get('summary', '')}*")
                        if player.get("insights"):
                            with st.expander("Full Report", expanded=False):
                                st.markdown('<div class="fv-report">', unsafe_allow_html=True)
                                st.markdown(player["insights"])
                                st.markdown('</div>', unsafe_allow_html=True)

            # ── Player Comparison ─────────────────────────────────────────────
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<p class="fv-label">Side-by-Side Comparison</p>', unsafe_allow_html=True)

            all_names = [p.get("name", p.get("label", f"Player {i+1}")) for i, p in enumerate(pool)]
            selected_names = st.multiselect(
                "Select 2–3 players to compare",
                options=all_names,
                max_selections=3,
                key="compare_select",
            )

            if len(selected_names) >= 2:
                compare_pool = [
                    p for p in pool
                    if p.get("name", p.get("label")) in selected_names
                ]
                cols = st.columns(len(compare_pool))
                for col, player in zip(cols, compare_pool):
                    grade  = player.get("grade", "?")
                    gcolor = GRADE_COLORS.get(grade, GRADE_COLORS["?"])
                    glabel = GRADE_LABELS.get(grade, "")
                    with col:
                        st.markdown(f"""
                            <div class="fv-card" style="text-align:center; border-top:3px solid {gcolor}; padding:1.25rem;">
                                <div style="font-weight:700;font-size:1rem;margin-bottom:0.5rem;">
                                    {player.get('name', player.get('label','?'))}
                                </div>
                                <div style="font-size:2.8rem;font-weight:800;color:{gcolor};line-height:1;">{grade}</div>
                                <div style="color:var(--text-muted);font-size:0.75rem;margin-top:4px;">{glabel}</div>
                                <div style="color:var(--text-subtle);font-size:0.8rem;margin-top:6px;">{player.get('position','?')}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown("**Strengths**")
                        for s in (player.get("strengths") or ["—"]):
                            st.markdown(f"✓ {s}")
                        st.markdown("**Concerns**")
                        for c in (player.get("concerns") or ["—"]):
                            st.markdown(f"⚠ {c}")
                        st.markdown(f"*{player.get('summary', '')}*")
            elif len(selected_names) == 1:
                st.info("Select at least one more player to compare.")
            else:
                st.markdown(
                    '<div style="color:var(--text-subtle);font-size:0.88rem;">'
                    'Select 2–3 players from the list above to see a side-by-side comparison.</div>',
                    unsafe_allow_html=True,
                )

    # =========================================================================
    # TAB 4 — Session Chat
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
