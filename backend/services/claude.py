"""Claude API service — analysis, chat, profile extraction, pitch interpretation."""
import json
import re
from functools import lru_cache

import anthropic

MODEL = "claude-sonnet-4-5"

SYSTEM_SCOUT = """You are FieldVision, an AI baseball scouting assistant trained on Branch Rickey's
1,919 historical scouting documents. Analyze player notes with precision, referencing Rickey's
evaluation frameworks when relevant. Be concise, insightful, and use baseball terminology naturally."""

SYSTEM_COACH = """You are FieldVision, a baseball analytics assistant for a college coaching staff.
You help interpret player data, Trackman metrics, and scouting reports. Speak plainly — coaches
need actionable insights, not jargon."""

GRADE_LABELS = {
    "A": "Elite prospect",
    "B": "Strong candidate",
    "C": "Developmental",
    "D": "Needs significant work",
    "F": "Not recommended",
}


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def analyze_notes(text: str, context: str = "") -> str:
    """Analyze raw scouting notes and return a structured report."""
    context_block = f"REFERENCE CONTEXT:\n{context}\n" if context else ""
    prompt = f"""Analyze these baseball scouting notes and produce a structured report.

{context_block}
SCOUTING NOTES:
{text}

Format your response as:
## Player Overview
## Key Strengths
## Areas of Concern
## Historical Comparison (if context available)
## Recommendation & Grade (A/B/C/D/F)"""

    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        system=SYSTEM_SCOUT,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def chat_reply(history: list[dict], context: str = "", session_context: str = "") -> str:
    """Continue a scouting chat conversation."""
    system = SYSTEM_SCOUT
    if session_context:
        system += f"\n\nSession context:\n{session_context}"
    if context:
        system += f"\n\n{context}"

    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=system,
        messages=history,
    )
    return resp.content[0].text


def extract_player_profile(label: str, insights_text: str) -> dict:
    """Extract a structured JSON player profile from analysis text."""
    prompt = f"""Extract a structured player profile from this scouting analysis.
Return ONLY a valid JSON object with exactly these keys:
{{"name": "string or null", "position": "string or null", "grade": "A/B/C/D/F",
  "strengths": ["list","of","strings"], "concerns": ["list","of","strings"],
  "summary": "1-2 sentence summary"}}

Label: {label}
INSIGHTS:
{insights_text[:1800]}"""

    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {
        "name": label, "position": None, "grade": "C",
        "strengths": [], "concerns": [],
        "summary": insights_text[:120],
    }


def interpret_pitch_metrics(summary: str, focus: str = "") -> str:
    """Translate Trackman pitch metrics into plain-language coach explanation."""
    focus_line = f"Focus area: {focus}" if focus else ""
    prompt = f"""You are a baseball analyst presenting Trackman pitch data objectively.
This data may include pitchers from multiple teams. Describe each pitcher's metrics factually —
velocity, spin, pitch mix, tendencies. Use neutral third-person language (e.g. "Smith throws..."
not "our guy" or "we need"). 2-3 sentences per pitcher. End with one cross-dataset observation.
{focus_line}

DATA SUMMARY:
{summary}"""

    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_COACH,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text
