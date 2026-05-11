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
    "A":  "Elite prospect",
    "A-": "Near-elite prospect",
    "B+": "Strong candidate, above average",
    "B":  "Solid candidate",
    "B-": "Above average with questions",
    "C+": "Average with upside",
    "C":  "Average / developmental",
    "C-": "Below average with some tools",
    "D+": "Significant concerns, one redeeming tool",
    "D":  "Needs significant work",
    "F":  "Not recommended",
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
## Recommendation & Grade (use full plus/minus scale: A, A-, B+, B, B-, C+, C, C-, D+, D, F)"""

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
{{"name": "string or null", "position": "string or null",
  "grade": "use full plus/minus scale: A, A-, B+, B, B-, C+, C, C-, D+, D, or F",
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


def extract_players_from_chat(reply: str, context: str = "") -> list[dict]:
    """
    Given an AI chat reply (and optional scouting context), extract structured
    profiles for any players that are meaningfully evaluated — not just named.
    Returns a list of profile dicts identical in shape to extract_player_profile().
    """
    context_block = (
        f"SCOUTING REPORTS (authoritative source for established grades):\n{context[:2500]}\n\n"
        if context else ""
    )
    prompt = f"""{context_block}Read the CHAT REPLY below and extract profiles for baseball players \
that are meaningfully evaluated — ranked, described, compared, or assessed. \
Skip players only mentioned by name with no evaluation.

CHAT REPLY:
{reply[:2500]}

GRADING INSTRUCTIONS — follow in this exact order:
1. If the SCOUTING REPORTS above contain an explicit grade for this player, \
use THAT grade exactly. Do not change it.
2. If no prior grade exists, assign using the full plus/minus scale based on evaluation language:
   A   — generational, elite at this level, can't-miss prospect
   A-  — elite with one minor question, clear top prospect
   B+  — above average, stands out vs. peers, has a plus tool
   B   — solid performer, above average overall, reliable
   B-  — above average but inconsistent or one clear weakness
   C+  — average with upside, shows flashes, developing tool
   C   — average, nothing jumps out, fits the level
   C-  — below average overall but shows something worth watching
   D+  — struggles in most areas, one redeeming quality
   D   — significant issues, below the level
   F   — not recommended, does not project
3. C is the baseline for an average player. B is above average. Most players are C or C+/C-.
4. When multiple players are in the same reply, their grades MUST differ if the language \
implies one outperformed another. Do not assign the same grade to everyone.
5. Use plus/minus to capture nuance — B+ and B- are different players.

Return ONLY a valid JSON array — no explanation, no markdown fences.
Each element must have exactly these keys:
{{"name": "Full Name", "position": "string or null",
  "grade": "one of: A, A-, B+, B, B-, C+, C, C-, D+, D, F",
  "strengths": ["list","of","strings"], "concerns": ["list","of","strings"],
  "summary": "1-2 sentence summary based on the evaluation"}}

Return [] if no players are meaningfully evaluated."""

    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if isinstance(result, list):
                    return result
            except Exception:
                pass
    return []


def profile_to_report(profile: dict) -> str:
    """Build a brief markdown report from a game player profile dict."""
    name      = profile.get("name", "Unknown")
    pos       = profile.get("position") or ""
    team      = profile.get("team") or ""
    grade     = profile.get("grade", "C")
    summary   = profile.get("summary", "")
    strengths = profile.get("strengths", [])
    concerns  = profile.get("concerns", [])
    header = f"{name} ({pos}) — {team}" if (pos or team) else name
    parts = [f"## Player Overview\n{header}"]
    if strengths:
        parts.append("## Key Strengths\n" + "\n".join(f"- {s}" for s in strengths))
    if concerns:
        parts.append("## Areas of Concern\n" + "\n".join(f"- {c}" for c in concerns))
    label = GRADE_LABELS.get(grade, "")
    parts.append(
        f"## Recommendation & Grade\n**Grade: {grade}**"
        + (f" — {label}" if label else "")
        + (f"\n\n{summary}" if summary else "")
    )
    return "\n\n".join(parts)


def analyze_game_document(text: str) -> list[dict]:
    """
    Given a full game play-by-play or box score, return the top 3 performers
    per team (6 total) as structured player profile dicts.
    """
    prompt = f"""Read this baseball game document carefully. It is a full play-by-play or box score.

GAME DOCUMENT:
{text[:5000]}

Step 1 — Identify BOTH teams and ALL players mentioned for each team.
Step 2 — Tally each player's specific contributions from the document:
  Batters: hits, extra-base hits (2B/3B/HR), RBIs, runs scored, walks, strikeouts, stolen bases
  Pitchers: innings pitched, strikeouts, runs/hits allowed
Step 3 — Rank players within each team by their game impact.
Step 4 — Select the TOP 3 PERFORMERS from EACH TEAM — exactly 6 players total.
  Priority: HR > multi-RBI hit > multiple hits > extra-base hit > walk/OBP > pitching > nothing

Grade each player on their performance in THIS game. C is the baseline for an average game.
Grades MUST differ within each team — not everyone played equally.
Be conservative: a solo HR in a lopsided loss is NOT an A-. Context matters.
  A   — Generational game: 3+ RBIs including HR, or completely dominated (pitcher)
  A-  — Elite game: HR + 2+ RBI, or 3+ hits with multiple RBIs, shaped the outcome
  B+  — Strong: solo HR, or 2 hits + 1 RBI, clearly above average today
  B   — Solid: 2 hits, or 1 hit + 1 RBI, consistent positive at-bats
  B-  — Above average: 1-2 hits with no RBI, or 1 hit + walk, productive
  C+  — Slightly above: 1 hit or 2 walks, one positive moment otherwise quiet
  C   — Average (baseline): reached base once, routine game, nothing notable
  C-  — Below average: 0 hits, 1 walk, quiet game, mild negative plays
  D+  — Poor: 0 hits, 2+ Ks, baserunning mistake, but one small positive
  D   — Very poor: 0 hits, multiple Ks, no contribution, hurt the team
  F   — Did not contribute in any way; actively hurt team performance

Return ONLY a valid JSON array of exactly 6 objects. No markdown fences, no explanation:
[{{"name": "Last, First", "position": "position abbreviation or null",
  "team": "team name exactly as shown in document",
  "grade": "letter grade with plus/minus if applicable",
  "strengths": ["specific contributions from THIS game only"],
  "concerns": ["specific weaknesses from THIS game only"],
  "summary": "1-2 sentences on this player's performance in this specific game"}}]"""

    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1400,
        system=SYSTEM_SCOUT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if isinstance(result, list):
                    return result
            except Exception:
                pass
    return []


def grade_pitchers_from_trackman(stats_summary: str) -> list[dict]:
    """
    Given a plain-text outcome stats summary per pitcher, return a grade + profile
    for each pitcher. Uses the same profile shape as other extraction functions.
    """
    prompt = f"""Grade each pitcher below based solely on their outcome stats from a Trackman CSV.

PITCHER STATS:
{stats_summary}

Grade each pitcher on this outing using the full plus/minus scale.
Context matters — a 7K/0HR performance is elite; 2HR/8R in 1 IP is poor.
Grades MUST differ between pitchers when stats clearly separate them.
  A   — Dominant: 6+ K, 0-1 BB, 0 HR, high whiff%, near-zero ERA
  A-  — Excellent: 4+ K, K > BB by large margin, minimal hits/HR
  B+  — Good: 3+ K, K > BB, few H, limited runs, above-average command
  B   — Solid: 2-3 K, reasonable control, average outing overall
  B-  — Above average with a flaw: decent K but walks or hits creeping in
  C+  — Average with a positive: K roughly = BB, some H, controlled damage
  C   — Average (baseline): mixed results, no clear strength or weakness
  C-  — Below average: BB >= K, multiple H, runs allowed, poor command
  D+  — Poor: high H+BB, HR allowed, significant runs, limited K
  D   — Very poor: blown outing, multiple HR, large run total
  F   — Catastrophic: gave up 2+ HR and many runs in minimal IP

Return ONLY a valid JSON array — no markdown fences, no explanation:
[{{"name": "Last, First", "position": "P", "team": "team code from stats",
  "grade": "letter grade with plus/minus if applicable",
  "strengths": ["specific stat-backed strength from this outing"],
  "concerns": ["specific stat-backed concern from this outing"],
  "summary": "1-2 sentence assessment of this specific outing"}}]"""

    client = _client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if isinstance(result, list):
                    return result
            except Exception:
                pass
    return []


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
