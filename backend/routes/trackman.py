"""POST /api/trackman — Trackman CSV upload, stats + AI interpretation."""
import io

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.services import claude

router = APIRouter()

PITCH_COLS = [
    "Pitcher", "PitcherTeam", "PitchType", "RelSpeed", "SpinRate",
    "InducedVertBreak", "HorzBreak", "PlateLocHeight", "PlateLocSide",
    "PitchCall", "TaggedPitchType", "AutoPitchType",
    # outcome stats
    "KorBB", "PlayResult", "OutsOnPlay", "RunsScored",
]

HITS_SET   = {"Single", "Double", "Triple", "HomeRun"}
STRIKE_SET = {"StrikeCalled", "StrikeSwinging", "FoulBallNotFieldable", "InPlay"}
SWING_SET  = {"StrikeSwinging", "FoulBallNotFieldable", "InPlay"}


def _safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


@router.post("/trackman")
async def trackman(
    file: UploadFile = File(...),
    focus: str = Form(""),
):
    try:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(400, "Only CSV files are supported")

        content = await file.read()
        try:
            df = pd.read_csv(io.StringIO(content.decode("utf-8", errors="replace")))
        except Exception as e:
            raise HTTPException(422, f"Could not parse CSV: {e}")

        available = _safe_cols(df, PITCH_COLS)
        if not available:
            raise HTTPException(422, "No recognized Trackman columns found. Make sure this is a Trackman export CSV.")

        df_clean = df[available].dropna(how="all")

        # Build summary stats per pitcher (also capture team if available)
        stats = {}
        if "Pitcher" in df_clean.columns:
            # Use original df to get PitcherTeam (it may not be in df_clean if not in PITCH_COLS hit)
            team_col_available = "PitcherTeam" in df.columns
            for pitcher, grp in df_clean.groupby("Pitcher"):
                pitcher_stats: dict = {"pitches": len(grp)}
                # Team affiliation
                if team_col_available:
                    team_vals = df.loc[df["Pitcher"] == pitcher, "PitcherTeam"].dropna()
                    pitcher_stats["team"] = str(team_vals.iloc[0]) if not team_vals.empty else ""
                else:
                    pitcher_stats["team"] = ""
                if "RelSpeed" in grp.columns:
                    pitcher_stats["avg_velo"] = round(float(grp["RelSpeed"].mean()), 1)
                    pitcher_stats["max_velo"] = round(float(grp["RelSpeed"].max()), 1)
                if "SpinRate" in grp.columns:
                    pitcher_stats["avg_spin"] = round(float(grp["SpinRate"].mean()), 0)
                if "PitchType" in grp.columns or "TaggedPitchType" in grp.columns:
                    col = "TaggedPitchType" if "TaggedPitchType" in grp.columns else "PitchType"
                    pitcher_stats["pitch_mix"] = grp[col].value_counts().to_dict()
                # Outcome stats
                if "KorBB" in grp.columns:
                    pitcher_stats["strikeouts"] = int((grp["KorBB"] == "Strikeout").sum())
                    pitcher_stats["walks"]      = int((grp["KorBB"] == "Walk").sum())
                if "PlayResult" in grp.columns:
                    pr = grp["PlayResult"].fillna("")
                    pitcher_stats["hits_allowed"] = int(pr.isin(HITS_SET).sum())
                    pitcher_stats["home_runs"]    = int((pr == "HomeRun").sum())
                if "RunsScored" in grp.columns:
                    pitcher_stats["runs_scored"] = int(grp["RunsScored"].fillna(0).sum())
                if "OutsOnPlay" in grp.columns:
                    total_outs = int(grp["OutsOnPlay"].fillna(0).sum())
                    pitcher_stats["outs_recorded"]   = total_outs
                    pitcher_stats["innings_pitched"]  = round(total_outs / 3, 1)
                if "PitchCall" in grp.columns:
                    total  = len(grp)
                    strikes = grp["PitchCall"].isin(STRIKE_SET).sum()
                    pitcher_stats["strike_pct"] = round(float(strikes / total * 100), 1) if total else 0.0
                    swings = grp["PitchCall"].isin(SWING_SET).sum()
                    whiffs = (grp["PitchCall"] == "StrikeSwinging").sum()
                    pitcher_stats["whiff_pct"] = round(float(whiffs / swings * 100), 1) if swings else 0.0
                stats[str(pitcher)] = pitcher_stats

        # Build teams grouping: { teamName: [pitcherName, ...] }
        teams: dict[str, list[str]] = {}
        for pitcher, s in stats.items():
            team = s.get("team") or "Unknown Team"
            teams.setdefault(team, []).append(pitcher)

        # Build plain-text summary for AI interpretation
        summary_lines = []
        grade_lines   = []
        for pitcher, s in stats.items():
            # Interpretation summary (velocity / spin / mix)
            line = f"{pitcher}: {s['pitches']} pitches"
            if "avg_velo" in s:
                line += f", avg {s['avg_velo']} mph (max {s['max_velo']})"
            if "avg_spin" in s:
                line += f", avg spin {int(s['avg_spin'])} rpm"
            if "pitch_mix" in s:
                mix = ", ".join(f"{k}:{v}" for k, v in list(s["pitch_mix"].items())[:4])
                line += f", mix: {mix}"
            summary_lines.append(line)
            # Grading summary (outcomes)
            gline = f"{pitcher} ({s.get('team','')}): {s.get('pitches',0)}P"
            if "innings_pitched" in s: gline += f" | {s['innings_pitched']}IP"
            if "strikeouts"      in s: gline += f" | {s['strikeouts']}K"
            if "walks"           in s: gline += f" | {s['walks']}BB"
            if "hits_allowed"    in s: gline += f" | {s['hits_allowed']}H"
            if "home_runs"       in s and s["home_runs"]: gline += f" | {s['home_runs']}HR"
            if "runs_scored"     in s: gline += f" | {s['runs_scored']}R"
            if "strike_pct"      in s: gline += f" | str%:{s['strike_pct']}%"
            if "whiff_pct"       in s: gline += f" | whiff%:{s['whiff_pct']}%"
            if "avg_velo"        in s: gline += f" | {s['avg_velo']}mph"
            grade_lines.append(gline)

        summary_text = "\n".join(summary_lines)
        grade_text   = "\n".join(grade_lines)

        if not summary_text:
            raise HTTPException(422, "No pitcher data found in this CSV.")

        # AI interpretation (narrative)
        interpretation = claude.interpret_pitch_metrics(summary_text, focus)

        # Pitcher grades from outcome stats
        pitcher_grades_list = claude.grade_pitchers_from_trackman(grade_text)
        pitcher_grades = {pg["name"]: pg for pg in pitcher_grades_list}

        return JSONResponse({
            "rows": len(df_clean),
            "pitchers": len(stats),
            "stats": stats,
            "teams": teams,
            "summary": summary_text,
            "interpretation": interpretation,
            "columns": available,
            "pitcher_grades": pitcher_grades,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Server error: {str(e)}")
