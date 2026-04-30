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
]


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
                stats[str(pitcher)] = pitcher_stats

        # Build teams grouping: { teamName: [pitcherName, ...] }
        teams: dict[str, list[str]] = {}
        for pitcher, s in stats.items():
            team = s.get("team") or "Unknown Team"
            teams.setdefault(team, []).append(pitcher)

        # Build plain-text summary for AI interpretation
        summary_lines = []
        for pitcher, s in stats.items():
            line = f"{pitcher}: {s['pitches']} pitches"
            if "avg_velo" in s:
                line += f", avg {s['avg_velo']} mph (max {s['max_velo']})"
            if "avg_spin" in s:
                line += f", avg spin {int(s['avg_spin'])} rpm"
            if "pitch_mix" in s:
                mix = ", ".join(f"{k}:{v}" for k, v in list(s["pitch_mix"].items())[:4])
                line += f", mix: {mix}"
            summary_lines.append(line)
        summary_text = "\n".join(summary_lines)

        if not summary_text:
            raise HTTPException(422, "No pitcher data found in this CSV.")

        # AI interpretation
        interpretation = claude.interpret_pitch_metrics(summary_text, focus)

        return JSONResponse({
            "rows": len(df_clean),
            "pitchers": len(stats),
            "stats": stats,
            "teams": teams,
            "summary": summary_text,
            "interpretation": interpretation,
            "columns": available,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Server error: {str(e)}")
