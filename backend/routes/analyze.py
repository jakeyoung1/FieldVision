"""POST /api/analyze — single or batch scouting note analysis."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.services import claude, files, rag

router = APIRouter()


def _is_game_document(text: str) -> bool:
    """Return True if the text looks like a game play-by-play or box score."""
    tl = text.lower()
    return ("batting" in tl and "inning" in tl) or ("score by innings" in tl)


class ExtractPlayersRequest(BaseModel):
    reply: str
    context: str = ""


@router.post("/extract-players")
async def extract_players(req: ExtractPlayersRequest):
    """Extract structured player profiles from a chat reply."""
    if not req.reply.strip():
        return JSONResponse({"profiles": []})
    try:
        profiles = claude.extract_players_from_chat(req.reply, req.context)
        return JSONResponse({"profiles": profiles})
    except Exception as e:
        raise HTTPException(500, f"Server error: {str(e)}")


@router.post("/analyze")
async def analyze(
    files_upload: list[UploadFile] = File(...),
    batch_mode: bool = Form(False),
    session_id: str = Form(""),
):
    """
    Analyze one or more scouting note files.
    Returns list of player results.
    """
    if not files_upload:
        raise HTTPException(400, "No files uploaded")

    try:
        raw_files = []
        for f in files_upload:
            content = await f.read()
            raw_files.append((f.filename, content))

        # Group multi-page PDFs by player name
        player_map = files.group_by_player(raw_files)

        results = []
        for label, text in player_map.items():
            if not text.strip():
                continue

            if _is_game_document(text):
                # Game play-by-play / box score — extract top 3 per team (6 total)
                game_profiles = claude.analyze_game_document(text)
                for gp in game_profiles:
                    player_name = gp.get("name", label)
                    team = gp.get("team", "")
                    display_label = f"{team} — {player_name}" if team else player_name
                    results.append({
                        "label": display_label,
                        "report": claude.profile_to_report(gp),
                        "profile": gp,
                        "context_used": False,
                    })
            else:
                # Standard individual scouting note
                context = rag.context_block(text[:500])
                report = claude.analyze_notes(text, context)
                profile = claude.extract_player_profile(label, report)
                results.append({
                    "label": label,
                    "report": report,
                    "profile": profile,
                    "context_used": bool(context),
                })

        return JSONResponse({"results": results, "count": len(results)})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Server error: {str(e)}")
