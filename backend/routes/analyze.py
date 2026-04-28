"""POST /api/analyze — single or batch scouting note analysis."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.services import claude, files, rag

router = APIRouter()


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
        # RAG context
        context = rag.context_block(text[:500])
        # Analysis
        report = claude.analyze_notes(text, context)
        # Profile extraction
        profile = claude.extract_player_profile(label, report)
        results.append({
            "label": label,
            "report": report,
            "profile": profile,
            "context_used": bool(context),
        })

    return JSONResponse({"results": results, "count": len(results)})
