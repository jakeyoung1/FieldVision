"""POST /api/chat — follow-up conversation on a scouting report."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.services import claude

router = APIRouter()


class ChatRequest(BaseModel):
    history: list[dict]       # [{role: user|assistant, content: str}]
    context: str = ""         # The report text to ground answers in
    session_context: str = "" # Brief session summary for system prompt


@router.post("/chat")
async def chat(req: ChatRequest):
    if not req.history:
        raise HTTPException(400, "history is required")

    reply = claude.chat_reply(req.history, req.context, req.session_context)
    return JSONResponse({"reply": reply})
