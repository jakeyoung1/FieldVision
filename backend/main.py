"""FieldVision — FastAPI backend entry point."""
import os
from pathlib import Path

# Load .env if present (local dev)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routes import analyze, chat, trackman

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FieldVision API",
    description="Baseball scouting intelligence powered by Claude + Branch Rickey RAG",
    version="2.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(analyze.router, prefix="/api")
app.include_router(chat.router,    prefix="/api")
app.include_router(trackman.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Static frontend ───────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent.parent / "static"
ROOT_HTML  = Path(__file__).parent.parent / "index.html"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_root():
    return FileResponse(str(ROOT_HTML))


@app.get("/{path:path}", include_in_schema=False)
def serve_spa(path: str):
    """Serve index.html for all non-API routes (SPA fallback)."""
    file = Path(__file__).parent.parent / path
    if file.exists() and file.is_file():
        return FileResponse(str(file))
    return FileResponse(str(ROOT_HTML))
