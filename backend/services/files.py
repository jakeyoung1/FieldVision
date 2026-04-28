"""File processing — PDF extraction, text normalization, Claude Vision OCR."""
import base64
import io
import re
from pathlib import Path


def extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from uploaded file bytes."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _pdf_to_text(content)
    elif ext in (".txt", ".md", ".csv"):
        return content.decode("utf-8", errors="replace")
    else:
        try:
            return content.decode("utf-8", errors="replace")
        except Exception:
            return ""


def _pdf_to_text(content: bytes) -> str:
    """
    Extract text from PDF. Tries pdfplumber first (fast, text-based PDFs).
    Falls back to Claude Vision for scanned / handwritten PDFs.
    """
    # ── Try pdfplumber (text-based PDFs) ─────────────────────────────────────
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
        if pages:
            return "\n\n".join(pages)
    except Exception:
        pass

    # ── Try pypdf (another text-based approach) ───────────────────────────────
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        texts = [page.extract_text() or "" for page in reader.pages]
        combined = "\n\n".join(t for t in texts if t.strip())
        if combined.strip():
            return combined
    except Exception:
        pass

    # ── Fallback: Claude Vision OCR (handles handwritten / scanned PDFs) ─────
    return _pdf_via_claude_vision(content)


def _pdf_via_claude_vision(content: bytes) -> str:
    """
    Send PDF to Claude as a base64 document and ask it to transcribe
    all text including handwritten notes.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()

        # Anthropic accepts PDFs directly as base64 documents
        pdf_b64 = base64.standard_b64encode(content).decode("utf-8")

        resp = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a baseball scouting document. "
                            "Please transcribe ALL text you can see, including handwritten notes, "
                            "abbreviations, numbers, and any annotations. "
                            "Preserve the structure as best you can. "
                            "Output only the transcribed text, nothing else."
                        ),
                    },
                ],
            }],
        )
        return resp.content[0].text.strip()

    except Exception as e:
        return f"[Vision OCR failed: {e}]"


def group_by_player(files: list[tuple[str, bytes]]) -> dict[str, str]:
    """
    Group files by player name (strips ' (page N)' suffixes from multi-page PDFs).
    Returns {player_label: combined_text}.
    """
    groups: dict[str, list[str]] = {}
    for filename, content in files:
        base = re.sub(
            r"\s*\(page \d+\)\s*$", "", Path(filename).stem, flags=re.IGNORECASE
        ).strip()
        text = extract_text(filename, content)
        groups.setdefault(base, []).append(text)

    return {label: "\n\n".join(texts) for label, texts in groups.items()}
