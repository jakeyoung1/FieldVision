"""File processing — PDF extraction, text normalization."""
import io
import re
from pathlib import Path


def extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from uploaded file bytes."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _pdf_to_text(content)
    elif ext in (".txt", ".md"):
        return content.decode("utf-8", errors="replace")
    elif ext == ".csv":
        return content.decode("utf-8", errors="replace")
    else:
        # Try UTF-8 decode as fallback
        try:
            return content.decode("utf-8", errors="replace")
        except Exception:
            return ""


def _pdf_to_text(content: bytes) -> str:
    """Convert PDF bytes to text using pdf2image + basic extraction."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
    except ImportError:
        pass

    # Fallback: try pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except ImportError:
        pass

    return "[PDF text extraction unavailable — install pdfplumber or pypdf]"


def group_by_player(files: list[tuple[str, bytes]]) -> dict[str, str]:
    """
    Group files by player name (strips ' (page N)' suffixes from multi-page PDFs).
    Returns {player_label: combined_text}.
    """
    groups: dict[str, list[str]] = {}
    for filename, content in files:
        base = re.sub(r"\s*\(page \d+\)\s*$", "", Path(filename).stem, flags=re.IGNORECASE).strip()
        text = extract_text(filename, content)
        groups.setdefault(base, []).append(text)

    return {label: "\n\n".join(texts) for label, texts in groups.items()}
