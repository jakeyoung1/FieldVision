"""
Standalone RAG evaluation — does NOT touch app.py or the live website.

Feed in the same kind of handwritten scouting documents the website accepts
(JPG / PNG / PDF). The script runs the exact same pipeline as the app
(OCR -> embed -> retrieve top-5 -> Claude-generated report) and then reports:

  (a) RETRIEVAL QUALITY
      Cosine similarity between the input transcription and each of the
      top-5 retrieved Branch Rickey reports. Higher = the historical context
      was more relevant to the input.

  (b) GENERATION FAITHFULNESS (grounding)
      Cosine similarity between the input transcription and the Claude-
      generated 3-section report. Higher = the generated report stayed
      closer to the content of the original notes.

Usage:
    # Pop up a file-chooser dialog (easiest):
    .venv/bin/python evaluate_similarity.py

    # Or pass paths directly on the command line:
    .venv/bin/python evaluate_similarity.py path/to/note1.jpg path/to/note2.pdf

Output:
    evaluation_report.pdf  (in the current directory)
"""

import argparse
import base64
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic
import faiss
import numpy as np
import pandas as pd
from fpdf import FPDF
from PIL import Image
from sentence_transformers import SentenceTransformer

try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ---------------------------------------------------------------------------
# Constants — identical to app.py so results are directly comparable
# ---------------------------------------------------------------------------
CLAUDE_MODEL    = "claude-sonnet-4-5"
EMBED_MODEL     = "all-MiniLM-L6-v2"
CSV_PATH        = "data/branch-rickey-scouting.csv"
EMBEDDINGS_PATH = "data/embeddings.npy"
TOP_K           = 5

SYSTEM_PROMPT = """You are an experienced baseball scout and analyst with decades of evaluating players at every level — high school, college, and professional. You have deep knowledge of hitting mechanics, pitching, fielding, baserunning, and long-term player development.

Your job is to read scouting notes and turn them into clear, useful intelligence for coaches and scouts.

When analyzing notes:
- Begin with a brief, direct summary of what the notes cover (2–3 sentences max)
- Follow with specific, actionable recommendations the coaching staff can act on immediately
- Be concrete — reference specific observations from the notes, not generic baseball advice
- Use standard scouting language and baseball terminology
- Draw on historical scouting standards when benchmarking a player
- If the user specifies a focus area or desired output, prioritize that above all else

When answering follow-up questions:
- Stay grounded in the scouting notes provided
- Be direct and concise
- If something isn't in the notes, say so rather than speculating
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    """Read the key from .streamlit/secrets.toml or ANTHROPIC_API_KEY env var."""
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ERROR: no ANTHROPIC_API_KEY found in .streamlit/secrets.toml or env.")
        sys.exit(1)
    return key


def pick_files_via_dialog() -> list[str]:
    """Pop up a native file-chooser dialog and return selected paths.

    Uses macOS AppleScript (always available on a Mac, no extra deps).
    Falls back to tkinter for other platforms.
    """
    import platform
    import subprocess

    # ----- macOS: use AppleScript (no deps) --------------------------------
    if platform.system() == "Darwin":
        script = '''
        set theFiles to choose file with prompt "Select handwritten scouting notes (JPG / PNG / PDF)" with multiple selections allowed
        set thePaths to {}
        repeat with f in theFiles
            set end of thePaths to POSIX path of f
        end repeat
        set AppleScript's text item delimiters to linefeed
        return thePaths as text
        '''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, check=True,
            )
            return [p.strip() for p in result.stdout.splitlines() if p.strip()]
        except subprocess.CalledProcessError as e:
            # Error -128 = user cancelled the dialog; treat as empty selection.
            err = (e.stderr or "")
            if "-128" in err or "User canceled" in err:
                return []
            print(f"osascript error: {err.strip() or e}")
            # fall through to tkinter fallback

    # ----- Cross-platform fallback: tkinter --------------------------------
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("ERROR: no file dialog available on this system. "
              "Pass file paths on the command line instead:\n"
              "  .venv/bin/python evaluate_similarity.py path/to/note.jpg")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()
    root.update()
    paths = filedialog.askopenfilenames(
        title="Select handwritten scouting notes (JPG / PNG / PDF)",
        filetypes=[
            ("Scouting notes", "*.jpg *.jpeg *.png *.pdf"),
            ("All files",      "*.*"),
        ],
    )
    root.destroy()
    return list(paths)


def collect_images(paths: list[str]) -> list[tuple[str, Image.Image]]:
    """Same behaviour as the website's collect_images."""
    images = []
    for path in paths:
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"  ! file not found: {path}")
            continue
        raw = path_obj.read_bytes()
        if path_obj.suffix.lower() == ".pdf":
            if not PDF_SUPPORT:
                print(f"  ! pdf2image not available — skipping {path}")
                continue
            pages = convert_from_bytes(raw)
            for i, page in enumerate(pages, 1):
                images.append((f"{path_obj.name} (page {i})", page))
        else:
            images.append((path_obj.name, Image.open(io.BytesIO(raw))))
    return images


def transcribe_image(client: anthropic.Anthropic, image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": image_data},
                },
                {
                    "type": "text",
                    "text": (
                        "You are transcribing a handwritten baseball scouting or match note. "
                        "Transcribe ALL text exactly as written, preserving line breaks, "
                        "abbreviations, shorthand, numbers, and player names. "
                        "If text is illegible indicate with [?]."
                    ),
                },
            ],
        }],
    )
    return response.content[0].text


def retrieve_context(query_text: str, index, df: pd.DataFrame, embed_model, k: int = TOP_K):
    """Same retrieval as the website — returns hits with cosine scores."""
    query_vec = embed_model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)
    hits = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx]
        hits.append({
            "score": float(score),
            "text": row["Transcription"],
            "item": row.get("Item", ""),
        })
    return hits


def generate_insights(client, transcription, historical_context) -> str:
    """Same insight-generation prompt as the website (no session context)."""
    historical_text = ""
    for i, ctx in enumerate(historical_context, 1):
        snippet = ctx["text"][:1000] + ("…" if len(ctx["text"]) > 1000 else "")
        historical_text += f"\n[Historical Report {i}]\n{snippet}\n"

    prompt = f"""Here are the scouting notes to analyze:

SCOUTING NOTES:
{transcription}

RELEVANT HISTORICAL BRANCH RICKEY SCOUTING REPORTS (for benchmarking):
{historical_text}

Provide:

## Summary
A brief 2–3 sentence summary of what these notes cover.

## Actionable Recommendations
Give exactly 2 recommendations grounded directly in the scouting notes above. Be specific and concrete.

## Bonus Insight from the Branch Rickey Papers
Draw one additional insight by connecting something in these notes to the historical Branch Rickey scouting reports provided. Frame it as a historical parallel or lesson from Rickey's scouting philosophy that applies to what you see in these notes."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def cosine(embed_model, a: str, b: str) -> float:
    """Cosine similarity between two texts using the SAME embedder as the RAG."""
    vecs = embed_model.encode([a, b], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(vecs)
    return float(np.dot(vecs[0], vecs[1]))


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
def sanitize(text: str) -> str:
    replacements = {
        "\u2026": "...", "\u2014": "--", "\u2013": "-",
        "\u2018": "'",  "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2022": "-",  "\u00a0": " ", "\u2012": "-", "\u2015": "--",
    }
    for ch, rp in replacements.items():
        text = text.replace(ch, rp)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def interpret(score: float) -> str:
    """Plain-language reading of a cosine score on text embeddings."""
    if score >= 0.70: return "very strong overlap"
    if score >= 0.50: return "strong overlap"
    if score >= 0.35: return "moderate overlap"
    if score >= 0.20: return "weak but non-trivial overlap"
    return "little overlap"


def build_evaluation_pdf(
    input_files: list[str],
    transcription: str,
    retrieval_hits: list[dict],
    generated_report: str,
    grounding_score: float,
    latency: dict,
    output_path: str,
):
    pdf = FPDF()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 33, 71)
    pdf.multi_cell(0, 9, sanitize("FieldVision RAG Evaluation Report"),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, sanitize(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"model={CLAUDE_MODEL} | embedder={EMBED_MODEL}"),
        new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Input files
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 6, sanitize("Input files"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for f in input_files:
        pdf.multi_cell(0, 5, sanitize(f"  - {f}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- Headline numbers ----------------------------------------------------
    scores = [h["score"] for h in retrieval_hits]
    mean_r = sum(scores) / len(scores) if scores else 0.0
    max_r  = max(scores) if scores else 0.0

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(200, 16, 46)
    pdf.multi_cell(0, 7, sanitize("Headline metrics"),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", size=10)
    metrics = [
        ("Retrieval quality (top-1 cosine)",  f"{max_r:.3f}",  interpret(max_r)),
        ("Retrieval quality (top-5 mean)",    f"{mean_r:.3f}", interpret(mean_r)),
        ("Generation faithfulness (cosine)",  f"{grounding_score:.3f}", interpret(grounding_score)),
    ]
    for label, val, note in metrics:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(80, 6, sanitize(label))
        pdf.set_font("Helvetica", size=10)
        pdf.cell(25, 6, sanitize(val))
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 6, sanitize(note), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 5, sanitize(
        f"Latency: OCR {latency['ocr']:.1f}s | retrieval {latency['retrieval']:.2f}s | "
        f"generation {latency['generation']:.1f}s | total {latency['total']:.1f}s"),
        new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # Scale note
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 5, sanitize(
        "Note on scale: cosine similarity on L2-normalised sentence-transformer "
        "embeddings typically falls in [0.0, 0.8] for English text. Values above "
        "0.5 are strong; values near 0 indicate no semantic overlap. Random pairs "
        "of unrelated English sentences embed to ~0.05-0.15."),
        new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # --- Retrieval table -----------------------------------------------------
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(200, 16, 46)
    pdf.multi_cell(0, 7, sanitize("Retrieval quality -- top-5 Branch Rickey hits"),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(12, 6, "Rank", border=1)
    pdf.cell(22, 6, "Cosine", border=1)
    pdf.cell(0,  6, "Item", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", size=9)
    for i, hit in enumerate(retrieval_hits, 1):
        pdf.cell(12, 6, str(i), border=1)
        pdf.cell(22, 6, f"{hit['score']:.3f}", border=1)
        item = str(hit.get("item", ""))[:80]
        pdf.cell(0, 6, sanitize(item), border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 5, sanitize(
        "Interpretation: each row is one retrieved historical Branch Rickey "
        "report. Higher cosine = more topically similar to the input notes. A "
        "high top-1 with a sharp drop to rank 5 indicates the retriever is "
        "discriminating well; flat scores suggest weak signal in the corpus."),
        new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # Retrieved snippets (truncated)
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 6, sanitize("Top-5 retrieved snippets (300 chars each)"),
                   new_x="LMARGIN", new_y="NEXT")
    for i, hit in enumerate(retrieval_hits, 1):
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(0, 5, sanitize(f"#{i}  cosine={hit['score']:.3f}"),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)
        snippet = str(hit["text"])[:300].replace("\n", " ")
        if len(hit["text"]) > 300:
            snippet += "..."
        pdf.multi_cell(0, 5, sanitize(snippet), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    # --- Grounding section ---------------------------------------------------
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(200, 16, 46)
    pdf.multi_cell(0, 7, sanitize("Generation faithfulness (grounding)"),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5.5, sanitize(
        f"Cosine between the input transcription and the Claude-generated "
        f"report: {grounding_score:.3f} ({interpret(grounding_score)}). "
        f"A higher score indicates the generated report stayed closer to the "
        f"semantics of the original notes -- i.e., it was grounded in the input "
        f"rather than drifting into generic baseball commentary."),
        new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- Appendix: transcription and generated report -----------------------
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(200, 16, 46)
    pdf.multi_cell(0, 7, sanitize("Input transcription (Claude OCR output)"),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(0, 5, sanitize(transcription), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(200, 16, 46)
    pdf.multi_cell(0, 7, sanitize("Generated report (the RAG output)"),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", size=10)
    for line in generated_report.split("\n"):
        clean = sanitize(line)
        if line.startswith("## "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.ln(1)
            pdf.multi_cell(0, 6, clean.replace("## ", ""), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
        else:
            pdf.multi_cell(0, 5.5, clean if clean.strip() else " ",
                           new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*",
                        help="Optional: JPG/PNG/PDF paths. If omitted, a file picker opens.")
    parser.add_argument("--output", default="evaluation_report.pdf",
                        help="Output PDF path (default: evaluation_report.pdf)")
    args = parser.parse_args()

    # If no files were passed on the CLI, open a native file-picker dialog.
    if not args.inputs:
        print("Opening file picker...")
        args.inputs = pick_files_via_dialog()
        if not args.inputs:
            print("No files selected. Exiting.")
            sys.exit(0)
        print(f"Selected {len(args.inputs)} file(s):")
        for p in args.inputs:
            print(f"  - {p}")

    t0 = time.time()

    # --- Load artifacts ------------------------------------------------------
    print("Loading Branch Rickey corpus and FAISS index...")
    if not os.path.exists(CSV_PATH) or not os.path.exists(EMBEDDINGS_PATH):
        print(f"ERROR: missing {CSV_PATH} or {EMBEDDINGS_PATH}. Run from the "
              f"FieldVision root and ensure precompute_embeddings.py has been run.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df = df[df["Transcription"].notna() & (df["Transcription"].str.strip() != "")].reset_index(drop=True)
    embeddings_np = np.load(EMBEDDINGS_PATH).astype(np.float32)
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)

    print(f"Loading embedder ({EMBED_MODEL})...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Connecting to Anthropic...")
    client = anthropic.Anthropic(api_key=load_api_key())

    # --- OCR -----------------------------------------------------------------
    print(f"Reading {len(args.inputs)} input file(s)...")
    images = collect_images(args.inputs)
    if not images:
        print("ERROR: no usable images extracted from inputs.")
        sys.exit(1)

    print(f"Transcribing {len(images)} page(s) via Claude Vision...")
    t_ocr = time.time()
    transcriptions = [transcribe_image(client, img) for _, img in images]
    combined = "\n\n".join(transcriptions)
    t_ocr = time.time() - t_ocr
    print(f"  OCR complete ({t_ocr:.1f}s, {len(combined)} chars)")

    # --- Retrieval -----------------------------------------------------------
    print("Retrieving top-5 historical reports...")
    t_ret = time.time()
    hits = retrieve_context(combined, index, df, embed_model, k=TOP_K)
    t_ret = time.time() - t_ret
    top1 = hits[0]["score"] if hits else 0.0
    mean5 = sum(h["score"] for h in hits) / len(hits) if hits else 0.0
    print(f"  top-1 cosine: {top1:.3f} | top-5 mean: {mean5:.3f}  ({t_ret:.2f}s)")

    # --- Generation ----------------------------------------------------------
    print("Generating RAG report via Claude...")
    t_gen = time.time()
    report = generate_insights(client, combined, hits)
    t_gen = time.time() - t_gen
    print(f"  generation complete ({t_gen:.1f}s, {len(report)} chars)")

    # --- Grounding cosine ----------------------------------------------------
    print("Computing grounding cosine (input vs. generated report)...")
    grounding = cosine(embed_model, combined, report)
    print(f"  grounding cosine: {grounding:.3f}")

    total = time.time() - t0

    # --- PDF -----------------------------------------------------------------
    print(f"Writing {args.output}...")
    build_evaluation_pdf(
        input_files=args.inputs,
        transcription=combined,
        retrieval_hits=hits,
        generated_report=report,
        grounding_score=grounding,
        latency={"ocr": t_ocr, "retrieval": t_ret, "generation": t_gen, "total": total},
        output_path=args.output,
    )
    print(f"Done. Total time: {total:.1f}s")
    print()
    print("SUMMARY")
    print(f"  Retrieval quality  top-1: {top1:.3f}  top-5 mean: {mean5:.3f}")
    print(f"  Generation faithfulness : {grounding:.3f}")
    print(f"  Report written to       : {args.output}")


if __name__ == "__main__":
    main()
