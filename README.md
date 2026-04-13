# ⚾ FieldVision — Baseball Scouting Intelligence

FieldVision is an AI-powered baseball scouting platform built for Saint Mary's College of California. It transforms handwritten scouting notes into structured, actionable intelligence by combining Claude AI with a historical scouting knowledge base built from the Branch Rickey Papers.

---

## Features

- **OCR Transcription** — Upload photos or PDFs of handwritten scouting notes; Claude Vision reads and digitizes them accurately, preserving abbreviations, shorthand, and player names
- **AI-Powered Analysis** — Generates a structured scouting report with a summary, concrete recommendations, and a historical parallel drawn from Branch Rickey's scouting philosophy
- **RAG Knowledge Base** — Semantic search against the Branch Rickey scouting archive using FAISS and local sentence embeddings for historically grounded context
- **Ask the Scout** — Follow-up chat interface with full context from the notes and analysis
- **PDF Export** — Download the digitized notes and full analysis report as clean, formatted PDFs

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| AI / LLM | Anthropic Claude (claude-haiku-4-5) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (CPU) |
| PDF Generation | fpdf2 |
| Image Processing | Pillow, pdf2image |

---

## Project Structure

```
FieldVision/
├── app.py                      # Main Streamlit application
├── precompute_embeddings.py    # One-time script to build the FAISS index
├── requirements.txt
├── packages.txt                # System-level dependencies (poppler)
├── data/
│   ├── branch-rickey-scouting.csv   # Historical scouting knowledge base
│   └── embeddings.npy               # Pre-computed embeddings (generated locally)
└── .streamlit/
    └── secrets.toml                 # API keys (not committed)
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/jakelyoung13-hash/FieldVision.git
cd FieldVision
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PDF support requires `poppler`. Install via Homebrew on macOS:
> ```bash
> brew install poppler
> ```

### 4. Add your Anthropic API key

Create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

### 5. Precompute embeddings

Run this once to build the FAISS index from the Branch Rickey knowledge base:

```bash
python precompute_embeddings.py
```

This generates `data/embeddings.npy` (~80 MB model download on first run).

### 6. Launch the app

```bash
streamlit run app.py
```

---

## Usage

1. Upload one or more images or PDFs of handwritten scouting notes
2. Optionally add context (e.g. *"High school pitching eval"*) and a focus area (e.g. *"Arm strength and projectability"*)
3. Click **Transcribe & Analyze**
4. Review the transcription, scouting report, and historical context
5. Use the **Ask the Scout** chat to dig deeper
6. Download the digitized notes or full analysis as a PDF

---

## Deploying to Streamlit Cloud

1. Push the repo to GitHub (ensure `data/embeddings.npy` is committed)
2. Connect the repo at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Add `ANTHROPIC_API_KEY` under **Settings → Secrets**
4. Deploy

---

## Roadmap

- Trackman data integration — overlay objective pitch/batted ball metrics alongside qualitative scouting notes
- Mass upload and batch processing
- Player comparison and talent pool filtering
- Expanded historical knowledge base

---

## Built By

Saint Mary's College of California Baseball  
Built with [Claude](https://anthropic.com) and [Streamlit](https://streamlit.io)
