# ⚾ FieldVision — Baseball Scouting Intelligence

AI-powered baseball analytics platform for Saint Mary's College of California. Transform handwritten scouting notes into structured reports, interpret Trackman pitching data, and chat with an AI scout — all grounded in Branch Rickey's 1,919 historical scouting documents.

**Live:** [fieldvision.onrender.com](https://fieldvision.onrender.com)

---

## Features

- **Handwritten Note OCR** — Upload PDF, TXT, or MD scouting notes. Claude Vision OCR handles scanned and handwritten PDFs that text extractors can't read
- **Branch Rickey RAG** — 1,919 historical scouting transcriptions indexed with TF-IDF (sklearn). Every report is benchmarked against decades of professional scouting wisdom
- **Trackman Analysis** — Upload a Trackman CSV export for post-game pitch analytics. Per-pitcher velocity, spin rate, and pitch mix with neutral AI interpretation
- **AI Chat Interface** — Follow-up questions grounded in your session. All uploads accumulate as context across the session
- **PDF Report Export** — Export any scouting report to a clean, print-ready PDF
- **Session Memory** — Upload multiple players in one session; the AI references all prior uploads when answering questions
- **Players Tab** — Auto-populated talent pool with grade badges, search, and grade filtering

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Native HTML / Tailwind CSS / Vanilla JS |
| Backend | FastAPI (Python 3.11) |
| AI / LLM | Anthropic Claude (Sonnet 4.5 + Opus 4.5 for Vision OCR) |
| RAG | scikit-learn TF-IDF (20K features, cosine similarity) |
| PDF Extraction | pdfplumber → pypdf → Claude Vision OCR (fallback chain) |
| Data | Pandas + NumPy |
| Deployment | Render (Docker) — tracked on `feature/full-stack` branch |

---

## Project Structure

```
FieldVision/
├── index.html                       # Full frontend (single-page app)
├── backend/
│   ├── main.py                      # FastAPI app entry point
│   ├── routes/
│   │   ├── analyze.py               # POST /api/analyze
│   │   ├── chat.py                  # POST /api/chat
│   │   └── trackman.py              # POST /api/trackman
│   └── services/
│       ├── claude.py                # All Claude API calls
│       ├── rag.py                   # TF-IDF retrieval from Branch Rickey CSV
│       └── files.py                 # PDF extraction + Claude Vision OCR fallback
├── static/
│   ├── logo.svg                     # Site logo (white + red FV mark)
│   ├── logo-share-card.png          # Shareable dark card (1200×1200)
│   └── logo-share-white.png         # Shareable white card (1200×1200)
├── data/
│   └── branch-rickey-scouting.csv   # Historical scouting knowledge base (1,919 docs)
├── requirements.txt
└── Dockerfile
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Upload scouting note files, returns structured reports |
| `POST` | `/api/chat` | Continue a scouting conversation with session context |
| `POST` | `/api/trackman` | Upload Trackman CSV, returns stats + AI interpretation |
| `GET/HEAD` | `/api/health` | Health check (used by UptimeRobot keep-alive) |

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/jakeyoung1/FieldVision.git
cd FieldVision
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Anthropic API key

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run the development server

```bash
uvicorn backend.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## Deployment

The app is deployed on **Render** using Docker. Render tracks the `feature/full-stack` branch.

**Branch sync pattern** — after every commit to `main`:
```bash
git checkout feature/full-stack && git merge main && git push origin feature/full-stack && git checkout main
```

**Environment variables required on Render:**
- `ANTHROPIC_API_KEY`

**Keep-alive:** UptimeRobot pings `/api/health` every 5 minutes to prevent Render's free tier from sleeping.

---

## Roadmap

- [ ] Mass upload — batch process multiple players at once
- [ ] Player comparison — side-by-side grade and strengths view
- [ ] Talent pool filtering — filter by grade, position, and metrics
- [ ] Custom domain
