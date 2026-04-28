#!/bin/bash
# FieldVision — Start the FastAPI backend locally
# Usage: ./start.sh
# Requires: .env file with ANTHROPIC_API_KEY=sk-ant-...

cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "⚠️  No .env file found. Copy .env.example and add your key:"
  echo "    cp .env.example .env && nano .env"
  exit 1
fi

export PYTHONPATH="$(pwd)"
/Users/jakel/Library/Python/3.9/bin/uvicorn backend.main:app --reload --port 8000
