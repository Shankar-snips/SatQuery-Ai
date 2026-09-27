#!/usr/bin/env bash
# Launches the SatQuery-AI backend (FastAPI) and reminds you how to open the frontend.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
echo "Starting SatQuery-AI backend on http://localhost:8000 ..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACK_PID=$!
echo "Backend PID: $BACK_PID"
echo "Open frontend/index.html in your browser (it talks to http://localhost:8000)."
wait $BACK_PID
