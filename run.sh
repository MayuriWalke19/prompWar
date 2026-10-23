#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  AttentionAI — startup script
#  Usage: ./run.sh
# ─────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
MODEL_FILE="$SCRIPT_DIR/face_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# ── Step 1: Create venv if missing ───────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "🔧 Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ── Step 2: Install dependencies ─────────────────────────────
echo "📦 Checking dependencies..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# ── Step 3: Download model if missing ────────────────────────
if [ ! -f "$MODEL_FILE" ]; then
  echo "⬇️  Downloading FaceLandmarker model (~3.6 MB)..."
  curl -sL "$MODEL_URL" -o "$MODEL_FILE"
  echo "✅ Model downloaded."
fi

# ── Step 4: Free port 5050 if already in use ─────────────────
if lsof -ti :5050 &>/dev/null; then
  echo "🔄 Stopping existing process on port 5050..."
  lsof -ti :5050 | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# ── Step 5: Launch ───────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  🧠  AttentionAI starting at http://localhost:5050"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  ⚠️  macOS: Grant Camera access if prompted."
echo "  Press Ctrl+C to stop."
echo ""

cd "$SCRIPT_DIR"
OPENCV_AVFOUNDATION_SKIP_AUTH=1 python3 main.py
