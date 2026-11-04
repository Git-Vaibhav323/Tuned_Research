"""
run_app.py — ResearchPilot launcher.

Pre-loads the embedding model into the HuggingFace cache,
then starts Streamlit. Because the model is already cached on disk,
Streamlit's first script run loads it in < 1 s instead of 6-18 s,
which prevents the WebSocket timeout / Connection error.

Usage:
    py -3.12 run_app.py
    py -3.12 run_app.py --port 8502
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def preload():
    print("ResearchPilot — pre-loading embedding model...")
    t0 = time.perf_counter()
    try:
        from rag.embeddings import get_model
        get_model()
        print(f"  Embedding model ready ({(time.perf_counter()-t0)*1000:.0f} ms)")
    except Exception as e:
        print(f"  Warning: could not pre-load model: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    preload()

    print(f"\nStarting Streamlit on http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "streamlit_app" / "app.py"),
        "--server.port",       str(args.port),
        "--server.headless",   "false",
        "--browser.gatherUsageStats", "false",
    ], cwd=str(ROOT))

if __name__ == "__main__":
    main()
