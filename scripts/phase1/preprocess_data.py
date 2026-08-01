"""Phase 1 — Cleaning and preprocessing entrypoint.

Delegates to preprocess_papers.py for the OpenAlex paper pipeline.
"""

from preprocess_papers import main


if __name__ == "__main__":
    raise SystemExit(main())
