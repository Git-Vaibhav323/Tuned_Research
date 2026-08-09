"""Phase 2 — Comparative analysis (delegates to M6 visualizations)."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("06_comparative_visualizations.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
