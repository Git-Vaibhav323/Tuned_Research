"""Phase 2 — Train models (delegates to M4 entrypoint)."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("04_train_models.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
