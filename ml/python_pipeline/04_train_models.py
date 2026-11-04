"""
Phase 2 / M4 — Train 10–15 ML algorithms on OA category classification.

Reads M3 matrices (data/ml/m3/oa_*.csv)
Writes: evaluation/reports/m4/, models/checkpoints/m4/, DB experiment_runs

Usage (from project root):
    python scripts/phase2/04_train_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.db import find_project_root
from researchpilot.models import run_m4


def main() -> None:
    root = find_project_root(PROJECT_ROOT)
    print("=== ResearchPilot M4 — Train Models (OA category) ===")
    print(f"Project root : {root}")

    required = [
        root / "data" / "ml" / "m3" / "oa_train.csv",
        root / "data" / "ml" / "m3" / "oa_train_y.csv",
        root / "data" / "ml" / "m3" / "oa_val.csv",
        root / "data" / "ml" / "m3" / "oa_val_y.csv",
        root / "data" / "ml" / "m3" / "oa_test.csv",
        root / "data" / "ml" / "m3" / "oa_test_y.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "M3 artifacts missing. Run:\n"
            "  python scripts/phase2/03_feature_selection.py\n"
            "Missing:\n  " + "\n  ".join(missing)
        )

    summary = run_m4(root)

    print("--- Training summary ---")
    print(f"Models OK     : {summary['n_models_ok']} / {summary['n_models_requested']}")
    print(f"Features      : {summary['n_features']}")
    print(f"Train/Val/Test: {summary['n_train']} / {summary['n_val']} / {summary['n_test']}")
    print(f"Champion      : {summary['champion']}")
    print(f"Test macro-F1 : {summary['champion_test_f1_macro']}")
    print(f"Test accuracy : {summary['champion_test_accuracy']}")
    print(f"Leaderboard   : {summary['leaderboard_csv']}")
    if summary.get("comparison_plot"):
        print(f"Comparison plot: {summary['comparison_plot']}")
    print(f"DB runs logged: {summary.get('db_logged_runs')}")
    print("Status         : OK - M4 training complete.")


if __name__ == "__main__":
    main()
