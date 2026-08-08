"""
Phase 2 / M5 — Hyperparameter tuning for top OA-category models.

Reads M3 matrices + M4 baseline summary
Writes: evaluation/reports/m5/, models/checkpoints/m5/

Usage (from project root):
    python scripts/phase2/05_tune_hyperparameters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.db import find_project_root
from researchpilot.models.tuning import run_m5


def main() -> None:
    root = find_project_root(PROJECT_ROOT)
    print("=== ResearchPilot M5 — Hyperparameter Tuning ===")
    print(f"Project root : {root}")

    required = [
        root / "data" / "ml" / "m3" / "oa_train.csv",
        root / "data" / "ml" / "m3" / "oa_val.csv",
        root / "data" / "ml" / "m3" / "oa_test.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "M3 artifacts missing. Run scripts/phase2/03_feature_selection.py first.\n"
            + "\n".join(missing)
        )

    summary = run_m5(root)

    print("--- Tuning summary ---")
    print(f"Models tuned OK     : {summary['n_ok']}")
    print(f"Champion (by val F1): {summary['champion_by_val_f1']}")
    print(f"Best params         : {summary['champion_best_params']}")
    print(f"CV best score       : {summary['champion_cv_best_score']:.4f}")
    print(f"Val macro-F1        : {summary['champion_val_f1_macro']:.4f}")
    print(f"Test macro-F1       : {summary['champion_test_f1_macro']:.4f}")
    print(f"Test accuracy       : {summary['champion_test_accuracy']:.4f}")
    print(f"Test ROC-AUC        : {summary['champion_test_roc_auc_ovr']}")
    if summary.get("m4_baseline_test_f1_macro") is not None:
        print(f"M4 champion F1      : {summary['m4_baseline_test_f1_macro']:.4f}")
        delta = summary.get("improvement_test_f1_vs_m4_champion")
        if delta is not None:
            print(f"Delta vs M4 champ   : {delta:+.4f}")
    print(f"Leaderboard         : {summary['leaderboard_csv']}")
    if summary.get("comparison_plot"):
        print(f"Comparison plot     : {summary['comparison_plot']}")
    print(f"DB runs logged      : {summary.get('db_logged_runs')}")
    print("Status               : OK - M5 tuning complete.")


if __name__ == "__main__":
    main()
