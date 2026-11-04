"""
Phase 2 / M7 — Impact-tier classification + light topic clustering.

Reads M3 impact matrices (no citation leakage).
Writes: evaluation/reports/m7/, models/checkpoints/m7/, reports/figures/m7/

Usage:
    python scripts/phase2/07_impact_and_clustering.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force CPU before XGBoost imports (Windows CUDA reliability).
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.db import find_project_root
from researchpilot.models.m7_secondary import run_m7


def main() -> None:
    root = find_project_root(PROJECT_ROOT)
    print("=== ResearchPilot M7 — Impact Tier + Clustering ===")
    print(f"Project root : {root}")

    required = [
        root / "data" / "ml" / "m3" / "X_impact_scaled.csv",
        root / "data" / "ml" / "m3" / "y_impact_final.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "M3 impact artifacts missing. Run scripts/phase2/03_feature_selection.py\n"
            + "\n".join(missing)
        )

    summary = run_m7(root)
    imp = summary["impact"]
    cl = summary["clustering"]

    print("--- Impact-tier ---")
    print(f"Features       : {imp['n_features']}")
    print(f"Train/Val/Test : {imp['n_train']} / {imp['n_val']} / {imp['n_test']}")
    print(f"Models attempted: {imp.get('n_models_attempted')}")
    print(f"Models OK      : {imp['n_models_ok']}")
    print(f"Models failed  : {imp.get('n_models_failed')}")
    print(f"XGBoost status : {imp.get('xgboost_status')}")
    if imp.get("xgboost_error"):
        print(f"XGBoost error  : {imp['xgboost_error']}")
    print(f"Champion (val) : {imp['champion_by_val_f1']}")
    print(f"Val macro-F1   : {imp['champion_val_f1_macro']}")
    print(f"Test macro-F1  : {imp['champion_test_f1_macro']}")
    print(f"Test accuracy  : {imp['champion_test_accuracy']}")
    print(f"Test ROC-AUC   : {imp['champion_test_roc_auc_ovr']}")
    print(f"Leakage cols   : {imp.get('leakage_columns_found')}")
    print("--- Clustering ---")
    if cl.get("enabled"):
        print(f"Best k         : {cl['best_k']}")
        print(f"Silhouette     : {cl['silhouette_train']}")
        print(f"Cluster plot   : {cl['cluster_png']}")
        print(f"Clusters interp: {len(cl.get('interpretation') or [])} summaries")
    print(f"Figures dir    : {root / 'reports' / 'figures' / 'm7'}")
    print(f"Reports dir    : {root / 'evaluation' / 'reports' / 'm7'}")
    print(f"DB runs logged : {summary.get('db_logged_runs')}")
    print("Status          : OK - M7 complete.")


if __name__ == "__main__":
    main()
