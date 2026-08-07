"""
Phase 2 / M3 — Feature selection comparison + train-ready preprocessing.

Reads M2 matrices from data/ml/
Writes: data/ml/m3/*  and reports/figures/m3_feature_selection_agreement.png

Usage (from project root):
    python scripts/phase2/03_feature_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.db import find_project_root
from researchpilot.features.m3_selection import run_m3


def main() -> None:
    root = find_project_root(PROJECT_ROOT)
    print("=== ResearchPilot M3 — Feature Selection ===")
    print(f"Project root : {root}")

    required = [
        root / "data" / "ml" / "X_oa_category_full.csv",
        root / "data" / "ml" / "y_oa_category.csv",
        root / "data" / "ml" / "X_impact_tier_full.csv",
        root / "data" / "ml" / "y_impact_tier.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "M2 artifacts missing. Run first:\n"
            "  python scripts/phase2/02_build_ml_features.py\n"
            "Missing:\n  " + "\n  ".join(missing)
        )

    summary = run_m3(root)

    print("--- Selection summary ---")
    print(f"OA input features     : {summary['n_features_oa_input']}")
    print(f"After variance filter : {summary['n_features_after_variance']}")
    print(f"Final selected (k)    : {summary['n_final_features']}")
    print(f"OA label classes      : {summary['oa_classes']}")
    print(f"Impact label classes  : {summary['impact_classes']}")
    print(f"Scaled structural OA  : {summary['scale_cols_oa']}")
    print("--- Method agreement (overlap with final) ---")
    for row in summary["method_agreement"]:
        print(
            f"  {row['method']:16s}  {row['overlap_with_final']}/{summary['n_final_features']}"
            f"  ({row['overlap_pct']}%)"
        )
    print(f"Artifacts             : {summary['m3_dir']}")
    if summary.get("agreement_plot"):
        print(f"Agreement plot        : {summary['agreement_plot']}")
    print("Status                : OK - M3 ready for M4 model training.")
    print("M4 entrypoint         : data/ml/m3/X_oa_final.csv + y_oa_final.csv")
    print("Preprocessor          : data/ml/m3/preprocessor_bundle.joblib")


if __name__ == "__main__":
    main()
