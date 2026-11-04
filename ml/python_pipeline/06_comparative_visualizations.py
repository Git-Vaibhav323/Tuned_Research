"""
Phase 2 / M6 — Comparative visualizations (ROC, PR, CM, feature importance).

Reads M4/M5 prediction CSVs and checkpoints.
Writes: reports/figures/m6/, evaluation/reports/m6/

Usage (from project root):
    python scripts/phase2/06_comparative_visualizations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.db import find_project_root
from researchpilot.evaluation import run_m6


def main() -> None:
    root = find_project_root(PROJECT_ROOT)
    print("=== ResearchPilot M6 — Comparative Visualizations ===")
    print(f"Project root : {root}")

    summary = run_m6(root)

    print("--- Summary ---")
    print(f"Models plotted : {summary['n_models_plotted']}")
    print(f"Models         : {', '.join(summary['models'])}")
    print(f"Best F1 model  : {summary['best_f1_model']} ({summary['best_f1_macro']:.4f})")
    print(f"Best ROC model : {summary['best_roc_model']}")
    print(f"Metrics table  : {summary['metrics_csv']}")
    figs = summary["figures"]
    print(f"CM figures     : {len(figs['confusion_matrices'])}")
    print(f"ROC figures    : {len(figs['roc'])}")
    print(f"PR figures     : {len(figs['pr'])}")
    print(f"Importance figs: {len(figs['feature_importance'])}")
    print(f"Figures dir    : {root / 'reports' / 'figures' / 'm6'}")
    print("Status          : OK - M6 comparative visualizations complete.")


if __name__ == "__main__":
    main()
