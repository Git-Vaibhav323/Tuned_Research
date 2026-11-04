"""
Phase 2 / M2 — Build ML features, impact tiers, and train/val/test splits.

Reads:  data/final/final_dataset.csv  (Phase 1, never modified)
        database/researchpilot.db     (papers must already exist from M1)
Writes: data/ml/*
        ml_features table in SQLite

Usage (from project root):
    python scripts/phase2/02_build_ml_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.db import connect, find_project_root, load_db_config
from researchpilot.features import build_m2_artifacts


ML_FEATURES_COLS = [
    "paper_id",
    "split",
    "impact_tier",
    "publication_year",
    "paper_age",
    "title_length",
    "abstract_length",
    "keyword_count",
    "concept_count",
    "is_open_access",
    "has_fulltext",
    "recent_paper",
    "has_doi",
    "oa_category",
    "feature_version",
]


def populate_ml_features(conn, base, feature_version: str = "v1") -> int:
    """Replace ml_features with one row per paper (summary store)."""
    conn.execute("DELETE FROM ml_features;")
    conn.commit()

    rows = base.copy()
    rows["paper_id"] = rows["id"]
    rows["feature_version"] = feature_version
    # SQLite INTEGER for bools
    for col in ("is_open_access", "has_fulltext", "recent_paper", "has_doi"):
        rows[col] = rows[col].fillna(0).astype(int)

    export = rows[ML_FEATURES_COLS]
    export.to_sql("ml_features", conn, if_exists="append", index=False)
    conn.commit()
    return int(conn.execute("SELECT COUNT(*) FROM ml_features;").fetchone()[0])


def verify_db(conn) -> dict:
    out = {}
    out["n_ml_features"] = conn.execute("SELECT COUNT(*) FROM ml_features;").fetchone()[0]
    out["splits"] = dict(
        conn.execute(
            "SELECT split, COUNT(*) FROM ml_features GROUP BY split ORDER BY split;"
        ).fetchall()
    )
    out["impact"] = dict(
        conn.execute(
            "SELECT impact_tier, COUNT(*) FROM ml_features GROUP BY impact_tier;"
        ).fetchall()
    )
    out["oa"] = dict(
        conn.execute(
            "SELECT oa_category, COUNT(*) FROM ml_features GROUP BY oa_category;"
        ).fetchall()
    )
    # FK sanity: every ml row exists in papers
    orphan = conn.execute(
        """
        SELECT COUNT(*) FROM ml_features m
        LEFT JOIN papers p ON p.id = m.paper_id
        WHERE p.id IS NULL;
        """
    ).fetchone()[0]
    out["orphan_rows"] = orphan
    return out


def main() -> None:
    root = find_project_root(PROJECT_ROOT)
    db_cfg = load_db_config(root)
    db_path = root / db_cfg["paths"]["database"]

    print("=== ResearchPilot M2 — Build ML Features ===")
    print(f"Project root : {root}")
    print(f"Database     : {db_path}")

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\nRun: python scripts/phase2/01_load_to_db.py"
        )

    conn = connect(db_path, db_cfg)
    try:
        n_papers = conn.execute("SELECT COUNT(*) FROM papers;").fetchone()[0]
        if n_papers != 2000:
            raise RuntimeError(
                f"Expected 2000 papers in DB, found {n_papers}. Re-run M1 load."
            )

        result = build_m2_artifacts(root)
        summary = result["summary"]
        version = result["cfg"]["feature_version"]

        n = populate_ml_features(conn, result["base"], feature_version=version)
        checks = verify_db(conn)

        print("--- Build summary ---")
        print(f"Rows              : {summary['n_rows']}")
        print(f"Splits            : {summary['split_counts']}")
        print(f"OA classes        : {summary['oa_counts']}")
        print(f"Impact tiers      : {summary['impact_counts']}")
        print(f"Impact thresholds : {summary['impact_thresholds']}")
        print(
            f"OA features       : {summary['n_features_oa_full']} full -> "
            f"{summary['n_features_oa_selected']} selected"
        )
        print(f"Impact features   : {summary['n_features_impact']}")
        print(f"Artifacts         : {summary['ml_dir']}")
        print("--- DB verification ---")
        print(f"ml_features rows  : {checks['n_ml_features']}")
        print(f"Splits in DB      : {checks['splits']}")
        print(f"Orphan FK rows    : {checks['orphan_rows']}")
        if checks["n_ml_features"] == 2000 and checks["orphan_rows"] == 0:
            print("Status            : OK — M2 feature store ready for M3/M4.")
        else:
            print("Status            : WARNING — check counts above.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
