"""
Phase 2 / M1 — Load Phase 1 final dataset into SQLite.

Reads:  data/final/final_dataset.csv  (never modified)
Writes: database/researchpilot.db

Usage (from project root):
    python scripts/phase2/01_load_to_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.db import apply_schemas, connect, find_project_root, load_db_config


BOOL_COLS = ("is_open_access", "has_fulltext", "recent_paper", "has_doi")

PAPERS_COLUMNS = [
    "id",
    "title",
    "abstract",
    "publication_year",
    "cited_by_count",
    "language",
    "type",
    "concepts",
    "keywords",
    "doi",
    "open_access",
    "concepts_clean",
    "keywords_clean",
    "is_open_access",
    "oa_status",
    "oa_url",
    "has_fulltext",
    "paper_age",
    "title_length",
    "abstract_length",
    "keyword_count",
    "concept_count",
    "citation_per_year",
    "citation_log",
    "recent_paper",
    "has_doi",
    "oa_category",
]


def _to_sqlite_bool(series: pd.Series) -> pd.Series:
    """Map truthy/falsy/NA to 1/0/NA for SQLite INTEGER."""

    def map_one(v):
        if pd.isna(v):
            return pd.NA
        if isinstance(v, str):
            low = v.strip().lower()
            if low in {"true", "1", "yes"}:
                return 1
            if low in {"false", "0", "no"}:
                return 0
            return pd.NA
        return int(bool(v))

    return series.map(map_one)


def prepare_papers_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PAPERS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    out = df[PAPERS_COLUMNS].copy()
    for col in BOOL_COLS:
        out[col] = _to_sqlite_bool(out[col])

    # Ensure string IDs; drop exact duplicate OpenAlex ids if any
    out["id"] = out["id"].astype(str)
    before = len(out)
    out = out.drop_duplicates(subset=["id"], keep="first")
    if len(out) < before:
        print(f"Dropped {before - len(out)} duplicate id row(s).")

    return out


def load_papers(conn, df: pd.DataFrame, replace: bool = True) -> int:
    if replace:
        conn.execute("DELETE FROM papers;")
        conn.commit()

    df.to_sql("papers", conn, if_exists="append", index=False)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM papers;").fetchone()[0]
    return int(n)


def write_manifest(conn, source_csv: Path, db_path: Path, n_rows: int, n_cols: int) -> None:
    conn.execute(
        """
        INSERT INTO load_manifest (source_csv, db_path, n_rows_loaded, n_columns, phase1_locked, notes)
        VALUES (?, ?, ?, ?, 1, ?)
        """,
        (
            str(source_csv).replace("\\", "/"),
            str(db_path).replace("\\", "/"),
            n_rows,
            n_cols,
            "M1: Phase 1 final_dataset.csv ingested into SQLite papers table.",
        ),
    )
    conn.commit()


def verify(conn) -> dict:
    checks = {}
    checks["n_papers"] = conn.execute("SELECT COUNT(*) FROM papers;").fetchone()[0]
    checks["n_oa_categories"] = conn.execute(
        "SELECT COUNT(DISTINCT oa_category) FROM papers;"
    ).fetchone()[0]
    checks["year_min"] = conn.execute("SELECT MIN(publication_year) FROM papers;").fetchone()[0]
    checks["year_max"] = conn.execute("SELECT MAX(publication_year) FROM papers;").fetchone()[0]
    checks["tables"] = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
    ]
    return checks


def main() -> None:
    root = find_project_root(PROJECT_ROOT)
    cfg = load_db_config(root)
    db_path = root / cfg["paths"]["database"]
    source_csv = root / cfg["paths"]["source_csv"]

    if not source_csv.exists():
        raise FileNotFoundError(f"Phase 1 CSV not found: {source_csv}")

    print("=== ResearchPilot M1 — Load to SQLite ===")
    print(f"Project root : {root}")
    print(f"Source CSV   : {source_csv}")
    print(f"Database     : {db_path}")

    df_raw = pd.read_csv(source_csv)
    print(f"CSV shape    : {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    df = prepare_papers_frame(df_raw)

    conn = connect(db_path, cfg)
    try:
        applied = apply_schemas(conn, cfg)
        print(f"Schemas      : {', '.join(applied)}")

        n = load_papers(conn, df, replace=bool(cfg.get("load", {}).get("replace_papers", True)))
        write_manifest(conn, source_csv, db_path, n, df.shape[1])

        checks = verify(conn)
        print("--- Verification ---")
        print(f"Rows in papers     : {checks['n_papers']}")
        print(f"Distinct oa_category: {checks['n_oa_categories']}")
        print(f"Year range         : {checks['year_min']}–{checks['year_max']}")
        print(f"Tables             : {', '.join(checks['tables'])}")

        if checks["n_papers"] != 2000:
            print(f"WARNING: expected 2000 papers, found {checks['n_papers']}")
        else:
            print("Status           : OK — 2000 papers loaded; Phase 1 CSV unchanged.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
