"""
DA2 / M8 — Database connectivity demonstration.

Shows: Python → SQLite → SQL → DataFrame / printed records.

Usage:
    python scripts/phase2/08_da2_database_demo.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "database" / "researchpilot.db"

QUERIES = {
    "1_recent_ai_ml_papers": """
        SELECT id, title, publication_year, cited_by_count, oa_category
        FROM papers
        WHERE publication_year >= 2023
        ORDER BY publication_year DESC, cited_by_count DESC
        LIMIT 10;
    """,
    "2_highly_cited_papers": """
        SELECT id, title, publication_year, cited_by_count, citation_per_year, oa_category
        FROM papers
        ORDER BY cited_by_count DESC
        LIMIT 10;
    """,
    "3_papers_by_oa_category": """
        SELECT
            oa_category,
            COUNT(*) AS n_papers,
            ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM papers), 2) AS pct,
            ROUND(AVG(cited_by_count), 1) AS avg_citations,
            ROUND(AVG(abstract_length), 1) AS avg_abstract_length
        FROM papers
        GROUP BY oa_category
        ORDER BY n_papers DESC;
    """,
}


def main() -> None:
    print("=== ResearchPilot DA2 — Database Demo ===")
    print(f"Database : {DB_PATH}")

    if not DB_PATH.exists():
        print("Database not found. Run first:")
        print("  python scripts/phase2/01_load_to_db.py")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        n = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        print(f"Connected: OK | papers rows = {n}")
        print()

        for name, sql in QUERIES.items():
            print("=" * 72)
            print(name)
            print("-" * 72)
            print(sql.strip())
            print("-" * 72)
            df = pd.read_sql_query(sql, conn)
            print(df.to_string(index=False))
            print(f"[{len(df)} row(s)]")
            print()
    finally:
        conn.close()

    print("Status: OK - database demonstration complete.")


if __name__ == "__main__":
    main()
