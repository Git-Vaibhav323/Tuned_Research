"""Run documented database/queries/*.sql against researchpilot.db and print results."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    db_path = ROOT / "database" / "researchpilot.db"
    qdir = ROOT / "database" / "queries"
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run: python scripts/phase2/01_load_to_db.py")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        for path in sorted(qdir.glob("*.sql")):
            print("=" * 60)
            print(path.name)
            print("-" * 60)
            sql = path.read_text(encoding="utf-8")
            cur = conn.execute(sql)
            rows = cur.fetchall()
            if not cur.description:
                print("(no result set)")
                continue
            cols = [d[0] for d in cur.description]
            print(" | ".join(cols))
            for row in rows:
                print(" | ".join(str(row[c]) for c in cols))
            print(f"[{len(rows)} row(s)]")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
