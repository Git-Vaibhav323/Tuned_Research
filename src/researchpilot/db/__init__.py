"""Database connectivity helpers for ResearchPilot Phase 2."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/phase2/db.yaml")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward until data/final/final_dataset.csv is found."""
    path = (start or Path.cwd()).resolve()
    for _ in range(8):
        if (path / "data" / "final" / "final_dataset.csv").exists():
            return path
        if path.parent == path:
            break
        path = path.parent
    raise FileNotFoundError(
        "Project root not found. Run from E:/Tuned_Research or set cwd correctly."
    )


def load_db_config(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or find_project_root()
    cfg_path = root / DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(root)
    return cfg


def get_db_path(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_db_config()
    root = Path(cfg["_project_root"])
    return root / cfg["paths"]["database"]


def connect(db_path: Path | None = None, cfg: dict[str, Any] | None = None) -> sqlite3.Connection:
    """Open SQLite connection with foreign keys enabled."""
    path = db_path or get_db_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def apply_schemas(conn: sqlite3.Connection, cfg: dict[str, Any] | None = None) -> list[str]:
    """Execute all schema SQL files listed in config. Returns applied filenames."""
    cfg = cfg or load_db_config()
    root = Path(cfg["_project_root"])
    schemas_dir = root / cfg["paths"]["schemas_dir"]
    applied: list[str] = []
    for name in cfg["schema_files"]:
        sql_path = schemas_dir / name
        sql = sql_path.read_text(encoding="utf-8")
        conn.executescript(sql)
        applied.append(name)
    conn.commit()
    return applied
