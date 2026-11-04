-- ResearchPilot Phase 2 — load provenance / schema metadata

CREATE TABLE IF NOT EXISTS load_manifest (
    load_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source_csv          TEXT NOT NULL,
    db_path             TEXT NOT NULL,
    n_rows_loaded       INTEGER NOT NULL,
    n_columns           INTEGER NOT NULL,
    phase1_locked       INTEGER NOT NULL DEFAULT 1,
    notes               TEXT,
    loaded_at           TEXT NOT NULL DEFAULT (datetime('now'))
);
