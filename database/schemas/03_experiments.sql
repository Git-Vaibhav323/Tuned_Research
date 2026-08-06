-- ResearchPilot Phase 2 — experiment tracking (populated in M4+)

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id              TEXT PRIMARY KEY,
    track               TEXT NOT NULL
        CHECK (track IN ('oa_category', 'impact_tier')),
    model_name          TEXT NOT NULL,
    params_json         TEXT,
    metrics_json        TEXT,
    cv_score            REAL,
    notes               TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_experiment_track ON experiment_runs (track);
CREATE INDEX IF NOT EXISTS idx_experiment_model ON experiment_runs (model_name);

CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    paper_id            TEXT NOT NULL,
    y_true              TEXT,
    y_pred              TEXT,
    y_proba_json        TEXT,
    FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id),
    FOREIGN KEY (paper_id) REFERENCES papers (id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions (run_id);

CREATE TABLE IF NOT EXISTS feature_importance (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    feature_name        TEXT NOT NULL,
    importance          REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id)
);

CREATE INDEX IF NOT EXISTS idx_fi_run ON feature_importance (run_id);
