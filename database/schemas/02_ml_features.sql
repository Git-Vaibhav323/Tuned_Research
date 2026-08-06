-- ResearchPilot Phase 2 — ML feature store (populated in M2+)
-- One row per paper used for supervised modeling

CREATE TABLE IF NOT EXISTS ml_features (
    paper_id            TEXT PRIMARY KEY,
    split               TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    impact_tier         TEXT CHECK (impact_tier IN ('low', 'medium', 'high')),
    publication_year    INTEGER,
    paper_age           INTEGER,
    title_length        INTEGER,
    abstract_length     INTEGER,
    keyword_count       INTEGER,
    concept_count       INTEGER,
    is_open_access      INTEGER,
    has_fulltext        INTEGER,
    recent_paper        INTEGER,
    has_doi             INTEGER,
    oa_category         TEXT,
    feature_version     TEXT NOT NULL DEFAULT 'v1',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (paper_id) REFERENCES papers (id)
);

CREATE INDEX IF NOT EXISTS idx_ml_features_split ON ml_features (split);
CREATE INDEX IF NOT EXISTS idx_ml_features_impact ON ml_features (impact_tier);
