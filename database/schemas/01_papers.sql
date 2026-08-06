-- ResearchPilot Phase 2 — papers table
-- Source of truth: data/final/final_dataset.csv (Phase 1, read-only)
-- Engine: SQLite

CREATE TABLE IF NOT EXISTS papers (
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    abstract            TEXT NOT NULL,
    publication_year    INTEGER NOT NULL,
    cited_by_count      INTEGER NOT NULL,
    language            TEXT NOT NULL,
    type                TEXT NOT NULL,
    concepts            TEXT,
    keywords            TEXT,
    doi                 TEXT,
    open_access         TEXT,
    concepts_clean      TEXT,
    keywords_clean      TEXT,
    is_open_access      INTEGER NOT NULL CHECK (is_open_access IN (0, 1)),
    oa_status           TEXT NOT NULL,
    oa_url              TEXT,
    has_fulltext        INTEGER CHECK (has_fulltext IN (0, 1)),
    paper_age           INTEGER NOT NULL,
    title_length        INTEGER NOT NULL,
    abstract_length     INTEGER NOT NULL,
    keyword_count       INTEGER NOT NULL,
    concept_count       INTEGER NOT NULL,
    citation_per_year   REAL NOT NULL,
    citation_log        REAL NOT NULL,
    recent_paper        INTEGER NOT NULL CHECK (recent_paper IN (0, 1)),
    has_doi             INTEGER NOT NULL CHECK (has_doi IN (0, 1)),
    oa_category         TEXT NOT NULL
        CHECK (oa_category IN ('fully_open', 'partially_open', 'closed')),
    loaded_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_papers_year ON papers (publication_year);
CREATE INDEX IF NOT EXISTS idx_papers_oa_category ON papers (oa_category);
CREATE INDEX IF NOT EXISTS idx_papers_oa_status ON papers (oa_status);
CREATE INDEX IF NOT EXISTS idx_papers_cited ON papers (cited_by_count);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers (doi);
