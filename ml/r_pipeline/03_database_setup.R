# =============================================================================
# ResearchPilot — Phase 2 DA2
# MILESTONE 3: Database Setup (R + SQLite)
# =============================================================================
# Input : data/final/final_dataset.csv
#         data/ml_r/engineered_features.csv
#         data/ml_r/selected_features_oa.csv
# Output: database/researchpilot_r.db   (SQLite database)
# =============================================================================
# Schema:
#   papers        — core paper metadata (2000 rows)
#   ml_features   — engineered ML features (2000 rows)
#   oa_features   — selected OA-task features (2000 rows)
#   model_results — placeholder for ML results written by 07_model_evaluation.R
# =============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(RSQLite)
  library(dplyr)
  library(readr)
})

cat("=== M3: SQLite Database Setup ===\n\n")

# ── 0. Paths ──────────────────────────────────────────────────────────────────
root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research")) {
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
}
if (is.null(root)) root <- getwd()
cat(sprintf("Project root : %s\n", root))

db_path      <- file.path(root, "database", "researchpilot_r.db")
final_csv    <- file.path(root, "data", "final", "final_dataset.csv")
eng_csv      <- file.path(root, "data", "ml_r", "engineered_features.csv")
oa_feat_csv  <- file.path(root, "data", "ml_r", "selected_features_oa.csv")

dir.create(dirname(db_path), showWarnings = FALSE, recursive = TRUE)

# Remove existing DB so we start clean
if (file.exists(db_path)) {
  file.remove(db_path)
  cat("Removed existing database.\n")
}

# ── 1. Load datasets ──────────────────────────────────────────────────────────
cat("\nLoading data...\n")
df_final    <- read_csv(final_csv,   show_col_types = FALSE)
df_eng      <- read_csv(eng_csv,     show_col_types = FALSE)
df_oa_feat  <- read_csv(oa_feat_csv, show_col_types = FALSE)

cat(sprintf("  final_dataset      : %d rows × %d cols\n", nrow(df_final), ncol(df_final)))
cat(sprintf("  engineered_features: %d rows × %d cols\n", nrow(df_eng),   ncol(df_eng)))
cat(sprintf("  selected_oa_feat   : %d rows × %d cols\n", nrow(df_oa_feat),ncol(df_oa_feat)))

# ── 2. Connect to SQLite ──────────────────────────────────────────────────────
cat(sprintf("\nConnecting to SQLite: %s\n", db_path))
con <- dbConnect(RSQLite::SQLite(), db_path)
cat("  [SUCCESS] Database connection established.\n\n")

# ── 3. TABLE 1: papers ────────────────────────────────────────────────────────
cat("Creating table: papers\n")

papers_tbl <- df_final %>%
  transmute(
    paper_id         = row_number(),
    openalex_id      = id,
    title            = title,
    abstract         = substr(abstract, 1, 2000),   # truncate for DB
    publication_year = publication_year,
    cited_by_count   = cited_by_count,
    language         = language,
    paper_type       = type,
    has_doi          = as.integer(has_doi),
    doi              = ifelse(is.na(doi), "", doi),
    is_open_access   = as.integer(is_open_access),
    oa_status        = oa_status,
    oa_category      = oa_category,
    paper_age        = paper_age,
    keyword_count    = keyword_count,
    concept_count    = concept_count
  )

dbWriteTable(con, "papers", papers_tbl, overwrite = TRUE, row.names = FALSE)
n_papers <- dbGetQuery(con, "SELECT COUNT(*) AS n FROM papers")$n
cat(sprintf("  Rows loaded: %d\n", n_papers))

# ── 4. TABLE 2: ml_features ───────────────────────────────────────────────────
cat("Creating table: ml_features\n")

ml_feat_tbl <- df_eng %>%
  transmute(
    paper_id                = row_number(),
    title_length            = title_length,
    abstract_length         = abstract_length,
    title_word_count        = title_word_count,
    abstract_word_count     = abstract_word_count,
    title_to_abstract_ratio = title_to_abstract_ratio,
    keyword_count           = keyword_count,
    concept_count           = concept_count,
    text_richness           = text_richness,
    recency_score           = recency_score,
    text_length_category    = text_length_category,
    abstract_keyword_overlap = abstract_keyword_overlap,
    recent_paper            = recent_paper,
    citation_per_year       = citation_per_year,
    citation_log            = citation_log,
    oa_category             = oa_category
  )

dbWriteTable(con, "ml_features", ml_feat_tbl, overwrite = TRUE, row.names = FALSE)
n_mlf <- dbGetQuery(con, "SELECT COUNT(*) AS n FROM ml_features")$n
cat(sprintf("  Rows loaded: %d\n", n_mlf))

# ── 5. TABLE 3: oa_features ───────────────────────────────────────────────────
cat("Creating table: oa_features\n")

oa_tbl <- df_oa_feat %>%
  mutate(paper_id = row_number()) %>%
  select(paper_id, everything())

dbWriteTable(con, "oa_features", oa_tbl, overwrite = TRUE, row.names = FALSE)
n_oa <- dbGetQuery(con, "SELECT COUNT(*) AS n FROM oa_features")$n
cat(sprintf("  Rows loaded: %d\n", n_oa))

# ── 6. TABLE 4: model_results ─────────────────────────────────────────────────
cat("Creating table: model_results (schema only)\n")

dbExecute(con, "
  CREATE TABLE IF NOT EXISTS model_results (
    result_id     INTEGER PRIMARY KEY,
    model_name    TEXT NOT NULL,
    task          TEXT NOT NULL,
    split         TEXT NOT NULL,
    accuracy      REAL,
    precision_macro REAL,
    recall_macro  REAL,
    f1_macro      REAL,
    roc_auc       REAL,
    run_date      TEXT,
    notes         TEXT
  )
")
cat("  Schema created (will be populated by 07_model_evaluation.R)\n")

# ── 7. Verify schema ──────────────────────────────────────────────────────────
cat("\n=== Database Schema Verification ===\n")
tables <- dbListTables(con)
cat(sprintf("Tables in database: %s\n\n", paste(tables, collapse=", ")))

for (tbl in tables) {
  cols  <- dbListFields(con, tbl)
  count_q <- tryCatch(
    dbGetQuery(con, sprintf("SELECT COUNT(*) AS n FROM %s", tbl))$n,
    error = function(e) "N/A"
  )
  cat(sprintf("  %-20s  %d rows  |  %d columns: %s\n",
              tbl, count_q, length(cols), paste(cols, collapse=", ")))
}

# ── 8. Create indexes for query performance ───────────────────────────────────
cat("\nCreating indexes...\n")
dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_papers_year     ON papers(publication_year)")
dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_papers_oa_cat   ON papers(oa_category)")
dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_papers_cited    ON papers(cited_by_count)")
dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_mlf_paper_id    ON ml_features(paper_id)")
cat("  Indexes created.\n")

# ── 9. Disconnect ─────────────────────────────────────────────────────────────
dbDisconnect(con)
cat(sprintf("\n[DONE] Database saved: %s\n", db_path))
cat(sprintf("  File size: %.1f KB\n", file.info(db_path)$size / 1024))

cat("\n=== M3 Database Setup COMPLETE ===\n")
cat(sprintf("  Tables: %s\n", paste(tables, collapse=", ")))
cat(sprintf("  papers.rows     : %d\n", n_papers))
cat(sprintf("  ml_features.rows: %d\n", n_mlf))
cat(sprintf("  oa_features.rows: %d\n", n_oa))
