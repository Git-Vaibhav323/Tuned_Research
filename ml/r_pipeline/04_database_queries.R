# =============================================================================
# ResearchPilot — Phase 2 DA2
# MILESTONE 3b: Database Queries (R + SQLite)
# =============================================================================
# Demonstrates R → DBI → SQLite → SQL → data.frame workflow
# Runs 7 meaningful SQL queries for faculty demonstration
# Saves query outputs to data/database_r/
# =============================================================================

suppressPackageStartupMessages({
  library(DBI)
  library(RSQLite)
  library(dplyr)
  library(readr)
  library(ggplot2)
})

cat("=== M3b: Database Queries ===\n")
cat("Demonstrating R + DBI + SQLite workflow\n\n")

# ── 0. Paths ──────────────────────────────────────────────────────────────────
root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research")) {
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
}
if (is.null(root)) root <- getwd()

db_path  <- file.path(root, "database", "researchpilot_r.db")
out_dir  <- file.path(root, "data", "database_r")
fig_dir  <- file.path(root, "reports", "figures", "phase2_r")
dir.create(out_dir, showWarnings=FALSE, recursive=TRUE)
dir.create(fig_dir, showWarnings=FALSE, recursive=TRUE)

# ── 1. Connect ────────────────────────────────────────────────────────────────
cat("─────────────────────────────────────────\n")
cat("STEP 1: Connecting to SQLite database\n")
cat("─────────────────────────────────────────\n")
con <- dbConnect(RSQLite::SQLite(), db_path)
cat(sprintf("  [SUCCESS] Connected to: %s\n", db_path))

tables <- dbListTables(con)
cat(sprintf("  Tables: %s\n", paste(tables, collapse=", ")))

for (tbl in tables) {
  n <- tryCatch(
    dbGetQuery(con, sprintf("SELECT COUNT(*) AS n FROM %s", tbl))$n, 
    error=function(e) "?")
  cat(sprintf("    %-20s  %s rows\n", tbl, n))
}
cat("\n")

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 1 — Most recently published papers
# ─────────────────────────────────────────────────────────────────────────────
cat("─────────────────────────────────────────\n")
cat("QUERY 1: Most recently published papers\n")
cat("─────────────────────────────────────────\n")

sql_q1 <- "
  SELECT
    paper_id,
    substr(title, 1, 70) AS title_short,
    publication_year,
    cited_by_count,
    oa_category
  FROM papers
  WHERE publication_year >= 2024
  ORDER BY publication_year DESC, cited_by_count DESC
  LIMIT 10
"
q1_result <- dbGetQuery(con, sql_q1)
cat(sprintf("SQL:\n%s\n", sql_q1))
cat("Result (top 10 most recent papers):\n")
print(q1_result)
write_csv(q1_result, file.path(out_dir, "q1_most_recent_papers.csv"))
cat(sprintf("\n  Saved: q1_most_recent_papers.csv (%d rows)\n\n", nrow(q1_result)))

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 2 — Most highly cited papers
# ─────────────────────────────────────────────────────────────────────────────
cat("─────────────────────────────────────────\n")
cat("QUERY 2: Most highly cited papers\n")
cat("─────────────────────────────────────────\n")

sql_q2 <- "
  SELECT
    paper_id,
    substr(title, 1, 70) AS title_short,
    publication_year,
    cited_by_count,
    oa_category,
    oa_status
  FROM papers
  ORDER BY cited_by_count DESC
  LIMIT 10
"
q2_result <- dbGetQuery(con, sql_q2)
cat(sprintf("SQL:\n%s\n", sql_q2))
cat("Result (top 10 most cited papers):\n")
print(q2_result)
write_csv(q2_result, file.path(out_dir, "q2_most_cited_papers.csv"))
cat(sprintf("\n  Saved: q2_most_cited_papers.csv (%d rows)\n\n", nrow(q2_result)))

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 3 — Papers by OA category
# ─────────────────────────────────────────────────────────────────────────────
cat("─────────────────────────────────────────\n")
cat("QUERY 3: Papers by OA category\n")
cat("─────────────────────────────────────────\n")

sql_q3 <- "
  SELECT
    oa_category,
    COUNT(*)                                    AS paper_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS percentage,
    ROUND(AVG(cited_by_count), 2)               AS avg_citations,
    MAX(cited_by_count)                         AS max_citations
  FROM papers
  GROUP BY oa_category
  ORDER BY paper_count DESC
"
q3_result <- dbGetQuery(con, sql_q3)
cat(sprintf("SQL:\n%s\n", sql_q3))
cat("Result (OA category distribution):\n")
print(q3_result)
write_csv(q3_result, file.path(out_dir, "q3_oa_category_distribution.csv"))
cat(sprintf("\n  Saved: q3_oa_category_distribution.csv\n\n"))

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 4 — Papers per publication year (aggregate count)
# ─────────────────────────────────────────────────────────────────────────────
cat("─────────────────────────────────────────\n")
cat("QUERY 4: Papers per publication year (aggregate)\n")
cat("─────────────────────────────────────────\n")

sql_q4 <- "
  SELECT
    publication_year,
    COUNT(*)                        AS paper_count,
    ROUND(AVG(cited_by_count), 2)   AS avg_citations,
    SUM(is_open_access)             AS open_access_count
  FROM papers
  GROUP BY publication_year
  ORDER BY publication_year
"
q4_result <- dbGetQuery(con, sql_q4)
cat(sprintf("SQL:\n%s\n", sql_q4))
cat("Result (papers per year):\n")
print(q4_result)
write_csv(q4_result, file.path(out_dir, "q4_papers_by_year.csv"))
cat(sprintf("\n  Saved: q4_papers_by_year.csv\n\n"))

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 5 — OA category breakdown by year (cross-tab)
# ─────────────────────────────────────────────────────────────────────────────
cat("─────────────────────────────────────────\n")
cat("QUERY 5: OA category breakdown by year\n")
cat("─────────────────────────────────────────\n")

sql_q5 <- "
  SELECT
    publication_year,
    SUM(CASE WHEN oa_category = 'fully_open'     THEN 1 ELSE 0 END) AS fully_open,
    SUM(CASE WHEN oa_category = 'partially_open' THEN 1 ELSE 0 END) AS partially_open,
    SUM(CASE WHEN oa_category = 'closed'         THEN 1 ELSE 0 END) AS closed,
    COUNT(*) AS total
  FROM papers
  GROUP BY publication_year
  ORDER BY publication_year
"
q5_result <- dbGetQuery(con, sql_q5)
cat(sprintf("SQL:\n%s\n", sql_q5))
cat("Result (OA by year):\n")
print(q5_result)
write_csv(q5_result, file.path(out_dir, "q5_oa_by_year.csv"))
cat(sprintf("\n  Saved: q5_oa_by_year.csv\n\n"))

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 6 — Join papers + ml_features: high-richness papers
# ─────────────────────────────────────────────────────────────────────────────
cat("─────────────────────────────────────────\n")
cat("QUERY 6: High text-richness papers (JOIN query)\n")
cat("─────────────────────────────────────────\n")

sql_q6 <- "
  SELECT
    p.paper_id,
    substr(p.title, 1, 60) AS title_short,
    p.publication_year,
    p.oa_category,
    m.text_richness,
    m.keyword_count,
    m.concept_count,
    m.abstract_word_count
  FROM papers p
  JOIN ml_features m ON p.paper_id = m.paper_id
  WHERE m.text_richness > 0.30
  ORDER BY m.text_richness DESC
  LIMIT 10
"
q6_result <- dbGetQuery(con, sql_q6)
cat(sprintf("SQL:\n%s\n", sql_q6))
cat("Result (top 10 most text-rich papers):\n")
print(q6_result)
write_csv(q6_result, file.path(out_dir, "q6_high_richness_papers.csv"))
cat(sprintf("\n  Saved: q6_high_richness_papers.csv\n\n"))

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 7 — Average features by OA category
# ─────────────────────────────────────────────────────────────────────────────
cat("─────────────────────────────────────────\n")
cat("QUERY 7: Average ML features by OA category\n")
cat("─────────────────────────────────────────\n")

sql_q7 <- "
  SELECT
    p.oa_category,
    COUNT(*)                            AS n_papers,
    ROUND(AVG(m.title_word_count), 2)   AS avg_title_words,
    ROUND(AVG(m.abstract_word_count), 2) AS avg_abstract_words,
    ROUND(AVG(m.keyword_count), 2)      AS avg_keywords,
    ROUND(AVG(m.concept_count), 2)      AS avg_concepts,
    ROUND(AVG(m.text_richness), 4)      AS avg_text_richness,
    ROUND(AVG(m.recency_score), 4)      AS avg_recency
  FROM papers p
  JOIN ml_features m ON p.paper_id = m.paper_id
  GROUP BY p.oa_category
  ORDER BY n_papers DESC
"
q7_result <- dbGetQuery(con, sql_q7)
cat(sprintf("SQL:\n%s\n", sql_q7))
cat("Result (avg features per OA class — key for ML insight):\n")
print(q7_result)
write_csv(q7_result, file.path(out_dir, "q7_avg_features_by_oa.csv"))
cat(sprintf("\n  Saved: q7_avg_features_by_oa.csv\n\n"))

# ── Quick visualisation from query results ─────────────────────────────────────
cat("Generating query visualisation...\n")

# OA by year stacked bar
q5_long <- q5_result %>%
  tidyr::pivot_longer(cols = c(fully_open, partially_open, closed),
                      names_to = "oa_category", values_to = "count") %>%
  mutate(oa_category = factor(oa_category,
                              levels = c("closed","partially_open","fully_open")))

p_oa_yr <- ggplot(q5_long, aes(x = factor(publication_year), y = count,
                               fill = oa_category)) +
  geom_col(position = "stack", alpha = 0.85) +
  scale_fill_manual(
    values = c("fully_open"="#1a9850","partially_open"="#fee08b","closed"="#d73027"),
    labels = c("Fully Open","Partially Open","Closed"),
    name   = "OA Category") +
  labs(title   = "Research Papers by Year and OA Category",
       subtitle = "ResearchPilot corpus — 2000 AI/ML papers (2022–2025)",
       x = "Publication Year", y = "Number of Papers",
       caption = "Source: OpenAlex API | ResearchPilot DA2 SQLite DB") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "top")

ggsave(file.path(fig_dir, "db_papers_by_year_oa.png"), p_oa_yr,
       width = 7, height = 5, dpi = 150)
cat(sprintf("  Saved: db_papers_by_year_oa.png\n"))

# ── Disconnect ────────────────────────────────────────────────────────────────
dbDisconnect(con)
cat("\n[DONE] Database connection closed.\n")

cat("\n=== M3b Database Queries COMPLETE ===\n")
cat(sprintf("  7 SQL queries executed\n"))
cat(sprintf("  Query CSVs saved to: %s\n", out_dir))
cat(sprintf("  Visualisation: db_papers_by_year_oa.png\n"))
cat("\nSUMMARY — Key findings from SQL:\n")
cat(sprintf("  Q3 OA distribution:\n"))
for (i in seq_len(nrow(q3_result))) {
  cat(sprintf("    %-20s: %d papers (%.1f%%), avg citations: %.0f\n",
              q3_result$oa_category[i],
              q3_result$paper_count[i],
              q3_result$percentage[i],
              q3_result$avg_citations[i]))
}
