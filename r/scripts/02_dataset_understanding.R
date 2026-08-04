# =============================================================================
# ResearchPilot — Faculty Demo: Dataset Understanding & Feature Description
# Rubric mapping : Dataset understanding and feature description (2 marks)
# Input          : data/final/final_dataset.csv
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(skimr)
  library(janitor)
})

find_project_root <- function() {
  path <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  for (i in seq_len(6)) {
    if (file.exists(file.path(path, "data", "final", "final_dataset.csv"))) {
      return(path)
    }
    parent <- dirname(path)
    if (identical(parent, path)) break
    path <- parent
  }
  stop("Project root not found. setwd('E:/Tuned_Research') first.")
}

project_root <- find_project_root()
final_csv <- file.path(project_root, "data", "final", "final_dataset.csv")
out_dir <- file.path(project_root, "reports", "tables")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("=== 1) Load final dataset ===")
papers <- read_csv(final_csv, show_col_types = FALSE)
cat("Rows:", nrow(papers), " | Columns:", ncol(papers), "\n")

message("\n=== 2) Structure overview (for faculty) ===")
glimpse(papers)
skim(papers)

message("\n=== 3) Feature dictionary (what each column means) ===")

feature_dictionary <- tribble(
  ~feature, ~type, ~description, ~phase_created,
  "id", "identifier", "OpenAlex work ID (unique paper URL)", "collection",
  "title", "text", "Paper title", "collection",
  "abstract", "text", "Plain-text abstract reconstructed from OpenAlex", "collection",
  "publication_year", "integer", "Year the paper was published (2022–2025)", "collection",
  "cited_by_count", "numeric", "Total citations recorded by OpenAlex", "collection",
  "language", "categorical", "Language code (kept English only)", "collection",
  "type", "categorical", "Work type (article)", "collection",
  "concepts", "json/text", "Raw OpenAlex concepts JSON", "collection",
  "keywords", "json/text", "Raw OpenAlex keywords JSON", "collection",
  "doi", "identifier", "Digital Object Identifier", "collection",
  "open_access", "json/text", "Raw open-access metadata JSON", "collection",
  "concepts_clean", "text", "Readable concept names joined by '; '", "feature extraction",
  "keywords_clean", "text", "Readable keyword names joined by '; '", "feature extraction",
  "is_open_access", "boolean", "TRUE if paper is open access", "feature extraction",
  "oa_status", "categorical", "OpenAlex OA status (gold/hybrid/green/...)", "feature extraction",
  "oa_url", "text", "URL to OA full text when available", "feature extraction",
  "has_fulltext", "boolean", "Whether a repository reports fulltext", "feature extraction",
  "paper_age", "integer", "Years since publication (reference year 2026)", "feature engineering",
  "title_length", "integer", "Character length of title", "feature engineering",
  "abstract_length", "integer", "Character length of abstract", "feature engineering",
  "keyword_count", "integer", "Number of keywords", "feature engineering",
  "concept_count", "integer", "Number of concepts", "feature engineering",
  "citation_per_year", "numeric", "cited_by_count / max(paper_age, 1)", "feature engineering",
  "citation_log", "numeric", "ln(1 + cited_by_count)", "feature engineering",
  "recent_paper", "binary", "1 if paper_age <= 2, else 0", "feature engineering",
  "has_doi", "binary", "1 if DOI present, else 0", "feature engineering",
  "oa_category", "categorical", "fully_open / partially_open / closed", "feature engineering"
)

print(feature_dictionary, n = 50)
write_csv(feature_dictionary, file.path(out_dir, "feature_dictionary.csv"))
message("Saved: reports/tables/feature_dictionary.csv")

message("\n=== 4) Feature groups for presentation ===")
feature_groups <- feature_dictionary %>%
  count(phase_created, type, name = "n_features") %>%
  arrange(phase_created, desc(n_features))
print(feature_groups)

message("\n=== 5) Target / analysis variable candidates (Phase 2 preview) ===")
candidates <- tribble(
  ~candidate_target, ~why_useful,
  "oa_category", "Predict open-access category from metadata/text proxies",
  "citation_log", "Model citation impact on a stabilized scale",
  "recent_paper", "Classify recent vs established papers",
  "has_fulltext", "Predict fulltext availability from bibliographic features"
)
print(candidates)

message("\n=== 6) Quick numeric profile of engineered features ===")
papers %>%
  summarise(
    across(
      c(paper_age, title_length, abstract_length, keyword_count, concept_count,
        citation_per_year, citation_log, recent_paper, has_doi),
      list(mean = ~mean(.x, na.rm = TRUE), median = ~median(.x, na.rm = TRUE)),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  glimpse()

message("\nDataset understanding script complete.")
message("Show faculty: feature_dictionary.csv + glimpse/skim output.")
