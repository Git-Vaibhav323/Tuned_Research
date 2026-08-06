# =============================================================================
# ResearchPilot: Data Preprocessing & Cleaning (in R)

# Input  : data/raw/raw_papers.csv
# Output : data/cleaned/cleaned_papers_r.csv
#          reports/tables/preprocessing_steps_r.csv
#          reports/preprocessing_report_r.md
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(janitor)
  library(scales)
})

find_project_root <- function() {
  path <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  for (i in seq_len(6)) {
    if (file.exists(file.path(path, "data", "raw", "raw_papers.csv"))) {
      return(path)
    }
    parent <- dirname(path)
    if (identical(parent, path)) break
    path <- parent
  }
  stop("Project root not found. setwd('E:/Tuned_Research') first.")
}

project_root <- find_project_root()
raw_csv <- file.path(project_root, "data", "raw", "raw_papers.csv")
out_csv <- file.path(project_root, "data", "cleaned", "cleaned_papers_r.csv")
table_dir <- file.path(project_root, "reports", "tables")
viz_dir <- file.path(project_root, "r", "visualizations")
report_md <- file.path(project_root, "reports", "preprocessing_report_r.md")
dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(viz_dir, recursive = TRUE, showWarnings = FALSE)

step_log <- list()
log_step <- function(step, detail, rows_before, rows_after) {
  step_log[[length(step_log) + 1]] <<- tibble(
    step = step,
    detail = detail,
    rows_before = rows_before,
    rows_after = rows_after,
    rows_removed = rows_before - rows_after
  )
}

# -----------------------------------------------------------------------------
# STEP 1 — Load raw data
# -----------------------------------------------------------------------------
message("STEP 1: Load raw OpenAlex data")
raw <- read_csv(raw_csv, show_col_types = FALSE)
cat("Raw shape:", nrow(raw), "x", ncol(raw), "\n")
glimpse(raw)
log_step("1_load_raw", "Read data/raw/raw_papers.csv", nrow(raw), nrow(raw))

# -----------------------------------------------------------------------------
# STEP 2 — Dataset summary BEFORE cleaning
# -----------------------------------------------------------------------------
message("\nSTEP 2: Before-cleaning summary")
before_missing <- raw %>%
  summarise(across(everything(), ~ sum(is.na(.x) | .x == ""))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") %>%
  arrange(desc(n_missing))
print(before_missing)

before_dup_rows <- sum(duplicated(raw))
before_dup_doi <- raw %>%
  filter(!is.na(doi), doi != "") %>%
  summarise(n = n() - n_distinct(tolower(str_trim(doi)))) %>%
  pull(n)

cat("Fully duplicate rows:", before_dup_rows, "\n")
cat("Duplicate DOIs (approx):", before_dup_doi, "\n")

p_miss_before <- before_missing %>%
  filter(n_missing > 0) %>%
  ggplot(aes(x = reorder(column, n_missing), y = n_missing)) +
  geom_col(fill = "#D95F02") +
  coord_flip() +
  labs(
    title = "Missing Values BEFORE Cleaning",
    x = NULL, y = "Missing count"
  )
ggsave(file.path(viz_dir, "21_missing_before_cleaning.png"), p_miss_before, width = 8, height = 5)

# -----------------------------------------------------------------------------
# STEP 3 — Clean title (whitespace / newlines)
# -----------------------------------------------------------------------------
message("\nSTEP 3: Clean title")
df <- raw
n0 <- nrow(df)
df <- df %>%
  mutate(
    title = title %>%
      str_replace_all("[\r\n]+", " ") %>%
      str_squish()
  )
log_step("3_clean_title", "Remove newlines and extra spaces in title", n0, nrow(df))

# -----------------------------------------------------------------------------
# STEP 4 — Clean abstract + drop empty abstracts
# -----------------------------------------------------------------------------
message("\nSTEP 4: Clean abstract and drop null/empty abstracts")
n0 <- nrow(df)
df <- df %>%
  mutate(
    abstract = abstract %>%
      as.character() %>%
      str_replace_all("[\r\n]+", " ") %>%
      str_squish()
  ) %>%
  filter(!is.na(abstract), abstract != "")
log_step("4_clean_abstract", "Normalize whitespace; remove empty abstracts", n0, nrow(df))

# -----------------------------------------------------------------------------
# STEP 5 — Type conversion (publication_year, cited_by_count)
# -----------------------------------------------------------------------------
message("\nSTEP 5: Convert numeric types")
n0 <- nrow(df)
df <- df %>%
  mutate(
    publication_year = as.integer(publication_year),
    cited_by_count = as.integer(cited_by_count)
  ) %>%
  filter(!is.na(publication_year), !is.na(cited_by_count))
log_step("5_type_conversion", "Cast year and citations to integer", n0, nrow(df))

# -----------------------------------------------------------------------------
# STEP 6 — Language filter (encoding / category consistency)
# -----------------------------------------------------------------------------
message("\nSTEP 6: Keep English only")
n0 <- nrow(df)
df <- df %>%
  mutate(language = str_to_lower(str_trim(as.character(language)))) %>%
  filter(language %in% c("en", "english"))
log_step("6_language_filter", "Keep only English records", n0, nrow(df))

# -----------------------------------------------------------------------------
# STEP 7 — DOI deduplication
# -----------------------------------------------------------------------------
message("\nSTEP 7: Remove duplicate DOIs (keep highest citations)")
n0 <- nrow(df)
df <- df %>%
  mutate(doi_norm = na_if(str_to_lower(str_trim(as.character(doi))), ""))

with_doi <- df %>%
  filter(!is.na(doi_norm)) %>%
  arrange(desc(cited_by_count), desc(publication_year)) %>%
  distinct(doi_norm, .keep_all = TRUE)

without_doi <- df %>% filter(is.na(doi_norm))
df <- bind_rows(with_doi, without_doi) %>% select(-doi_norm)
log_step("7_doi_dedup", "Drop duplicate DOIs; retain highest cited_by_count", n0, nrow(df))

# -----------------------------------------------------------------------------
# STEP 8 — Simple encoding example (for faculty rubric)
# -----------------------------------------------------------------------------
message("\nSTEP 8: Categorical encoding example (language/type)")
# Demonstrate label encoding style for presentation (not required for final file)
df_encoded_demo <- df %>%
  mutate(
    language_code = as.integer(factor(language)),
    type_code = as.integer(factor(type))
  ) %>%
  select(id, language, language_code, type, type_code) %>%
  slice_head(n = 10)
print(df_encoded_demo)

# -----------------------------------------------------------------------------
# STEP 9 — Normalization example (for faculty rubric)
# -----------------------------------------------------------------------------
message("\nSTEP 9: Normalization example (min-max on cited_by_count)")
# Show formula; keep original column unchanged in saved cleaned file
cite_min <- min(df$cited_by_count, na.rm = TRUE)
cite_max <- max(df$cited_by_count, na.rm = TRUE)
df_norm_demo <- df %>%
  mutate(
    cited_by_count_minmax = (cited_by_count - cite_min) / (cite_max - cite_min)
  ) %>%
  select(title, cited_by_count, cited_by_count_minmax) %>%
  slice_head(n = 10)
print(df_norm_demo)
cat("Min-max formula: (x - min) / (max - min)\n")

# Note: engineered features like citation_log are a better production transform
# and are created later in feature engineering.

# -----------------------------------------------------------------------------
# STEP 10 — After-cleaning summary + save
# -----------------------------------------------------------------------------
message("\nSTEP 10: After-cleaning summary and save")
after_missing <- df %>%
  summarise(across(everything(), ~ sum(is.na(.x) | .x == ""))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") %>%
  arrange(desc(n_missing))
print(after_missing)

p_miss_after <- after_missing %>%
  filter(n_missing > 0) %>%
  {
    if (nrow(.) == 0) {
      ggplot(data.frame(x = 1, y = 1), aes(x, y)) +
        annotate("text", x = 1, y = 1, label = "No / very few missing values after cleaning", size = 5) +
        theme_void() +
        labs(title = "Missing Values AFTER Cleaning")
    } else {
      ggplot(., aes(x = reorder(column, n_missing), y = n_missing)) +
        geom_col(fill = "#1B9E77") +
        coord_flip() +
        labs(title = "Missing Values AFTER Cleaning", x = NULL, y = "Missing count")
    }
  }
ggsave(file.path(viz_dir, "22_missing_after_cleaning.png"), p_miss_after, width = 8, height = 5)

write_csv(df, out_csv)
message("Saved cleaned file: ", out_csv)

steps_tbl <- bind_rows(step_log)
print(steps_tbl)
write_csv(steps_tbl, file.path(table_dir, "preprocessing_steps_r.csv"))

# -----------------------------------------------------------------------------
# STEP 11 — Write preprocessing report for documentation marks
# -----------------------------------------------------------------------------
report <- paste0(
  "# Preprocessing Report (R Demonstration)\n\n",
  "**Project:** ResearchPilot  \n",
  "**Input:** `data/raw/raw_papers.csv`  \n",
  "**Output:** `data/cleaned/cleaned_papers_r.csv`  \n\n",
  "## Steps demonstrated\n\n",
  "1. Load raw data\n",
  "2. Missing-value audit (before)\n",
  "3. Title cleaning (newlines / whitespace)\n",
  "4. Abstract cleaning + drop empty abstracts\n",
  "5. Type conversion (`publication_year`, `cited_by_count` → integer)\n",
  "6. Language filter (English only)\n",
  "7. DOI deduplication\n",
  "8. Encoding example (`language_code`, `type_code`)\n",
  "9. Normalization example (min-max on citations)\n",
  "10. Missing-value audit (after) + save\n\n",
  "## Row impact\n\n",
  "| Step | Rows before | Rows after | Removed |\n",
  "|------|------------:|-----------:|--------:|\n",
  paste0(
    apply(steps_tbl, 1, function(r) {
      sprintf("| %s | %s | %s | %s |", r[["step"]], r[["rows_before"]], r[["rows_after"]], r[["rows_removed"]])
    }),
    collapse = "\n"
  ),
  "\n\n## Notes for evaluation\n\n",
  "- Collection already applied year/language/type/abstract filters via OpenAlex API.\n",
  "- This R script re-validates and demonstrates preprocessing transparently.\n",
  "- Production engineered features (log citations, age-normalized citations) are in `feature_engineering.py` / `final_dataset.csv`.\n"
)
writeLines(report, report_md)
message("Saved report: ", report_md)

message("\nPreprocessing demo complete. Show faculty steps 1–11 live.")
