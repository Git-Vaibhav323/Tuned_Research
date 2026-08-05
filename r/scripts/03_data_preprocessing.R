# =============================================================================
# ResearchPilot — Data Preprocessing
# Input : data/raw/raw_papers.csv
# Output: data/cleaned/cleaned_papers_r.csv
#         reports/tables/preprocessing_steps.csv
#         reports/preprocessing_report_r.md
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
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
  stop("Project root not found. Use setwd('E:/Tuned_Research') first.")
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
# 1. Load raw data
# -----------------------------------------------------------------------------
cat("=== 1. LOAD RAW DATA ===\n")
raw <- read_csv(raw_csv, show_col_types = FALSE)
cat("Shape:", nrow(raw), "x", ncol(raw), "\n")
glimpse(raw)
log_step("load_raw", "Read raw_papers.csv", nrow(raw), nrow(raw))

# -----------------------------------------------------------------------------
# 2. Missing values before cleaning
# -----------------------------------------------------------------------------
cat("\n=== 2. MISSING VALUES (BEFORE) ===\n")
missing_before <- raw %>%
  summarise(across(everything(), ~ sum(is.na(.x) | .x == ""))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") %>%
  mutate(pct_missing = round(100 * n_missing / nrow(raw), 2)) %>%
  arrange(desc(n_missing))

print(missing_before)

p_miss_before <- ggplot(
  missing_before,
  aes(x = reorder(column, n_missing), y = n_missing)
) +
  geom_col(fill = "#D95F02") +
  coord_flip() +
  labs(
    title = "Missing Values Before Cleaning",
    x = NULL,
    y = "Missing count"
  ) +
  theme_minimal(base_size = 13)

print(p_miss_before)
ggsave(file.path(viz_dir, "21_missing_before_cleaning.png"), p_miss_before, width = 8, height = 5)

# -----------------------------------------------------------------------------
# 3. Clean title
# -----------------------------------------------------------------------------
cat("\n=== 3. CLEAN TITLE ===\n")
df <- raw
n0 <- nrow(df)
df <- df %>%
  mutate(
    title = title %>%
      str_replace_all("[\r\n]+", " ") %>%
      str_squish()
  )
log_step("clean_title", "Remove newlines and extra spaces in title", n0, nrow(df))
cat("Rows after title cleaning:", nrow(df), "\n")

# -----------------------------------------------------------------------------
# 4. Clean abstract
# -----------------------------------------------------------------------------
cat("\n=== 4. CLEAN ABSTRACT ===\n")
n0 <- nrow(df)
df <- df %>%
  mutate(
    abstract = abstract %>%
      as.character() %>%
      str_replace_all("[\r\n]+", " ") %>%
      str_squish()
  ) %>%
  filter(!is.na(abstract), abstract != "")
log_step("clean_abstract", "Normalize whitespace; remove empty abstracts", n0, nrow(df))
cat("Rows after abstract cleaning:", nrow(df), "\n")

# -----------------------------------------------------------------------------
# 5. Type conversion
# -----------------------------------------------------------------------------
cat("\n=== 5. TYPE CONVERSION ===\n")
n0 <- nrow(df)
df <- df %>%
  mutate(
    publication_year = as.integer(publication_year),
    cited_by_count = as.integer(cited_by_count)
  ) %>%
  filter(!is.na(publication_year), !is.na(cited_by_count))
log_step("type_conversion", "Convert year and citations to integer", n0, nrow(df))
cat("Rows after type conversion:", nrow(df), "\n")

# -----------------------------------------------------------------------------
# 6. Language filter
# -----------------------------------------------------------------------------
cat("\n=== 6. LANGUAGE FILTER ===\n")
n0 <- nrow(df)
df <- df %>%
  mutate(language = str_to_lower(str_trim(as.character(language)))) %>%
  filter(language %in% c("en", "english"))
log_step("language_filter", "Keep English records only", n0, nrow(df))
cat("Rows after language filter:", nrow(df), "\n")

# -----------------------------------------------------------------------------
# 7. DOI deduplication
# -----------------------------------------------------------------------------
cat("\n=== 7. DOI DEDUPLICATION ===\n")
n0 <- nrow(df)
df <- df %>%
  mutate(doi_norm = na_if(str_to_lower(str_trim(as.character(doi))), ""))

with_doi <- df %>%
  filter(!is.na(doi_norm)) %>%
  arrange(desc(cited_by_count), desc(publication_year)) %>%
  distinct(doi_norm, .keep_all = TRUE)

without_doi <- df %>% filter(is.na(doi_norm))
df <- bind_rows(with_doi, without_doi) %>% select(-doi_norm)
log_step("doi_dedup", "Remove duplicate DOIs; keep highest citations", n0, nrow(df))
cat("Rows after DOI deduplication:", nrow(df), "\n")

# -----------------------------------------------------------------------------
# 8. Encoding
# -----------------------------------------------------------------------------
cat("\n=== 8. CATEGORICAL ENCODING ===\n")
encoding_demo <- df %>%
  mutate(
    language_code = as.integer(factor(language)),
    type_code = as.integer(factor(type))
  ) %>%
  select(title, language, language_code, type, type_code) %>%
  mutate(title = substr(title, 1, 50)) %>%
  slice_head(n = 8)

print(encoding_demo)
write_csv(encoding_demo, file.path(table_dir, "encoding_example.csv"))

# -----------------------------------------------------------------------------
# 9. Normalization
# -----------------------------------------------------------------------------
cat("\n=== 9. MIN-MAX NORMALIZATION ===\n")
cite_min <- min(df$cited_by_count, na.rm = TRUE)
cite_max <- max(df$cited_by_count, na.rm = TRUE)

normalization_demo <- df %>%
  mutate(
    cited_by_count_minmax = (cited_by_count - cite_min) / (cite_max - cite_min)
  ) %>%
  select(title, cited_by_count, cited_by_count_minmax) %>%
  mutate(title = substr(title, 1, 50)) %>%
  slice_head(n = 8)

print(normalization_demo)
cat("Formula: (x - min) / (max - min)\n")
cat("Min:", cite_min, "| Max:", cite_max, "\n")
write_csv(normalization_demo, file.path(table_dir, "normalization_example.csv"))

# -----------------------------------------------------------------------------
# 10. Missing values after cleaning + save
# -----------------------------------------------------------------------------
cat("\n=== 10. MISSING VALUES (AFTER) ===\n")
missing_after <- df %>%
  summarise(across(everything(), ~ sum(is.na(.x) | .x == ""))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") %>%
  mutate(pct_missing = round(100 * n_missing / nrow(df), 2)) %>%
  arrange(desc(n_missing))

print(missing_after)

p_miss_after <- ggplot(
  missing_after,
  aes(x = reorder(column, n_missing), y = n_missing)
) +
  geom_col(fill = "#1B9E77") +
  coord_flip() +
  labs(
    title = "Missing Values After Cleaning",
    x = NULL,
    y = "Missing count"
  ) +
  theme_minimal(base_size = 13)

print(p_miss_after)
ggsave(file.path(viz_dir, "22_missing_after_cleaning.png"), p_miss_after, width = 8, height = 5)

write_csv(df, out_csv)
cat("\nSaved cleaned dataset:", out_csv, "\n")

# -----------------------------------------------------------------------------
# 11. Preprocessing pipeline summary
# -----------------------------------------------------------------------------
cat("\n=== 11. PREPROCESSING PIPELINE SUMMARY ===\n")
steps_tbl <- bind_rows(step_log)
print(steps_tbl)
write_csv(steps_tbl, file.path(table_dir, "preprocessing_steps.csv"))

pipeline_summary <- tribble(
  ~step, ~operation, ~result,
  "1", "Load raw data", paste0(nrow(raw), " rows, ", ncol(raw), " columns"),
  "2", "Missing value audit", paste0(sum(missing_before$n_missing), " missing cells before cleaning"),
  "3", "Clean title and abstract", "Removed newlines and extra spaces",
  "4", "Type conversion", "publication_year and cited_by_count as integer",
  "5", "Language filter", "Kept English records only",
  "6", "DOI deduplication", "Removed duplicate DOIs",
  "7", "Encoding", "language and type converted to numeric codes",
  "8", "Normalization", "cited_by_count scaled to [0, 1]",
  "9", "Final cleaned data", paste0(nrow(df), " rows, ", ncol(df), " columns")
)

print(pipeline_summary)
write_csv(pipeline_summary, file.path(table_dir, "preprocessing_pipeline_summary.csv"))

report <- paste0(
  "# Preprocessing Report\n\n",
  "**Input:** `data/raw/raw_papers.csv`  \n",
  "**Output:** `data/cleaned/cleaned_papers_r.csv`  \n\n",
  "## Pipeline\n\n",
  "1. Load raw data\n",
  "2. Missing value audit\n",
  "3. Title and abstract cleaning\n",
  "4. Type conversion\n",
  "5. Language filter\n",
  "6. DOI deduplication\n",
  "7. Categorical encoding\n",
  "8. Min-max normalization\n",
  "9. Save cleaned dataset\n\n",
  "## Row impact\n\n",
  "| Step | Detail | Before | After | Removed |\n",
  "|------|--------|-------:|------:|--------:|\n",
  paste0(
    apply(steps_tbl, 1, function(r) {
      sprintf(
        "| %s | %s | %s | %s | %s |",
        r[["step"]], r[["detail"]], r[["rows_before"]], r[["rows_after"]], r[["rows_removed"]]
      )
    }),
    collapse = "\n"
  ),
  "\n"
)
writeLines(report, report_md)

cat("\nPreprocessing complete.\n")
cat("Report:", report_md, "\n")
