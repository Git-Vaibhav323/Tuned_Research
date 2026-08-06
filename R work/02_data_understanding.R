# =============================================================================
# ResearchPilot — Data Understanding
# Dataset: data/final/final_dataset.csv
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(skimr)
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
  stop("Project root not found. Use setwd('E:/Tuned_Research') first.")
}

project_root <- find_project_root()
input_csv <- file.path(project_root, "data", "final", "final_dataset.csv")
out_dir <- file.path(project_root, "reports", "tables")
viz_dir <- file.path(project_root, "r", "visualizations")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(viz_dir, recursive = TRUE, showWarnings = FALSE)

papers <- read_csv(input_csv, show_col_types = FALSE)

cat("=== DATASET OVERVIEW ===\n")
cat("Rows   :", nrow(papers), "\n")
cat("Columns:", ncol(papers), "\n\n")

cat("=== STRUCTURE ===\n")
glimpse(papers)

cat("\n=== SUMMARY STATISTICS ===\n")
print(skim(papers))

feature_dictionary <- tribble(
  ~feature, ~type, ~description,
  "id", "identifier", "OpenAlex work ID",
  "title", "text", "Paper title",
  "abstract", "text", "Paper abstract",
  "publication_year", "integer", "Publication year (2022-2025)",
  "cited_by_count", "numeric", "Total citation count",
  "language", "categorical", "Language code",
  "type", "categorical", "Work type (article)",
  "concepts", "json/text", "Raw OpenAlex concepts",
  "keywords", "json/text", "Raw OpenAlex keywords",
  "doi", "identifier", "Digital Object Identifier",
  "open_access", "json/text", "Raw open-access metadata",
  "concepts_clean", "text", "Readable concept names",
  "keywords_clean", "text", "Readable keyword names",
  "is_open_access", "boolean", "Whether the paper is open access",
  "oa_status", "categorical", "OpenAlex OA status",
  "oa_url", "text", "Open-access fulltext URL",
  "has_fulltext", "boolean", "Whether fulltext is available",
  "paper_age", "integer", "Years since publication",
  "title_length", "integer", "Title length in characters",
  "abstract_length", "integer", "Abstract length in characters",
  "keyword_count", "integer", "Number of keywords",
  "concept_count", "integer", "Number of concepts",
  "citation_per_year", "numeric", "Citations divided by paper age",
  "citation_log", "numeric", "Natural log of (1 + citations)",
  "recent_paper", "binary", "1 if paper age <= 2, else 0",
  "has_doi", "binary", "1 if DOI is present, else 0",
  "oa_category", "categorical", "fully_open / partially_open / closed"
)

print(feature_dictionary, n = 50)
write_csv(feature_dictionary, file.path(out_dir, "feature_dictionary.csv"))

feature_categories <- feature_dictionary %>%
  count(type, name = "n_features") %>%
  arrange(desc(n_features))

print(feature_categories)

p_categories <- ggplot(feature_categories, aes(x = reorder(type, n_features), y = n_features, fill = type)) +
  geom_col(show.legend = FALSE, width = 0.7) +
  geom_text(aes(label = n_features), hjust = -0.2, size = 4) +
  coord_flip() +
  labs(title = "Feature Categories by Data Type", x = NULL, y = "Number of features") +
  theme_minimal(base_size = 13) +
  expand_limits(y = max(feature_categories$n_features) + 2)

print(p_categories)
ggsave(file.path(viz_dir, "feature_categories_by_type.png"), p_categories, width = 8, height = 5)

schema <- tibble(
  column = names(papers),
  dtype = vapply(papers, function(x) class(x)[1], character(1)),
  n_missing = vapply(papers, function(x) sum(is.na(x)), integer(1)),
  pct_missing = round(100 * n_missing / nrow(papers), 2)
)
print(schema, n = 50)
write_csv(schema, file.path(out_dir, "dataset_schema.csv"))

sample_rows <- papers %>%
  select(title, publication_year, cited_by_count, oa_category,
         paper_age, citation_log, keyword_count, concept_count) %>%
  mutate(title = substr(title, 1, 60)) %>%
  slice_head(n = 5)

print(sample_rows)
write_csv(sample_rows, file.path(out_dir, "final_dataset_sample5.csv"))