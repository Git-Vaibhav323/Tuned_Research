# =============================================================================
# ResearchPilot — Exploratory Data Analysis (EDA)
# Dataset: data/final/final_dataset.csv
# Outputs: r/visualizations/*.png , reports/tables/*.csv
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(scales)
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
viz_dir <- file.path(project_root, "r", "visualizations")
table_dir <- file.path(project_root, "reports", "tables")
dir.create(viz_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

theme_set(theme_minimal(base_size = 13))

save_plot <- function(plot_obj, filename, width = 8, height = 5) {
  print(plot_obj)
  ggsave(file.path(viz_dir, filename), plot_obj, width = width, height = height, dpi = 150)
  message("Saved: ", file.path(viz_dir, filename))
}

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
papers <- read_csv(input_csv, show_col_types = FALSE) %>%
  mutate(
    publication_year = as.integer(publication_year),
    cited_by_count = as.numeric(cited_by_count),
    oa_category = factor(
      oa_category,
      levels = c("fully_open", "partially_open", "closed", "unknown")
    )
  )

cat("Loaded:", nrow(papers), "rows x", ncol(papers), "columns\n")

# -----------------------------------------------------------------------------
# 2. Missing values
# -----------------------------------------------------------------------------
missing_tbl <- papers %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") %>%
  mutate(pct_missing = round(100 * n_missing / nrow(papers), 2)) %>%
  arrange(desc(n_missing))

print(missing_tbl)

p_missing <- ggplot(
  missing_tbl %>% filter(n_missing > 0),
  aes(x = reorder(column, n_missing), y = n_missing)
) +
  geom_col(fill = "#2C7FB8") +
  coord_flip() +
  labs(title = "Missing Values by Column", x = NULL, y = "Missing count")

save_plot(p_missing, "01_missing_values.png")

# -----------------------------------------------------------------------------
# 3. Publication year
# -----------------------------------------------------------------------------
year_counts <- papers %>%
  count(publication_year, name = "n_papers") %>%
  arrange(publication_year)

print(year_counts)

p_year <- ggplot(year_counts, aes(x = factor(publication_year), y = n_papers)) +
  geom_col(fill = "#1B9E77", width = 0.7) +
  geom_text(aes(label = n_papers), vjust = -0.4, size = 4) +
  labs(
    title = "Papers by Publication Year",
    x = "Publication Year",
    y = "Number of Papers"
  ) +
  expand_limits(y = max(year_counts$n_papers) * 1.1)

save_plot(p_year, "02_papers_by_year.png")

# -----------------------------------------------------------------------------
# 4. Citation histograms
# -----------------------------------------------------------------------------
p_cite_raw <- ggplot(papers, aes(x = cited_by_count)) +
  geom_histogram(bins = 40, fill = "#D95F02", color = "white") +
  scale_x_continuous(labels = comma) +
  labs(
    title = "Citation Count Distribution",
    x = "cited_by_count",
    y = "Number of Papers"
  )

save_plot(p_cite_raw, "04_citation_count_raw.png")

p_cite_log <- ggplot(papers, aes(x = citation_log)) +
  geom_histogram(bins = 40, fill = "#E7298A", color = "white") +
  labs(
    title = "Log Citation Distribution",
    subtitle = "citation_log = ln(1 + cited_by_count)",
    x = "citation_log",
    y = "Number of Papers"
  )

save_plot(p_cite_log, "05_citation_log.png")

# -----------------------------------------------------------------------------
# 5. Open access
# -----------------------------------------------------------------------------
oa_counts <- papers %>%
  count(oa_category, name = "n") %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  arrange(desc(n))

print(oa_counts)

p_oa <- ggplot(oa_counts, aes(x = reorder(oa_category, n), y = n, fill = oa_category)) +
  geom_col(show.legend = FALSE, width = 0.7) +
  geom_text(aes(label = paste0(n, " (", pct, "%)")), hjust = -0.05, size = 3.8) +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Open Access Category Distribution",
    x = NULL,
    y = "Number of Papers"
  ) +
  expand_limits(y = max(oa_counts$n) * 1.2)

save_plot(p_oa, "14_oa_category.png")

p_oa_cite <- ggplot(papers, aes(x = oa_category, y = citation_log, fill = oa_category)) +
  geom_boxplot(show.legend = FALSE, outlier.alpha = 0.35) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Log Citations by Open Access Category",
    x = "oa_category",
    y = "citation_log"
  )

save_plot(p_oa_cite, "16_oa_category_vs_citation_log.png")

# -----------------------------------------------------------------------------
# 6. Text and topic features
# -----------------------------------------------------------------------------
p_title <- ggplot(papers, aes(x = title_length)) +
  geom_histogram(bins = 35, fill = "#1F78B4", color = "white") +
  labs(title = "Title Length Distribution", x = "title_length", y = "Count")

save_plot(p_title, "08_title_length.png")

p_kw_concept <- ggplot(papers, aes(x = keyword_count, y = concept_count)) +
  geom_point(alpha = 0.25, color = "#7570B3") +
  geom_smooth(method = "lm", se = TRUE, color = "#E7298A") +
  labs(
    title = "Keyword Count vs Concept Count",
    x = "keyword_count",
    y = "concept_count"
  )

save_plot(p_kw_concept, "13_keyword_vs_concept.png")

# -----------------------------------------------------------------------------
# 7. Recent vs older papers
# -----------------------------------------------------------------------------
recent_tbl <- papers %>%
  mutate(recency = if_else(recent_paper == 1, "Recent (age <= 2)", "Older (age > 2)")) %>%
  count(recency, name = "n") %>%
  mutate(pct = round(100 * n / sum(n), 1))

print(recent_tbl)

p_recent <- ggplot(recent_tbl, aes(x = recency, y = n, fill = recency)) +
  geom_col(show.legend = FALSE, width = 0.6) +
  geom_text(aes(label = paste0(n, " (", pct, "%)")), vjust = -0.4) +
  scale_fill_manual(values = c("#FC8D62", "#8DA0CB")) +
  labs(title = "Recent vs Older Papers", x = NULL, y = "Number of Papers")

save_plot(p_recent, "17_recent_vs_older.png")

# -----------------------------------------------------------------------------
# 8. Correlation heatmap
# -----------------------------------------------------------------------------
numeric_feats <- papers %>%
  select(
    publication_year, cited_by_count, paper_age, title_length,
    abstract_length, keyword_count, concept_count,
    citation_per_year, citation_log
  )

corr_mat <- round(cor(numeric_feats, use = "pairwise.complete.obs"), 2)
print(corr_mat)

corr_long <- as.data.frame(as.table(corr_mat)) %>%
  rename(feature_x = Var1, feature_y = Var2, correlation = Freq)

p_corr <- ggplot(corr_long, aes(x = feature_x, y = feature_y, fill = correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", correlation)), size = 3) +
  scale_fill_gradient2(low = "#B2182B", mid = "white", high = "#2166AC", midpoint = 0) +
  labs(
    title = "Correlation Heatmap of Numeric Features",
    x = NULL,
    y = NULL,
    fill = "r"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

save_plot(p_corr, "19_correlation_heatmap.png", width = 9, height = 7)

# -----------------------------------------------------------------------------
# 9. Summary tables
# -----------------------------------------------------------------------------
summary_by_year <- papers %>%
  group_by(publication_year) %>%
  summarise(
    n_papers = n(),
    median_citations = median(cited_by_count),
    median_citation_log = median(citation_log),
    pct_open = mean(is_open_access, na.rm = TRUE) * 100,
    .groups = "drop"
  )

print(summary_by_year)
write_csv(summary_by_year, file.path(table_dir, "summary_by_year.csv"))

summary_by_oa <- papers %>%
  group_by(oa_category) %>%
  summarise(
    n_papers = n(),
    median_citations = median(cited_by_count),
    median_citation_per_year = median(citation_per_year),
    .groups = "drop"
  ) %>%
  arrange(desc(n_papers))

print(summary_by_oa)
write_csv(summary_by_oa, file.path(table_dir, "summary_by_oa_category.csv"))

cat("\nEDA complete.\n")
cat("Figures saved in:", viz_dir, "\n")
cat("Tables saved in :", table_dir, "\n")
