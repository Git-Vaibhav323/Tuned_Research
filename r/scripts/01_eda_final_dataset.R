# =============================================================================
# ResearchPilot — Phase 1 Exploratory Data Analysis (R)
# Dataset : data/final/final_dataset.csv
# Purpose : Step-by-step EDA with tidyverse / ggplot2
# Outputs : r/visualizations/*.png  (+ optional reports/figures/)
# =============================================================================
#
# HOW TO RUN
# ----------
# 1. Open this file in RStudio.
# 2. Run Step 0 once (package install).
# 3. Run each STEP block in order (Ctrl+Enter / Source).
# 4. Figures are saved under r/visualizations/.
#
# Required packages: tidyverse, skimr, janitor, scales, GGally (optional)
# =============================================================================


# =============================================================================
# STEP 0 — Install packages (run once per machine)
# =============================================================================

# Prefer: run r/scripts/00_install_packages.R once, then continue here.
# Do NOT use dependencies = TRUE (it installs a huge optional package tree).


# =============================================================================
# STEP 1 — Load libraries
# =============================================================================

# Load packages individually (more reliable than library(tidyverse) on Windows)
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(forcats)
  library(readr)
  library(ggplot2)
  library(scales)
  library(skimr)
  library(janitor)
})

# GGally is optional (pairwise plot only)
has_ggally <- requireNamespace("GGally", quietly = TRUE)
if (!has_ggally) {
  message("Note: GGally is not installed. Step 12 pairwise plot will be skipped.")
}

theme_set(theme_minimal(base_size = 12))

# Compatibility helpers if tidyverse-style pipes are expected
if (!exists("%||%", mode = "function")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}


# =============================================================================
# STEP 2 — Locate project root and define paths
# =============================================================================

find_project_root <- function() {
  path <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  for (i in seq_len(6)) {
    candidate <- file.path(path, "data", "final", "final_dataset.csv")
    if (file.exists(candidate)) {
      return(path)
    }
    parent <- dirname(path)
    if (identical(parent, path)) break
    path <- parent
  }
  stop(
    "Could not find project root containing data/final/final_dataset.csv. ",
    "Set the working directory to the ResearchPilot project folder first."
  )
}

project_root <- find_project_root()

input_csv <- file.path(project_root, "data", "final", "final_dataset.csv")
viz_dir   <- file.path(project_root, "r", "visualizations")
fig_dir   <- file.path(project_root, "reports", "figures")

dir.create(viz_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

save_plot <- function(plot, filename, width = 9, height = 6) {
  out1 <- file.path(viz_dir, filename)
  out2 <- file.path(fig_dir, filename)
  ggsave(out1, plot = plot, width = width, height = height, dpi = 150)
  ggsave(out2, plot = plot, width = width, height = height, dpi = 150)
  message("Saved: ", out1)
  invisible(plot)
}


# =============================================================================
# STEP 3 — Load the final dataset
# =============================================================================

papers <- read_csv(
  input_csv,
  show_col_types = FALSE,
  locale = locale(encoding = "UTF-8")
)

# Helpful factor conversions for categorical EDA
papers <- papers %>%
  mutate(
    publication_year = as.integer(publication_year),
    cited_by_count   = as.numeric(cited_by_count),
    is_open_access   = as.logical(is_open_access),
    has_fulltext     = as.logical(has_fulltext),
    recent_paper     = as.integer(recent_paper),
    has_doi          = as.integer(has_doi),
    oa_status        = factor(oa_status),
    oa_category      = factor(
      oa_category,
      levels = c("fully_open", "partially_open", "closed", "unknown")
    ),
    language = factor(language),
    type     = factor(type)
  )

cat("Loaded rows:", nrow(papers), "\n")
cat("Loaded cols:", ncol(papers), "\n")
glimpse(papers)


# =============================================================================
# STEP 4 — Dataset structure overview
# =============================================================================

# 4.1 Dimensions
dim(papers)

# 4.2 Column names
names(papers)

# 4.3 Compact skim summary (excellent for EDA reports)
skim(papers)

# 4.4 First few rows of key analytical columns
papers %>%
  select(
    title, publication_year, cited_by_count,
    paper_age, title_length, abstract_length,
    keyword_count, concept_count, citation_per_year, citation_log,
    recent_paper, has_doi, oa_category, is_open_access
  ) %>%
  head(10)


# =============================================================================
# STEP 5 — Missing values audit
# =============================================================================

missing_tbl <- papers %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") %>%
  mutate(
    pct_missing = round(100 * n_missing / nrow(papers), 2)
  ) %>%
  arrange(desc(n_missing))

print(missing_tbl)

p_missing <- missing_tbl %>%
  filter(n_missing > 0) %>%
  ggplot(aes(x = reorder(column, n_missing), y = n_missing)) +
  geom_col(fill = "#2C7FB8") +
  coord_flip() +
  labs(
    title = "Missing Values by Column",
    subtitle = "ResearchPilot final_dataset.csv",
    x = NULL,
    y = "Number of missing values"
  )

# If no missing values, create a simple annotation plot
if (nrow(missing_tbl %>% filter(n_missing > 0)) == 0) {
  p_missing <- ggplot(data.frame(x = 1, y = 1), aes(x, y)) +
    annotate("text", x = 1, y = 1, label = "No missing values in final_dataset", size = 5) +
    theme_void() +
    labs(title = "Missing Values Audit")
}

save_plot(p_missing, "01_missing_values.png", width = 8, height = 5)


# =============================================================================
# STEP 6 — Univariate analysis: publication year & paper age
# =============================================================================

# 6.1 Papers per year
year_counts <- papers %>%
  count(publication_year, name = "n_papers") %>%
  arrange(publication_year)

print(year_counts)

p_year <- ggplot(year_counts, aes(x = factor(publication_year), y = n_papers)) +
  geom_col(fill = "#1B9E77") +
  geom_text(aes(label = n_papers), vjust = -0.4, size = 3.5) +
  labs(
    title = "Number of Papers by Publication Year",
    subtitle = "AI / Machine Learning articles (2022–2025)",
    x = "Publication year",
    y = "Number of papers"
  )

save_plot(p_year, "02_papers_by_year.png", width = 8, height = 5)

# 6.2 Paper age distribution
p_age <- ggplot(papers, aes(x = paper_age)) +
  geom_bar(fill = "#7570B3", width = 0.7) +
  labs(
    title = "Distribution of Paper Age",
    subtitle = paste0("Reference year used in feature engineering"),
    x = "Paper age (years)",
    y = "Count"
  )

save_plot(p_age, "03_paper_age_distribution.png", width = 8, height = 5)


# =============================================================================
# STEP 7 — Citation analysis
# =============================================================================

# 7.1 Summary statistics
papers %>%
  summarise(
    mean_citations   = mean(cited_by_count, na.rm = TRUE),
    median_citations = median(cited_by_count, na.rm = TRUE),
    sd_citations     = sd(cited_by_count, na.rm = TRUE),
    min_citations    = min(cited_by_count, na.rm = TRUE),
    max_citations    = max(cited_by_count, na.rm = TRUE),
    mean_cites_year  = mean(citation_per_year, na.rm = TRUE),
    median_cites_year = median(citation_per_year, na.rm = TRUE)
  ) %>%
  print()

# 7.2 Raw citation distribution (expect strong right skew)
p_cite_raw <- ggplot(papers, aes(x = cited_by_count)) +
  geom_histogram(bins = 40, fill = "#D95F02", color = "white") +
  scale_x_continuous(labels = comma) +
  labs(
    title = "Distribution of Raw Citation Counts",
    subtitle = "Expect strong right skew from a few highly cited papers",
    x = "cited_by_count",
    y = "Number of papers"
  )

save_plot(p_cite_raw, "04_citation_count_raw.png")

# 7.3 Log-transformed citations
p_cite_log <- ggplot(papers, aes(x = citation_log)) +
  geom_histogram(bins = 40, fill = "#E7298A", color = "white") +
  labs(
    title = "Distribution of Log Citations (log1p)",
    subtitle = "citation_log = ln(1 + cited_by_count)",
    x = "citation_log",
    y = "Number of papers"
  )

save_plot(p_cite_log, "05_citation_log.png")

# 7.4 Citations per year
p_cite_year <- ggplot(papers, aes(x = citation_per_year)) +
  geom_histogram(bins = 40, fill = "#66A61E", color = "white") +
  scale_x_continuous(labels = comma) +
  labs(
    title = "Distribution of Citations per Year",
    subtitle = "Age-normalized impact: cited_by_count / max(paper_age, 1)",
    x = "citation_per_year",
    y = "Number of papers"
  )

save_plot(p_cite_year, "06_citation_per_year.png")

# 7.5 Boxplot of log citations by year
p_cite_by_year <- ggplot(papers, aes(x = factor(publication_year), y = citation_log)) +
  geom_boxplot(fill = "#A6761D", alpha = 0.8, outlier.alpha = 0.4) +
  labs(
    title = "Log Citations by Publication Year",
    x = "Publication year",
    y = "citation_log"
  )

save_plot(p_cite_by_year, "07_citation_log_by_year.png", width = 8, height = 5)


# =============================================================================
# STEP 8 — Text length features (title / abstract)
# =============================================================================

papers %>%
  summarise(
    across(
      c(title_length, abstract_length),
      list(mean = mean, median = median, sd = sd, min = min, max = max),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  print()

p_title_len <- ggplot(papers, aes(x = title_length)) +
  geom_histogram(bins = 35, fill = "#1F78B4", color = "white") +
  labs(
    title = "Title Length Distribution",
    subtitle = "Character count after stripping whitespace",
    x = "title_length (characters)",
    y = "Count"
  )

save_plot(p_title_len, "08_title_length.png", width = 8, height = 5)

p_abs_len <- ggplot(papers, aes(x = abstract_length)) +
  geom_histogram(bins = 35, fill = "#33A02C", color = "white") +
  labs(
    title = "Abstract Length Distribution",
    subtitle = "Character count after stripping whitespace",
    x = "abstract_length (characters)",
    y = "Count"
  )

save_plot(p_abs_len, "09_abstract_length.png", width = 8, height = 5)

# Scatter: abstract length vs log citations
p_abs_vs_cite <- ggplot(papers, aes(x = abstract_length, y = citation_log)) +
  geom_point(alpha = 0.25, color = "#1B9E77") +
  geom_smooth(method = "lm", se = TRUE, color = "#D95F02") +
  labs(
    title = "Abstract Length vs Log Citations",
    x = "abstract_length",
    y = "citation_log"
  )

save_plot(p_abs_vs_cite, "10_abstract_length_vs_citation_log.png")


# =============================================================================
# STEP 9 — Keyword and concept richness
# =============================================================================

papers %>%
  summarise(
    across(
      c(keyword_count, concept_count),
      list(mean = mean, median = median, sd = sd, min = min, max = max),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  print()

p_kw <- ggplot(papers, aes(x = keyword_count)) +
  geom_histogram(bins = 25, fill = "#6A3D9A", color = "white") +
  labs(
    title = "Keyword Count Distribution",
    x = "keyword_count",
    y = "Count"
  )

save_plot(p_kw, "11_keyword_count.png", width = 8, height = 5)

p_concept <- ggplot(papers, aes(x = concept_count)) +
  geom_histogram(bins = 25, fill = "#B15928", color = "white") +
  labs(
    title = "Concept Count Distribution",
    x = "concept_count",
    y = "Count"
  )

save_plot(p_concept, "12_concept_count.png", width = 8, height = 5)

p_kw_vs_concept <- ggplot(papers, aes(x = keyword_count, y = concept_count)) +
  geom_point(alpha = 0.25, color = "#7570B3") +
  geom_smooth(method = "lm", se = TRUE, color = "#E7298A") +
  labs(
    title = "Keyword Count vs Concept Count",
    x = "keyword_count",
    y = "concept_count"
  )

save_plot(p_kw_vs_concept, "13_keyword_vs_concept.png")


# =============================================================================
# STEP 10 — Open access analysis
# =============================================================================

# 10.1 OA category counts
oa_cat_tbl <- papers %>%
  count(oa_category, name = "n") %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  arrange(desc(n))

print(oa_cat_tbl)

p_oa_cat <- ggplot(oa_cat_tbl, aes(x = reorder(oa_category, n), y = n, fill = oa_category)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = paste0(n, " (", pct, "%)")), hjust = -0.05, size = 3.3) +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  expand_limits(y = max(oa_cat_tbl$n) * 1.18) +
  labs(
    title = "Open Access Category Distribution",
    subtitle = "fully_open / partially_open / closed",
    x = NULL,
    y = "Number of papers"
  )

save_plot(p_oa_cat, "14_oa_category.png", width = 9, height = 5)

# 10.2 Detailed oa_status
oa_status_tbl <- papers %>%
  count(oa_status, name = "n") %>%
  mutate(pct = round(100 * n / sum(n), 1)) %>%
  arrange(desc(n))

print(oa_status_tbl)

p_oa_status <- ggplot(oa_status_tbl, aes(x = reorder(oa_status, n), y = n)) +
  geom_col(fill = "#80B1D3") +
  coord_flip() +
  labs(
    title = "OpenAlex OA Status Distribution",
    x = "oa_status",
    y = "Number of papers"
  )

save_plot(p_oa_status, "15_oa_status.png", width = 8, height = 5)

# 10.3 OA vs citations
p_oa_cite <- ggplot(papers, aes(x = oa_category, y = citation_log, fill = oa_category)) +
  geom_boxplot(alpha = 0.85, show.legend = FALSE, outlier.alpha = 0.35) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Log Citations by Open Access Category",
    x = "oa_category",
    y = "citation_log"
  )

save_plot(p_oa_cite, "16_oa_category_vs_citation_log.png", width = 8, height = 5)

# 10.4 Boolean OA / fulltext rates
papers %>%
  summarise(
    pct_open_access = mean(is_open_access, na.rm = TRUE) * 100,
    pct_has_fulltext = mean(has_fulltext, na.rm = TRUE) * 100,
    pct_has_doi = mean(has_doi, na.rm = TRUE) * 100,
    pct_recent = mean(recent_paper, na.rm = TRUE) * 100
  ) %>%
  print()


# =============================================================================
# STEP 11 — Recent papers vs older papers
# =============================================================================

recent_tbl <- papers %>%
  mutate(recency = if_else(recent_paper == 1, "Recent (age ≤ 2)", "Older (age > 2)")) %>%
  count(recency, name = "n") %>%
  mutate(pct = round(100 * n / sum(n), 1))

print(recent_tbl)

p_recent <- ggplot(recent_tbl, aes(x = recency, y = n, fill = recency)) +
  geom_col(show.legend = FALSE, width = 0.6) +
  geom_text(aes(label = paste0(n, " (", pct, "%)")), vjust = -0.4) +
  scale_fill_manual(values = c("#FC8D62", "#8DA0CB")) +
  labs(
    title = "Recent vs Older Papers",
    x = NULL,
    y = "Number of papers"
  )

save_plot(p_recent, "17_recent_vs_older.png", width = 7, height = 5)

p_recent_cite <- papers %>%
  mutate(recency = if_else(recent_paper == 1, "Recent", "Older")) %>%
  ggplot(aes(x = recency, y = citation_per_year, fill = recency)) +
  geom_boxplot(show.legend = FALSE, outlier.alpha = 0.3) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "Citations per Year: Recent vs Older Papers",
    x = NULL,
    y = "citation_per_year"
  )

save_plot(p_recent_cite, "18_recent_vs_citation_per_year.png", width = 7, height = 5)


# =============================================================================
# STEP 12 — Correlation analysis (numeric engineered features)
# =============================================================================

numeric_feats <- papers %>%
  select(
    publication_year,
    cited_by_count,
    paper_age,
    title_length,
    abstract_length,
    keyword_count,
    concept_count,
    citation_per_year,
    citation_log,
    recent_paper,
    has_doi
  )

corr_mat <- cor(numeric_feats, use = "pairwise.complete.obs")
print(round(corr_mat, 2))

# Long-format correlation heatmap
corr_long <- as.data.frame(as.table(corr_mat)) %>%
  rename(feature_x = Var1, feature_y = Var2, correlation = Freq)

p_corr <- ggplot(corr_long, aes(x = feature_x, y = feature_y, fill = correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", correlation)), size = 2.6) +
  scale_fill_gradient2(low = "#B2182B", mid = "white", high = "#2166AC", midpoint = 0) +
  labs(
    title = "Correlation Heatmap of Numeric Features",
    x = NULL,
    y = NULL,
    fill = "r"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

save_plot(p_corr, "19_correlation_heatmap.png", width = 10, height = 8)

# Optional pairwise plot (requires GGally)
if (has_ggally) {
  p_pairs <- GGally::ggpairs(
    numeric_feats %>%
      select(paper_age, title_length, abstract_length, keyword_count,
             concept_count, citation_log, citation_per_year),
    progress = FALSE
  )

  ggsave(
    file.path(viz_dir, "20_pairwise_numeric.png"),
    plot = p_pairs,
    width = 12,
    height = 12,
    dpi = 120
  )
  ggsave(
    file.path(fig_dir, "20_pairwise_numeric.png"),
    plot = p_pairs,
    width = 12,
    height = 12,
    dpi = 120
  )
  message("Saved pairwise plot to visualizations/")
} else {
  message("Skipped pairwise plot (install GGally to enable it).")
}


# =============================================================================
# STEP 13 — Top papers by citations (quick qualitative check)
# =============================================================================

top_cited <- papers %>%
  arrange(desc(cited_by_count)) %>%
  select(
    title, publication_year, cited_by_count, citation_per_year,
    oa_category, keyword_count, concept_count, doi
  ) %>%
  slice_head(n = 15)

print(top_cited)

# Save a CSV snapshot for the report appendix
write_csv(top_cited, file.path(project_root, "reports", "tables", "top15_cited_papers.csv"))
message("Saved reports/tables/top15_cited_papers.csv")


# =============================================================================
# STEP 14 — EDA summary tables for the report
# =============================================================================

summary_by_year <- papers %>%
  group_by(publication_year) %>%
  summarise(
    n_papers = n(),
    median_citations = median(cited_by_count),
    median_citation_log = median(citation_log),
    median_abstract_length = median(abstract_length),
    pct_open = mean(is_open_access, na.rm = TRUE) * 100,
    .groups = "drop"
  )

print(summary_by_year)
write_csv(summary_by_year, file.path(project_root, "reports", "tables", "summary_by_year.csv"))

summary_by_oa <- papers %>%
  group_by(oa_category) %>%
  summarise(
    n_papers = n(),
    median_citations = median(cited_by_count),
    median_citation_per_year = median(citation_per_year),
    median_abstract_length = median(abstract_length),
    mean_keyword_count = mean(keyword_count),
    mean_concept_count = mean(concept_count),
    .groups = "drop"
  ) %>%
  arrange(desc(n_papers))

print(summary_by_oa)
write_csv(summary_by_oa, file.path(project_root, "reports", "tables", "summary_by_oa_category.csv"))


# =============================================================================
# STEP 15 — Interpretation checklist (fill during write-up)
# =============================================================================
#
# After reviewing the plots, answer in your Phase 1 report:
#
# 1. How are papers distributed across 2022–2025?
# 2. How skewed are citations? Did citation_log help?
# 3. Do open-access categories differ in citation impact?
# 4. Are title/abstract lengths associated with citations?
# 5. Do keyword/concept counts suggest topical richness?
# 6. What share of papers are recent (age ≤ 2)?
# 7. Which numeric features are most correlated?
# 8. Which features look promising as Phase 2 model inputs?
# 9. Are there outliers that need special handling before ML?
# 10. What biases might exist (topic, year, OA, citation truncation)?
#
# =============================================================================


message("====================================================")
message("EDA complete.")
message("Figures : ", viz_dir)
message("Also in : ", fig_dir)
message("Tables  : ", file.path(project_root, "reports", "tables"))
message("====================================================")
