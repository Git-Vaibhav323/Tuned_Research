# =============================================================================
# ResearchPilot — Phase 2 DA2 (ENHANCED)
# MILESTONE 2v2: Feature Selection on Enriched Feature Matrix
# =============================================================================
# Input : data/ml_r/enriched_features_v2.csv   (2000 × 206)
# Output: data/ml_r/selected_features_v2.csv   (2000 × selected+1)
#         data/ml_r/selected_feature_names.rds  (character vector)
# Methods:
#   1. Near-zero variance filter
#   2. Correlation filter  (|r| > 0.95 — relaxed for TF-IDF columns)
#   3. Random Forest importance (top 60 features)
# Note: With 205 features, mutual information is computationally expensive.
#       We use RF importance (Gini) as the primary ranking method.
# =============================================================================
suppressPackageStartupMessages({ library(dplyr); library(readr); library(stringr) })
set.seed(42)
cat("=== M2v2: Feature Selection (Enriched) ===\n\n")

root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research"))
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) { root <- normalizePath(cand); break }
if (is.null(root)) root <- getwd()

data_in  <- file.path(root, "data", "ml_r", "enriched_features_v2.csv")
out_dir  <- file.path(root, "data", "ml_r")
fig_dir  <- file.path(root, "reports", "figures", "phase2_r")
rep_dir  <- file.path(root, "reports", "phase2_r")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(rep_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. Load ────────────────────────────────────────────────────────────────────
cat("Loading enriched features...\n")
df <- read_csv(data_in, show_col_types = FALSE)
df$oa_category <- factor(df$oa_category)
feat_cols <- setdiff(names(df), "oa_category")
cat(sprintf("  %d rows × %d candidate features\n\n", nrow(df), length(feat_cols)))

# ── 2. Stratified train split (70%) — all selection on TRAIN only ─────────────
set.seed(42)
classes   <- levels(df$oa_category)
train_idx <- unlist(lapply(classes, function(cls) {
  idx <- which(df$oa_category == cls)
  sample(idx, round(0.70 * length(idx)))
}))
df_train  <- df[train_idx, ]
cat(sprintf("Training set: %d rows\n\n", nrow(df_train)))

# ── 3. Step 1 — Near-Zero Variance ────────────────────────────────────────────
cat("Step 1: Near-Zero Variance filter...\n")
X_train <- df_train[, feat_cols, drop = FALSE]
X_train <- as.data.frame(lapply(X_train, function(x)
  if (is.logical(x)) as.integer(x) else x))

nzv_flags <- vapply(names(X_train), function(nm) {
  col <- as.numeric(X_train[[nm]]); col <- col[!is.na(col)]
  if (length(col) == 0) return(TRUE)
  if (var(col) < 1e-10)  return(TRUE)
  ft <- sort(table(col), decreasing = TRUE)
  if (length(ft) <= 1) return(TRUE)
  (ft[1] / ft[2] > 20) & (length(ft) / length(col) < 0.05)
}, logical(1))

nzv_removed <- names(X_train)[nzv_flags]
nzv_kept    <- names(X_train)[!nzv_flags]
cat(sprintf("  Removed (NZV) : %d\n", length(nzv_removed)))
cat(sprintf("  Retained      : %d\n\n", length(nzv_kept)))

# ── 4. Step 2 — Correlation filter (TF-IDF columns only, r > 0.95) ────────────
cat("Step 2: Correlation filter (TF-IDF pairs |r| > 0.95)...\n")
# Only apply within TF-IDF group — cross-group correlation is OK
# (publisher vs domain features can be correlated and both informative)
tfidf_kept    <- grep("^tf_",  nzv_kept, value = TRUE)
nontfidf_kept <- grep("^tf_",  nzv_kept, value = TRUE, invert = TRUE)

corr_remove_tfidf <- character(0)
if (length(tfidf_kept) > 1) {
  X_tfidf <- as.matrix(X_train[, tfidf_kept, drop = FALSE])
  cm  <- cor(X_tfidf, use = "pairwise.complete.obs")
  cm[upper.tri(cm, diag = TRUE)] <- NA
  hi  <- which(abs(cm) > 0.95, arr.ind = TRUE)
  if (nrow(hi) > 0) {
    for (i in seq_len(nrow(hi))) {
      row_nm <- rownames(cm)[hi[i, 1]]
      if (!row_nm %in% corr_remove_tfidf)
        corr_remove_tfidf <- c(corr_remove_tfidf, row_nm)
    }
  }
}
kept_after_corr <- c(nontfidf_kept,
                     setdiff(tfidf_kept, corr_remove_tfidf))
cat(sprintf("  TF-IDF corr-removed: %d\n", length(corr_remove_tfidf)))
cat(sprintf("  Retained            : %d\n\n", length(kept_after_corr)))

# ── 5. Step 3 — Random Forest Importance (top 80 features) ────────────────────
cat("Step 3: Random Forest importance ranking...\n")
suppressPackageStartupMessages(library(randomForest))

X_rf <- as.matrix(X_train[, kept_after_corr, drop = FALSE])
cc   <- complete.cases(X_rf)
set.seed(42)
rf_m <- randomForest(
  x = X_rf[cc, ], y = df_train$oa_category[cc],
  ntree = 300, mtry = max(1L, floor(sqrt(ncol(X_rf)))),
  importance = TRUE, do.trace = FALSE
)
imp_vals <- importance(rf_m, type = 2)[, 1]
imp_sorted <- sort(imp_vals, decreasing = TRUE)

# Keep top 80 features (or all if fewer available)
n_keep <- min(80L, length(imp_sorted))
selected_features <- names(imp_sorted)[seq_len(n_keep)]

cat(sprintf("  Top %d features selected by RF importance\n\n", n_keep))
cat("Top 20 features:\n")
for (i in seq_len(min(20L, n_keep))) {
  grp <- case_when(
    startsWith(selected_features[i], "tf_")  ~ "TF-IDF",
    startsWith(selected_features[i], "dom_") |
      startsWith(selected_features[i], "cn_") ~ "domain",
    startsWith(selected_features[i], "pub_") ~ "publisher",
    TRUE ~ "structural"
  )
  cat(sprintf("  %2d. %-35s %.4f  [%s]\n",
              i, selected_features[i], imp_sorted[i], grp))
}

# ── 6. Save selected feature matrix ───────────────────────────────────────────
cat("\nSaving selected features...\n")
df_selected <- df[, c(selected_features, "oa_category"), drop = FALSE]
write_csv(df_selected, file.path(out_dir, "selected_features_v2.csv"))
saveRDS(selected_features, file.path(out_dir, "selected_feature_names.rds"))
cat(sprintf("  Saved: selected_features_v2.csv  (%d × %d)\n",
            nrow(df_selected), ncol(df_selected)))

# ── 7. Feature group breakdown in selection ────────────────────────────────────
grp_counts <- table(case_when(
  startsWith(selected_features, "tf_")  ~ "TF-IDF",
  startsWith(selected_features, "dom_") |
    startsWith(selected_features, "cn_") ~ "domain",
  startsWith(selected_features, "pub_") ~ "publisher",
  TRUE ~ "structural"
))
cat("\nSelected feature groups:\n"); print(grp_counts)

# ── 8. Visualisation — top feature importance bar ─────────────────────────────
suppressPackageStartupMessages(library(ggplot2))
imp_df <- data.frame(
  feature   = names(imp_sorted)[seq_len(min(25L, length(imp_sorted)))],
  importance = imp_sorted[seq_len(min(25L, length(imp_sorted)))],
  group      = case_when(
    startsWith(names(imp_sorted)[seq_len(min(25L, length(imp_sorted)))], "tf_")  ~ "TF-IDF",
    startsWith(names(imp_sorted)[seq_len(min(25L, length(imp_sorted)))], "dom_") |
      startsWith(names(imp_sorted)[seq_len(min(25L, length(imp_sorted)))], "cn_")  ~ "Domain",
    startsWith(names(imp_sorted)[seq_len(min(25L, length(imp_sorted)))], "pub_") ~ "Publisher",
    TRUE ~ "Structural"
  ),
  stringsAsFactors = FALSE
)
imp_df$feature <- str_remove(imp_df$feature, "^tf_|^dom_|^pub_|^cn_")
imp_df$feature <- factor(imp_df$feature,
                          levels = imp_df$feature[order(imp_df$importance)])

p_imp <- ggplot(imp_df, aes(x = feature, y = importance, fill = group)) +
  geom_col(alpha = 0.85) +
  scale_fill_manual(
    values = c("TF-IDF" = "#4292c6", "Domain" = "#41ab5d",
               "Publisher" = "#d7301f", "Structural" = "#f16913"),
    name = "Feature Group") +
  coord_flip() +
  labs(title = "Top 25 Feature Importances — Enriched Feature Set (RF Gini)",
       subtitle = "Publisher DOI and domain indicators dominate",
       x = NULL, y = "Mean Decrease in Gini Impurity",
       caption = "Features selected for OA category classification (leakage-free)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(fig_dir, "feature_importance_enriched.png"),
       p_imp, width = 9, height = 7, dpi = 150)
cat(sprintf("Saved: feature_importance_enriched.png\n"))

# ── 9. Write feature selection report ─────────────────────────────────────────
rpt <- c(
  "# Phase 2 DA2 — Feature Selection Report v2 (Enriched Features)",
  "",
  paste0("> **Date**: ", Sys.Date(), "  |  **Script**: `r/scripts/phase2/02_feature_selection_v2.R`"),
  "",
  "## Summary",
  "",
  sprintf("| Stage | Features |"),
  sprintf("|-------|---------|"),
  sprintf("| Input (enriched matrix) | %d |", length(feat_cols)),
  sprintf("| After NZV filter | %d |", length(nzv_kept)),
  sprintf("| After correlation filter | %d |", length(kept_after_corr)),
  sprintf("| Final (RF top-80) | %d |", n_keep),
  "",
  "## Selected Feature Groups",
  "",
  paste0("| Group | Count |"),
  paste0("|-------|-------|"),
  paste0(sprintf("| %s | %d |", names(grp_counts), as.integer(grp_counts)),
         collapse = "\n"),
  "",
  "## Top 20 Selected Features",
  "",
  "| Rank | Feature | Importance | Group |",
  "|------|---------|------------|-------|",
  paste0(sprintf("| %d | `%s` | %.4f | %s |",
                 seq_len(min(20L, n_keep)), selected_features[seq_len(min(20L, n_keep))],
                 imp_sorted[seq_len(min(20L, n_keep))],
                 case_when(
                   startsWith(selected_features[seq_len(min(20L, n_keep))], "tf_")  ~ "TF-IDF",
                   startsWith(selected_features[seq_len(min(20L, n_keep))], "dom_") |
                     startsWith(selected_features[seq_len(min(20L, n_keep))], "cn_")  ~ "Domain",
                   startsWith(selected_features[seq_len(min(20L, n_keep))], "pub_") ~ "Publisher",
                   TRUE ~ "Structural")),
         collapse = "\n"),
  "",
  "## Why These Features Work",
  "",
  "The publisher DOI prefix features (pub_ieee, pub_mdpi, pub_elsevier etc.)",
  "are the most discriminative because OA status is largely determined by",
  "journal/publisher policy, which is encoded in the DOI prefix.",
  "Domain features (dom_medical, dom_education) capture field-level OA",
  "mandates (e.g. NIH mandate for medical research, open pedagogy in education).",
  "TF-IDF terms capture vocabulary patterns correlated with specific venues.",
  "",
  paste0("*Generated: ", Sys.time(), "*")
)
writeLines(rpt, file.path(rep_dir, "feature_selection_v2_report.md"))
cat(sprintf("Saved: feature_selection_v2_report.md\n"))

cat("\n=== M2v2 COMPLETE ===\n")
cat(sprintf("  Candidate features : %d\n", length(feat_cols)))
cat(sprintf("  Selected features  : %d\n", n_keep))
cat(sprintf("  Feature groups     : %s\n",
            paste(names(grp_counts), grp_counts, sep = "=", collapse = ", ")))
