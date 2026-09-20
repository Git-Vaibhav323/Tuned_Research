# =============================================================================
# ResearchPilot — Phase 2 DA2
# MILESTONE 6: Model Evaluation & Leaderboard (R)
# =============================================================================
suppressPackageStartupMessages({ library(dplyr); library(readr); library(stringr) })
cat("=== M6: Model Evaluation & Leaderboard ===\n\n")

root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research")) {
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
}
if (is.null(root)) root <- getwd()

ml_dir   <- file.path(root, "data",    "ml_r")
rep_dir  <- file.path(root, "reports", "tables")
db_path  <- file.path(root, "database","researchpilot_r.db")
dir.create(rep_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. Load ────────────────────────────────────────────────────────────────────
baseline_path <- file.path(ml_dir, "model_metrics_oa.csv")
tuning_path   <- file.path(ml_dir, "tuning_results.csv")
if (!file.exists(baseline_path)) stop("Run 05_ml_models.R first")

df_base <- read_csv(baseline_path, show_col_types = FALSE)
cat(sprintf("Baseline: %d rows, %d models\n",
            nrow(df_base), length(unique(df_base$model))))

# ── 2. Tidy baseline ───────────────────────────────────────────────────────────
base_tidy <- df_base %>%
  rename(acc  = accuracy,
         prec = precision,
         rec  = recall,
         f1   = f1_macro,
         roc  = roc_auc) %>%
  mutate(model_type  = "baseline",
         cv_f1       = NA_real_,
         best_params = "default") %>%
  select(model, split, acc, prec, rec, f1, roc, model_type, cv_f1, best_params)

# ── 3. Tidy tuned (if available) ──────────────────────────────────────────────
tune_tidy <- data.frame()
if (file.exists(tuning_path)) {
  df_tune <- read_csv(tuning_path, show_col_types = FALSE)
  cat(sprintf("Tuned  : %d rows\n", nrow(df_tune)))
  if (nrow(df_tune) > 0 && "accuracy" %in% names(df_tune)) {
    df_tune <- df_tune %>% filter(!is.na(accuracy))
    # cv_f1_macro column may or may not exist
    has_cv_f1 <- "cv_f1_macro" %in% names(df_tune)
    tune_tidy <- df_tune %>%
      mutate(
        acc        = accuracy,
        f1         = f1_macro,
        prec       = NA_real_,
        rec        = NA_real_,
        roc        = NA_real_,
        model_type = "tuned",
        cv_f1      = if (has_cv_f1) cv_f1_macro else NA_real_
      ) %>%
      select(model, split, acc, prec, rec, f1, roc, model_type, cv_f1, best_params)
  }
}

# ── 4. Combine + test-set leaderboard ─────────────────────────────────────────
all_results <- bind_rows(base_tidy, tune_tidy) %>%
  arrange(split, desc(f1))

test_board <- all_results %>%
  filter(split == "test") %>%
  arrange(desc(f1)) %>%
  mutate(rank = row_number())

cat("\n====== TEST-SET LEADERBOARD ======\n")
cat(sprintf("%-4s %-30s %-9s %7s %7s %7s %7s %7s\n",
            "Rank","Model","Type","Acc","Prec","Rec","F1","ROC-AUC"))
cat(paste(rep("-",82), collapse=""), "\n")
for (i in seq_len(nrow(test_board))) {
  r <- test_board[i,]
  cat(sprintf("%4d %-30s %-9s %7.4f %7s %7s %7.4f %7s\n",
              r$rank, r$model, r$model_type, r$acc,
              ifelse(is.na(r$prec), "   N/A", sprintf("%7.4f", r$prec)),
              ifelse(is.na(r$rec),  "   N/A", sprintf("%7.4f", r$rec)),
              r$f1,
              ifelse(is.na(r$roc),  "   N/A", sprintf("%7.4f", r$roc))))
}

# ── 5. Category winners ────────────────────────────────────────────────────────
cat("\nCategory Winners:\n")
safe_best <- function(col) {
  v <- test_board[[col]]; v[is.na(v)] <- -Inf
  test_board[which.max(v), c("model", col)]
}
bA <- safe_best("acc"); bF <- safe_best("f1"); bR <- safe_best("roc")
cat(sprintf("  Best Accuracy : %-28s %.4f\n", bA$model, bA$acc))
cat(sprintf("  Best Macro-F1 : %-28s %.4f\n", bF$model, bF$f1))
cat(sprintf("  Best ROC-AUC  : %-28s %.4f\n", bR$model, bR$roc))
cat("\n  NOTE: Different metrics have different leaders.\n")

# ── 6. Baseline vs tuned ───────────────────────────────────────────────────────
if (nrow(tune_tidy) > 0) {
  cat("\nBaseline vs Tuned (test set):\n")
  tuned_models <- unique(tune_tidy$model)
  for (tm in tuned_models) {
    bm <- str_remove(tm, "_tuned$")
    b  <- base_tidy %>% filter(model == bm, split == "test")
    t  <- tune_tidy %>% filter(model == tm, split == "test")
    if (nrow(b) == 0 || nrow(t) == 0) next
    cat(sprintf("  %-25s baseline F1=%.4f  tuned F1=%.4f  delta=%+.4f\n",
                tm, b$f1, t$f1, t$f1 - b$f1))
  }
}

# ── 7. Save CSV ────────────────────────────────────────────────────────────────
write_csv(test_board,    file.path(rep_dir, "r_model_leaderboard.csv"))
write_csv(all_results,   file.path(ml_dir,  "all_results_combined.csv"))
cat(sprintf("\nSaved: r_model_leaderboard.csv (%d models)\n", nrow(test_board)))

# ── 8. Markdown leaderboard ────────────────────────────────────────────────────
cat("Writing markdown leaderboard...\n")

n_test <- if (nrow(test_board) > 0) test_board$n[1] else 301

md_rows <- paste0(sprintf(
  "| %d | `%s` | %s | %.4f | %s | %s | **%.4f** | %s |",
  test_board$rank, test_board$model, test_board$model_type,
  test_board$acc,
  ifelse(is.na(test_board$prec), "—", sprintf("%.4f", test_board$prec)),
  ifelse(is.na(test_board$rec),  "—", sprintf("%.4f", test_board$rec)),
  test_board$f1,
  ifelse(is.na(test_board$roc),  "—", sprintf("%.4f", test_board$roc))),
  collapse = "\n")

writeLines(c(
  "# ResearchPilot DA2 — R Model Leaderboard",
  "",
  paste0("> **Task**: OA Category Classification (fully_open / partially_open / closed)  "),
  paste0("> **Test set**: ", n_test, " papers  |  **Metrics**: macro-averaged  |  **Date**: ", Sys.Date()),
  "",
  "| Rank | Model | Type | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |",
  "|------|-------|------|----------|-----------|--------|----------|---------|",
  md_rows,
  "",
  "## Notes",
  "- All metrics computed on the **held-out test set** (15% of data).",
  "- Precision / Recall / F1 are **macro-averaged** across all 3 classes.",
  "- ROC-AUC uses **one-vs-rest (OVR)** macro averaging.",
  "- `—` = metric not computed for tuned model variants.",
  "",
  "## Python Baseline Reference",
  "| Metric | Python M4/M5 | Note |",
  "|--------|-------------|------|",
  "| Best Accuracy | 0.4817 (AdaBoost) | R uses different random split |",
  "| Best Macro-F1 | 0.4517 (AdaBoost) | Expected range: 0.40–0.50 |",
  "| Best ROC-AUC  | 0.6278 (Extra Trees tuned) | Expected range: 0.55–0.65 |",
  "",
  paste0("*Generated: ", Sys.time(), "*")
), file.path(rep_dir, "r_model_leaderboard.md"))
cat("Saved: r_model_leaderboard.md\n")

# ── 9. Write to SQLite ─────────────────────────────────────────────────────────
if (requireNamespace("DBI",quietly=TRUE) && requireNamespace("RSQLite",quietly=TRUE) &&
    file.exists(db_path) && nrow(test_board) > 0) {
  suppressPackageStartupMessages({ library(DBI); library(RSQLite) })
  tryCatch({
    con    <- dbConnect(RSQLite::SQLite(), db_path)
    db_out <- data.frame(
      result_id       = seq_len(nrow(test_board)),
      model_name      = test_board$model,
      task            = "oa_category_classification",
      split           = "test",
      accuracy        = test_board$acc,
      precision_macro = test_board$prec,
      recall_macro    = test_board$rec,
      f1_macro        = test_board$f1,
      roc_auc         = test_board$roc,
      run_date        = as.character(Sys.Date()),
      notes           = paste0("type=", test_board$model_type)
    )
    dbWriteTable(con, "model_results", db_out, overwrite = TRUE, row.names = FALSE)
    cat(sprintf("SQLite model_results: %d rows written.\n", nrow(db_out)))
    dbDisconnect(con)
  }, error = function(e) cat(sprintf("SQLite skipped: %s\n", e$message)))
}

cat("\n=== M6 COMPLETE ===\n")
cat(sprintf("  Models in leaderboard: %d\n", nrow(test_board)))
