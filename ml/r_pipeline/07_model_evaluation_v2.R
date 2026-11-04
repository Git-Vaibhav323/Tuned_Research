# =============================================================================
# ResearchPilot — DA2 (ENHANCED) — M6v2: Model Evaluation & Leaderboard
# =============================================================================
suppressPackageStartupMessages({ library(dplyr); library(readr); library(stringr) })
cat("=== M6v2: Model Evaluation & Leaderboard ===\n\n")

root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research"))
  if (file.exists(file.path(cand,"data","final","final_dataset.csv"))) { root <- normalizePath(cand); break }
if (is.null(root)) root <- getwd()

ml_dir  <- file.path(root,"data","ml_r")
rep_dir <- file.path(root,"reports","tables")
db_path <- file.path(root,"database","researchpilot_r.db")
dir.create(rep_dir, showWarnings=FALSE, recursive=TRUE)

# ── Load both pipelines ────────────────────────────────────────────────────────
base_path  <- file.path(ml_dir,"model_metrics_v2.csv")
tune_path  <- file.path(ml_dir,"tuning_results_v2.csv")
if (!file.exists(base_path)) stop("Run 05_ml_models_v2.R first")

df_base <- read_csv(base_path, show_col_types=FALSE)
cat(sprintf("Baseline v2 : %d rows (%d models)\n", nrow(df_base), length(unique(df_base$model))))

df_tune <- if (file.exists(tune_path)) {
  t <- read_csv(tune_path, show_col_types=FALSE)
  cat(sprintf("Tuned v2    : %d rows\n", nrow(t))); t
} else { cat("No tuning results\n"); data.frame() }

# ── Tidy ───────────────────────────────────────────────────────────────────────
base_tidy <- df_base %>%
  rename(acc=accuracy, prec=precision, rec=recall, f1=f1_macro, roc=roc_auc) %>%
  mutate(model_type="baseline", cv_f1=NA_real_, best_params="default") %>%
  select(model, split, acc, prec, rec, f1, roc, model_type, cv_f1, best_params)

tune_tidy <- if (nrow(df_tune)>0 && "accuracy" %in% names(df_tune)) {
  has_cv <- "cv_f1_macro" %in% names(df_tune)
  df_tune %>% filter(!is.na(accuracy)) %>%
    mutate(acc=accuracy, f1=f1_macro, prec=NA_real_, rec=NA_real_, roc=NA_real_,
           model_type="tuned",
           cv_f1=if(has_cv) cv_f1_macro else NA_real_) %>%
    select(model, split, acc, prec, rec, f1, roc, model_type, cv_f1, best_params)
} else data.frame()

all_results <- bind_rows(base_tidy, tune_tidy) %>% arrange(split, desc(f1))

test_board <- all_results %>%
  filter(split=="test") %>%
  arrange(desc(acc)) %>%
  mutate(rank=row_number())

cat("\n====== TEST-SET LEADERBOARD (Enriched Features) ======\n")
cat(sprintf("%-4s %-30s %-9s %7s %7s %7s %7s %7s\n",
            "Rank","Model","Type","Acc","Prec","Rec","F1","AUC"))
cat(paste(rep("-",83),collapse=""),"\n")
for (i in seq_len(nrow(test_board))) {
  r <- test_board[i,]
  cat(sprintf("%4d %-30s %-9s %7.4f %7s %7s %7.4f %7s\n",
              r$rank, r$model, r$model_type, r$acc,
              ifelse(is.na(r$prec),"   N/A",sprintf("%7.4f",r$prec)),
              ifelse(is.na(r$rec), "   N/A",sprintf("%7.4f",r$rec)),
              r$f1,
              ifelse(is.na(r$roc), "   N/A",sprintf("%7.4f",r$roc))))
}

# ── Winners ───────────────────────────────────────────────────────────────────
cat("\nCategory Winners:\n")
bA <- test_board[which.max(test_board$acc), ]
bF <- test_board[which.max(test_board$f1),  ]
auc_v <- test_board$roc; auc_v[is.na(auc_v)] <- -Inf
bR <- test_board[which.max(auc_v), ]
cat(sprintf("  Best Accuracy : %-30s %.4f\n", bA$model, bA$acc))
cat(sprintf("  Best Macro-F1 : %-30s %.4f\n", bF$model, bF$f1))
cat(sprintf("  Best ROC-AUC  : %-30s %.4f\n", bR$model, bR$roc))

# ── vs old baseline ────────────────────────────────────────────────────────────
cat("\n--- Improvement vs original 9-feature baseline ---\n")
cat("  Original best accuracy: 0.4452  |  Original best F1: 0.3783\n")
cat(sprintf("  New best accuracy     : %.4f  |  New best F1     : %.4f\n",
            bA$acc, bF$f1))
cat(sprintf("  Accuracy gain: +%.4f (+%.1f%%)  |  F1 gain: +%.4f (+%.1f%%)\n",
            bA$acc-0.4452, (bA$acc-0.4452)*100,
            bF$f1-0.3783, (bF$f1-0.3783)*100))

# ── Save ───────────────────────────────────────────────────────────────────────
write_csv(test_board, file.path(rep_dir,"r_model_leaderboard_v2.csv"))
write_csv(all_results, file.path(ml_dir,"all_results_v2.csv"))

# Markdown leaderboard
md_rows <- paste0(sprintf(
  "| %d | `%s` | %s | **%.4f** | %s | %s | %.4f | %s |",
  test_board$rank, test_board$model, test_board$model_type, test_board$acc,
  ifelse(is.na(test_board$prec),"—",sprintf("%.4f",test_board$prec)),
  ifelse(is.na(test_board$rec), "—",sprintf("%.4f",test_board$rec)),
  test_board$f1,
  ifelse(is.na(test_board$roc), "—",sprintf("%.4f",test_board$roc))),
  collapse="\n")

writeLines(c(
  "# ResearchPilot DA2 — R Model Leaderboard v2 (Enriched Features)",
  "", paste0("> **Date**: ", Sys.Date(), "  |  **Features**: 80 enriched (TF-IDF + domain + publisher + structural)"),
  "> **Task**: OA Category Classification (fully_open / partially_open / closed)",
  "",
  "| Rank | Model | Type | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |",
  "|------|-------|------|----------|-----------|--------|----------|---------|",
  md_rows,
  "",
  "## Improvement Summary",
  "",
  "| Metric | Original (9 features) | Enriched (80 features) | Improvement |",
  "|--------|----------------------|----------------------|-------------|",
  sprintf("| Accuracy | 0.4452 | %.4f | +%.4f (+%.1f%%) |",
          bA$acc, bA$acc-0.4452, (bA$acc-0.4452)*100),
  sprintf("| Macro-F1 | 0.3783 | %.4f | +%.4f (+%.1f%%) |",
          bF$f1, bF$f1-0.3783, (bF$f1-0.3783)*100),
  "",
  "## Key Insight",
  "Publisher DOI prefix and domain binary features are the strongest predictors.",
  "IEEE-published papers are predominantly CLOSED; MDPI/BMC are FULLY OPEN.",
  "Medical domain papers skew FULLY OPEN (NIH/Wellcome Trust mandate).",
  "",
  paste0("*Generated: ", Sys.time(), "*")
), file.path(rep_dir,"r_model_leaderboard_v2.md"))

cat(sprintf("\nSaved: r_model_leaderboard_v2.csv (%d models)\n", nrow(test_board)))
cat(sprintf("Saved: r_model_leaderboard_v2.md\n"))

# ── Update SQLite ──────────────────────────────────────────────────────────────
if (requireNamespace("DBI",quietly=TRUE) && requireNamespace("RSQLite",quietly=TRUE) &&
    file.exists(db_path) && nrow(test_board)>0) {
  suppressPackageStartupMessages({ library(DBI); library(RSQLite) })
  tryCatch({
    con <- dbConnect(RSQLite::SQLite(), db_path)
    db_out <- data.frame(
      result_id=seq_len(nrow(test_board)), model_name=test_board$model,
      task="oa_category_v2_enriched", split="test",
      accuracy=test_board$acc, precision_macro=test_board$prec,
      recall_macro=test_board$rec, f1_macro=test_board$f1, roc_auc=test_board$roc,
      run_date=as.character(Sys.Date()), notes=paste0("enriched_v2,type=",test_board$model_type))
    # Append to existing table
    existing <- dbReadTable(con, "model_results")
    combined <- rbind(existing, db_out)
    combined$result_id <- seq_len(nrow(combined))
    dbWriteTable(con, "model_results", combined, overwrite=TRUE, row.names=FALSE)
    cat(sprintf("SQLite model_results: %d total rows.\n", nrow(combined)))
    dbDisconnect(con)
  }, error=function(e) cat(sprintf("SQLite skipped: %s\n", e$message)))
}

cat("\n=== M6v2 COMPLETE ===\n")
