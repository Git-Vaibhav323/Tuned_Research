# =============================================================================
# ResearchPilot — Phase 2 DA2
# MILESTONE 4: Machine Learning Models (R)
# Task    : OA Category Classification  (fully_open / partially_open / closed)
# Input   : data/ml_r/selected_features_oa.csv
# Outputs : data/ml_r/model_metrics_oa.csv
#           data/ml_r/model_objects_oa.rds
#           data/ml_r/ml_data_splits.rds
#           data/ml_r/rf_importance_oa.rds
#           reports/tables/r_model_leaderboard_baseline.csv
# Models  : 13 algorithms across 7 families
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(readr)
})
set.seed(42)
cat("=== M4: ML Models — OA Category Classification ===\n\n")

# ── 0. Project root ────────────────────────────────────────────────────────────
root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research")) {
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
}
if (is.null(root)) root <- getwd()

data_in <- file.path(root, "data", "ml_r", "selected_features_oa.csv")
out_dir <- file.path(root, "data", "ml_r")
rep_dir <- file.path(root, "reports", "tables")
dir.create(rep_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. Load + split ────────────────────────────────────────────────────────────
cat("Loading data...\n")
df <- read_csv(data_in, show_col_types = FALSE)
df$oa_category <- factor(df$oa_category)
feat_cols      <- setdiff(names(df), "oa_category")
CLASSES        <- levels(df$oa_category)
cat(sprintf("  %d rows × %d features | Classes: %s\n\n",
            nrow(df), length(feat_cols), paste(CLASSES, collapse = ", ")))

# Stratified 70 / 15 / 15
set.seed(42)
train_idx <- unlist(lapply(CLASSES, function(cls) {
  idx <- which(df$oa_category == cls)
  sample(idx, round(0.70 * length(idx)))
}))
remaining <- setdiff(seq_len(nrow(df)), train_idx)
val_idx   <- unlist(lapply(CLASSES, function(cls) {
  idx <- remaining[df$oa_category[remaining] == cls]
  sample(idx, round(0.50 * length(idx)))
}))
test_idx <- setdiff(remaining, val_idx)

X_tr <- as.matrix(df[train_idx, feat_cols]); y_tr <- df$oa_category[train_idx]
X_vl <- as.matrix(df[val_idx,   feat_cols]); y_vl <- df$oa_category[val_idx]
X_te <- as.matrix(df[test_idx,  feat_cols]); y_te <- df$oa_category[test_idx]

cat(sprintf("Split | Train: %d  Val: %d  Test: %d\n", nrow(X_tr), nrow(X_vl), nrow(X_te)))
cat("Train:\n"); print(table(y_tr)); cat("Test:\n"); print(table(y_te)); cat("\n")

# Scale (fit on train only)
sc_mu  <- colMeans(X_tr, na.rm = TRUE)
sc_sd  <- apply(X_tr, 2, function(x) { s <- sd(x, na.rm=TRUE); if (s < 1e-10) 1.0 else s })
scale_X <- function(X) scale(X, center = sc_mu, scale = sc_sd)
X_tr_s <- scale_X(X_tr); X_vl_s <- scale_X(X_vl); X_te_s <- scale_X(X_te)

# Save splits for downstream scripts
saveRDS(
  list(X_train=X_tr, X_val=X_vl, X_test=X_te,
       X_train_s=X_tr_s, X_val_s=X_vl_s, X_test_s=X_te_s,
       y_train=y_tr, y_val=y_vl, y_test=y_te,
       scale_params=list(mean=sc_mu, sd=sc_sd),
       train_idx=train_idx, val_idx=val_idx, test_idx=test_idx),
  file.path(out_dir, "ml_data_splits.rds"))
cat("Splits saved.\n\n")

# ── 2. Evaluation helpers ──────────────────────────────────────────────────────
# Safely ensure prob matrix has right columns and no NAs
safe_prob_matrix <- function(pp, classes) {
  if (is.null(pp)) return(NULL)
  if (is.vector(pp) && !is.list(pp)) pp <- matrix(pp, nrow = 1)
  if (!is.matrix(pp) && is.data.frame(pp)) pp <- as.matrix(pp)
  if (!is.matrix(pp)) return(NULL)
  if (ncol(pp) != length(classes)) return(NULL)
  if (is.null(colnames(pp))) colnames(pp) <- classes
  # Align columns to class order
  if (!all(classes %in% colnames(pp))) return(NULL)
  pp <- pp[, classes, drop = FALSE]
  if (anyNA(pp)) { pp[is.na(pp)] <- 1 / length(classes) }
  pp
}

compute_auc_ovr <- function(pp, y_true, classes) {
  pp <- safe_prob_matrix(pp, classes)
  if (is.null(pp)) return(NA_real_)
  auc_vals <- vapply(seq_along(classes), function(i) {
    cls   <- classes[i]
    score <- pp[, i]
    bin   <- as.integer(y_true == cls)
    n_pos <- sum(bin); n_neg <- length(bin) - n_pos
    if (n_pos == 0L || n_neg == 0L) return(NA_real_)
    ord   <- order(score, decreasing = TRUE)
    bin_o <- bin[ord]
    tp    <- cumsum(bin_o); fp <- cumsum(1L - bin_o)
    tpr   <- c(0, tp / n_pos, 1); fpr <- c(0, fp / n_neg, 1)
    sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
  }, numeric(1))
  mean(auc_vals, na.rm = TRUE)
}

evaluate_predictions <- function(pred_class, pred_prob, y_true, model_nm, split_nm) {
  classes <- levels(y_true)
  pc      <- factor(as.character(pred_class), levels = classes)
  acc     <- mean(pc == y_true, na.rm = TRUE)

  per_class <- lapply(classes, function(cls) {
    tp <- sum(pc == cls & y_true == cls, na.rm = TRUE)
    fp <- sum(pc == cls & y_true != cls, na.rm = TRUE)
    fn <- sum(pc != cls & y_true == cls, na.rm = TRUE)
    p  <- if ((tp + fp) > 0) tp / (tp + fp) else 0.0
    r  <- if ((tp + fn) > 0) tp / (tp + fn) else 0.0
    f1 <- if ((p + r)   > 0) 2 * p * r / (p + r) else 0.0
    list(p = p, r = r, f1 = f1)
  })
  macro_p  <- mean(vapply(per_class, `[[`, numeric(1), "p"))
  macro_r  <- mean(vapply(per_class, `[[`, numeric(1), "r"))
  macro_f1 <- mean(vapply(per_class, `[[`, numeric(1), "f1"))
  auc      <- compute_auc_ovr(pred_prob, y_true, classes)

  data.frame(model      = model_nm,
             split      = split_nm,
             n          = length(y_true),
             accuracy   = round(acc,      4),
             precision  = round(macro_p,  4),
             recall     = round(macro_r,  4),
             f1_macro   = round(macro_f1, 4),
             roc_auc    = round(auc,      4),
             stringsAsFactors = FALSE)
}

# Convenience: evaluate on val + test and append to list
add_results <- function(result_list, model_nm, get_preds_fn) {
  for (sp in c("val", "test")) {
    X_s  <- if (sp == "val") X_vl   else X_te
    Xs_s <- if (sp == "val") X_vl_s else X_te_s
    y_s  <- if (sp == "val") y_vl   else y_te
    res  <- tryCatch(get_preds_fn(X_s, Xs_s, y_s, sp),
                     error = function(e) {
                       cat(sprintf("    EVAL ERROR (%s/%s): %s\n",
                                   model_nm, sp, conditionMessage(e)))
                       NULL
                     })
    if (!is.null(res)) result_list[[paste0(model_nm, "_", sp)]] <- res
  }
  result_list
}

# ── 3. Model training ──────────────────────────────────────────────────────────
metrics_list  <- list()
model_objects <- list()

safe_train <- function(label, expr) {
  cat(sprintf("  %-40s ... ", label)); flush.console()
  m <- tryCatch(expr, error = function(e) {
    cat(sprintf("FAIL (%s)\n", conditionMessage(e))); NULL
  })
  if (!is.null(m)) cat("OK\n")
  m
}

# ─────────────────────────────────────────────────────────────
cat("--- Group 1: Linear / Regularised Models ---\n")

# 1. Logistic Regression (multinomial via nnet::multinom)
suppressPackageStartupMessages(library(nnet))
lr_train_df <- as.data.frame(X_tr_s)
lr_train_df$y_tr <- y_tr
m_lr <- safe_train("Logistic Regression (multinom)", {
  nnet::multinom(y_tr ~ ., data = lr_train_df,
                 MaxNWts = 5000, maxit = 500, trace = FALSE)
})
if (!is.null(m_lr)) {
  model_objects[["logistic_regression"]] <- m_lr
  lr_feat_names <- colnames(X_tr_s)
  metrics_list <- add_results(metrics_list, "logistic_regression",
    function(X, Xs, y, sp) {
      Xs_df <- as.data.frame(Xs)
      names(Xs_df) <- lr_feat_names
      pc  <- as.character(predict(m_lr, Xs_df, type = "class"))
      pp  <- predict(m_lr, Xs_df, type = "probs")
      if (!is.matrix(pp)) pp <- matrix(as.numeric(pp), nrow = length(pc),
                                       dimnames = list(NULL, CLASSES))
      evaluate_predictions(pc, pp, y, "logistic_regression", sp)
    })
}

# 2. Elastic Net Logistic Regression (glmnet)
if (requireNamespace("glmnet", quietly = TRUE)) {
  suppressPackageStartupMessages(library(glmnet))
  m_en <- safe_train("Elastic Net (alpha=0.5, cv.glmnet)", {
    glmnet::cv.glmnet(X_tr_s, y_tr, family = "multinomial",
                      alpha = 0.5, nfolds = 5, type.measure = "class")
  })
  if (!is.null(m_en)) {
    model_objects[["elastic_net"]] <- m_en
    metrics_list <- add_results(metrics_list, "elastic_net",
      function(X, Xs, y, sp) {
        pc  <- as.vector(predict(m_en, Xs, s = "lambda.min", type = "class"))
        ppa <- predict(m_en, Xs, s = "lambda.min", type = "response")
        pp  <- matrix(ppa[, , 1], ncol = dim(ppa)[2],
                      dimnames = list(NULL, dimnames(ppa)[[2]]))
        evaluate_predictions(pc, pp, y, "elastic_net", sp)
      })
  }
}

# ─────────────────────────────────────────────────────────────
cat("\n--- Group 2: Tree-Based Models ---\n")
suppressPackageStartupMessages(library(rpart))

# 3. CART Decision Tree
m_dt <- safe_train("CART Decision Tree (rpart)", {
  rpart::rpart(y_tr ~ ., data = data.frame(X_tr, y_tr = y_tr),
               method = "class",
               control = rpart::rpart.control(cp = 0.001, maxdepth = 10))
})
if (!is.null(m_dt)) {
  model_objects[["decision_tree"]] <- m_dt
  metrics_list <- add_results(metrics_list, "decision_tree",
    function(X, Xs, y, sp) {
      pp <- predict(m_dt, data.frame(X), type = "prob")
      pc <- predict(m_dt, data.frame(X), type = "class")
      evaluate_predictions(pc, pp, y, "decision_tree", sp)
    })
}

# 4. Random Forest
suppressPackageStartupMessages(library(randomForest))
m_rf <- safe_train("Random Forest (500 trees, randomForest)", {
  randomForest::randomForest(
    x = X_tr, y = y_tr, ntree = 500,
    mtry = max(1L, floor(sqrt(ncol(X_tr)))),
    importance = TRUE, do.trace = FALSE)
})
if (!is.null(m_rf)) {
  model_objects[["random_forest"]] <- m_rf
  saveRDS(randomForest::importance(m_rf),
          file.path(out_dir, "rf_importance_oa.rds"))
  metrics_list <- add_results(metrics_list, "random_forest",
    function(X, Xs, y, sp) {
      pp <- predict(m_rf, X, type = "prob")
      pc <- predict(m_rf, X, type = "class")
      evaluate_predictions(pc, pp, y, "random_forest", sp)
    })
}

# ─────────────────────────────────────────────────────────────
cat("\n--- Group 3: Boosting Models ---\n")

# 5. Gradient Boosting Machine (gbm)
if (requireNamespace("gbm", quietly = TRUE)) {
  suppressPackageStartupMessages(library(gbm))
  m_gbm <- safe_train("Gradient Boosting Machine (gbm, 300 trees)", {
    gbm::gbm(y_tr ~ ., data = data.frame(X_tr, y_tr = y_tr),
             distribution = "multinomial",
             n.trees = 300, interaction.depth = 3,
             shrinkage = 0.05, bag.fraction = 0.8, verbose = FALSE)
  })
  if (!is.null(m_gbm)) {
    model_objects[["gradient_boosting"]] <- m_gbm
    metrics_list <- add_results(metrics_list, "gradient_boosting",
      function(X, Xs, y, sp) {
        raw  <- gbm::predict.gbm(m_gbm, data.frame(X), n.trees = 300,
                                 type = "response")
        pp   <- raw[, , 1]; colnames(pp) <- CLASSES
        pc   <- CLASSES[apply(pp, 1, which.max)]
        evaluate_predictions(pc, pp, y, "gradient_boosting", sp)
      })
  }
}

# 6. XGBoost
if (requireNamespace("xgboost", quietly = TRUE)) {
  suppressPackageStartupMessages(library(xgboost))
  y_int   <- as.integer(y_tr) - 1L
  n_cls   <- length(CLASSES)
  dtrain  <- xgboost::xgb.DMatrix(X_tr, label = y_int)

  m_xgb <- safe_train("XGBoost (100 rounds)", {
    xgboost::xgb.train(
      params  = list(objective = "multi:softprob", num_class = n_cls,
                     max_depth = 4, eta = 0.1, subsample = 0.8,
                     eval_metric = "mlogloss", verbose = 0),
      data    = dtrain, nrounds = 100, verbose = 0)
  })
  if (!is.null(m_xgb)) {
    model_objects[["xgboost"]] <- m_xgb
    metrics_list <- add_results(metrics_list, "xgboost",
      function(X, Xs, y, sp) {
        raw <- predict(m_xgb, xgboost::xgb.DMatrix(X))
        pp  <- matrix(raw, ncol = n_cls, byrow = TRUE,
                      dimnames = list(NULL, CLASSES))
        pc  <- CLASSES[apply(pp, 1, which.max)]
        evaluate_predictions(pc, pp, y, "xgboost", sp)
      })
  }
}

# 7. AdaBoost (adabag)
if (requireNamespace("adabag", quietly = TRUE)) {
  suppressPackageStartupMessages(library(adabag))
  m_ada <- safe_train("AdaBoost (adabag, 100 iters)", {
    adabag::boosting(y_tr ~ ., data = data.frame(X_tr, y_tr = y_tr),
                     mfinal = 100, coeflearn = "Breiman",
                     control = rpart::rpart.control(maxdepth = 3))
  })
  if (!is.null(m_ada)) {
    model_objects[["adaboost"]] <- m_ada
    metrics_list <- add_results(metrics_list, "adaboost",
      function(X, Xs, y, sp) {
        res <- predict(m_ada, data.frame(X))
        pp  <- res$prob; colnames(pp) <- CLASSES
        pc  <- res$class
        evaluate_predictions(pc, pp, y, "adaboost", sp)
      })
  }
}

# ─────────────────────────────────────────────────────────────
cat("\n--- Group 4: Support Vector Machines ---\n")
suppressPackageStartupMessages(library(e1071))

# 8. SVM — RBF kernel
m_svm_rbf <- safe_train("SVM RBF kernel (e1071)", {
  e1071::svm(x = X_tr_s, y = y_tr, kernel = "radial",
             cost = 1, probability = TRUE)
})
if (!is.null(m_svm_rbf)) {
  model_objects[["svm_rbf"]] <- m_svm_rbf
  metrics_list <- add_results(metrics_list, "svm_rbf",
    function(X, Xs, y, sp) {
      pr  <- predict(m_svm_rbf, Xs, probability = TRUE)
      pp  <- attr(pr, "probabilities")[, CLASSES, drop = FALSE]
      evaluate_predictions(as.character(pr), pp, y, "svm_rbf", sp)
    })
}

# 9. SVM — Linear kernel
m_svm_lin <- safe_train("SVM Linear kernel (e1071)", {
  e1071::svm(x = X_tr_s, y = y_tr, kernel = "linear",
             cost = 1, probability = TRUE)
})
if (!is.null(m_svm_lin)) {
  model_objects[["svm_linear"]] <- m_svm_lin
  metrics_list <- add_results(metrics_list, "svm_linear",
    function(X, Xs, y, sp) {
      pr  <- predict(m_svm_lin, Xs, probability = TRUE)
      pp  <- attr(pr, "probabilities")[, CLASSES, drop = FALSE]
      evaluate_predictions(as.character(pr), pp, y, "svm_linear", sp)
    })
}

# ─────────────────────────────────────────────────────────────
cat("\n--- Group 5: Probabilistic Models ---\n")

# 10. Naive Bayes
m_nb <- safe_train("Naive Bayes (e1071::naiveBayes)", {
  e1071::naiveBayes(X_tr, y_tr)
})
if (!is.null(m_nb)) {
  model_objects[["naive_bayes"]] <- m_nb
  metrics_list <- add_results(metrics_list, "naive_bayes",
    function(X, Xs, y, sp) {
      pc  <- predict(m_nb, X, type = "class")
      pp  <- predict(m_nb, X, type = "raw")
      if (!is.factor(pc)) pc <- factor(pc, levels = CLASSES)
      evaluate_predictions(pc, pp, y, "naive_bayes", sp)
    })
}

# ─────────────────────────────────────────────────────────────
cat("\n--- Group 6: Discriminant Analysis ---\n")
suppressPackageStartupMessages(library(MASS))

# 11. LDA
m_lda <- safe_train("Linear Discriminant Analysis (MASS::lda)", {
  MASS::lda(x = X_tr_s, grouping = y_tr)
})
if (!is.null(m_lda)) {
  model_objects[["lda"]] <- m_lda
  metrics_list <- add_results(metrics_list, "lda",
    function(X, Xs, y, sp) {
      res <- predict(m_lda, Xs)
      pp  <- res$posterior[, CLASSES, drop = FALSE]
      evaluate_predictions(res$class, pp, y, "lda", sp)
    })
}

# ─────────────────────────────────────────────────────────────
cat("\n--- Group 7: Instance-Based + Neural Network ---\n")

# 12. K-Nearest Neighbours (kknn)
if (requireNamespace("kknn", quietly = TRUE)) {
  suppressPackageStartupMessages(library(kknn))
  cat(sprintf("  %-40s ... ", "KNN k=7 (kknn)")); flush.console()
  knn_ok <- TRUE
  for (sp in c("val", "test")) {
    tryCatch({
      Xtr_df <- as.data.frame(X_tr_s); Xtr_df$y_tr <- y_tr
      Xsp_df <- as.data.frame(if (sp=="val") X_vl_s else X_te_s)
      names(Xsp_df) <- colnames(X_tr_s)
      y_s    <- if (sp=="val") y_vl else y_te
      res_k  <- kknn::kknn(y_tr ~ ., train = Xtr_df, test = Xsp_df,
                            k = 7, kernel = "rectangular")
      pp <- res_k$prob; colnames(pp) <- CLASSES
      pc <- as.character(fitted(res_k))
      metrics_list[[paste0("knn_", sp)]] <<-
        evaluate_predictions(pc, pp, y_s, "knn", sp)
    }, error = function(e) {
      cat(sprintf("FAIL (%s)\n", conditionMessage(e))); knn_ok <<- FALSE
    })
  }
  if (knn_ok) {
    cat("OK\n")
    model_objects[["knn"]] <- list(type="kknn", k=7)  # store params (no model object for kknn)
  }
}

# 13. MLP Neural Network (nnet)
m_mlp <- safe_train("MLP Neural Network (nnet, size=50)", {
  nnet::nnet(x = X_tr_s,
             y = nnet::class.ind(y_tr),
             size    = 50,
             maxit   = 500,
             decay   = 0.01,
             softmax = TRUE,
             trace   = FALSE,
             MaxNWts = 10000)
})
if (!is.null(m_mlp)) {
  model_objects[["mlp"]] <- m_mlp
  metrics_list <- add_results(metrics_list, "mlp",
    function(X, Xs, y, sp) {
      pp_raw <- predict(m_mlp, Xs, type = "raw")
      # Ensure named matrix
      if (!is.matrix(pp_raw)) pp_raw <- matrix(pp_raw, nrow = nrow(Xs))
      if (is.null(colnames(pp_raw)) || ncol(pp_raw) != length(CLASSES))
        colnames(pp_raw) <- CLASSES
      pc <- CLASSES[apply(pp_raw, 1, which.max)]
      evaluate_predictions(pc, pp_raw, y, "mlp", sp)
    })
}

# ── 4. Compile leaderboard ─────────────────────────────────────────────────────
cat("\n\n=== RESULTS ===\n")
metrics_df <- do.call(rbind, metrics_list)
rownames(metrics_df) <- NULL

val_df  <- metrics_df[metrics_df$split == "val",  ]
test_df <- metrics_df[metrics_df$split == "test", ]
test_df <- test_df[order(-test_df$f1_macro), ]

cat(sprintf("\nModels evaluated: %d\n", length(model_objects)))
cat("\nTEST SET LEADERBOARD:\n")
cat(sprintf("%-30s %7s %7s %7s %7s %7s\n",
            "Model","Acc","Prec","Rec","F1","AUC"))
cat(paste(rep("-", 75), collapse = ""), "\n")
for (i in seq_len(nrow(test_df))) {
  r <- test_df[i, ]
  cat(sprintf("%-30s %7.4f %7.4f %7.4f %7.4f %s\n",
              r$model, r$accuracy, r$precision, r$recall, r$f1_macro,
              ifelse(is.na(r$roc_auc), "    N/A", sprintf("%7.4f", r$roc_auc))))
}

# Best per metric
if (nrow(test_df) > 0) {
  cat("\nCategory Winners (Test Set):\n")
  safe_winner <- function(col) {
    v <- test_df[[col]]; v[is.na(v)] <- -Inf
    test_df$model[which.max(v)]
  }
  cat(sprintf("  Best Accuracy  : %-28s %.4f\n", safe_winner("accuracy"),
              max(test_df$accuracy, na.rm=TRUE)))
  cat(sprintf("  Best Macro-F1  : %-28s %.4f\n", safe_winner("f1_macro"),
              max(test_df$f1_macro,  na.rm=TRUE)))
  auc_vals <- test_df$roc_auc; auc_vals[is.na(auc_vals)] <- -Inf
  cat(sprintf("  Best ROC-AUC   : %-28s %.4f\n", safe_winner("roc_auc"),
              max(test_df$roc_auc, na.rm=TRUE)))
}

# ── 5. Save ────────────────────────────────────────────────────────────────────
write_csv(metrics_df, file.path(out_dir, "model_metrics_oa.csv"))
write_csv(test_df,    file.path(rep_dir, "r_model_leaderboard_baseline.csv"))
saveRDS(model_objects, file.path(out_dir, "model_objects_oa.rds"))
cat(sprintf("\nSaved model_metrics_oa.csv       (%d rows)\n", nrow(metrics_df)))
cat(sprintf("Saved r_model_leaderboard_baseline.csv (%d models)\n", nrow(test_df)))
cat(sprintf("Saved model_objects_oa.rds        (%d models)\n", length(model_objects)))
cat("\n=== M4 COMPLETE ===\n")
