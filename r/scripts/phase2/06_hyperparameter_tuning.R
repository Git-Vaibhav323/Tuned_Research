# =============================================================================
# ResearchPilot — Phase 2 DA2
# MILESTONE 5: Hyperparameter Tuning (R)
# =============================================================================
# Tunes 4 models using 5-fold cross-validation on training data ONLY.
# Models tuned: Random Forest, XGBoost, SVM-RBF, Gradient Boosting
# Output: data/ml_r/tuning_results.csv
#         data/ml_r/tuned_model_objects.rds
#         reports/tables/r_tuning_leaderboard.csv
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(readr)
})
set.seed(42)
cat("=== M5: Hyperparameter Tuning ===\n\n")

# ── 0. Paths ──────────────────────────────────────────────────────────────────
root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research")) {
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
}
if (is.null(root)) root <- getwd()

splits_path  <- file.path(root, "data", "ml_r", "ml_data_splits.rds")
metrics_path <- file.path(root, "data", "ml_r", "model_metrics_oa.csv")
out_dir      <- file.path(root, "data", "ml_r")
rep_dir      <- file.path(root, "reports", "tables")
dir.create(rep_dir, showWarnings=FALSE, recursive=TRUE)

# Check prerequisites
if (!file.exists(splits_path)) {
  stop("Run 05_ml_models.R first to generate ml_data_splits.rds")
}

# ── 1. Load data ──────────────────────────────────────────────────────────────
cat("Loading data splits from 05_ml_models.R output...\n")
splits    <- readRDS(splits_path)
X_train   <- splits$X_train;   X_train_s <- splits$X_train_s
X_val     <- splits$X_val;     X_val_s   <- splits$X_val_s
X_test    <- splits$X_test;    X_test_s  <- splits$X_test_s
y_train   <- splits$y_train;   y_val     <- splits$y_val;   y_test <- splits$y_test
scale_p   <- splits$scale_params

cat(sprintf("  Train: %d  Val: %d  Test: %d\n",
            length(y_train), length(y_val), length(y_test)))
cat(sprintf("  Features: %d | Classes: %s\n\n",
            ncol(X_train), paste(levels(y_train), collapse=", ")))

# ── 2. Evaluation helper ──────────────────────────────────────────────────────
eval_preds <- function(pred_c, pred_p, y_true) {
  acc     <- mean(pred_c == y_true, na.rm=TRUE)
  cls     <- levels(y_true)
  f1_list <- sapply(cls, function(c) {
    tp <- sum(pred_c==c & y_true==c)
    fp <- sum(pred_c==c & y_true!=c)
    fn <- sum(pred_c!=c & y_true==c)
    p  <- if ((tp+fp)>0) tp/(tp+fp) else 0
    r  <- if ((tp+fn)>0) tp/(tp+fn) else 0
    if ((p+r)>0) 2*p*r/(p+r) else 0
  })
  f1_macro <- mean(f1_list)
  list(accuracy=round(acc,4), f1_macro=round(f1_macro,4))
}

# ── 3. 5-fold CV helper ───────────────────────────────────────────────────────
cv_folds <- 5
fold_ids  <- sample(rep(1:cv_folds, length.out=length(y_train)))

cv_score <- function(train_fn, pred_fn) {
  # Returns mean macro-F1 across 5 folds
  fold_f1s <- sapply(1:cv_folds, function(k) {
    val_k   <- which(fold_ids == k)
    train_k <- which(fold_ids != k)
    model_k <- train_fn(train_k)
    if (is.null(model_k)) return(NA_real_)
    preds_k <- pred_fn(model_k, train_k, val_k)
    if (is.null(preds_k)) return(NA_real_)
    m <- eval_preds(preds_k$class, preds_k$prob, y_train[val_k])
    m$f1_macro
  })
  mean(fold_f1s, na.rm=TRUE)
}

tuning_log <- list()

# ── 4. Tune Random Forest ─────────────────────────────────────────────────────
cat("--- Tuning: Random Forest ---\n")
if (requireNamespace("randomForest", quietly=TRUE)) {
  suppressPackageStartupMessages(library(randomForest))

  rf_grid <- expand.grid(
    ntree = c(200, 500),
    mtry  = c(2, 3, 4)
  )
  cat(sprintf("  Grid: %d combinations\n", nrow(rf_grid)))

  rf_cv_results <- lapply(seq_len(nrow(rf_grid)), function(i) {
    nt <- rf_grid$ntree[i]; mt <- rf_grid$mtry[i]
    cat(sprintf("    ntree=%d mtry=%d ... ", nt, mt)); flush.console()
    cv_f1 <- cv_score(
      train_fn = function(idx) {
        tryCatch(
          randomForest(x=X_train[idx,], y=y_train[idx], ntree=nt, mtry=mt),
          error=function(e) NULL)
      },
      pred_fn  = function(m, tr_idx, val_idx) {
        pc <- predict(m, X_train[val_idx,], type="class")
        pp <- predict(m, X_train[val_idx,], type="prob")
        list(class=pc, prob=pp)
      }
    )
    cat(sprintf("CV F1=%.4f\n", ifelse(is.na(cv_f1),0,cv_f1)))
    data.frame(model="random_forest", ntree=nt, mtry=mt,
               cv_f1_macro=round(cv_f1,4), stringsAsFactors=FALSE)
  })

  rf_cv_df <- do.call(rbind, rf_cv_results)
  best_rf_params <- rf_cv_df[which.max(rf_cv_df$cv_f1_macro), ]
  cat(sprintf("  Best: ntree=%d, mtry=%d, CV-F1=%.4f\n\n",
              best_rf_params$ntree, best_rf_params$mtry,
              best_rf_params$cv_f1_macro))

  # Refit best RF on full training set
  rf_tuned <- randomForest(
    x=X_train, y=y_train,
    ntree=best_rf_params$ntree,
    mtry=best_rf_params$mtry,
    importance=TRUE)

  # Evaluate on val and test
  for (split_nm in c("val","test")) {
    X_s <- if (split_nm=="val") X_val else X_test
    y_s <- if (split_nm=="val") y_val  else y_test
    pc  <- predict(rf_tuned, X_s, type="class")
    pp  <- predict(rf_tuned, X_s, type="prob")
    m   <- eval_preds(pc, pp, y_s)
    row <- data.frame(model="random_forest_tuned", split=split_nm,
                      best_params=paste0("ntree=",best_rf_params$ntree,
                                         ",mtry=",best_rf_params$mtry),
                      cv_f1_macro=best_rf_params$cv_f1_macro,
                      accuracy=m$accuracy, f1_macro=m$f1_macro,
                      stringsAsFactors=FALSE)
    tuning_log[[paste0("rf_tuned_",split_nm)]] <- row
  }

  tuning_log[["rf_cv_grid"]] <- rf_cv_df
  saveRDS(rf_tuned, file.path(out_dir, "rf_tuned.rds"))
  cat(sprintf("  Saved: rf_tuned.rds\n"))
}

# ── 5. Tune XGBoost ───────────────────────────────────────────────────────────
cat("\n--- Tuning: XGBoost ---\n")
if (requireNamespace("xgboost", quietly=TRUE)) {
  suppressPackageStartupMessages(library(xgboost))

  xgb_grid <- expand.grid(
    max_depth = c(3, 5),
    eta       = c(0.05, 0.10),
    nrounds   = c(100, 200)
  )
  cat(sprintf("  Grid: %d combinations\n", nrow(xgb_grid)))

  y_int <- as.integer(y_train) - 1L
  n_cls <- length(levels(y_train))

  xgb_cv_results <- lapply(seq_len(nrow(xgb_grid)), function(i) {
    md <- xgb_grid$max_depth[i]; et <- xgb_grid$eta[i]; nr <- xgb_grid$nrounds[i]
    cat(sprintf("    depth=%d eta=%.2f nrounds=%d ... ", md, et, nr)); flush.console()

    # XGBoost built-in CV
    dtrain_full <- xgb.DMatrix(X_train, label=y_int)
    cv_res <- tryCatch(
      xgb.cv(
        params = list(objective="multi:softprob", num_class=n_cls,
                      max_depth=md, eta=et, eval_metric="mlogloss", verbose=0),
        data=dtrain_full, nrounds=nr, nfold=5, verbose=0, showsd=FALSE),
      error=function(e) NULL)

    cv_logloss <- if (!is.null(cv_res)) {
      min(cv_res$evaluation_log$test_mlogloss_mean, na.rm=TRUE)
    } else NA_real_

    cat(sprintf("CV-logloss=%.4f\n", ifelse(is.na(cv_logloss),99,cv_logloss)))
    data.frame(model="xgboost", max_depth=md, eta=et, nrounds=nr,
               cv_logloss=round(cv_logloss,4), stringsAsFactors=FALSE)
  })

  xgb_cv_df <- do.call(rbind, xgb_cv_results)
  best_xgb  <- xgb_cv_df[which.min(xgb_cv_df$cv_logloss), ]
  cat(sprintf("  Best: depth=%d eta=%.2f nrounds=%d CV-logloss=%.4f\n\n",
              best_xgb$max_depth, best_xgb$eta, best_xgb$nrounds, best_xgb$cv_logloss))

  # Refit
  dtrain <- xgb.DMatrix(X_train, label=y_int)
  xgb_tuned <- xgb.train(
    params  = list(objective="multi:softprob", num_class=n_cls,
                   max_depth=best_xgb$max_depth, eta=best_xgb$eta,
                   eval_metric="mlogloss", verbose=0),
    data=dtrain, nrounds=best_xgb$nrounds, verbose=0)

  for (split_nm in c("val","test")) {
    X_s <- if (split_nm=="val") X_val else X_test
    y_s <- if (split_nm=="val") y_val  else y_test
    raw <- predict(xgb_tuned, xgb.DMatrix(X_s))
    pp  <- matrix(raw, ncol=n_cls, byrow=TRUE)
    colnames(pp) <- levels(y_train)
    pc  <- factor(levels(y_train)[apply(pp,1,which.max)], levels=levels(y_train))
    m   <- eval_preds(pc, pp, y_s)
    row <- data.frame(model="xgboost_tuned", split=split_nm,
                      best_params=paste0("depth=",best_xgb$max_depth,
                                         ",eta=",best_xgb$eta,
                                         ",nr=",best_xgb$nrounds),
                      cv_f1_macro=NA, accuracy=m$accuracy, f1_macro=m$f1_macro,
                      stringsAsFactors=FALSE)
    tuning_log[[paste0("xgb_tuned_",split_nm)]] <- row
  }
  saveRDS(xgb_tuned, file.path(out_dir, "xgb_tuned.rds"))
  cat(sprintf("  Saved: xgb_tuned.rds\n"))
}

# ── 6. Tune SVM ───────────────────────────────────────────────────────────────
cat("\n--- Tuning: SVM (RBF) ---\n")
if (requireNamespace("e1071", quietly=TRUE)) {
  suppressPackageStartupMessages(library(e1071))

  svm_grid <- expand.grid(
    cost  = c(0.1, 1, 10),
    gamma = c(0.01, 0.1, 1/ncol(X_train))
  )
  cat(sprintf("  Grid: %d combinations\n", nrow(svm_grid)))

  svm_cv_results <- lapply(seq_len(nrow(svm_grid)), function(i) {
    C <- svm_grid$cost[i]; g <- svm_grid$gamma[i]
    cat(sprintf("    cost=%.1f gamma=%.4f ... ", C, g)); flush.console()

    cv_f1 <- cv_score(
      train_fn = function(idx) {
        tryCatch(
          e1071::svm(x=X_train_s[idx,], y=y_train[idx], kernel="radial",
                     cost=C, gamma=g, probability=TRUE),
          error=function(e) NULL)
      },
      pred_fn = function(m, tr_idx, val_idx) {
        pr  <- predict(m, X_train_s[val_idx,], probability=TRUE)
        pp  <- attr(pr, "probabilities")[, levels(y_train), drop=FALSE]
        pc  <- factor(as.character(pr), levels=levels(y_train))
        list(class=pc, prob=pp)
      }
    )
    cat(sprintf("CV F1=%.4f\n", ifelse(is.na(cv_f1),0,cv_f1)))
    data.frame(model="svm_rbf", cost=C, gamma=g,
               cv_f1_macro=round(cv_f1,4), stringsAsFactors=FALSE)
  })

  svm_cv_df <- do.call(rbind, svm_cv_results)
  best_svm  <- svm_cv_df[which.max(svm_cv_df$cv_f1_macro), ]
  cat(sprintf("  Best: cost=%.1f gamma=%.4f CV-F1=%.4f\n\n",
              best_svm$cost, best_svm$gamma, best_svm$cv_f1_macro))

  svm_tuned <- e1071::svm(x=X_train_s, y=y_train, kernel="radial",
                           cost=best_svm$cost, gamma=best_svm$gamma, probability=TRUE)

  for (split_nm in c("val","test")) {
    X_s <- if (split_nm=="val") X_val_s else X_test_s
    y_s <- if (split_nm=="val") y_val    else y_test
    pr  <- predict(svm_tuned, X_s, probability=TRUE)
    pp  <- attr(pr, "probabilities")[, levels(y_train), drop=FALSE]
    pc  <- factor(as.character(pr), levels=levels(y_train))
    m   <- eval_preds(pc, pp, y_s)
    row <- data.frame(model="svm_tuned", split=split_nm,
                      best_params=paste0("cost=",best_svm$cost,",gamma=",round(best_svm$gamma,4)),
                      cv_f1_macro=best_svm$cv_f1_macro, accuracy=m$accuracy, f1_macro=m$f1_macro,
                      stringsAsFactors=FALSE)
    tuning_log[[paste0("svm_tuned_",split_nm)]] <- row
  }
  saveRDS(svm_tuned, file.path(out_dir, "svm_tuned.rds"))
  cat(sprintf("  Saved: svm_tuned.rds\n"))
}

# ── 7. Compile tuning summary ─────────────────────────────────────────────────
cat("\n=== Tuning Summary ===\n")

tune_rows <- Filter(function(x) "accuracy" %in% names(x), tuning_log)
tune_df   <- do.call(rbind, tune_rows)
rownames(tune_df) <- NULL

test_tune <- tune_df %>% filter(split=="test") %>% arrange(desc(f1_macro))
cat(sprintf("\n%-25s  %-35s  %6s  %6s\n","Model","Best Params","Acc","F1"))
cat(paste(rep("-",80),collapse=""),"\n")
for (i in seq_len(nrow(test_tune))) {
  cat(sprintf("%-25s  %-35s  %.4f  %.4f\n",
              test_tune$model[i], test_tune$best_params[i],
              test_tune$accuracy[i], test_tune$f1_macro[i]))
}

# ── 8. Save outputs ───────────────────────────────────────────────────────────
write_csv(tune_df, file.path(out_dir, "tuning_results.csv"))
write_csv(test_tune, file.path(rep_dir, "r_tuning_leaderboard.csv"))
cat(sprintf("\nSaved: tuning_results.csv (%d rows)\n", nrow(tune_df)))
cat(sprintf("Saved: r_tuning_leaderboard.csv (%d rows)\n", nrow(test_tune)))

cat("\n=== M5 Tuning COMPLETE ===\n")
