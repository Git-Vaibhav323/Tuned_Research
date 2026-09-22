# =============================================================================
# ResearchPilot — DA2 (ENHANCED) — M5v2: Hyperparameter Tuning
# Tunes RF, XGBoost, SVM on the enriched 80-feature set
# =============================================================================
suppressPackageStartupMessages({ library(dplyr); library(readr) })
set.seed(42)
cat("=== M5v2: Hyperparameter Tuning (Enriched Features) ===\n\n")

root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research"))
  if (file.exists(file.path(cand,"data","final","final_dataset.csv"))) { root <- normalizePath(cand); break }
if (is.null(root)) root <- getwd()

splits <- readRDS(file.path(root,"data","ml_r","ml_data_splits_v2.rds"))
X_tr <- splits$X_train;  X_vl <- splits$X_val;  X_te <- splits$X_test
X_tr_s <- splits$X_train_s; X_vl_s <- splits$X_val_s; X_te_s <- splits$X_test_s
y_tr <- splits$y_train;  y_vl <- splits$y_val;  y_te <- splits$y_test
CLASSES <- levels(y_tr)

out_dir <- file.path(root,"data","ml_r")
rep_dir <- file.path(root,"reports","tables")
dir.create(rep_dir, showWarnings=FALSE, recursive=TRUE)

cat(sprintf("Train: %d | Val: %d | Test: %d | Features: %d\n\n",
            nrow(X_tr), nrow(X_vl), nrow(X_te), ncol(X_tr)))

# Eval helper
eval_simple <- function(pc, y) {
  pc  <- factor(as.character(pc), levels=levels(y))
  acc <- mean(pc==y, na.rm=TRUE)
  f1s <- vapply(levels(y), function(c) {
    tp<-sum(pc==c&y==c); fp<-sum(pc==c&y!=c); fn<-sum(pc!=c&y==c)
    p<-if((tp+fp)>0)tp/(tp+fp) else 0; r<-if((tp+fn)>0)tp/(tp+fn) else 0
    if((p+r)>0) 2*p*r/(p+r) else 0
  }, numeric(1))
  list(accuracy=round(acc,4), f1_macro=round(mean(f1s),4))
}

# 5-fold CV indices on training set
fold_ids <- sample(rep(1:5, length.out=nrow(X_tr)))

cv_rf <- function(ntree, mtry) {
  f1s <- vapply(1:5, function(k) {
    tr_k <- which(fold_ids!=k); val_k <- which(fold_ids==k)
    suppressPackageStartupMessages(library(randomForest))
    m  <- tryCatch(randomForest::randomForest(x=X_tr[tr_k,], y=y_tr[tr_k],
                                              ntree=ntree, mtry=mtry, do.trace=FALSE),
                   error=function(e) NULL)
    if (is.null(m)) return(NA_real_)
    eval_simple(predict(m, X_tr[val_k,], type="class"), y_tr[val_k])$f1_macro
  }, numeric(1))
  mean(f1s, na.rm=TRUE)
}

# ── Tune Random Forest ────────────────────────────────────────────────────────
cat("--- Tuning: Random Forest ---\n")
suppressPackageStartupMessages(library(randomForest))

rf_grid <- expand.grid(ntree=c(300,500,800), mtry=c(5,8,12,15))
cat(sprintf("  Grid: %d combinations × 5 folds\n", nrow(rf_grid)))

rf_results <- lapply(seq_len(nrow(rf_grid)), function(i) {
  nt <- rf_grid$ntree[i]; mt <- rf_grid$mtry[i]
  cat(sprintf("    ntree=%d mtry=%d ... ", nt, mt)); flush.console()
  cv_f1 <- cv_rf(nt, mt)
  cat(sprintf("CV F1=%.4f\n", cv_f1))
  data.frame(model="random_forest", ntree=nt, mtry=mt, cv_f1=round(cv_f1,4))
})
rf_cv_df  <- do.call(rbind, rf_results)
best_rf   <- rf_cv_df[which.max(rf_cv_df$cv_f1), ]
cat(sprintf("  Best: ntree=%d, mtry=%d, CV-F1=%.4f\n\n",
            best_rf$ntree, best_rf$mtry, best_rf$cv_f1))

# Refit best RF on full train
rf_tuned <- randomForest::randomForest(
  x=X_tr, y=y_tr, ntree=best_rf$ntree, mtry=best_rf$mtry,
  importance=TRUE, do.trace=FALSE)

tuning_rows <- list()
for (sp in c("val","test")) {
  X_s <- if(sp=="val") X_vl else X_te; y_s <- if(sp=="val") y_vl else y_te
  pc  <- predict(rf_tuned, X_s, type="class")
  m   <- eval_simple(pc, y_s)
  tuning_rows[[paste0("random_forest_tuned_",sp)]] <- data.frame(
    model="random_forest_tuned", split=sp,
    best_params=sprintf("ntree=%d,mtry=%d",best_rf$ntree,best_rf$mtry),
    cv_f1_macro=best_rf$cv_f1, accuracy=m$accuracy, f1_macro=m$f1_macro)
}
saveRDS(rf_tuned, file.path(out_dir,"rf_tuned_v2.rds"))
cat(sprintf("  Saved: rf_tuned_v2.rds\n"))

# ── Tune XGBoost ──────────────────────────────────────────────────────────────
cat("\n--- Tuning: XGBoost ---\n")
suppressPackageStartupMessages(library(xgboost))
y_int <- as.integer(y_tr)-1L; n_cls <- length(CLASSES)
dtrain <- xgboost::xgb.DMatrix(X_tr, label=y_int)

xgb_grid <- expand.grid(max_depth=c(4,6,8), eta=c(0.05,0.10), nrounds=c(150,300))
cat(sprintf("  Grid: %d combinations\n", nrow(xgb_grid)))

xgb_results <- lapply(seq_len(nrow(xgb_grid)), function(i) {
  md<-xgb_grid$max_depth[i]; et<-xgb_grid$eta[i]; nr<-xgb_grid$nrounds[i]
  cat(sprintf("    depth=%d eta=%.2f rounds=%d ... ", md, et, nr)); flush.console()
  cv_res <- tryCatch(xgboost::xgb.cv(
    params=list(objective="multi:softprob", num_class=n_cls,
                max_depth=md, eta=et, subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", verbose=0),
    data=dtrain, nrounds=nr, nfold=5, verbose=0, showsd=FALSE),
    error=function(e) NULL)
  logloss <- if(!is.null(cv_res)) min(cv_res$evaluation_log$test_mlogloss_mean,na.rm=TRUE) else 99
  cat(sprintf("logloss=%.4f\n", logloss))
  data.frame(model="xgboost", max_depth=md, eta=et, nrounds=nr, cv_logloss=round(logloss,4))
})
xgb_cv_df <- do.call(rbind, xgb_results)
best_xgb  <- xgb_cv_df[which.min(xgb_cv_df$cv_logloss), ]
cat(sprintf("  Best: depth=%d eta=%.2f rounds=%d logloss=%.4f\n\n",
            best_xgb$max_depth, best_xgb$eta, best_xgb$nrounds, best_xgb$cv_logloss))

xgb_tuned <- xgboost::xgb.train(
  params=list(objective="multi:softprob", num_class=n_cls,
              max_depth=best_xgb$max_depth, eta=best_xgb$eta,
              subsample=0.8, colsample_bytree=0.8,
              eval_metric="mlogloss", verbose=0),
  data=dtrain, nrounds=best_xgb$nrounds, verbose=0)

for (sp in c("val","test")) {
  X_s <- if(sp=="val") X_vl else X_te; y_s <- if(sp=="val") y_vl else y_te
  raw <- predict(xgb_tuned, xgboost::xgb.DMatrix(X_s))
  pp  <- matrix(raw, ncol=n_cls, byrow=TRUE)
  pc  <- CLASSES[apply(pp,1,which.max)]
  m   <- eval_simple(pc, y_s)
  tuning_rows[[paste0("xgboost_tuned_",sp)]] <- data.frame(
    model="xgboost_tuned", split=sp,
    best_params=sprintf("depth=%d,eta=%.2f,rounds=%d",
                        best_xgb$max_depth,best_xgb$eta,best_xgb$nrounds),
    cv_f1_macro=NA_real_, accuracy=m$accuracy, f1_macro=m$f1_macro)
}
saveRDS(xgb_tuned, file.path(out_dir,"xgb_tuned_v2.rds"))
cat(sprintf("  Saved: xgb_tuned_v2.rds\n"))

# ── Tune SVM ──────────────────────────────────────────────────────────────────
cat("\n--- Tuning: SVM (RBF) ---\n")
suppressPackageStartupMessages(library(e1071))
svm_grid <- expand.grid(cost=c(1,10,100), gamma=c(0.001,0.01,0.1))
cat(sprintf("  Grid: %d combinations × 5 folds\n", nrow(svm_grid)))

svm_results <- lapply(seq_len(nrow(svm_grid)), function(i) {
  C <- svm_grid$cost[i]; g <- svm_grid$gamma[i]
  cat(sprintf("    cost=%.0f gamma=%.4f ... ", C, g)); flush.console()
  f1s <- vapply(1:5, function(k) {
    tr_k <- which(fold_ids!=k); vl_k <- which(fold_ids==k)
    m <- tryCatch(e1071::svm(x=X_tr_s[tr_k,], y=y_tr[tr_k],
                             kernel="radial", cost=C, gamma=g, probability=TRUE),
                  error=function(e) NULL)
    if (is.null(m)) return(NA_real_)
    pc <- as.character(predict(m, X_tr_s[vl_k,]))
    eval_simple(pc, y_tr[vl_k])$f1_macro
  }, numeric(1))
  cv_f1 <- mean(f1s, na.rm=TRUE)
  cat(sprintf("CV F1=%.4f\n", cv_f1))
  data.frame(model="svm_rbf", cost=C, gamma=g, cv_f1=round(cv_f1,4))
})
svm_cv_df <- do.call(rbind, svm_results)
best_svm  <- svm_cv_df[which.max(svm_cv_df$cv_f1), ]
cat(sprintf("  Best: cost=%.0f gamma=%.4f CV-F1=%.4f\n\n",
            best_svm$cost, best_svm$gamma, best_svm$cv_f1))

svm_tuned <- e1071::svm(x=X_tr_s, y=y_tr, kernel="radial",
                         cost=best_svm$cost, gamma=best_svm$gamma, probability=TRUE)
for (sp in c("val","test")) {
  X_s <- if(sp=="val") X_vl_s else X_te_s; y_s <- if(sp=="val") y_vl else y_te
  pr  <- predict(svm_tuned, X_s, probability=TRUE)
  pc  <- as.character(pr)
  m   <- eval_simple(pc, y_s)
  tuning_rows[[paste0("svm_tuned_",sp)]] <- data.frame(
    model="svm_tuned", split=sp,
    best_params=sprintf("cost=%.0f,gamma=%.4f",best_svm$cost,best_svm$gamma),
    cv_f1_macro=best_svm$cv_f1, accuracy=m$accuracy, f1_macro=m$f1_macro)
}
saveRDS(svm_tuned, file.path(out_dir,"svm_tuned_v2.rds"))
cat(sprintf("  Saved: svm_tuned_v2.rds\n"))

# ── Summary ────────────────────────────────────────────────────────────────────
tune_df <- do.call(rbind, Filter(function(x) "accuracy" %in% names(x), tuning_rows))
rownames(tune_df) <- NULL

cat("\n=== Tuning Summary ===\n")
cat(sprintf("%-25s %-35s %7s %7s\n","Model","Best Params","Acc","F1"))
cat(paste(rep("-",80),collapse=""),"\n")
test_tune <- tune_df[tune_df$split=="test", ]
for (i in seq_len(nrow(test_tune))) {
  cat(sprintf("%-25s %-35s %7.4f %7.4f\n",
              test_tune$model[i], test_tune$best_params[i],
              test_tune$accuracy[i], test_tune$f1_macro[i]))
}

write_csv(tune_df, file.path(out_dir,"tuning_results_v2.csv"))
write_csv(test_tune, file.path(rep_dir,"r_tuning_leaderboard_v2.csv"))
cat(sprintf("\nSaved: tuning_results_v2.csv\nSaved: r_tuning_leaderboard_v2.csv\n"))
cat("\n=== M5v2 Tuning COMPLETE ===\n")
