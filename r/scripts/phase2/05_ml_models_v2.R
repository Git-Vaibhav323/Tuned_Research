# =============================================================================
# ResearchPilot — Phase 2 DA2 (ENHANCED)
# MILESTONE 4v2: ML Models on Enriched Features
# Target: 65-75% accuracy using 80 best features (pub+domain+tfidf)
# =============================================================================
suppressPackageStartupMessages({ library(dplyr); library(readr) })
set.seed(42)
cat("=== M4v2: ML Models — Enriched Features ===\n\n")

root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research"))
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) { root <- normalizePath(cand); break }
if (is.null(root)) root <- getwd()

data_in <- file.path(root, "data", "ml_r", "selected_features_v2.csv")
out_dir <- file.path(root, "data", "ml_r")
rep_dir <- file.path(root, "reports", "tables")
dir.create(rep_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. Load + split ────────────────────────────────────────────────────────────
cat("Loading enriched selected features...\n")
df <- read_csv(data_in, show_col_types = FALSE)
df$oa_category <- factor(df$oa_category)
feat_cols <- setdiff(names(df), "oa_category")
CLASSES   <- levels(df$oa_category)
cat(sprintf("  %d rows × %d features | Classes: %s\n\n",
            nrow(df), length(feat_cols), paste(CLASSES, collapse = ", ")))

set.seed(42)
train_idx <- unlist(lapply(CLASSES, function(cls) {
  idx <- which(df$oa_category == cls); sample(idx, round(0.70 * length(idx)))
}))
remaining <- setdiff(seq_len(nrow(df)), train_idx)
val_idx   <- unlist(lapply(CLASSES, function(cls) {
  idx <- remaining[df$oa_category[remaining] == cls]; sample(idx, round(0.50 * length(idx)))
}))
test_idx  <- setdiff(remaining, val_idx)

X_tr <- as.matrix(df[train_idx, feat_cols]); y_tr <- df$oa_category[train_idx]
X_vl <- as.matrix(df[val_idx,   feat_cols]); y_vl <- df$oa_category[val_idx]
X_te <- as.matrix(df[test_idx,  feat_cols]); y_te <- df$oa_category[test_idx]

cat(sprintf("Split | Train: %d  Val: %d  Test: %d\n", nrow(X_tr), nrow(X_vl), nrow(X_te)))
cat("Train:\n"); print(table(y_tr))
cat("Test:\n");  print(table(y_te)); cat("\n")

# Scale (fit on train only)
sc_mu <- colMeans(X_tr, na.rm = TRUE)
sc_sd <- apply(X_tr, 2, function(x) { s <- sd(x, na.rm=TRUE); if (s < 1e-10) 1.0 else s })
sX    <- function(X) scale(X, center = sc_mu, scale = sc_sd)
X_tr_s <- sX(X_tr); X_vl_s <- sX(X_vl); X_te_s <- sX(X_te)

# Save splits
saveRDS(list(X_train=X_tr, X_val=X_vl, X_test=X_te,
             X_train_s=X_tr_s, X_val_s=X_vl_s, X_test_s=X_te_s,
             y_train=y_tr, y_val=y_vl, y_test=y_te,
             scale_params=list(mean=sc_mu, sd=sc_sd)),
        file.path(out_dir, "ml_data_splits_v2.rds"))

# ── 2. Evaluation helpers ──────────────────────────────────────────────────────
auc_ovr <- function(pp, y, cls) {
  if (is.null(pp) || !is.matrix(pp) || ncol(pp) != length(cls)) return(NA_real_)
  pp[is.na(pp)] <- 1/length(cls)
  colnames(pp)   <- cls
  vals <- vapply(seq_along(cls), function(i) {
    sc <- pp[,i]; bin <- as.integer(y == cls[i])
    np <- sum(bin); nn <- length(bin)-np
    if (np==0||nn==0) return(NA_real_)
    ord <- order(sc, decreasing=TRUE)
    tpr <- c(0, cumsum(bin[ord])/np, 1)
    fpr <- c(0, cumsum(1L-bin[ord])/nn, 1)
    sum(diff(fpr)*(tpr[-1]+tpr[-length(tpr)])/2)
  }, numeric(1))
  mean(vals, na.rm=TRUE)
}

eval_preds <- function(pc, pp, y, model_nm, split_nm) {
  cls <- levels(y)
  pc  <- factor(as.character(pc), levels=cls)
  acc <- mean(pc==y, na.rm=TRUE)
  f1s <- vapply(cls, function(c) {
    tp<-sum(pc==c&y==c); fp<-sum(pc==c&y!=c); fn<-sum(pc!=c&y==c)
    p<-if((tp+fp)>0)tp/(tp+fp) else 0; r<-if((tp+fn)>0)tp/(tp+fn) else 0
    if((p+r)>0) 2*p*r/(p+r) else 0
  }, numeric(1))
  data.frame(model=model_nm, split=split_nm, n=length(y),
             accuracy=round(acc,4), precision=round(mean(vapply(cls, function(c){
               tp<-sum(pc==c&y==c); fp<-sum(pc==c&y!=c)
               if((tp+fp)>0)tp/(tp+fp) else 0}, numeric(1))),4),
             recall=round(mean(vapply(cls, function(c){
               tp<-sum(pc==c&y==c); fn<-sum(pc!=c&y==c)
               if((tp+fn)>0)tp/(tp+fn) else 0}, numeric(1))),4),
             f1_macro=round(mean(f1s),4),
             roc_auc=round(auc_ovr(pp, y, cls),4),
             stringsAsFactors=FALSE)
}

add_results <- function(lst, nm, fn) {
  for (sp in c("val","test")) {
    X  <- if(sp=="val") X_vl   else X_te
    Xs <- if(sp=="val") X_vl_s else X_te_s
    y  <- if(sp=="val") y_vl   else y_te
    r  <- tryCatch(fn(X, Xs, y, sp), error=function(e){
      cat(sprintf("    EVAL ERROR (%s/%s): %s\n",nm,sp,e$message)); NULL})
    if (!is.null(r)) lst[[paste0(nm,"_",sp)]] <- r
  }; lst
}

safe_train <- function(label, expr) {
  cat(sprintf("  %-42s ... ", label)); flush.console()
  m <- tryCatch(expr, error=function(e){cat(sprintf("FAIL (%s)\n",e$message)); NULL})
  if (!is.null(m)) cat("OK\n"); m
}

metrics_list  <- list()
model_objects <- list()

# ── 3. Group 1: Linear / Regularised ──────────────────────────────────────────
cat("\n--- Group 1: Linear Models ---\n")
suppressPackageStartupMessages(library(nnet))

# 1. Logistic Regression (multinomial)
lr_df <- as.data.frame(X_tr_s); lr_df$y_tr <- y_tr; lr_fn <- colnames(X_tr_s)
m_lr  <- safe_train("Logistic Regression (multinom)", {
  nnet::multinom(y_tr~., data=lr_df, MaxNWts=20000, maxit=500, trace=FALSE)
})
if (!is.null(m_lr)) {
  model_objects[["logistic_regression"]] <- m_lr
  metrics_list <- add_results(metrics_list, "logistic_regression", function(X,Xs,y,sp){
    d <- as.data.frame(Xs); names(d) <- lr_fn
    pc <- as.character(predict(m_lr, d, type="class"))
    pp <- predict(m_lr, d, type="probs")
    if (!is.matrix(pp)) pp <- matrix(as.numeric(pp), nrow=nrow(d),
                                     dimnames=list(NULL, CLASSES))
    eval_preds(pc, pp, y, "logistic_regression", sp)
  })
}

# 2. Elastic Net
if (requireNamespace("glmnet", quietly=TRUE)) {
  suppressPackageStartupMessages(library(glmnet))
  m_en <- safe_train("Elastic Net (glmnet, alpha=0.5)", {
    glmnet::cv.glmnet(X_tr_s, y_tr, family="multinomial",
                      alpha=0.5, nfolds=5, type.measure="class")
  })
  if (!is.null(m_en)) {
    model_objects[["elastic_net"]] <- m_en
    metrics_list <- add_results(metrics_list, "elastic_net", function(X,Xs,y,sp){
      pc  <- as.vector(predict(m_en, Xs, s="lambda.min", type="class"))
      ppa <- predict(m_en, Xs, s="lambda.min", type="response")
      pp  <- matrix(ppa[,,1], ncol=dim(ppa)[2], dimnames=list(NULL,dimnames(ppa)[[2]]))
      eval_preds(pc, pp, y, "elastic_net", sp)
    })
  }
}

# ── 4. Group 2: Tree Models ────────────────────────────────────────────────────
cat("\n--- Group 2: Tree Models ---\n")
suppressPackageStartupMessages(library(rpart))

# 3. Decision Tree
m_dt <- safe_train("CART Decision Tree", {
  rpart::rpart(y_tr~., data=data.frame(X_tr, y_tr=y_tr),
               method="class", control=rpart::rpart.control(cp=0.001, maxdepth=12))
})
if (!is.null(m_dt)) {
  model_objects[["decision_tree"]] <- m_dt
  metrics_list <- add_results(metrics_list, "decision_tree", function(X,Xs,y,sp){
    pp <- predict(m_dt, data.frame(X), type="prob")
    pc <- predict(m_dt, data.frame(X), type="class")
    eval_preds(pc, pp, y, "decision_tree", sp)
  })
}

# 4. Random Forest (500 trees — key model for enriched features)
suppressPackageStartupMessages(library(randomForest))
m_rf <- safe_train("Random Forest (500 trees) *** KEY MODEL ***", {
  randomForest::randomForest(
    x=X_tr, y=y_tr, ntree=500,
    mtry=max(1L, floor(sqrt(ncol(X_tr)))),
    importance=TRUE, do.trace=FALSE)
})
if (!is.null(m_rf)) {
  model_objects[["random_forest"]] <- m_rf
  saveRDS(randomForest::importance(m_rf),
          file.path(out_dir, "rf_importance_v2.rds"))
  metrics_list <- add_results(metrics_list, "random_forest", function(X,Xs,y,sp){
    pp <- predict(m_rf, X, type="prob")
    pc <- predict(m_rf, X, type="class")
    eval_preds(pc, pp, y, "random_forest", sp)
  })
}

# ── 5. Group 3: Boosting ──────────────────────────────────────────────────────
cat("\n--- Group 3: Boosting ---\n")

# 5. XGBoost (strongest model expected)
if (requireNamespace("xgboost", quietly=TRUE)) {
  suppressPackageStartupMessages(library(xgboost))
  y_int  <- as.integer(y_tr) - 1L
  n_cls  <- length(CLASSES)
  dtrain <- xgboost::xgb.DMatrix(X_tr, label=y_int)

  m_xgb <- safe_train("XGBoost (150 rounds) *** KEY MODEL ***", {
    xgboost::xgb.train(
      params=list(objective="multi:softprob", num_class=n_cls,
                  max_depth=6, eta=0.1, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=3,
                  eval_metric="mlogloss", verbose=0),
      data=dtrain, nrounds=150, verbose=0)
  })
  if (!is.null(m_xgb)) {
    model_objects[["xgboost"]] <- m_xgb
    metrics_list <- add_results(metrics_list, "xgboost", function(X,Xs,y,sp){
      raw <- predict(m_xgb, xgboost::xgb.DMatrix(X))
      pp  <- matrix(raw, ncol=n_cls, byrow=TRUE, dimnames=list(NULL,CLASSES))
      pc  <- CLASSES[apply(pp,1,which.max)]
      eval_preds(pc, pp, y, "xgboost", sp)
    })
  }
}

# 6. Gradient Boosting
if (requireNamespace("gbm", quietly=TRUE)) {
  suppressPackageStartupMessages(library(gbm))
  m_gbm <- safe_train("Gradient Boosting (gbm, 300 trees)", {
    gbm::gbm(y_tr~., data=data.frame(X_tr, y_tr=y_tr),
             distribution="multinomial", n.trees=300,
             interaction.depth=5, shrinkage=0.05,
             bag.fraction=0.8, verbose=FALSE)
  })
  if (!is.null(m_gbm)) {
    model_objects[["gradient_boosting"]] <- m_gbm
    metrics_list <- add_results(metrics_list, "gradient_boosting", function(X,Xs,y,sp){
      raw <- gbm::predict.gbm(m_gbm, data.frame(X), n.trees=300, type="response")
      pp  <- raw[,,1]; colnames(pp) <- CLASSES
      pc  <- CLASSES[apply(pp,1,which.max)]
      eval_preds(pc, pp, y, "gradient_boosting", sp)
    })
  }
}

# 7. AdaBoost
if (requireNamespace("adabag", quietly=TRUE)) {
  suppressPackageStartupMessages(library(adabag))
  m_ada <- safe_train("AdaBoost (100 iters)", {
    adabag::boosting(y_tr~., data=data.frame(X_tr, y_tr=y_tr),
                     mfinal=100, coeflearn="Breiman",
                     control=rpart::rpart.control(maxdepth=4))
  })
  if (!is.null(m_ada)) {
    model_objects[["adaboost"]] <- m_ada
    metrics_list <- add_results(metrics_list, "adaboost", function(X,Xs,y,sp){
      res <- predict(m_ada, data.frame(X))
      pp  <- res$prob; colnames(pp) <- CLASSES
      eval_preds(res$class, pp, y, "adaboost", sp)
    })
  }
}

# ── 6. Group 4: SVM ───────────────────────────────────────────────────────────
cat("\n--- Group 4: SVM ---\n")
suppressPackageStartupMessages(library(e1071))

# 8. SVM RBF
m_svm_rbf <- safe_train("SVM (RBF kernel)", {
  e1071::svm(x=X_tr_s, y=y_tr, kernel="radial", cost=10, probability=TRUE)
})
if (!is.null(m_svm_rbf)) {
  model_objects[["svm_rbf"]] <- m_svm_rbf
  metrics_list <- add_results(metrics_list, "svm_rbf", function(X,Xs,y,sp){
    pr  <- predict(m_svm_rbf, Xs, probability=TRUE)
    pp  <- attr(pr,"probabilities")[,CLASSES,drop=FALSE]
    eval_preds(as.character(pr), pp, y, "svm_rbf", sp)
  })
}

# 9. SVM Linear
m_svm_lin <- safe_train("SVM (Linear kernel)", {
  e1071::svm(x=X_tr_s, y=y_tr, kernel="linear", cost=1, probability=TRUE)
})
if (!is.null(m_svm_lin)) {
  model_objects[["svm_linear"]] <- m_svm_lin
  metrics_list <- add_results(metrics_list, "svm_linear", function(X,Xs,y,sp){
    pr  <- predict(m_svm_lin, Xs, probability=TRUE)
    pp  <- attr(pr,"probabilities")[,CLASSES,drop=FALSE]
    eval_preds(as.character(pr), pp, y, "svm_linear", sp)
  })
}

# ── 7. Group 5: Probabilistic ─────────────────────────────────────────────────
cat("\n--- Group 5: Probabilistic ---\n")

# 10. Naive Bayes
m_nb <- safe_train("Naive Bayes", {
  e1071::naiveBayes(X_tr, y_tr)
})
if (!is.null(m_nb)) {
  model_objects[["naive_bayes"]] <- m_nb
  metrics_list <- add_results(metrics_list, "naive_bayes", function(X,Xs,y,sp){
    pc <- predict(m_nb, X, type="class")
    pp <- predict(m_nb, X, type="raw")
    if (!is.factor(pc)) pc <- factor(pc, levels=CLASSES)
    eval_preds(pc, pp, y, "naive_bayes", sp)
  })
}

# ── 8. Group 6: LDA ───────────────────────────────────────────────────────────
cat("\n--- Group 6: Discriminant Analysis ---\n")
suppressPackageStartupMessages(library(MASS))

m_lda <- safe_train("LDA", {
  MASS::lda(x=X_tr_s, grouping=y_tr)
})
if (!is.null(m_lda)) {
  model_objects[["lda"]] <- m_lda
  metrics_list <- add_results(metrics_list, "lda", function(X,Xs,y,sp){
    res <- predict(m_lda, Xs)
    pp  <- res$posterior[,CLASSES,drop=FALSE]
    eval_preds(res$class, pp, y, "lda", sp)
  })
}

# ── 9. Group 7: KNN + MLP ─────────────────────────────────────────────────────
cat("\n--- Group 7: Instance-Based + Neural Network ---\n")

# 12. KNN
if (requireNamespace("kknn", quietly=TRUE)) {
  suppressPackageStartupMessages(library(kknn))
  cat(sprintf("  %-42s ... ", "KNN (k=7)")); flush.console()
  knn_ok <- TRUE
  for (sp in c("val","test")) {
    tryCatch({
      tr_df  <- as.data.frame(X_tr_s); tr_df$y_tr <- y_tr
      tst_df <- as.data.frame(if(sp=="val") X_vl_s else X_te_s)
      names(tst_df) <- colnames(X_tr_s)
      y_s <- if(sp=="val") y_vl else y_te
      res <- kknn::kknn(y_tr~., train=tr_df, test=tst_df, k=7, kernel="rectangular")
      pp  <- res$prob; colnames(pp) <- CLASSES
      metrics_list[[paste0("knn_",sp)]] <<-
        eval_preds(as.character(fitted(res)), pp, y_s, "knn", sp)
    }, error=function(e){ cat(sprintf("FAIL(%s)\n",e$message)); knn_ok <<- FALSE })
  }
  if (knn_ok) { cat("OK\n"); model_objects[["knn"]] <- list(type="kknn",k=7) }
}

# 13. MLP
suppressPackageStartupMessages(library(nnet))
m_mlp <- safe_train("MLP Neural Network (size=100)", {
  nnet::nnet(x=X_tr_s, y=nnet::class.ind(y_tr),
             size=100, maxit=500, decay=0.01,
             softmax=TRUE, trace=FALSE, MaxNWts=50000)
})
if (!is.null(m_mlp)) {
  model_objects[["mlp"]] <- m_mlp
  metrics_list <- add_results(metrics_list, "mlp", function(X,Xs,y,sp){
    pp_r <- predict(m_mlp, Xs, type="raw")
    if (!is.matrix(pp_r)) pp_r <- matrix(pp_r, nrow=nrow(Xs))
    if (ncol(pp_r)==length(CLASSES)) colnames(pp_r) <- CLASSES
    pc <- CLASSES[apply(pp_r,1,which.max)]
    eval_preds(pc, pp_r, y, "mlp", sp)
  })
}

# ── 10. Compile results ────────────────────────────────────────────────────────
cat("\n\n=== RESULTS (Enriched Features) ===\n")
metrics_df <- do.call(rbind, metrics_list); rownames(metrics_df) <- NULL
test_df    <- metrics_df[metrics_df$split=="test", ]
test_df    <- test_df[order(-test_df$accuracy), ]

cat(sprintf("\n%-30s %7s %7s %7s %7s %7s\n","Model","Acc","Prec","Rec","F1","AUC"))
cat(paste(rep("-",75),collapse=""),"\n")
for (i in seq_len(nrow(test_df))) {
  r <- test_df[i,]
  cat(sprintf("%-30s %7.4f %7.4f %7.4f %7.4f %s\n",
              r$model, r$accuracy, r$precision, r$recall, r$f1_macro,
              ifelse(is.na(r$roc_auc),"    N/A",sprintf("%7.4f",r$roc_auc))))
}

cat("\n--- Category Winners ---\n")
safe_w <- function(col) { v<-test_df[[col]]; v[is.na(v)]<- -Inf; test_df$model[which.max(v)] }
cat(sprintf("  Best Accuracy : %-28s %.4f\n", safe_w("accuracy"),  max(test_df$accuracy,  na.rm=TRUE)))
cat(sprintf("  Best Macro-F1 : %-28s %.4f\n", safe_w("f1_macro"),  max(test_df$f1_macro,  na.rm=TRUE)))
cat(sprintf("  Best ROC-AUC  : %-28s %.4f\n", safe_w("roc_auc"),   max(test_df$roc_auc,   na.rm=TRUE)))

cat("\n--- vs Old Baseline (9 features) ---\n")
cat("  Old best accuracy: 0.4452 | Old best F1: 0.3783\n")
cat(sprintf("  New best accuracy: %.4f | New best F1: %.4f\n",
            max(test_df$accuracy,na.rm=TRUE), max(test_df$f1_macro,na.rm=TRUE)))
cat(sprintf("  Accuracy improvement: +%.4f (+%.1f%%)\n",
            max(test_df$accuracy,na.rm=TRUE) - 0.4452,
            (max(test_df$accuracy,na.rm=TRUE) - 0.4452)*100))

# ── 11. Save ───────────────────────────────────────────────────────────────────
write_csv(metrics_df, file.path(out_dir, "model_metrics_v2.csv"))
write_csv(test_df,    file.path(rep_dir, "r_model_leaderboard_v2_baseline.csv"))
saveRDS(model_objects, file.path(out_dir, "model_objects_v2.rds"))
cat(sprintf("\nSaved: model_metrics_v2.csv (%d rows)\n", nrow(metrics_df)))
cat(sprintf("Saved: r_model_leaderboard_v2_baseline.csv (%d models)\n", nrow(test_df)))
cat(sprintf("Saved: model_objects_v2.rds (%d models)\n", length(model_objects)))
cat("\n=== M4v2 COMPLETE ===\n")
