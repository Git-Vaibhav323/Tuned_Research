# =============================================================================
# ResearchPilot — DA2 (ENHANCED) — M7av2: Impact-Tier Classification
# Uses enriched features (TF-IDF + domain + publisher) for impact prediction
# Verified Python baseline: AdaBoost test-F1=0.543, test-acc=0.551, ROC-AUC=0.680
# =============================================================================
suppressPackageStartupMessages({ library(dplyr); library(readr); library(ggplot2) })
set.seed(42)
cat("=== M7av2: Impact-Tier Classification (Enriched Features) ===\n\n")

root <- NULL
for (cand in c(getwd(),"E:/Tuned_Research","e:/Tuned_Research"))
  if (file.exists(file.path(cand,"data","final","final_dataset.csv"))) { root <- normalizePath(cand); break }
if (is.null(root)) root <- getwd()

out_dir  <- file.path(root,"data","ml_r")
rep_dir  <- file.path(root,"reports","tables")
fig_dir  <- file.path(root,"reports","figures","phase2_r")
thr_file <- file.path(out_dir,"impact_tier_thresholds.csv")
dir.create(rep_dir, showWarnings=FALSE, recursive=TRUE)

# Load enriched features + add impact_tier target
enrich_path <- file.path(out_dir,"enriched_features_v2.csv")
final_path  <- file.path(root,"data","final","final_dataset.csv")

cat("Loading data...\n")
df_enrich <- read_csv(enrich_path, show_col_types=FALSE)
df_final  <- read_csv(final_path,  show_col_types=FALSE)

# Merge citation_per_year back in (needed to define target)
df_enrich$citation_per_year <- df_final$citation_per_year
cat(sprintf("  %d rows × %d cols (enriched + citation_per_year)\n\n",
            nrow(df_enrich), ncol(df_enrich)))

# ── 1. Stratified split ────────────────────────────────────────────────────────
set.seed(42)
# Temporary split just to get training-set citation_per_year for thresholds
temp_idx <- sample(seq_len(nrow(df_enrich)), round(0.70*nrow(df_enrich)))
cpy_train <- df_enrich$citation_per_year[temp_idx]

# Load/compute thresholds
if (file.exists(thr_file)) {
  thr     <- read_csv(thr_file, show_col_types=FALSE)
  q_low   <- thr$value[thr$threshold=="q_low"]
  q_high  <- thr$value[thr$threshold=="q_high"]
  cat(sprintf("Loaded thresholds: q_low=%.4f  q_high=%.4f\n\n", q_low, q_high))
} else {
  q_low  <- quantile(cpy_train, 1/3, na.rm=TRUE)
  q_high <- quantile(cpy_train, 2/3, na.rm=TRUE)
  cat(sprintf("Computed thresholds: q_low=%.4f  q_high=%.4f\n\n", q_low, q_high))
}

# Add target
df_enrich$impact_tier <- factor(dplyr::case_when(
  df_enrich$citation_per_year <= q_low  ~ "low",
  df_enrich$citation_per_year <= q_high ~ "medium",
  TRUE                                   ~ "high"
), levels=c("low","medium","high"))
cat("Impact-tier distribution:\n"); print(table(df_enrich$impact_tier))

# ── 2. Feature columns (exclude OA target + citation leakage) ─────────────────
exclude_cols <- c("oa_category","oa_category_encoded","citation_per_year",
                  "cited_by_count","citation_log","citation_per_year",
                  "impact_tier")
feat_cols <- setdiff(names(df_enrich), exclude_cols)
feat_cols <- feat_cols[!grepl("^oa_cat_|^is_open_access|^has_fulltext", feat_cols)]

cat(sprintf("\nFeature columns for impact-tier: %d\n\n", length(feat_cols)))

# ── 3. Split ───────────────────────────────────────────────────────────────────
CLASSES   <- levels(df_enrich$impact_tier)
train_idx <- unlist(lapply(CLASSES, function(cls) {
  idx <- which(df_enrich$impact_tier==cls); sample(idx, round(0.70*length(idx)))
}))
remaining <- setdiff(seq_len(nrow(df_enrich)), train_idx)
val_idx   <- unlist(lapply(CLASSES, function(cls) {
  idx <- remaining[df_enrich$impact_tier[remaining]==cls]
  sample(idx, round(0.50*length(idx)))
}))
test_idx  <- setdiff(remaining, val_idx)

X_raw_tr <- as.matrix(df_enrich[train_idx, feat_cols])
X_raw_vl <- as.matrix(df_enrich[val_idx,   feat_cols])
X_raw_te <- as.matrix(df_enrich[test_idx,  feat_cols])
y_tr <- df_enrich$impact_tier[train_idx]
y_vl <- df_enrich$impact_tier[val_idx]
y_te <- df_enrich$impact_tier[test_idx]

# Impute NAs with column median
for (j in seq_len(ncol(X_raw_tr))) {
  med <- median(X_raw_tr[,j], na.rm=TRUE)
  if (is.na(med)) med <- 0
  X_raw_tr[is.na(X_raw_tr[,j]),j] <- med
  X_raw_vl[is.na(X_raw_vl[,j]),j] <- med
  X_raw_te[is.na(X_raw_te[,j]),j] <- med
}

# Scale
sc_mu <- colMeans(X_raw_tr, na.rm=TRUE)
sc_sd <- apply(X_raw_tr, 2, function(x){s<-sd(x,na.rm=TRUE); if(s<1e-10) 1 else s})
X_tr_s <- scale(X_raw_tr, sc_mu, sc_sd)
X_vl_s <- scale(X_raw_vl, sc_mu, sc_sd)
X_te_s <- scale(X_raw_te, sc_mu, sc_sd)

cat(sprintf("Split: Train=%d  Val=%d  Test=%d\n\n", nrow(X_raw_tr), nrow(X_raw_vl), nrow(X_raw_te)))

# ── 4. Eval helper ─────────────────────────────────────────────────────────────
eval_m <- function(pc, pp, y, nm, sp) {
  cls <- levels(y); pc <- factor(as.character(pc), levels=cls)
  acc <- mean(pc==y, na.rm=TRUE)
  f1s <- vapply(cls, function(c){
    tp<-sum(pc==c&y==c); fp<-sum(pc==c&y!=c); fn<-sum(pc!=c&y==c)
    p<-if((tp+fp)>0)tp/(tp+fp) else 0; r<-if((tp+fn)>0)tp/(tp+fn) else 0
    if((p+r)>0) 2*p*r/(p+r) else 0}, numeric(1))
  roc <- if(!is.null(pp)&&is.matrix(pp)&&ncol(pp)==length(cls)){
    pp[is.na(pp)] <- 1/length(cls); colnames(pp) <- cls
    mean(vapply(seq_along(cls), function(i){
      sc<-pp[,i]; bin<-as.integer(y==cls[i])
      np<-sum(bin); nn<-length(bin)-np
      if(np==0||nn==0) return(NA_real_)
      ord<-order(sc,decreasing=TRUE)
      tpr<-c(0,cumsum(bin[ord])/np,1); fpr<-c(0,cumsum(1L-bin[ord])/nn,1)
      sum(diff(fpr)*(tpr[-1]+tpr[-length(tpr)])/2)}, numeric(1)), na.rm=TRUE)
  } else NA_real_
  data.frame(model=nm, split=sp, accuracy=round(acc,4),
             f1_macro=round(mean(f1s),4), roc_auc=round(roc,4),
             stringsAsFactors=FALSE)
}

safe_fit <- function(nm, expr) {
  cat(sprintf("  %-35s ... ", nm)); flush.console()
  m <- tryCatch(expr, error=function(e){cat(sprintf("FAIL: %s\n",e$message)); NULL})
  if (!is.null(m)) cat("OK\n"); m
}

all_metrics <- list()

# ── 5. Train models ────────────────────────────────────────────────────────────
cat("--- Training Impact-Tier Classifiers ---\n")

# 1. Logistic Regression
suppressPackageStartupMessages(library(nnet))
lr_df <- as.data.frame(X_tr_s); lr_df$y_tr <- y_tr; lr_fn <- colnames(X_tr_s)
m_lr  <- safe_fit("Logistic Regression", {
  nnet::multinom(y_tr~., data=lr_df, maxit=500, trace=FALSE, MaxNWts=20000)
})
if (!is.null(m_lr)) for (sp in c("val","test")) {
  Xd <- as.data.frame(if(sp=="val")X_vl_s else X_te_s); names(Xd) <- lr_fn
  ys <- if(sp=="val")y_vl else y_te
  pc <- as.character(predict(m_lr,Xd,type="class"))
  pp <- predict(m_lr,Xd,type="probs")
  if(!is.matrix(pp)) pp <- matrix(as.numeric(pp),nrow=nrow(Xd),dimnames=list(NULL,CLASSES))
  all_metrics[[paste0("logistic_regression_",sp)]] <- eval_m(pc,pp,ys,"logistic_regression",sp)
}

# 2. Decision Tree
suppressPackageStartupMessages(library(rpart))
m_dt <- safe_fit("Decision Tree", {
  rpart::rpart(y_tr~., data=data.frame(X_raw_tr,y_tr=y_tr),
               method="class", control=rpart::rpart.control(cp=0.001,maxdepth=10))
})
if (!is.null(m_dt)) for (sp in c("val","test")) {
  Xr <- if(sp=="val")X_raw_vl else X_raw_te; ys <- if(sp=="val")y_vl else y_te
  pp <- predict(m_dt,data.frame(Xr),type="prob")
  pc <- predict(m_dt,data.frame(Xr),type="class")
  all_metrics[[paste0("decision_tree_",sp)]] <- eval_m(pc,pp,ys,"decision_tree",sp)
}

# 3. Random Forest
suppressPackageStartupMessages(library(randomForest))
m_rf <- safe_fit("Random Forest (300 trees)", {
  randomForest::randomForest(x=X_raw_tr, y=y_tr, ntree=300,
                              mtry=max(1L,floor(sqrt(ncol(X_raw_tr)))),
                              importance=TRUE, do.trace=FALSE)
})
if (!is.null(m_rf)) for (sp in c("val","test")) {
  Xr <- if(sp=="val")X_raw_vl else X_raw_te; ys <- if(sp=="val")y_vl else y_te
  pp <- predict(m_rf,Xr,type="prob"); pc <- predict(m_rf,Xr,type="class")
  all_metrics[[paste0("random_forest_",sp)]] <- eval_m(pc,pp,ys,"random_forest",sp)
}

# 4. GBM
if (requireNamespace("gbm",quietly=TRUE)) {
  suppressPackageStartupMessages(library(gbm))
  m_gbm <- safe_fit("Gradient Boosting", {
    gbm::gbm(y_tr~., data=data.frame(X_raw_tr,y_tr=y_tr),
             distribution="multinomial", n.trees=200,
             interaction.depth=4, shrinkage=0.05, verbose=FALSE)
  })
  if (!is.null(m_gbm)) for (sp in c("val","test")) {
    Xr <- if(sp=="val")X_raw_vl else X_raw_te; ys <- if(sp=="val")y_vl else y_te
    raw <- gbm::predict.gbm(m_gbm,data.frame(Xr),n.trees=200,type="response")
    pp  <- raw[,,1]; colnames(pp) <- CLASSES
    pc  <- CLASSES[apply(pp,1,which.max)]
    all_metrics[[paste0("gradient_boosting_",sp)]] <- eval_m(pc,pp,ys,"gradient_boosting",sp)
  }
}

# 5. XGBoost
if (requireNamespace("xgboost",quietly=TRUE)) {
  suppressPackageStartupMessages(library(xgboost))
  y_int <- as.integer(y_tr)-1L; n_cls <- length(CLASSES)
  m_xgb <- safe_fit("XGBoost (200 rounds)", {
    xgboost::xgb.train(
      params=list(objective="multi:softprob",num_class=n_cls,
                  max_depth=6, eta=0.1, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="mlogloss", verbose=0),
      data=xgboost::xgb.DMatrix(X_raw_tr,label=y_int), nrounds=200, verbose=0)
  })
  if (!is.null(m_xgb)) for (sp in c("val","test")) {
    Xr <- if(sp=="val")X_raw_vl else X_raw_te; ys <- if(sp=="val")y_vl else y_te
    raw <- predict(m_xgb,xgboost::xgb.DMatrix(Xr))
    pp  <- matrix(raw,ncol=n_cls,byrow=TRUE); colnames(pp) <- CLASSES
    pc  <- CLASSES[apply(pp,1,which.max)]
    all_metrics[[paste0("xgboost_",sp)]] <- eval_m(pc,pp,ys,"xgboost",sp)
  }
}

# 6. AdaBoost
if (requireNamespace("adabag",quietly=TRUE)) {
  suppressPackageStartupMessages(library(adabag))
  m_ada <- safe_fit("AdaBoost (100 iters) [CHAMPION]", {
    adabag::boosting(y_tr~., data=data.frame(X_raw_tr,y_tr=y_tr),
                     mfinal=100, coeflearn="Breiman",
                     control=rpart::rpart.control(maxdepth=4))
  })
  if (!is.null(m_ada)) for (sp in c("val","test")) {
    Xdf <- data.frame(if(sp=="val")X_raw_vl else X_raw_te)
    ys  <- if(sp=="val")y_vl else y_te
    res <- predict(m_ada, Xdf)
    pp  <- res$prob; colnames(pp) <- CLASSES
    all_metrics[[paste0("adaboost_",sp)]] <- eval_m(res$class,pp,ys,"adaboost",sp)
  }
}

# 7. SVM
suppressPackageStartupMessages(library(e1071))
m_svm <- safe_fit("SVM (RBF, cost=10)", {
  e1071::svm(x=X_tr_s, y=y_tr, kernel="radial", cost=10, probability=TRUE)
})
if (!is.null(m_svm)) for (sp in c("val","test")) {
  Xs <- if(sp=="val")X_vl_s else X_te_s; ys <- if(sp=="val")y_vl else y_te
  pr <- predict(m_svm,Xs,probability=TRUE)
  pp <- attr(pr,"probabilities")[,CLASSES,drop=FALSE]
  all_metrics[[paste0("svm_rbf_",sp)]] <- eval_m(as.character(pr),pp,ys,"svm_rbf",sp)
}

# 8. LDA
suppressPackageStartupMessages(library(MASS))
m_lda <- safe_fit("LDA", {
  MASS::lda(x=X_tr_s, grouping=y_tr)
})
if (!is.null(m_lda)) for (sp in c("val","test")) {
  Xs <- if(sp=="val")X_vl_s else X_te_s; ys <- if(sp=="val")y_vl else y_te
  res <- predict(m_lda,Xs)
  pp  <- res$posterior[,CLASSES,drop=FALSE]
  all_metrics[[paste0("lda_",sp)]] <- eval_m(res$class,pp,ys,"lda",sp)
}

# 9. Naive Bayes
m_nb <- safe_fit("Naive Bayes", { e1071::naiveBayes(X_raw_tr, y_tr) })
if (!is.null(m_nb)) for (sp in c("val","test")) {
  Xr <- if(sp=="val")X_raw_vl else X_raw_te; ys <- if(sp=="val")y_vl else y_te
  pc <- predict(m_nb,Xr,type="class"); pp <- predict(m_nb,Xr,type="raw")
  if (!is.factor(pc)) pc <- factor(pc, levels=CLASSES)
  all_metrics[[paste0("naive_bayes_",sp)]] <- eval_m(pc,pp,ys,"naive_bayes",sp)
}

# 10. MLP
suppressPackageStartupMessages(library(nnet))
m_mlp <- safe_fit("MLP (size=50)", {
  nnet::nnet(x=X_tr_s, y=nnet::class.ind(y_tr), size=50,
             maxit=500, decay=0.01, softmax=TRUE, trace=FALSE, MaxNWts=20000)
})
if (!is.null(m_mlp)) for (sp in c("val","test")) {
  Xs <- if(sp=="val")X_vl_s else X_te_s; ys <- if(sp=="val")y_vl else y_te
  pp <- predict(m_mlp,Xs,type="raw")
  if (!is.matrix(pp)) pp <- matrix(pp,nrow=nrow(Xs))
  if (ncol(pp)==length(CLASSES)) colnames(pp) <- CLASSES
  pc <- CLASSES[apply(pp,1,which.max)]
  all_metrics[[paste0("mlp_",sp)]] <- eval_m(pc,pp,ys,"mlp",sp)
}

# ── 6. Compile results ─────────────────────────────────────────────────────────
cat("\n=== Impact-Tier Results (Enriched Features) ===\n\n")
metrics_df <- do.call(rbind, all_metrics); rownames(metrics_df) <- NULL
test_m <- metrics_df[metrics_df$split=="test",]; test_m <- test_m[order(-test_m$accuracy),]

cat("TEST SET:\n")
cat(sprintf("%-30s  %7s  %7s  %7s\n","Model","Acc","F1","AUC"))
cat(paste(rep("-",57),collapse=""),"\n")
for (i in seq_len(nrow(test_m))) {
  cat(sprintf("%-30s  %7.4f  %7.4f  %s\n", test_m$model[i],
              test_m$accuracy[i], test_m$f1_macro[i],
              ifelse(is.na(test_m$roc_auc[i]),"    N/A",sprintf("%7.4f",test_m$roc_auc[i]))))
}

cat("\n--- vs Python M7 Baseline (AdaBoost) ---\n")
cat("  Python: test-F1=0.543  test-acc=0.551  ROC-AUC=0.680\n")
best_row <- test_m[1,]
cat(sprintf("  R best: test-F1=%.3f  test-acc=%.3f  ROC-AUC=%s  [%s]\n",
            best_row$f1_macro, best_row$accuracy,
            ifelse(is.na(best_row$roc_auc),"N/A",sprintf("%.3f",best_row$roc_auc)),
            best_row$model))

# ── 7. Visualisations ─────────────────────────────────────────────────────────
cat("\nGenerating figures...\n")
test_m$model_label <- factor(gsub("_"," ",test_m$model),
                              levels=gsub("_"," ",test_m$model[order(test_m$f1_macro)]))

p_imp <- ggplot(test_m, aes(x=model_label, y=accuracy)) +
  geom_col(fill="#2c7bb6", alpha=0.85) +
  geom_hline(yintercept=0.551, linetype="dashed", colour="#d7301f", linewidth=0.8) +
  annotate("text", x=Inf, y=0.551, label=" Python AdaBoost\n acc=0.551",
           hjust=1.1, vjust=-0.3, size=3, colour="#d7301f") +
  coord_flip() +
  labs(title="Impact-Tier Classification — Accuracy (Enriched Features)",
       subtitle="All classifiers on enriched TF-IDF + domain + publisher features",
       x=NULL, y="Accuracy (test set)",
       caption="Red dashed = Python M7 AdaBoost baseline (0.551)") +
  theme_minimal(base_size=11) + theme(plot.title=element_text(face="bold")) +
  scale_y_continuous(limits=c(0,1))

ggsave(file.path(fig_dir,"v2_impact_model_comparison.png"), p_imp, width=8, height=5, dpi=150)
cat("  Saved: v2_impact_model_comparison.png\n")

# Tier distribution
tier_dist <- data.frame(tier=CLASSES, count=as.integer(table(df_enrich$impact_tier)))
p_dist <- ggplot(tier_dist, aes(x=tier, y=count, fill=tier)) +
  geom_col(alpha=0.85, show.legend=FALSE) +
  scale_fill_manual(values=c("low"="#2171b5","medium"="#6baed6","high"="#d7301f")) +
  labs(title="Impact-Tier Class Distribution",
       subtitle=sprintf("q_low=%.1f  q_high=%.1f  (training-set tertiles)",q_low,q_high),
       x="Impact Tier", y="Count") +
  theme_minimal(base_size=11) + theme(plot.title=element_text(face="bold"))
ggsave(file.path(fig_dir,"v2_impact_class_distribution.png"), p_dist, width=5, height=4, dpi=150)
cat("  Saved: v2_impact_class_distribution.png\n")

# ── 8. Save ────────────────────────────────────────────────────────────────────
write_csv(metrics_df, file.path(out_dir,"impact_tier_metrics_v2.csv"))
write_csv(test_m,     file.path(rep_dir,"r_impact_leaderboard_v2.csv"))
cat(sprintf("\nSaved: impact_tier_metrics_v2.csv\nSaved: r_impact_leaderboard_v2.csv\n"))
cat("\n=== M7av2 COMPLETE ===\n")
cat(sprintf("  Best model : %s  acc=%.4f  F1=%.4f\n",
            test_m$model[1], test_m$accuracy[1], test_m$f1_macro[1]))
