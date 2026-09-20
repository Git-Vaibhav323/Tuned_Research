# =============================================================================
# ResearchPilot — Phase 2 DA2
# MILESTONE 7a: Impact-Tier Classification (R)
# =============================================================================
# Target   : impact_tier (low / medium / high) based on citation_per_year
# Thresholds: calculated from training data (q33/q67 of citation_per_year)
# Verified baselines:
#   AdaBoost val-F1=0.600, test-F1=0.543, test-acc=0.551, test-ROC-AUC=0.680
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(ggplot2)
})
set.seed(42)
cat("=== M7a: Impact-Tier Classification ===\n\n")

# ── 0. Paths ──────────────────────────────────────────────────────────────────
root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research")) {
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
}
if (is.null(root)) root <- getwd()

data_in  <- file.path(root, "data", "ml_r", "selected_features_impact.csv")
thr_file <- file.path(root, "data", "ml_r", "impact_tier_thresholds.csv")
out_dir  <- file.path(root, "data", "ml_r")
rep_dir  <- file.path(root, "reports", "tables")
fig_dir  <- file.path(root, "reports", "figures", "phase2_r")
dir.create(rep_dir, showWarnings=FALSE, recursive=TRUE)

# ── 1. Load data ──────────────────────────────────────────────────────────────
cat("Loading impact-tier feature matrix...\n")
df <- read_csv(data_in, show_col_types=FALSE)
df$impact_tier <- factor(df$impact_tier, levels=c("low","medium","high"))
cat(sprintf("  %d rows × %d cols | Classes: %s\n",
            nrow(df), ncol(df), paste(levels(df$impact_tier), collapse=", ")))
print(table(df$impact_tier))
cat("\n")

# Load thresholds
thr <- tryCatch(read_csv(thr_file, show_col_types=FALSE), error=function(e) NULL)
if (!is.null(thr)) {
  q_low  <- thr$value[thr$threshold=="q_low"]
  q_high <- thr$value[thr$threshold=="q_high"]
  cat(sprintf("Impact-tier thresholds (from M2 training data):\n"))
  cat(sprintf("  q_low  (33rd pct): %.4f  [baseline: ~87.33]\n", q_low))
  cat(sprintf("  q_high (67th pct): %.4f  [baseline: ~134.0]\n\n", q_high))
}

# ── 2. Split ──────────────────────────────────────────────────────────────────
feat_cols <- setdiff(names(df), "impact_tier")

# Impute NAs with column median (affects has_fulltext_int: 9 NAs)
for (col in feat_cols) {
  if (anyNA(df[[col]])) {
    med <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med
    cat(sprintf("  Imputed NAs in %-25s with median=%.2f\n", col, med))
  }
}
set.seed(42)
classes <- levels(df$impact_tier)
train_idx <- unlist(lapply(classes, function(cls) {
  idx <- which(df$impact_tier == cls)
  sample(idx, round(0.70*length(idx)))
}))
remaining <- setdiff(seq_len(nrow(df)), train_idx)
val_idx   <- unlist(lapply(classes, function(cls) {
  idx <- remaining[df$impact_tier[remaining]==cls]
  sample(idx, round(0.50*length(idx)))
}))
test_idx <- setdiff(remaining, val_idx)

X_train_raw <- as.matrix(df[train_idx, feat_cols])
X_val_raw   <- as.matrix(df[val_idx,   feat_cols])
X_test_raw  <- as.matrix(df[test_idx,  feat_cols])
y_train     <- df$impact_tier[train_idx]
y_val       <- df$impact_tier[val_idx]
y_test      <- df$impact_tier[test_idx]

# Scale
sc_mean <- apply(X_train_raw, 2, mean, na.rm=TRUE)
sc_sd   <- apply(X_train_raw, 2, function(x) { s<-sd(x,na.rm=TRUE); if(s<1e-10) 1 else s })
X_train <- scale(X_train_raw, center=sc_mean, scale=sc_sd)
X_val   <- scale(X_val_raw,   center=sc_mean, scale=sc_sd)
X_test  <- scale(X_test_raw,  center=sc_mean, scale=sc_sd)

cat(sprintf("Split | Train: %d | Val: %d | Test: %d\n",
            length(y_train), length(y_val), length(y_test)))
cat("Train distribution:\n"); print(table(y_train))
cat("\n")

# ── 3. Evaluate helper ───────────────────────────────────────────────────────
eval_m <- function(pc, pp, yt, model_nm, split_nm) {
  classes <- levels(yt)
  pc      <- factor(as.character(pc), levels = classes)
  acc     <- mean(pc == yt, na.rm = TRUE)
  f1s <- vapply(classes, function(c) {
    tp <- sum(pc == c & yt == c, na.rm=TRUE)
    fp <- sum(pc == c & yt != c, na.rm=TRUE)
    fn <- sum(pc != c & yt == c, na.rm=TRUE)
    p  <- if ((tp + fp) > 0L) tp / (tp + fp) else 0.0
    r  <- if ((tp + fn) > 0L) tp / (tp + fn) else 0.0
    if ((p + r) > 0) 2 * p * r / (p + r) else 0.0
  }, numeric(1))
  roc <- NA_real_
  if (!is.null(pp) && is.matrix(pp) && ncol(pp) == length(classes)) {
    pp[is.na(pp)] <- 1 / length(classes)
    colnames(pp)  <- classes
    auc_v <- vapply(seq_along(classes), function(i) {
      sc  <- pp[, i]
      bin <- as.integer(yt == classes[i])
      np  <- sum(bin); nn <- length(bin) - np
      if (np == 0L || nn == 0L) return(NA_real_)
      ord  <- order(sc, decreasing = TRUE)
      tpr  <- c(0, cumsum(bin[ord]) / np, 1)
      fpr  <- c(0, cumsum(1L - bin[ord]) / nn, 1)
      sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
    }, numeric(1))
    roc <- mean(auc_v, na.rm = TRUE)
  }
  data.frame(model = model_nm, split = split_nm,
             accuracy = round(acc, 4), f1_macro = round(mean(f1s), 4),
             roc_auc  = round(roc, 4), stringsAsFactors = FALSE)
}

all_metrics <- list()

safe_fit <- function(nm, expr) {
  cat(sprintf("  Training: %-30s ... ", nm)); flush.console()
  tryCatch({ r<-expr; cat("OK\n"); r }, error=function(e){ cat(sprintf("FAIL: %s\n",conditionMessage(e))); NULL })
}

# ── 4. Train models ───────────────────────────────────────────────────────────
cat("\n--- Training Impact-Tier Classifiers ---\n")

# Logistic Regression (multinomial)
suppressPackageStartupMessages(library(nnet))
lr_train_df      <- as.data.frame(X_train); lr_train_df$y_train <- y_train
lr_feat_names_it <- colnames(X_train)
lr_m <- safe_fit("Logistic Regression", {
  nnet::multinom(y_train ~ ., data = lr_train_df, maxit = 500, trace = FALSE, MaxNWts = 5000)
})
if (!is.null(lr_m)) {
  for (sp in c("val","test")) {
    Xr  <- if (sp=="val") X_val_raw else X_test_raw
    Xdf <- as.data.frame(Xr); names(Xdf) <- lr_feat_names_it
    ys  <- if (sp=="val") y_val else y_test
    pc  <- as.character(predict(lr_m, Xdf, type="class"))
    pp  <- predict(lr_m, Xdf, type="probs")
    if (!is.matrix(pp)) pp <- matrix(as.numeric(pp), nrow=nrow(Xdf),
                                     dimnames=list(NULL, levels(y_train)))
    all_metrics[[paste0("logistic_regression_",sp)]] <- eval_m(pc, pp, ys, "logistic_regression", sp)
  }
}

# Decision Tree
suppressPackageStartupMessages(library(rpart))
dt_m <- safe_fit("Decision Tree", {
  rpart(y_train~., data=data.frame(X_train_raw, y_train=y_train),
        method="class", control=rpart.control(cp=0.005, maxdepth=8))
})
if (!is.null(dt_m)) {
  for (sp in c("val","test")) {
    Xr<-if(sp=="val") X_val_raw else X_test_raw; ys<-if(sp=="val") y_val else y_test
    pp<-predict(dt_m,data.frame(Xr),type="prob")
    pc<-predict(dt_m,data.frame(Xr),type="class")
    if(!is.factor(pc)) pc<-factor(pc,levels=levels(y_train))
    all_metrics[[paste0("decision_tree_",sp)]] <- eval_m(pc,pp,ys,"decision_tree",sp)
  }
}

# Random Forest
if (requireNamespace("randomForest",quietly=TRUE)) {
  suppressPackageStartupMessages(library(randomForest))
  rf_m <- safe_fit("Random Forest (300 trees)", {
    randomForest(x=X_train_raw, y=y_train, ntree=300, importance=TRUE)
  })
  if (!is.null(rf_m)) {
    for (sp in c("val","test")) {
      Xr<-if(sp=="val") X_val_raw else X_test_raw; ys<-if(sp=="val") y_val else y_test
      pc<-predict(rf_m,Xr,type="class"); pp<-predict(rf_m,Xr,type="prob")
      all_metrics[[paste0("random_forest_",sp)]] <- eval_m(pc,pp,ys,"random_forest",sp)
    }
  }
}

# GBM
if (requireNamespace("gbm",quietly=TRUE)) {
  suppressPackageStartupMessages(library(gbm))
  gbm_m <- safe_fit("Gradient Boosting (gbm)", {
    gbm(y_train~., data=data.frame(X_train_raw,y_train=y_train),
        distribution="multinomial", n.trees=200, interaction.depth=3,
        shrinkage=0.05, verbose=FALSE)
  })
  if (!is.null(gbm_m)) {
    for (sp in c("val","test")) {
      Xr<-if(sp=="val") X_val_raw else X_test_raw; ys<-if(sp=="val") y_val else y_test
      raw<-predict(gbm_m,data.frame(Xr),n.trees=200,type="response")
      pp<-raw[,,1]; colnames(pp)<-levels(y_train)
      pc<-factor(levels(y_train)[apply(pp,1,which.max)],levels=levels(y_train))
      all_metrics[[paste0("gradient_boosting_",sp)]] <- eval_m(pc,pp,ys,"gradient_boosting",sp)
    }
  }
}

# XGBoost
if (requireNamespace("xgboost",quietly=TRUE)) {
  suppressPackageStartupMessages(library(xgboost))
  y_int<-as.integer(y_train)-1L
  n_cls<-length(levels(y_train))
  xgb_m <- safe_fit("XGBoost", {
    xgb.train(params=list(objective="multi:softprob",num_class=n_cls,
                           max_depth=4,eta=0.1,eval_metric="mlogloss",verbose=0),
              data=xgb.DMatrix(X_train_raw,label=y_int),nrounds=150,verbose=0)
  })
  if (!is.null(xgb_m)) {
    for (sp in c("val","test")) {
      Xr<-if(sp=="val") X_val_raw else X_test_raw; ys<-if(sp=="val") y_val else y_test
      raw<-predict(xgb_m,xgb.DMatrix(Xr))
      pp<-matrix(raw,ncol=n_cls,byrow=TRUE); colnames(pp)<-levels(y_train)
      pc<-factor(levels(y_train)[apply(pp,1,which.max)],levels=levels(y_train))
      all_metrics[[paste0("xgboost_",sp)]] <- eval_m(pc,pp,ys,"xgboost",sp)
    }
  }
}

# AdaBoost
if (requireNamespace("adabag",quietly=TRUE)) {
  suppressPackageStartupMessages(library(adabag))
  ada_m <- safe_fit("AdaBoost (100 iter) [CHAMPION]", {
    adabag::boosting(y_train~., data=data.frame(X_train_raw,y_train=y_train),
                     mfinal=100, coeflearn="Breiman",
                     control=rpart.control(maxdepth=3))
  })
  if (!is.null(ada_m)) {
    for (sp in c("val","test")) {
      Xdf<-if(sp=="val") data.frame(X_val_raw) else data.frame(X_test_raw)
      ys<-if(sp=="val") y_val else y_test
      res<-predict(ada_m,Xdf)
      pc<-factor(res$class,levels=levels(y_train))
      pp<-res$prob; colnames(pp)<-levels(y_train)
      all_metrics[[paste0("adaboost_",sp)]] <- eval_m(pc,pp,ys,"adaboost",sp)
    }
  }
}

# SVM
if (requireNamespace("e1071",quietly=TRUE)) {
  suppressPackageStartupMessages(library(e1071))
  svm_m <- safe_fit("SVM (RBF)", {
    e1071::svm(x=X_train,y=y_train,kernel="radial",probability=TRUE,cost=1)
  })
  if (!is.null(svm_m)) {
    for (sp in c("val","test")) {
      Xs<-if(sp=="val") X_val else X_test; ys<-if(sp=="val") y_val else y_test
      res<-predict(svm_m,Xs,probability=TRUE)
      pp<-attr(res,"probabilities")[,levels(y_train),drop=FALSE]
      pc<-factor(as.character(res),levels=levels(y_train))
      all_metrics[[paste0("svm_",sp)]] <- eval_m(pc,pp,ys,"svm_rbf",sp)
    }
  }
}

# LDA
suppressPackageStartupMessages(library(MASS))
lda_m <- safe_fit("LDA", {
  MASS::lda(x=X_train, grouping=y_train)
})
if (!is.null(lda_m)) {
  for (sp in c("val","test")) {
    Xs<-if(sp=="val") X_val else X_test; ys<-if(sp=="val") y_val else y_test
    res<-predict(lda_m,Xs)
    pc<-factor(res$class,levels=levels(y_train))
    pp<-res$posterior[,levels(y_train),drop=FALSE]
    all_metrics[[paste0("lda_",sp)]] <- eval_m(pc,pp,ys,"lda",sp)
  }
}

# Naive Bayes
suppressPackageStartupMessages(library(e1071))
nb_m <- safe_fit("Naive Bayes (e1071)", {
  e1071::naiveBayes(X_train_raw, y_train)
})
if (!is.null(nb_m)) {
  for (sp in c("val","test")) {
    Xr<-if(sp=="val") X_val_raw else X_test_raw; ys<-if(sp=="val") y_val else y_test
    pc<-predict(nb_m,Xr,type="class"); pp<-predict(nb_m,Xr,type="raw")
    if(!is.factor(pc)) pc<-factor(pc,levels=levels(y_train))
    all_metrics[[paste0("naive_bayes_",sp)]] <- eval_m(pc,pp,ys,"naive_bayes",sp)
  }
}

# MLP
suppressPackageStartupMessages(library(nnet))
mlp_m <- safe_fit("MLP Neural Network", {
  nnet::nnet(x=X_train, y=class.ind(y_train), size=50, maxit=500,
             decay=0.01, softmax=TRUE, trace=FALSE, MaxNWts=5000)
})
if (!is.null(mlp_m)) {
  for (sp in c("val","test")) {
    Xs<-if(sp=="val") X_val else X_test; ys<-if(sp=="val") y_val else y_test
    pp_raw<-predict(mlp_m,Xs,type="raw")
    if(is.vector(pp_raw)) pp_raw<-matrix(pp_raw,ncol=length(levels(y_train)))
    colnames(pp_raw)<-levels(y_train)
    pc<-factor(levels(y_train)[apply(pp_raw,1,which.max)],levels=levels(y_train))
    all_metrics[[paste0("mlp_",sp)]] <- eval_m(pc,pp_raw,ys,"mlp",sp)
  }
}

# ── 5. Compile and display ────────────────────────────────────────────────────
cat("\n=== Impact-Tier Results ===\n\n")
metrics_df <- do.call(rbind, all_metrics); rownames(metrics_df) <- NULL
test_m     <- metrics_df %>% filter(split=="test") %>% arrange(desc(f1_macro))
val_m      <- metrics_df %>% filter(split=="val")  %>% arrange(desc(f1_macro))

cat("TEST SET RESULTS:\n")
cat(sprintf("%-30s  %6s  %6s  %6s\n","Model","Acc","F1","ROC-AUC"))
cat(paste(rep("-",55),collapse=""),"\n")
for (i in seq_len(nrow(test_m))) {
  cat(sprintf("%-30s  %.4f  %.4f  %s\n",
              test_m$model[i], test_m$accuracy[i], test_m$f1_macro[i],
              ifelse(is.na(test_m$roc_auc[i]),"   N/A",sprintf("%.4f",test_m$roc_auc[i]))))
}

# Compare vs Python baseline
cat("\n--- Comparison with Python M7 Baseline ---\n")
cat("  Python AdaBoost: test-F1=0.543  test-acc=0.551  test-ROC-AUC=0.680\n")
ada_row <- test_m %>% filter(model=="adaboost")
if (nrow(ada_row) > 0) {
  cat(sprintf("  R     AdaBoost: test-F1=%.3f  test-acc=%.3f  test-ROC-AUC=%s\n",
              ada_row$f1_macro[1], ada_row$accuracy[1],
              ifelse(is.na(ada_row$roc_auc[1]),"N/A",sprintf("%.3f",ada_row$roc_auc[1]))))
  cat("  Note: Minor differences expected due to R vs Python train/split implementation.\n")
}

# ── 6. Visualisation — model comparison bar ───────────────────────────────────
cat("\nGenerating impact-tier figures...\n")
test_m$model_label <- factor(gsub("_"," ",test_m$model),
                              levels=gsub("_"," ",test_m$model[order(test_m$f1_macro)]))

p_imp <- ggplot(test_m, aes(x=model_label, y=f1_macro)) +
  geom_col(fill="#2c7bb6", alpha=0.85) +
  geom_hline(yintercept=0.543, linetype="dashed", colour="#d7301f", linewidth=0.8) +
  annotate("text", x=Inf, y=0.543, label=" Python baseline\n AdaBoost F1=0.543",
           hjust=1.1, vjust=-0.3, size=3.2, colour="#d7301f") +
  coord_flip() +
  labs(title="Impact-Tier Classification — Macro-F1 Comparison",
       subtitle="Target: low / medium / high citation impact tier | R Implementation",
       x=NULL, y="Macro-F1 Score (test set)",
       caption="Dashed red line = Python M7 AdaBoost baseline (F1=0.543)") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold")) +
  scale_y_continuous(limits=c(0,1))

ggsave(file.path(fig_dir,"impact_tier_model_comparison.png"), p_imp,
       width=8, height=5, dpi=150)
cat("  Saved: impact_tier_model_comparison.png\n")

# Class distribution
tier_dist <- data.frame(
  tier  = levels(df$impact_tier),
  count = as.integer(table(df$impact_tier))
)
p_dist <- ggplot(tier_dist, aes(x=tier, y=count, fill=tier)) +
  geom_col(alpha=0.85, show.legend=FALSE) +
  scale_fill_manual(values=c("low"="#2171b5","medium"="#6baed6","high"="#d7301f")) +
  labs(title="Impact-Tier Class Distribution",
       subtitle=sprintf("q_low=%.1f  q_high=%.1f (citation_per_year thresholds, training data)",
                        ifelse(exists("q_low"),q_low,87.28),
                        ifelse(exists("q_high"),q_high,137.22)),
       x="Impact Tier", y="Count") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"))

ggsave(file.path(fig_dir,"impact_tier_class_distribution.png"), p_dist,
       width=5, height=4, dpi=150)
cat("  Saved: impact_tier_class_distribution.png\n")

# ── 7. Save results ───────────────────────────────────────────────────────────
write_csv(metrics_df, file.path(out_dir, "impact_tier_metrics.csv"))
write_csv(test_m,     file.path(rep_dir, "r_impact_leaderboard.csv"))
cat(sprintf("\nSaved: impact_tier_metrics.csv\nSaved: r_impact_leaderboard.csv\n"))

cat("\n=== M7a Impact Tier COMPLETE ===\n")
cat(sprintf("  Models trained: %d\n", nrow(test_m)))
cat(sprintf("  Best model    : %s (F1=%.4f)\n",
            test_m$model[1], test_m$f1_macro[1]))
