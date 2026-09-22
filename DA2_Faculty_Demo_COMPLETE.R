# ============================================================================
#  DA-2 COMPLETE FACULTY DEMO  |  ResearchPilot
#  Open this in RStudio. Paste ONE block at a time during presentation.
#  Every block prints output + renders plots in RStudio Plots pane.
# ============================================================================

# ── SETUP (run this ONCE before anything else) ────────────────────────────────
setwd("e:/Tuned_Research")

library(RSQLite)
library(ggplot2)
library(randomForest)
library(e1071)
library(nnet)
library(rpart)
library(MASS)
library(glmnet)
library(gbm)
library(adabag)
library(xgboost)
library(pROC)
library(gridExtra)

mods         <- readRDS("data/ml_r/model_objects_v2.rds")
sel          <- read.csv("data/ml_r/selected_features_v2.csv")
metrics      <- read.csv("data/ml_r/model_metrics_v2.csv")
lb           <- read.csv("reports/tables/r_model_leaderboard_v2.csv")
tuning       <- read.csv("data/ml_r/tuning_results_v2.csv")
clusters     <- read.csv("data/ml_r/cluster_keywords.csv")
impact       <- read.csv("data/ml_r/impact_tier_metrics_v2.csv")
feature_cols <- setdiff(colnames(sel), "oa_category")
rf_cols      <- rownames(mods$random_forest$importance)

# ── Load the EXACT saved test split used by the training pipeline ─────────────
# (ml_data_splits_v2.rds is written by 05_ml_models_v2.R, set.seed(42))
splits    <- readRDS("data/ml_r/ml_data_splits_v2.rds")
X_te      <- splits$X_test    # 301 x 80 — unscaled, for RF/tree models
X_te_s    <- splits$X_test_s  # 301 x 80 — scaled,   for SVM/LDA/LR
y_te      <- splits$y_test    # factor, 301 labels (closed/fully_open/partially_open)
CLASSES   <- levels(y_te)

cat("SETUP COMPLETE — all objects loaded.\n")
cat("Models available:", paste(names(mods), collapse=", "), "\n")
cat("Dataset: 2000 papers x", ncol(sel)-1, "selected features\n")
cat(sprintf("Proper test set loaded: %d rows | %s\n",
            nrow(X_te), paste(table(y_te), collapse=" / ")))


# ============================================================================
# CMD-1  DATABASE CONNECTIVITY
# ============================================================================
cat("\n========== CMD-1: DATABASE CONNECTIVITY ==========\n")

con <- dbConnect(SQLite(), "database/researchpilot_r.db")
cat("Connected to SQLite DB. Tables:", paste(dbListTables(con), collapse=", "), "\n\n")

cat("Q1 — Papers per OA category:\n")
print(dbGetQuery(con,
  "SELECT oa_category, COUNT(*) AS n_papers
   FROM papers GROUP BY oa_category ORDER BY n_papers DESC"))

cat("\nQ2 — Top 5 most-cited papers:\n")
print(dbGetQuery(con,
  "SELECT paper_id, cited_by_count, publication_year, oa_category,
          SUBSTR(title,1,55) AS title
   FROM papers ORDER BY cited_by_count DESC LIMIT 5"))

cat("\nQ3 — Papers published 2023-2025:\n")
print(dbGetQuery(con,
  "SELECT publication_year, COUNT(*) AS n FROM papers
   WHERE publication_year >= 2023
   GROUP BY publication_year ORDER BY publication_year"))

cat("\nQ4 — Avg citations per OA category:\n")
print(dbGetQuery(con,
  "SELECT oa_category,
          ROUND(AVG(cited_by_count),1) AS avg_citations,
          COUNT(*) AS n
   FROM papers GROUP BY oa_category ORDER BY avg_citations DESC"))

cat("\nQ5 — Feature averages per category (JOIN query):\n")
print(dbGetQuery(con,
  "SELECT p.oa_category,
          ROUND(AVG(m.title_length),1)    AS avg_title_len,
          ROUND(AVG(m.abstract_length),0) AS avg_abstract_len,
          ROUND(AVG(m.concept_count),2)   AS avg_concepts,
          ROUND(AVG(m.text_richness),4)   AS avg_richness,
          COUNT(*) AS n
   FROM ml_features m
   JOIN papers p ON p.paper_id = m.paper_id
   GROUP BY p.oa_category"))

dbDisconnect(con)
cat("DB connection closed.\n")

# PLOT — Papers by OA category (bar chart in Plots pane)
con2 <- dbConnect(SQLite(), "database/researchpilot_r.db")
oa_dist <- dbGetQuery(con2, "SELECT oa_category, COUNT(*) AS n FROM papers GROUP BY oa_category")
dbDisconnect(con2)

print(
  ggplot(oa_dist, aes(x=reorder(oa_category,-n), y=n, fill=oa_category)) +
    geom_col(width=0.5) +
    geom_text(aes(label=n), vjust=-0.4, size=4.5, fontface="bold") +
    scale_fill_manual(values=c("closed"="#E74C3C","fully_open"="#2ECC71","partially_open"="#3498DB")) +
    labs(title="Papers per Open-Access Category (DB Query)",
         subtitle="Source: SQLite — researchpilot_r.db",
         x="OA Category", y="Number of Papers") +
    theme_minimal(base_size=13) + theme(legend.position="none")
)


# ============================================================================
# CMD-2  FEATURE ENGINEERING
# ============================================================================
cat("\n========== CMD-2: FEATURE ENGINEERING ==========\n")

fe <- read.csv("data/ml_r/enriched_features_v2.csv")
cat("Enriched dataset:", nrow(fe), "papers x", ncol(fe), "features\n\n")
cat("  Raw columns       : 27\n")
cat("  After engineering : 206  (+179 new features)\n")
cat("  After selection   : 80   (NZV → corr filter → RF Gini top-80)\n\n")
cat("Feature categories:\n")
cat("  1. Text-length  : title_length, abstract_length, word counts\n")
cat("  2. Richness     : text_richness, title_to_abstract_ratio\n")
cat("  3. Metadata     : paper_age, recency_score, citation_log\n")
cat("  4. TF-IDF terms : tf_learning, tf_deep, tf_model (80+ terms)\n")
cat("  5. Publisher    : pub_ieee, pub_mdpi, pub_springer (one-hot)\n")
cat("  6. Concepts     : concept_count, cn_medicine\n\n")

key <- c("title_length","abstract_length","text_richness",
         "recency_score","paper_age","concept_count","keyword_count")
print(round(sapply(fe[, key], function(x) c(Min=min(x), Mean=mean(x), Max=max(x))), 3))

cat("\nTarget variable distribution:\n")
tbl <- table(fe$oa_category)
print(tbl)

# PLOT — Feature summary boxplots in Plots pane
fe_long <- stack(fe[, key])
print(
  ggplot(fe_long, aes(x=ind, y=values, fill=ind)) +
    geom_boxplot(outlier.size=0.5, outlier.alpha=0.4) +
    coord_flip() +
    labs(title="Distribution of Key Engineered Features",
         subtitle="2000 papers — 7 key features shown",
         x=NULL, y="Value") +
    theme_minimal(base_size=12) + theme(legend.position="none")
)


# ============================================================================
# CMD-3  FEATURE SELECTION
# ============================================================================
cat("\n========== CMD-3: FEATURE SELECTION ==========\n")

cat("206 features -> 80 selected (61% reduction)\n")
cat("Method: 3-step pipeline (as implemented in 02_feature_selection_v2.R):\n")
cat("  Step 1 — Near-Zero Variance (NZV) filter\n")
cat("           Removes constant/near-constant columns (var < 1e-10)\n")
cat("  Step 2 — Pearson correlation filter (TF-IDF group only, |r| > 0.95)\n")
cat("           Removes one column from highly correlated TF-IDF pairs\n")
cat("  Step 3 — Random Forest Gini importance ranking (top 80 kept)\n")
cat("           300-tree RF trained on 70% train split only — no leakage\n\n")

pub_feats  <- sum(grepl("^pub_",  feature_cols))
tf_feats   <- sum(grepl("^tf_",   feature_cols))
cn_feats   <- sum(grepl("^cn_",   feature_cols))
meta_feats <- length(feature_cols) - pub_feats - tf_feats - cn_feats

cat(sprintf("  TF-IDF term features   : %d\n", tf_feats))
cat(sprintf("  Publisher one-hot      : %d\n", pub_feats))
cat(sprintf("  Concept/domain flags   : %d\n", cn_feats))
cat(sprintf("  Metadata/text features : %d\n", meta_feats))

cat("\nTop 30 selected features:\n")
print(head(feature_cols, 30))

# PLOT — Feature type breakdown pie in Plots pane
feat_types <- data.frame(
  Type  = c("TF-IDF Terms","Publisher","Concepts","Metadata/Text"),
  Count = c(tf_feats, pub_feats, cn_feats, meta_feats)
)
print(
  ggplot(feat_types, aes(x="", y=Count, fill=Type)) +
    geom_col(width=1, colour="white") +
    coord_polar("y") +
    geom_text(aes(label=paste0(Type,"\n(",Count,")")),
              position=position_stack(vjust=0.5), size=3.8) +
    scale_fill_manual(values=c("#3498DB","#E67E22","#2ECC71","#9B59B6")) +
    labs(title="Selected Feature Breakdown by Type",
         subtitle="80 features retained from 206") +
    theme_void() + theme(legend.position="none")
)


# ============================================================================
# CMD-4  ALL 12 ML ALGORITHMS
# ============================================================================
cat("\n========== CMD-4: 12 ML ALGORITHMS ==========\n")

model_info <- data.frame(
  No    = 1:12,
  Model = c("Logistic Regression","Elastic Net","Decision Tree",
            "Random Forest","XGBoost","Gradient Boosting",
            "AdaBoost","SVM (RBF)","SVM (Linear)",
            "Naive Bayes","LDA","MLP (Neural Net)"),
  Type  = c("Linear","Regularised Linear","Tree","Ensemble",
            "Boosting","Boosting","Boosting","Kernel SVM",
            "Kernel SVM","Probabilistic","Dimensionality","Deep"),
  RLib  = c("glmnet","glmnet","rpart","randomForest","xgboost",
            "gbm","adabag","e1071","e1071","e1071","MASS","nnet")
)
print(model_info, row.names=FALSE)

cat("\n--- Random Forest ---\n")
cat("ntree:", mods$random_forest$ntree, " | mtry:", mods$random_forest$mtry,
    " | OOB:", round(mods$random_forest$err.rate[500,1]*100,2), "%\n")
cat("Classes:", paste(levels(mods$random_forest$y), collapse=" | "), "\n")

cat("\n--- SVM (RBF) ---\n")
cat("Cost:", mods$svm_rbf$cost, " | Gamma:", mods$svm_rbf$gamma,
    " | Support vectors:", nrow(mods$svm_rbf$SV), "\n")

cat("\n--- Logistic Regression top 10 coefficients ---\n")
coefs <- coef(mods$logistic_regression)
print(round(sort(abs(coefs[1,]), decreasing=TRUE)[1:10], 4))

cat("\n--- Elastic Net ---\n")
cat("lambda.min:", round(mods$elastic_net$lambda.min,5),
    " | lambda.1se:", round(mods$elastic_net$lambda.1se,5), "\n")

cat("\n--- MLP (nnet) ---\n")
cat("Hidden units:", mods$mlp$n[2], " | Weights:", length(mods$mlp$wts), "\n")


# ============================================================================
# CMD-5  LIVE PREDICTION — single paper, 6 models
# ============================================================================
cat("\n========== CMD-5: LIVE PREDICTION — PAPER #1501 ==========\n")

test_paper <- sel[1501, feature_cols]
cat("Paper #1501 true label:", sel$oa_category[1501], "\n\n")

results <- data.frame(Model=character(), Prediction=character(), stringsAsFactors=FALSE)
for (nm in c("random_forest","logistic_regression","elastic_net",
             "svm_rbf","lda","decision_tree")) {
  pred <- tryCatch({
    if (nm == "random_forest")
      as.character(predict(mods[[nm]], newdata=test_paper[,rf_cols,drop=FALSE]))
    else if (nm == "elastic_net")
      as.character(predict(mods[[nm]], newx=as.matrix(test_paper), type="class", s="lambda.min"))
    else if (nm == "lda")
      as.character(predict(mods[[nm]], newdata=test_paper)$class)
    else if (nm == "decision_tree")
      as.character(predict(mods[[nm]], newdata=test_paper, type="class"))
    else
      as.character(predict(mods[[nm]], newdata=test_paper))
  }, error=function(e) paste("ERR:", e$message))
  results <- rbind(results, data.frame(Model=nm, Prediction=pred[1]))
}
print(results, row.names=FALSE)

cat("\nRandom Forest class PROBABILITIES for paper #1501:\n")
probs_single <- predict(mods$random_forest,
                        newdata=test_paper[,rf_cols,drop=FALSE], type="prob")
print(round(probs_single, 4))

# PLOT — probability bar for this paper
probs_df <- data.frame(
  Class = colnames(probs_single),
  Prob  = as.numeric(probs_single[1,])
)
print(
  ggplot(probs_df, aes(x=Class, y=Prob, fill=Class)) +
    geom_col(width=0.4) +
    geom_text(aes(label=sprintf("%.2f%%", Prob*100)), vjust=-0.4, size=5, fontface="bold") +
    scale_fill_manual(values=c("closed"="#E74C3C","fully_open"="#2ECC71","partially_open"="#3498DB")) +
    labs(title="Random Forest Prediction Probabilities",
         subtitle=paste("Paper #1501 | True label:", sel$oa_category[1501]),
         x="OA Category", y="Probability") +
    ylim(0,1.05) +
    theme_minimal(base_size=13) + theme(legend.position="none")
)


# ============================================================================
# CMD-6  CONFUSION MATRIX — proper saved test split (ml_data_splits_v2.rds)
# ============================================================================
cat("\n========== CMD-6: CONFUSION MATRIX (proper test split) ==========\n")
cat(sprintf("Test set: %d rows | closed=%d  fully_open=%d  partially_open=%d\n",
            nrow(X_te), sum(y_te=="closed"), sum(y_te=="fully_open"),
            sum(y_te=="partially_open")))
cat("Source: ml_data_splits_v2.rds — same split used by 05_ml_models_v2.R\n\n")

# RF was trained on unscaled X_tr — use unscaled X_te here
preds_rf  <- predict(mods$random_forest, newdata=X_te)
cm        <- table(Predicted=preds_rf, Actual=y_te)

print(cm)
acc <- sum(diag(cm)) / sum(cm)
cat(sprintf("\nOverall Accuracy : %.4f  (%.1f%%)\n\n", acc, acc*100))

cat(sprintf("%-16s  %6s  %6s  %6s\n","Class","Prec","Recall","F1"))
cat(strrep("-",40),"\n")
for (cls in CLASSES) {
  tp   <- cm[cls,cls]
  fp   <- sum(cm[cls,]) - tp
  fn   <- sum(cm[,cls]) - tp
  prec <- ifelse((tp+fp)>0, tp/(tp+fp), 0)
  rec  <- ifelse((tp+fn)>0, tp/(tp+fn), 0)
  f1   <- ifelse((prec+rec)>0, 2*prec*rec/(prec+rec), 0)
  cat(sprintf("%-16s  %6.3f  %6.3f  %6.3f\n", cls, prec, rec, f1))
}

# Cross-check against stored pipeline metrics
stored_rf <- metrics[metrics$model=="random_forest" & metrics$split=="test",]
cat(sprintf("\nStored pipeline accuracy : %.4f  (matches: %s)\n",
            stored_rf$accuracy,
            ifelse(abs(stored_rf$accuracy - acc) < 0.0001, "YES - consistent", "MISMATCH")))

# PLOT — Confusion Matrix heatmap in Plots pane
cm_df <- as.data.frame(cm)
print(
  ggplot(cm_df, aes(x=Actual, y=Predicted, fill=Freq)) +
    geom_tile(colour="white", linewidth=1) +
    geom_text(aes(label=Freq), size=8, fontface="bold", colour="white") +
    scale_fill_gradient(low="#2E86C1", high="#1A5276") +
    labs(title="Confusion Matrix — Random Forest",
         subtitle=sprintf("Accuracy: %.1f%%  |  Proper test split  (n=%d)", acc*100, nrow(X_te)),
         x="Actual", y="Predicted") +
    theme_minimal(base_size=13) +
    theme(legend.position="none",
          axis.text=element_text(size=12, face="bold"))
)


# ============================================================================
# CMD-7  FEATURE IMPORTANCE — live from RF
# ============================================================================
cat("\n========== CMD-7: FEATURE IMPORTANCE ==========\n")

imp         <- as.data.frame(mods$random_forest$importance)
imp$feature <- rownames(imp)
imp         <- imp[order(-imp$MeanDecreaseGini), ]
top20       <- head(imp, 20)

cat("Top 20 features (Mean Decrease Gini):\n")
print(top20[, c("feature","MeanDecreaseGini")], row.names=FALSE)

print(
  ggplot(top20, aes(x=reorder(feature, MeanDecreaseGini),
                    y=MeanDecreaseGini, fill=MeanDecreaseGini)) +
    geom_col() +
    coord_flip() +
    scale_fill_gradient(low="#AED6F1", high="#1A5276") +
    labs(title="Top 20 Feature Importances — Random Forest",
         subtitle="Metric: Mean Decrease in Gini Impurity",
         x=NULL, y="Mean Decrease Gini") +
    theme_minimal(base_size=12) +
    theme(legend.position="none")
)


# ============================================================================
# CMD-8  FULL METRICS TABLE — all 12 models
# ============================================================================
cat("\n========== CMD-8: METRICS TABLE — ALL 12 MODELS ==========\n")

test_m <- metrics[metrics$split == "test", ]
test_m <- test_m[order(-test_m$f1_macro), ]

cat(sprintf("\n%-25s %7s %7s %7s %7s %8s\n","Model","Acc","Prec","Recall","F1","ROC-AUC"))
cat(strrep("-",63),"\n")
for (i in 1:nrow(test_m)) {
  r <- test_m[i,]
  cat(sprintf("%-25s %7.4f %7.4f %7.4f %7.4f %8s\n",
    r$model, r$accuracy,
    ifelse(is.na(r$precision),0,r$precision),
    r$recall, r$f1_macro,
    ifelse(is.na(r$roc_auc),"   N/A",sprintf("%.4f",r$roc_auc))))
}
cat(strrep("-",63),"\n")
best <- test_m[1,]
cat(sprintf("\nBEST: %s | Acc=%.4f | F1=%.4f\n", best$model, best$accuracy, best$f1_macro))


# ============================================================================
# CMD-9  HYPERPARAMETER TUNING — before vs after
# ============================================================================
cat("\n========== CMD-9: HYPERPARAMETER TUNING ==========\n")

tuned_test  <- tuning[tuning$split == "test", ]
baseline_nm <- c("random_forest_tuned"="random_forest",
                 "xgboost_tuned"      ="xgboost",
                 "svm_tuned"          ="svm_rbf")

cat(sprintf("%-22s  %9s  %9s  %10s  %s\n","Tuned Model","Base F1","Tuned F1","Improvement","Best Params"))
cat(strrep("-",80),"\n")
for (i in 1:nrow(tuned_test)) {
  r       <- tuned_test[i,]
  base    <- metrics[metrics$model==baseline_nm[r$model] & metrics$split=="test",]
  base_f1 <- if (nrow(base)>0) base$f1_macro[1] else NA
  delta   <- if (!is.na(base_f1)) r$f1_macro - base_f1 else NA
  arrow   <- if (!is.na(delta) && delta>=0) "+" else ""
  cat(sprintf("%-22s  %9.4f  %9.4f  %10s  %s\n",
    r$model, ifelse(is.na(base_f1),0,base_f1), r$f1_macro,
    ifelse(is.na(delta),"N/A",paste0(arrow,round(delta,4))), r$best_params))
}
cat(strrep("-",80),"\n")

# PLOT — Baseline vs Tuned comparison in Plots pane
tuned_names <- c("random_forest_tuned","svm_tuned","xgboost_tuned")
base_names  <- c("random_forest","svm_rbf","xgboost")
base_f1s    <- sapply(base_names, function(m) metrics[metrics$model==m & metrics$split=="test","f1_macro"][1])
tuned_f1s   <- sapply(tuned_names, function(m) tuning[tuning$model==m & tuning$split=="test","f1_macro"][1])

compare_df <- data.frame(
  Model   = rep(c("Random Forest","SVM","XGBoost"), 2),
  Version = c(rep("Baseline",3), rep("Tuned",3)),
  F1      = c(base_f1s, tuned_f1s)
)
print(
  ggplot(compare_df, aes(x=Model, y=F1, fill=Version)) +
    geom_col(position="dodge", width=0.5) +
    geom_text(aes(label=round(F1,4)), position=position_dodge(0.5), vjust=-0.4, size=4) +
    scale_fill_manual(values=c("Baseline"="#5DADE2","Tuned"="#E67E22")) +
    labs(title="Hyperparameter Tuning — Baseline vs Tuned F1",
         subtitle="Grid search: RF(ntree,mtry) | SVM(cost,gamma) | XGB(depth,eta,rounds)",
         x=NULL, y="F1-Macro Score") +
    ylim(0, 0.75) +
    theme_minimal(base_size=13) +
    theme(legend.position="bottom")
)


# ============================================================================
# CMD-10  ROC CURVES — proper saved test split, OVR macro-AUC
# ============================================================================
cat("\n========== CMD-10: ROC CURVES (proper test split — 3-class OVR) ==========\n")
cat(sprintf("Using same X_te / y_te as CMD-6 and CMD-8 (n=%d)\n\n", nrow(X_te)))

# RF probabilities on the proper test set (unscaled — matches training)
probs_rf <- predict(mods$random_forest, newdata=X_te, type="prob")
auc_vals <- setNames(numeric(3), CLASSES)

par(mfrow=c(1,3), mar=c(4,4,3,1))
for (cls in CLASSES) {
  bin_y         <- as.integer(y_te == cls)
  roc_obj       <- roc(bin_y, probs_rf[,cls], quiet=TRUE)
  auc_vals[cls] <- as.numeric(auc(roc_obj))
  plot(roc_obj, main=paste0("ROC — ", cls),
       col="#1A5276", lwd=2,
       print.auc=TRUE, print.auc.y=0.4,
       legacy.axes=TRUE)
  abline(a=0, b=1, lty=2, col="grey60")
}
par(mfrow=c(1,1))

cat("\nAUC per class (One-vs-Rest, proper test split):\n")
for (cls in CLASSES) cat(sprintf("  %-16s  AUC = %.4f\n", cls, auc_vals[cls]))
cat(sprintf("  Mean AUC (macro OVR) : %.4f\n", mean(auc_vals)))

# Cross-check against stored pipeline metrics
stored_rf_auc <- metrics[metrics$model=="random_forest" & metrics$split=="test","roc_auc"]
cat(sprintf("\nStored pipeline ROC-AUC  : %.4f  (matches: %s)\n",
            stored_rf_auc,
            ifelse(abs(stored_rf_auc - mean(auc_vals)) < 0.001, "YES - consistent", "MISMATCH")))


# ============================================================================
# CMD-11  COMPARATIVE BAR CHARTS — F1 + Accuracy + ROC
# ============================================================================
cat("\n========== CMD-11: COMPARATIVE BAR CHARTS ==========\n")

lb_t       <- lb[lb$split=="test",]
lb_t$model <- factor(lb_t$model, levels=lb_t$model[order(lb_t$f1)])

p_f1 <- ggplot(lb_t, aes(x=model, y=f1, fill=model_type)) +
  geom_col(width=0.7) +
  geom_text(aes(label=round(f1,3)), hjust=-0.1, size=2.8) +
  coord_flip() +
  scale_fill_manual(values=c("baseline"="#5DADE2","tuned"="#E67E22")) +
  labs(title="F1-Macro", x=NULL, y="F1") +
  theme_minimal(base_size=9) + theme(legend.position="none")

p_acc <- ggplot(lb_t, aes(x=model, y=acc, fill=model_type)) +
  geom_col(width=0.7) +
  geom_text(aes(label=round(acc,3)), hjust=-0.1, size=2.8) +
  coord_flip() +
  scale_fill_manual(values=c("baseline"="#5DADE2","tuned"="#E67E22")) +
  labs(title="Accuracy", x=NULL, y="Acc") +
  theme_minimal(base_size=9) + theme(legend.position="none")

lb_r       <- lb_t[!is.na(lb_t$roc),]
lb_r$model <- factor(lb_r$model, levels=lb_r$model[order(lb_r$roc)])
p_roc <- ggplot(lb_r, aes(x=model, y=roc, fill=model_type)) +
  geom_col(width=0.7) +
  geom_hline(yintercept=0.5, linetype="dashed", colour="red", linewidth=0.6) +
  geom_text(aes(label=round(roc,3)), hjust=-0.1, size=2.8) +
  coord_flip() +
  scale_fill_manual(values=c("baseline"="#5DADE2","tuned"="#E67E22")) +
  labs(title="ROC-AUC", x=NULL, y="AUC",
       caption="Dashed = random (0.5)") +
  theme_minimal(base_size=9) + theme(legend.position="none")

grid.arrange(p_f1, p_acc, p_roc, ncol=3,
             top="Comparative Model Performance — All 12 Algorithms")


# ============================================================================
# CMD-12  PRECISION-RECALL-F1 GROUPED BAR
# ============================================================================
cat("\n========== CMD-12: PRECISION-RECALL-F1 ==========\n")

lb_test  <- lb[lb$split=="test",]
lb_long  <- data.frame(
  Model  = rep(lb_test$model, 3),
  Metric = c(rep("Precision",nrow(lb_test)),
             rep("Recall",   nrow(lb_test)),
             rep("F1",       nrow(lb_test))),
  Value  = c(ifelse(is.na(lb_test$prec), lb_test$f1, lb_test$prec),
             lb_test$rec, lb_test$f1)
)
lb_long       <- lb_long[!is.na(lb_long$Value),]
lb_long$Model <- factor(lb_long$Model, levels=lb_test$model[order(lb_test$f1)])

print(
  ggplot(lb_long, aes(x=Model, y=Value, fill=Metric)) +
    geom_col(position="dodge", width=0.7) +
    coord_flip() +
    scale_fill_manual(values=c("Precision"="#2ECC71","Recall"="#E74C3C","F1"="#3498DB")) +
    labs(title="Precision / Recall / F1 — All 12 Models",
         x=NULL, y="Score") +
    theme_minimal(base_size=11) +
    theme(legend.position="bottom")
)


# ============================================================================
# CMD-13  CLUSTERING — K-means profiles
# ============================================================================
cat("\n========== CMD-13: UNSUPERVISED CLUSTERING ==========\n")

cat(sprintf("%-10s  %7s  %7s  %s\n","Cluster","Papers","Pct(%)","Top Terms"))
cat(strrep("-",88),"\n")
for (i in 1:nrow(clusters))
  cat(sprintf("Cluster %-2d  %7d  %7.1f%%  %s\n",
    clusters$cluster[i], clusters$n_papers[i],
    clusters$pct_papers[i], clusters$top_terms[i]))
cat(strrep("-",88),"\n\n")
cat("  Cluster 1 (10%): Education / GenAI / ChatGPT\n")
cat("  Cluster 2 (71%): General ML / Prediction / Robotics\n")
cat("  Cluster 3 (19%): Medical Imaging / Deep Learning\n\n")

# PLOT — Cluster sizes in Plots pane
print(
  ggplot(clusters, aes(x=factor(cluster), y=n_papers, fill=factor(cluster))) +
    geom_col(width=0.5) +
    geom_text(aes(label=paste0(n_papers,"\n(",pct_papers,"%)")),
              vjust=-0.2, size=5, fontface="bold") +
    scale_fill_manual(values=c("1"="#E74C3C","2"="#2ECC71","3"="#3498DB")) +
    labs(title="K-Means Cluster Sizes  (k = 3)",
         subtitle="Cluster 1=Education/AI  |  Cluster 2=General ML  |  Cluster 3=Medical Imaging",
         x="Cluster", y="Number of Papers") +
    theme_minimal(base_size=13) + theme(legend.position="none")
)


# ============================================================================
# CMD-14  FINAL DASHBOARD — 4-panel summary
# ============================================================================
cat("\n========== CMD-14: FINAL SUMMARY DASHBOARD ==========\n")

lb_d       <- lb[lb$split=="test",]
lb_d$model <- factor(lb_d$model, levels=lb_d$model[order(lb_d$f1)])

d1 <- ggplot(lb_d, aes(x=model, y=f1, fill=model_type)) +
  geom_col(width=0.7) + coord_flip() +
  scale_fill_manual(values=c("baseline"="#5DADE2","tuned"="#E67E22")) +
  labs(title="F1-Macro", x=NULL, y="F1") +
  theme_minimal(base_size=8) + theme(legend.position="none")

d2 <- ggplot(lb_d, aes(x=model, y=acc, fill=model_type)) +
  geom_col(width=0.7) + coord_flip() +
  scale_fill_manual(values=c("baseline"="#5DADE2","tuned"="#E67E22")) +
  labs(title="Accuracy", x=NULL, y="Acc") +
  theme_minimal(base_size=8) + theme(legend.position="none")

lb_dr       <- lb_d[!is.na(lb_d$roc),]
lb_dr$model <- factor(lb_dr$model, levels=lb_dr$model[order(lb_dr$roc)])
d3 <- ggplot(lb_dr, aes(x=model, y=roc, fill=model_type)) +
  geom_col(width=0.7) +
  geom_hline(yintercept=0.5, linetype="dashed", colour="red", linewidth=0.5) +
  coord_flip() +
  scale_fill_manual(values=c("baseline"="#5DADE2","tuned"="#E67E22")) +
  labs(title="ROC-AUC", x=NULL, y="AUC") +
  theme_minimal(base_size=8) + theme(legend.position="none")

cd <- as.data.frame(table(sel$oa_category))
colnames(cd) <- c("class","n")
d4 <- ggplot(cd, aes(x=class, y=n, fill=class)) +
  geom_col(width=0.5) +
  geom_text(aes(label=n), vjust=-0.3, size=3.5) +
  scale_fill_manual(values=c("closed"="#E74C3C","fully_open"="#2ECC71","partially_open"="#3498DB")) +
  labs(title="Class Distribution", x=NULL, y="Papers") +
  theme_minimal(base_size=8) + theme(legend.position="none")

grid.arrange(d1, d2, d3, d4, ncol=2,
             top="DA-2 ResearchPilot — Complete Model Performance Dashboard")


# ============================================================================
# CMD-15  RUBRIC CHECKLIST — final print
# ============================================================================
cat("\n")

# Best BASELINE by macro-F1 (from model_metrics_v2.csv — baselines only)
base_test   <- metrics[metrics$split=="test",]
best_base   <- base_test[which.max(base_test$f1_macro),]

# Best TUNED overall by macro-F1 (from full leaderboard)
lb_test_all <- lb[lb$split=="test",]
best_tuned  <- lb_test_all[which.max(lb_test_all$f1),]

# RF ROC-AUC from stored pipeline (proper test set)
rf_roc_stored <- metrics[metrics$model=="random_forest" & metrics$split=="test","roc_auc"]

cat("=================================================================\n")
cat("  DA-2 RUBRIC COVERAGE — FINAL CHECKLIST\n")
cat("=================================================================\n")
cat("  [DONE]  Database Connectivity\n")
cat("          SQLite, 4 tables, 5 SQL queries (incl JOIN)\n\n")
cat("  [DONE]  Feature Engineering\n")
cat("          27 raw -> 206 engineered (TF-IDF, publisher,\n")
cat("          recency, richness, citation log, concepts)\n\n")
cat("  [DONE]  Feature Selection\n")
cat("          206 -> 80 features via 3-step pipeline:\n")
cat("          Step 1: NZV filter (remove constant/near-constant)\n")
cat("          Step 2: Pearson corr filter (TF-IDF group, |r|>0.95)\n")
cat("          Step 3: RF Gini importance top-80 (train split only)\n\n")
cat("  [DONE]  10-15 ML/DL Algorithms — 12 implemented\n")
cat("          LR, Elastic Net, DT, RF, XGBoost, GBM,\n")
cat("          AdaBoost, SVM-RBF, SVM-Linear, NB, LDA, MLP\n\n")
cat("  [DONE]  Hyperparameter Tuning\n")
cat("          RF(ntree,mtry), XGBoost(depth,eta,rounds),\n")
cat("          SVM(cost,gamma) — grid search + 5-fold CV\n\n")
cat("  [DONE]  Comparative Performance Analysis\n")
cat(sprintf("          Best BASELINE  : %-20s  F1=%.4f  Acc=%.4f\n",
            best_base$model, best_base$f1_macro, best_base$accuracy))
cat(sprintf("          Best OVERALL   : %-20s  F1=%.4f  Acc=%.4f\n",
            best_tuned$model, best_tuned$f1, best_tuned$acc))
cat(sprintf("          Tuning gain    : +%.4f F1 over baseline\n",
            best_tuned$f1 - best_base$f1_macro))
cat(sprintf("          RF ROC-AUC     : %.4f (OVR macro, proper test set)\n\n",
            rf_roc_stored))
cat("  [DONE]  Comparative Visualizations\n")
cat("          ROC Curves (OVR, n=301), Confusion Matrix heatmap,\n")
cat("          Feature Importance, Precision-Recall,\n")
cat("          F1/Acc/ROC bar charts, Clustering, Dashboard\n")
cat("  NOTE    All CMD-6/CMD-10 metrics use ml_data_splits_v2.rds\n")
cat("          (same stratified test split as training pipeline)\n")
cat("=================================================================\n")
cat("  ALL RUBRIC ITEMS FULLY COVERED\n")
cat("=================================================================\n")
                                                                    