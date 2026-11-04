# =============================================================================
# ResearchPilot — DA2 (ENHANCED) — M6bv2: Comparative Visualizations
# =============================================================================
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(ggplot2); library(tidyr); library(stringr)
})
set.seed(42)
cat("=== M6bv2: Comparative Visualizations (Enriched) ===\n\n")

root <- NULL
for (cand in c(getwd(),"E:/Tuned_Research","e:/Tuned_Research"))
  if (file.exists(file.path(cand,"data","final","final_dataset.csv"))) { root <- normalizePath(cand); break }
if (is.null(root)) root <- getwd()

ml_dir  <- file.path(root,"data","ml_r")
rep_dir <- file.path(root,"reports","tables")
fig_dir <- file.path(root,"reports","figures","phase2_r")
dir.create(fig_dir, showWarnings=FALSE, recursive=TRUE)

sv <- function(p, name, w=9, h=6) {
  ggsave(file.path(fig_dir, paste0(name,".png")), p, width=w, height=h, dpi=150)
  cat(sprintf("  Saved: %s.png\n", name))
}

# ── Load leaderboard ───────────────────────────────────────────────────────────
lb_path <- file.path(rep_dir,"r_model_leaderboard_v2.csv")
if (!file.exists(lb_path)) stop("Run 07_model_evaluation_v2.R first")
lb <- read_csv(lb_path, show_col_types=FALSE)

# Clean labels
lb$model_label <- str_replace_all(lb$model,"_"," ")
lb$model_label <- factor(lb$model_label, levels=lb$model_label[order(lb$acc)])

# ── 1. Accuracy comparison ────────────────────────────────────────────────────
cat("Plot 1: Accuracy comparison\n")
p1 <- ggplot(lb, aes(x=model_label, y=acc,
                     fill=ifelse(model_type=="tuned","Tuned","Baseline"))) +
  geom_col(alpha=0.85) +
  geom_hline(yintercept=0.70, linetype="dashed", colour="#d62728", linewidth=0.8) +
  geom_hline(yintercept=0.4452, linetype="dotted", colour="grey50", linewidth=0.8) +
  scale_fill_manual(values=c("Baseline"="#4292c6","Tuned"="#41ab5d"), name="") +
  coord_flip() +
  scale_y_continuous(limits=c(0,1), labels=scales::percent_format(1)) +
  labs(title="Model Accuracy — Enriched Features (TF-IDF + Domain + Publisher)",
       subtitle="Red dashed = 70% target | Grey dotted = old 9-feature baseline",
       x=NULL, y="Accuracy", caption="Test set | OA Category Classification") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"), legend.position="top")
sv(p1, "v2_01_accuracy_comparison")

# ── 2. Macro-F1 comparison ────────────────────────────────────────────────────
cat("Plot 2: Macro-F1 comparison\n")
lb2 <- lb; lb2$model_label <- factor(lb2$model_label,
                                      levels=lb2$model_label[order(lb2$f1)])
p2 <- ggplot(lb2, aes(x=model_label, y=f1,
                      fill=ifelse(model_type=="tuned","Tuned","Baseline"))) +
  geom_col(alpha=0.85) +
  geom_hline(yintercept=0.3783, linetype="dotted", colour="grey50", linewidth=0.8) +
  scale_fill_manual(values=c("Baseline"="#4292c6","Tuned"="#41ab5d"), name="") +
  coord_flip() +
  scale_y_continuous(limits=c(0,1)) +
  labs(title="Macro-F1 Score — Enriched vs Original Features",
       subtitle="Grey dotted = old 9-feature best (F1=0.3783)",
       x=NULL, y="Macro-F1") +
  theme_minimal(base_size=11) +
  theme(plot.title=element_text(face="bold"), legend.position="top")
sv(p2, "v2_02_f1_comparison")

# ── 3. Old vs New comparison bar ──────────────────────────────────────────────
cat("Plot 3: Old vs New feature set comparison\n")
compare_df <- data.frame(
  feature_set = c("Original\n(9 features)", "Original\n(9 features)",
                  "Enriched\n(80 features)", "Enriched\n(80 features)"),
  metric      = c("Accuracy", "Macro-F1", "Accuracy", "Macro-F1"),
  value       = c(0.4452, 0.3783,
                  max(lb$acc, na.rm=TRUE), max(lb$f1, na.rm=TRUE))
)
p3 <- ggplot(compare_df, aes(x=feature_set, y=value, fill=metric)) +
  geom_col(position=position_dodge(0.7), width=0.6, alpha=0.85) +
  geom_hline(yintercept=0.70, linetype="dashed", colour="#d62728", linewidth=0.8) +
  scale_fill_manual(values=c("Accuracy"="#2166ac","Macro-F1"="#d7301f"), name="Metric") +
  scale_y_continuous(limits=c(0,1), labels=scales::percent_format(1)) +
  labs(title="Feature Engineering Impact on Model Performance",
       subtitle="Adding TF-IDF + domain + publisher features",
       x=NULL, y="Score",
       caption="Red dashed = 70% target accuracy") +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold"), legend.position="top") +
  geom_text(aes(label=sprintf("%.3f", value)),
            position=position_dodge(0.7), vjust=-0.4, size=3.5, fontface="bold")
sv(p3, "v2_03_old_vs_new_comparison", w=7, h=5)

# ── 4. ROC-AUC bar ─────────────────────────────────────────────────────────────
cat("Plot 4: ROC-AUC comparison\n")
lb_roc <- lb %>% filter(!is.na(roc)) %>%
  mutate(model_label=factor(str_replace_all(model,"_"," "),
                             levels=str_replace_all(model,"_"," ")[order(roc)]))
if (nrow(lb_roc)>0) {
  p4 <- ggplot(lb_roc, aes(x=model_label, y=roc,
                           fill=ifelse(model_type=="tuned","Tuned","Baseline"))) +
    geom_col(alpha=0.85) +
    geom_hline(yintercept=0.5, linetype="dotted", colour="grey60") +
    scale_fill_manual(values=c("Baseline"="#4292c6","Tuned"="#41ab5d"), name="") +
    coord_flip() + scale_y_continuous(limits=c(0,1)) +
    labs(title="ROC-AUC (OVR) — Enriched Features", x=NULL, y="ROC-AUC") +
    theme_minimal(base_size=11) +
    theme(plot.title=element_text(face="bold"), legend.position="top")
  sv(p4, "v2_04_roc_auc_comparison")
}

# ── 5. Precision / Recall / F1 grouped ────────────────────────────────────────
cat("Plot 5: Precision/Recall/F1 grouped\n")
lb_base <- lb %>% filter(model_type=="baseline", !is.na(prec)) %>%
  select(model_label, prec, rec, f1) %>%
  pivot_longer(-model_label, names_to="metric", values_to="value") %>%
  mutate(metric=case_when(metric=="prec"~"Precision",
                          metric=="rec"~"Recall", TRUE~"Macro-F1"))
if (nrow(lb_base)>0) {
  f1_order <- lb_base %>% filter(metric=="Macro-F1") %>%
    arrange(value) %>% pull(model_label)
  lb_base$model_label <- factor(lb_base$model_label, levels=f1_order)
  p5 <- ggplot(lb_base, aes(x=model_label, y=value, fill=metric)) +
    geom_col(position="dodge", alpha=0.85) +
    scale_fill_manual(values=c("Precision"="#1f77b4","Recall"="#ff7f0e","Macro-F1"="#2ca02c"),
                      name="Metric") +
    coord_flip() + scale_y_continuous(limits=c(0,1)) +
    labs(title="Precision / Recall / F1 — Enriched Features",
         subtitle="Macro-averaged across fully_open, partially_open, closed",
         x=NULL, y="Score") +
    theme_minimal(base_size=11) +
    theme(plot.title=element_text(face="bold"), legend.position="top")
  sv(p5, "v2_05_precision_recall_f1", w=10, h=7)
}

# ── 6. Confusion matrix for best model ────────────────────────────────────────
cat("Plot 6: Confusion matrix for best model\n")
splits  <- tryCatch(readRDS(file.path(ml_dir,"ml_data_splits_v2.rds")), error=function(e) NULL)
models  <- tryCatch(readRDS(file.path(ml_dir,"model_objects_v2.rds")),  error=function(e) NULL)

if (!is.null(splits) && !is.null(models)) {
  X_te   <- splits$X_test; y_te <- splits$y_test; CLASSES <- levels(y_te)

  # Best model by accuracy
  best_nm <- lb$model[which.max(lb$acc)]
  cat(sprintf("  Best model: %s (acc=%.4f)\n", best_nm, max(lb$acc,na.rm=TRUE)))

  get_preds <- function(nm) {
    m <- models[[nm]]; if (is.null(m)) return(NULL)
    tryCatch({
      if (nm=="random_forest") {
        suppressPackageStartupMessages(library(randomForest))
        predict(m, X_te, type="class")
      } else if (nm=="xgboost") {
        suppressPackageStartupMessages(library(xgboost))
        raw <- predict(m, xgboost::xgb.DMatrix(X_te))
        pp  <- matrix(raw, ncol=length(CLASSES), byrow=TRUE)
        factor(CLASSES[apply(pp,1,which.max)], levels=CLASSES)
      } else if (nm=="gradient_boosting") {
        suppressPackageStartupMessages(library(gbm))
        raw <- gbm::predict.gbm(m, data.frame(X_te), n.trees=300, type="response")
        pp  <- raw[,,1]; colnames(pp) <- CLASSES
        factor(CLASSES[apply(pp,1,which.max)], levels=CLASSES)
      } else if (nm %in% c("svm_rbf","svm_linear")) {
        suppressPackageStartupMessages(library(e1071))
        factor(as.character(predict(m, splits[[if(nm=="svm_rbf")"X_test_s" else "X_test_s"]])),
               levels=CLASSES)
      } else if (nm=="lda") {
        suppressPackageStartupMessages(library(MASS))
        predict(m, splits$X_test_s)$class
      } else if (nm=="decision_tree") {
        suppressPackageStartupMessages(library(rpart))
        predict(m, data.frame(X_te), type="class")
      } else if (nm=="naive_bayes") {
        suppressPackageStartupMessages(library(e1071))
        predict(m, X_te, type="class")
      } else if (nm=="mlp") {
        suppressPackageStartupMessages(library(nnet))
        pp <- predict(m, splits$X_test_s, type="raw")
        if (!is.matrix(pp)) pp <- matrix(pp, nrow=nrow(splits$X_test_s))
        factor(CLASSES[apply(pp,1,which.max)], levels=CLASSES)
      } else NULL
    }, error=function(e) NULL)
  }

  plot_cm <- function(pc, y, title, fname) {
    if (is.null(pc)) return()
    pc  <- factor(as.character(pc), levels=CLASSES)
    cm  <- as.data.frame(table(Predicted=pc, Actual=factor(y, levels=CLASSES)))
    cm  <- cm %>% group_by(Actual) %>% mutate(Prop=Freq/sum(Freq)) %>% ungroup()
    p   <- ggplot(cm, aes(x=Actual, y=Predicted, fill=Prop)) +
      geom_tile(colour="white", linewidth=0.5) +
      geom_text(aes(label=sprintf("%d\n(%.0f%%)", Freq, Prop*100)),
                size=3.5, colour="black") +
      scale_fill_gradient(low="#ffffff", high="#2166ac", limits=c(0,1)) +
      scale_x_discrete(labels=function(x) str_replace_all(x,"_","\n")) +
      scale_y_discrete(labels=function(x) str_replace_all(x,"_","\n")) +
      labs(title=title, x="Actual", y="Predicted",
           caption="Diagonal = correct | Off-diagonal = errors") +
      theme_minimal(base_size=11) +
      theme(plot.title=element_text(face="bold"))
    sv(p, fname, w=6, h=5)
  }

  # Plot CMs for best and RF
  for (nm in unique(c(best_nm, "random_forest", "xgboost"))) {
    if (nm %in% names(models)) {
      pc <- get_preds(nm)
      if (!is.null(pc)) {
        plot_cm(pc, y_te,
          sprintf("Confusion Matrix — %s\n(Enriched Features, acc=%.3f)",
                  str_replace_all(nm,"_"," "),
                  mean(factor(as.character(pc),levels=CLASSES)==y_te,na.rm=TRUE)),
          sprintf("v2_cm_%s", nm))
      }
    }
  }
}

# ── 7. Feature importance (enriched) ─────────────────────────────────────────
cat("Plot 7: Feature importance (enriched)\n")
rf_imp <- tryCatch(readRDS(file.path(ml_dir,"rf_importance_v2.rds")), error=function(e) NULL)
if (!is.null(rf_imp)) {
  imp_df <- data.frame(
    feature    = rownames(rf_imp),
    importance = rf_imp[,"MeanDecreaseGini"],
    stringsAsFactors = FALSE
  ) %>% arrange(desc(importance)) %>% head(25) %>%
    mutate(
      feat_clean = str_remove(feature, "^tf_|^dom_|^pub_|^cn_"),
      group = case_when(
        startsWith(feature,"tf_")  ~ "TF-IDF",
        startsWith(feature,"dom_") | startsWith(feature,"cn_") ~ "Domain",
        startsWith(feature,"pub_") ~ "Publisher",
        TRUE ~ "Structural"
      )
    )
  imp_df$feat_clean <- factor(imp_df$feat_clean,
                               levels=imp_df$feat_clean[order(imp_df$importance)])

  p7 <- ggplot(imp_df, aes(x=feat_clean, y=importance, fill=group)) +
    geom_col(alpha=0.85) +
    scale_fill_manual(
      values=c("TF-IDF"="#4292c6","Domain"="#41ab5d",
               "Publisher"="#d7301f","Structural"="#f16913"),
      name="Feature Group") +
    coord_flip() +
    labs(title="Top 25 Feature Importances — Enriched Pipeline (RF Gini)",
         subtitle="Publisher DOI prefix = strongest discriminator for OA status",
         x=NULL, y="Mean Decrease in Gini Impurity",
         caption="pub_ieee most important: IEEE papers are overwhelmingly closed-access") +
    theme_minimal(base_size=11) +
    theme(plot.title=element_text(face="bold"), legend.position="top")
  sv(p7, "v2_07_feature_importance_enriched", w=9, h=7)
}

# ── 8. Baseline vs Tuned ──────────────────────────────────────────────────────
cat("Plot 8: Baseline vs Tuned\n")
lb_bt <- lb %>%
  mutate(base_model = str_remove(model,"_tuned$"),
         label = str_replace_all(model,"_"," "))
if (any(lb_bt$model_type=="tuned")) {
  p8 <- ggplot(lb_bt, aes(x=base_model, y=acc, fill=model_type)) +
    geom_col(position=position_dodge(0.7), width=0.6, alpha=0.85) +
    scale_fill_manual(values=c("baseline"="#4292c6","tuned"="#41ab5d"),
                      labels=c("Baseline","Tuned"), name="") +
    labs(title="Baseline vs Tuned Models (Enriched Features)",
         x=NULL, y="Accuracy", subtitle="Effect of hyperparameter tuning") +
    theme_minimal(base_size=11) +
    theme(plot.title=element_text(face="bold"),
          axis.text.x=element_text(angle=30, hjust=1), legend.position="top") +
    scale_y_continuous(limits=c(0,1), labels=scales::percent_format(1))
  sv(p8, "v2_08_baseline_vs_tuned", w=8, h=5)
}

cat(sprintf("\n=== M6bv2 Visualizations COMPLETE ===\n"))
cat(sprintf("  Figures saved to: %s\n", fig_dir))
cat(sprintf("  Total v2 figures: %d\n",
            length(list.files(fig_dir, "^v2_.*\\.png"))))
