# =============================================================================
# ResearchPilot — Phase 2 DA2
# MILESTONE 6b: Comparative Visualizations (R + ggplot2)
# =============================================================================
# Generates all 10 required DA2 figures:
#   1. Model accuracy comparison
#   2. Macro-F1 comparison
#   3. ROC curves (OVR per class)
#   4. ROC-AUC bar chart
#   5. Precision-Recall curves
#   6. Confusion matrices (best model + best tuned)
#   7. Feature importance (Random Forest)
#   8. Precision comparison
#   9. Recall comparison
#  10. Baseline vs tuned comparison
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(ggplot2); library(tidyr); library(stringr)
})
set.seed(42)
cat("=== M6b: Comparative Visualizations ===\n\n")

# ── 0. Paths ──────────────────────────────────────────────────────────────────
root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research")) {
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
}
if (is.null(root)) root <- getwd()

ml_dir  <- file.path(root, "data", "ml_r")
rep_dir <- file.path(root, "reports", "tables")
fig_dir <- file.path(root, "reports", "figures", "phase2_r")
dir.create(fig_dir, showWarnings=FALSE, recursive=TRUE)

# ── 1. Load data ──────────────────────────────────────────────────────────────
leaderboard <- tryCatch(
  read_csv(file.path(rep_dir, "r_model_leaderboard.csv"), show_col_types=FALSE),
  error=function(e) { cat("WARNING: Leaderboard not found. Run 07_model_evaluation.R\n"); NULL }
)

splits    <- tryCatch(readRDS(file.path(ml_dir, "ml_data_splits.rds")), error=function(e) NULL)
models    <- tryCatch(readRDS(file.path(ml_dir, "model_objects_oa.rds")), error=function(e) NULL)
rf_imp    <- tryCatch(readRDS(file.path(ml_dir, "rf_importance_oa.rds")), error=function(e) NULL)

# Colour palette for models
MODEL_COLORS <- c(
  "logistic_regression" = "#1f77b4",
  "elastic_net"         = "#aec7e8",
  "decision_tree"       = "#ff7f0e",
  "random_forest"       = "#2ca02c",
  "gradient_boosting"   = "#d62728",
  "xgboost"             = "#9467bd",
  "adaboost"            = "#8c564b",
  "svm_rbf"             = "#e377c2",
  "svm_linear"          = "#f7b6d2",
  "naive_bayes"         = "#7f7f7f",
  "lda"                 = "#bcbd22",
  "knn"                 = "#17becf",
  "mlp"                 = "#393b79",
  "random_forest_tuned" = "#74c476",
  "xgboost_tuned"       = "#c5b0d5",
  "svm_tuned"           = "#f7b6d2"
)

save_fig <- function(p, name, w=9, h=6) {
  path <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(path, p, width=w, height=h, dpi=150)
  cat(sprintf("  Saved: %s.png\n", name))
}

# ─────────────────────────────────────────────────────────────────────────────
if (!is.null(leaderboard)) {

  lb <- leaderboard
  lb$model_label <- str_replace_all(lb$model, "_", " ")
  lb$model_label <- factor(lb$model_label,
                           levels=lb$model_label[order(lb$f1)])

  # ── PLOT 1: Accuracy comparison ──────────────────────────────────────────
  cat("Plot 1: Accuracy comparison\n")
  p1 <- ggplot(lb, aes(x=model_label, y=acc,
                       fill=ifelse(model_type=="tuned","Tuned","Baseline"))) +
    geom_col(alpha=0.85, show.legend=TRUE) +
    geom_hline(yintercept=max(lb$acc, na.rm=TRUE), linetype="dashed",
               colour="#d62728", linewidth=0.7) +
    scale_fill_manual(values=c("Baseline"="#4292c6","Tuned"="#41ab5d"), name="") +
    coord_flip() +
    labs(title="Model Accuracy Comparison — OA Category Classification",
         subtitle="Held-out test set | 3-class multiclass problem",
         x=NULL, y="Accuracy",
         caption="Dashed line = best accuracy achieved") +
    theme_minimal(base_size=11) +
    theme(plot.title=element_text(face="bold"), legend.position="top") +
    scale_y_continuous(limits=c(0,1), labels=scales::percent_format(1))
  save_fig(p1, "01_model_accuracy_comparison")

  # ── PLOT 2: Macro-F1 comparison ───────────────────────────────────────────
  cat("Plot 2: Macro-F1 comparison\n")
  p2 <- ggplot(lb, aes(x=model_label, y=f1,
                       fill=ifelse(model_type=="tuned","Tuned","Baseline"))) +
    geom_col(alpha=0.85) +
    geom_hline(yintercept=max(lb$f1, na.rm=TRUE), linetype="dashed",
               colour="#d62728", linewidth=0.7) +
    scale_fill_manual(values=c("Baseline"="#4292c6","Tuned"="#41ab5d"), name="") +
    coord_flip() +
    labs(title="Macro-F1 Score Comparison — OA Category Classification",
         subtitle="Macro-F1 = unweighted average of per-class F1 scores",
         x=NULL, y="Macro-F1 Score",
         caption="Dashed line = best Macro-F1 achieved") +
    theme_minimal(base_size=11) +
    theme(plot.title=element_text(face="bold"), legend.position="top") +
    scale_y_continuous(limits=c(0,1))
  save_fig(p2, "02_model_f1_comparison")

  # ── PLOT 4: ROC-AUC bar ───────────────────────────────────────────────────
  cat("Plot 4: ROC-AUC bar chart\n")
  lb_roc <- lb %>% filter(!is.na(roc)) %>%
    mutate(model_label=factor(model_label, levels=model_label[order(roc)]))
  if (nrow(lb_roc) > 0) {
    p4 <- ggplot(lb_roc, aes(x=model_label, y=roc,
                             fill=ifelse(model_type=="tuned","Tuned","Baseline"))) +
      geom_col(alpha=0.85) +
      geom_hline(yintercept=0.5, linetype="dotted", colour="grey50") +
      scale_fill_manual(values=c("Baseline"="#4292c6","Tuned"="#41ab5d"), name="") +
      coord_flip() +
      labs(title="ROC-AUC Comparison (One-vs-Rest)",
           subtitle="Macro-averaged OVR ROC-AUC across 3 OA classes",
           x=NULL, y="ROC-AUC (macro OVR)",
           caption="Dotted line = random classifier (AUC = 0.5)") +
      theme_minimal(base_size=11) +
      theme(plot.title=element_text(face="bold"), legend.position="top") +
      scale_y_continuous(limits=c(0,1))
    save_fig(p4, "04_roc_auc_comparison")
  }

  # ── PLOT 8: Precision comparison ──────────────────────────────────────────
  cat("Plot 8: Precision comparison\n")
  lb_prec <- lb %>% filter(!is.na(prec)) %>%
    mutate(model_label=factor(model_label, levels=model_label[order(prec)]))
  if (nrow(lb_prec) > 0) {
    p8 <- ggplot(lb_prec, aes(x=model_label, y=prec, fill=model_type)) +
      geom_col(alpha=0.85) +
      scale_fill_manual(values=c("baseline"="#4292c6","tuned"="#41ab5d"), name="") +
      coord_flip() +
      labs(title="Macro-Precision Comparison",
           subtitle="Macro-averaged precision across fully_open, partially_open, closed",
           x=NULL, y="Macro Precision") +
      theme_minimal(base_size=11) +
      theme(plot.title=element_text(face="bold")) +
      scale_y_continuous(limits=c(0,1))
    save_fig(p8, "08_precision_comparison")
  }

  # ── PLOT 9: Recall comparison ─────────────────────────────────────────────
  cat("Plot 9: Recall comparison\n")
  lb_rec <- lb %>% filter(!is.na(rec)) %>%
    mutate(model_label=factor(model_label, levels=model_label[order(rec)]))
  if (nrow(lb_rec) > 0) {
    p9 <- ggplot(lb_rec, aes(x=model_label, y=rec, fill=model_type)) +
      geom_col(alpha=0.85) +
      scale_fill_manual(values=c("baseline"="#4292c6","tuned"="#41ab5d"), name="") +
      coord_flip() +
      labs(title="Macro-Recall Comparison",
           subtitle="Macro-averaged recall across fully_open, partially_open, closed",
           x=NULL, y="Macro Recall") +
      theme_minimal(base_size=11) +
      theme(plot.title=element_text(face="bold")) +
      scale_y_continuous(limits=c(0,1))
    save_fig(p9, "09_recall_comparison")
  }

  # ── PLOT 10: Precision / Recall / F1 grouped bar ─────────────────────────
  cat("Plot 10: Precision-Recall-F1 grouped\n")
  lb_base <- lb %>% filter(model_type=="baseline", !is.na(prec)) %>%
    select(model_label, prec, rec, f1) %>%
    pivot_longer(-model_label, names_to="metric", values_to="value") %>%
    mutate(metric=case_when(
      metric=="prec" ~ "Precision",
      metric=="rec"  ~ "Recall",
      metric=="f1"   ~ "Macro-F1"))

  if (nrow(lb_base) > 0) {
    # Reorder by F1
    f1_order <- lb_base %>% filter(metric=="Macro-F1") %>%
      arrange(value) %>% pull(model_label)
    lb_base$model_label <- factor(lb_base$model_label, levels=f1_order)

    p10 <- ggplot(lb_base, aes(x=model_label, y=value, fill=metric)) +
      geom_col(position="dodge", alpha=0.85) +
      scale_fill_manual(
        values=c("Precision"="#1f77b4","Recall"="#ff7f0e","Macro-F1"="#2ca02c"),
        name="Metric") +
      coord_flip() +
      labs(title="Precision / Recall / F1 Comparison",
           subtitle="Why accuracy alone is insufficient: class imbalance changes metric rankings",
           x=NULL, y="Score",
           caption=paste0("fully_open:834  partially_open:723  closed:443 — ",
                          "imbalanced target requires macro averaging")) +
      theme_minimal(base_size=11) +
      theme(plot.title=element_text(face="bold"), legend.position="top") +
      scale_y_continuous(limits=c(0,1))
    save_fig(p10, "10_precision_recall_f1_grouped", w=10, h=7)
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# ROC Curves require predictions per model — use models object
# ─────────────────────────────────────────────────────────────────────────────
if (!is.null(splits) && !is.null(models)) {
  X_test   <- splits$X_test
  X_test_s <- splits$X_test_s
  y_test   <- splits$y_test
  classes  <- levels(y_test)

  compute_roc_df <- function(scores, labels, class_name) {
    # Returns data.frame(fpr, tpr, threshold, class, model)
    bin  <- as.integer(labels == class_name)
    if (sum(bin)==0 || sum(bin)==length(bin)) return(NULL)
    ord  <- order(scores, decreasing=TRUE)
    bin_o <- bin[ord]; score_o <- scores[ord]
    tp <- cumsum(bin_o);  fp <- cumsum(1-bin_o)
    tpr <- tp/max(tp);    fpr <- fp/max(fp)
    data.frame(fpr=c(0,fpr,1), tpr=c(0,tpr,1),
               class_name=class_name, stringsAsFactors=FALSE)
  }

  roc_data_list <- list()

  # Helper: get probabilities from each model
  get_probs <- function(model_nm) {
    m <- models[[model_nm]]
    if (is.null(m)) return(NULL)
    tryCatch({
      if (model_nm %in% c("random_forest")) {
        predict(m, X_test, type="prob")
      } else if (model_nm %in% c("svm_rbf","svm_linear")) {
        suppressPackageStartupMessages(library(e1071))
        pr <- predict(m, X_test_s, probability=TRUE)
        attr(pr, "probabilities")[, classes, drop=FALSE]
      } else if (model_nm == "decision_tree") {
        suppressPackageStartupMessages(library(rpart))
        predict(m, data.frame(X_test), type="prob")
      } else if (model_nm == "lda") {
        suppressPackageStartupMessages(library(MASS))
        predict(m, X_test_s)$posterior[, classes, drop=FALSE]
      } else if (model_nm == "naive_bayes") {
        if (inherits(m, "naive_bayes")) {
          suppressPackageStartupMessages(library(naivebayes))
          predict(m, X_test, type="prob")
        } else {
          suppressPackageStartupMessages(library(e1071))
          predict(m, X_test, type="raw")
        }
      } else if (model_nm %in% c("logistic_regression")) {
        suppressPackageStartupMessages(library(nnet))
        p <- predict(m, data.frame(X_test_s), type="probs")
        if (is.vector(p)) matrix(p, ncol=length(classes),
                                  dimnames=list(NULL,classes)) else p
      } else if (model_nm == "mlp") {
        suppressPackageStartupMessages(library(nnet))
        p <- predict(m, X_test_s, type="raw")
        if (is.vector(p)) matrix(p,ncol=length(classes)) else p
      } else if (model_nm == "knn") {
        NULL  # skip — kknn needs re-prediction
      } else if (model_nm == "gradient_boosting") {
        suppressPackageStartupMessages(library(gbm))
        raw <- predict(m, data.frame(X_test), n.trees=300, type="response")
        raw[,,1]
      } else if (model_nm == "xgboost") {
        suppressPackageStartupMessages(library(xgboost))
        raw <- predict(m, xgboost::xgb.DMatrix(X_test))
        matrix(raw, ncol=length(classes), byrow=TRUE)
      } else {
        NULL
      }
    }, error=function(e) NULL)
  }

  # Build ROC for top 5 models by F1 (if leaderboard available)
  top5_models <- if (!is.null(leaderboard)) {
    leaderboard %>% filter(model_type=="baseline") %>%
      arrange(desc(f1)) %>% head(5) %>% pull(model)
  } else names(models)[1:min(5,length(models))]

  cat("\nPlot 3: ROC Curves (OVR)\n")
  for (mn in top5_models) {
    pp <- get_probs(mn)
    if (is.null(pp)) next
    pp <- tryCatch({
      if (is.null(colnames(pp)) || !all(classes %in% colnames(pp))) {
        colnames(pp) <- classes[seq_len(ncol(pp))]
      }
      pp
    }, error=function(e) NULL)
    if (is.null(pp)) next

    for (cls in classes) {
      if (!cls %in% colnames(pp)) next
      rdf <- compute_roc_df(pp[,cls], y_test, cls)
      if (!is.null(rdf)) {
        rdf$model <- mn
        roc_data_list[[paste0(mn,"_",cls)]] <- rdf
      }
    }
  }

  if (length(roc_data_list) > 0) {
    roc_df <- do.call(rbind, roc_data_list) %>%
      mutate(model_label = str_replace_all(model,"_"," "),
             class_label = str_replace_all(class_name,"_"," "))

    p3 <- ggplot(roc_df, aes(x=fpr, y=tpr, colour=model_label)) +
      geom_line(linewidth=0.8, alpha=0.85) +
      geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey70") +
      facet_wrap(~class_label, nrow=1) +
      scale_colour_brewer(palette="Set1", name="Model") +
      labs(title="ROC Curves — One-vs-Rest (Top 5 Models by F1)",
           subtitle="Each panel shows the ROC for predicting that OA class vs all others",
           x="False Positive Rate", y="True Positive Rate",
           caption="Dashed line = random classifier") +
      theme_minimal(base_size=11) +
      theme(plot.title=element_text(face="bold"), legend.position="bottom",
            strip.text=element_text(face="bold"))
    save_fig(p3, "03_roc_curves_ovr", w=12, h=5)
  }

  # ── PLOT 5: Precision-Recall curve ───────────────────────────────────────
  cat("Plot 5: Precision-Recall curves\n")
  pr_data_list <- list()
  for (mn in top5_models) {
    pp <- get_probs(mn)
    if (is.null(pp)) next
    if (is.null(colnames(pp)) || !all(classes %in% colnames(pp))) {
      colnames(pp) <- classes[seq_len(ncol(pp))]
    }
    for (cls in classes) {
      if (!cls %in% colnames(pp)) next
      scores <- pp[,cls]
      bin    <- as.integer(y_test == cls)
      if (sum(bin)==0) next
      ord    <- order(scores, decreasing=TRUE)
      tp_c   <- cumsum(bin[ord])
      fp_c   <- cumsum(1-bin[ord])
      prec_c <- tp_c / (tp_c + fp_c)
      rec_c  <- tp_c / sum(bin)
      prec_c[is.nan(prec_c)] <- 0
      pr_data_list[[paste0(mn,"_",cls)]] <- data.frame(
        recall=c(0,rec_c,1), precision=c(1,prec_c,0),
        model=mn, class_name=cls, stringsAsFactors=FALSE)
    }
  }
  if (length(pr_data_list) > 0) {
    pr_df <- do.call(rbind, pr_data_list) %>%
      mutate(model_label=str_replace_all(model,"_"," "),
             class_label=str_replace_all(class_name,"_"," "))
    p5 <- ggplot(pr_df, aes(x=recall, y=precision, colour=model_label)) +
      geom_line(linewidth=0.8, alpha=0.85) +
      facet_wrap(~class_label, nrow=1) +
      scale_colour_brewer(palette="Set1", name="Model") +
      labs(title="Precision-Recall Curves — Top 5 Models",
           subtitle="Higher area under curve = better model for that class",
           x="Recall", y="Precision") +
      theme_minimal(base_size=11) +
      theme(plot.title=element_text(face="bold"), legend.position="bottom",
            strip.text=element_text(face="bold"))
    save_fig(p5, "05_precision_recall_curves", w=12, h=5)
  }

  # ── PLOT 6: Confusion matrices ────────────────────────────────────────────
  cat("Plot 6: Confusion matrices\n")
  plot_confusion_matrix <- function(pred_c, true_labels, title, fname) {
    classes <- levels(true_labels)
    cm_df   <- as.data.frame(table(Predicted=factor(pred_c,levels=classes),
                                   Actual=factor(true_labels,levels=classes)))
    # Normalise by actual (per-column)
    cm_df <- cm_df %>%
      group_by(Actual) %>%
      mutate(Proportion=Freq/sum(Freq)) %>%
      ungroup()

    p <- ggplot(cm_df, aes(x=Actual, y=Predicted, fill=Proportion)) +
      geom_tile(colour="white", linewidth=0.5) +
      geom_text(aes(label=sprintf("%d\n(%.0f%%)", Freq, Proportion*100)),
                size=3.5, colour="black") +
      scale_fill_gradient(low="#ffffff", high="#2166ac",
                          limits=c(0,1), name="Proportion") +
      scale_x_discrete(labels=function(x) str_replace_all(x,"_","\n")) +
      scale_y_discrete(labels=function(x) str_replace_all(x,"_","\n")) +
      labs(title=title, x="Actual Class", y="Predicted Class",
           caption="Diagonal = correct predictions | Off-diagonal = errors") +
      theme_minimal(base_size=11) +
      theme(plot.title=element_text(face="bold"),
            axis.text=element_text(size=10))
    save_fig(p, fname, w=6, h=5)
  }

  # Best model confusion matrix
  if (!is.null(leaderboard) && nrow(leaderboard) > 0) {
    best_model_nm <- leaderboard$model[which.max(leaderboard$f1)]
    pp <- get_probs(best_model_nm)
    if (!is.null(pp)) {
      if (is.null(colnames(pp))) colnames(pp) <- classes
      pc <- factor(classes[apply(pp,1,which.max)], levels=classes)
      plot_confusion_matrix(pc, y_test,
        sprintf("Confusion Matrix — %s (Best F1)", str_replace_all(best_model_nm,"_"," ")),
        "06a_confusion_matrix_best_model")
    }

    # Random Forest confusion matrix (always useful for feature importance comparison)
    if ("random_forest" %in% names(models)) {
      rf_pp <- tryCatch(predict(models$random_forest, X_test, type="prob"), error=function(e) NULL)
      if (!is.null(rf_pp)) {
        rf_pc <- factor(classes[apply(rf_pp,1,which.max)], levels=classes)
        plot_confusion_matrix(rf_pc, y_test,
          "Confusion Matrix — Random Forest",
          "06b_confusion_matrix_rf")
      }
    }
  }
}

# ── PLOT 7: Feature Importance (Random Forest) ────────────────────────────────
if (!is.null(rf_imp)) {
  cat("Plot 7: Feature importance\n")
  imp_df <- data.frame(
    feature    = rownames(rf_imp),
    importance = rf_imp[,"MeanDecreaseGini"],
    stringsAsFactors=FALSE
  ) %>%
    arrange(desc(importance)) %>%
    head(15) %>%
    mutate(
      feature_label = str_replace_all(feature,"_"," "),
      feature_type  = case_when(
        feature %in% c("publication_year","paper_age","has_doi","recent_paper") ~ "Metadata",
        feature %in% c("keyword_count","concept_count","abstract_word_count","title_word_count") ~ "Content",
        TRUE ~ "Engineered"
      )
    )
  imp_df$feature_label <- factor(imp_df$feature_label,
                                  levels=imp_df$feature_label[order(imp_df$importance)])

  p7 <- ggplot(imp_df, aes(x=feature_label, y=importance, fill=feature_type)) +
    geom_col(alpha=0.85) +
    scale_fill_manual(
      values=c("Metadata"="#4292c6","Content"="#41ab5d","Engineered"="#fd8d3c"),
      name="Feature Type") +
    coord_flip() +
    labs(title="Feature Importance — Random Forest (Gini)",
         subtitle="Top features contributing to OA category predictions",
         x=NULL, y="Mean Decrease in Gini Impurity",
         caption="Note: importance measures contribution to model predictions,\nnot causal effect on OA status") +
    theme_minimal(base_size=11) +
    theme(plot.title=element_text(face="bold"), legend.position="top")
  save_fig(p7, "07_feature_importance_rf", w=8, h=6)
}

# ── PLOT: Baseline vs Tuned ───────────────────────────────────────────────────
if (!is.null(leaderboard)) {
  cat("Plot 10: Baseline vs Tuned\n")
  lb_compare <- leaderboard %>%
    filter(model_type %in% c("baseline","tuned")) %>%
    mutate(
      base_model = str_remove(model, "_tuned$"),
      model_label = str_replace_all(model,"_"," ")
    )

  if (any(lb_compare$model_type=="tuned")) {
    p_bvt <- ggplot(lb_compare, aes(x=base_model, y=f1,
                                    fill=model_type, group=model_type)) +
      geom_col(position=position_dodge(0.7), width=0.6, alpha=0.85) +
      scale_fill_manual(values=c("baseline"="#4292c6","tuned"="#41ab5d"),
                        labels=c("Baseline","Tuned"), name="") +
      labs(title="Baseline vs Tuned Model Comparison",
           subtitle="Effect of hyperparameter tuning on Macro-F1 (test set)",
           x=NULL, y="Macro-F1 Score",
           caption="Tuning used 5-fold CV on training data only") +
      theme_minimal(base_size=11) +
      theme(plot.title=element_text(face="bold"),
            axis.text.x=element_text(angle=30, hjust=1), legend.position="top") +
      scale_y_continuous(limits=c(0,1))
    save_fig(p_bvt, "10_baseline_vs_tuned", w=8, h=5)
  }
}

cat("\n=== M6b Visualizations COMPLETE ===\n")
cat(sprintf("  All figures saved to: %s\n", fig_dir))
figs <- list.files(fig_dir, "*.png")
cat(sprintf("  Total figures: %d\n", length(figs)))
