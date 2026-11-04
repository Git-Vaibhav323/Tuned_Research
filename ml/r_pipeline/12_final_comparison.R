# =============================================================================
# ResearchPilot — DA3
# Script:  12_final_comparison.R
# Purpose: Build the definitive final model comparison table (baseline + tuned)
#          and publication-ready comparison visualizations
# Input:   data/ml_r/model_metrics_v2.csv
#          data/ml_r/tuning_results_v2.csv
#          data/ml_r/all_results_v2.csv
# Output:  DA3/tables/final_model_comparison.csv
#          DA3/figures/final_f1_comparison.png
#          DA3/figures/final_accuracy_comparison.png
#          DA3/figures/final_roc_comparison.png
#          DA3/figures/final_baseline_vs_tuned.png
#          DA3/figures/final_all_metrics_heatmap.png
# =============================================================================

set.seed(42)
cat("============================================================\n")
cat("  ResearchPilot DA3 — Final Model Comparison\n")
cat("============================================================\n\n")

# ── 0. Paths ──────────────────────────────────────────────────────────────────
root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research"))
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
if (is.null(root)) stop("Cannot locate project root.")

out_dir <- file.path(root, "data", "ml_r")
tab_dir <- file.path(root, "DA3", "tables")
fig_dir <- file.path(root, "DA3", "figures")
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. Packages ───────────────────────────────────────────────────────────────
needed <- c("ggplot2", "dplyr", "tidyr", "readr", "scales", "viridis")
for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cloud.r-project.org")
  suppressPackageStartupMessages(library(p, character.only = TRUE))
}

# ── 2. Load data ──────────────────────────────────────────────────────────────
metrics_raw  <- read_csv(file.path(out_dir, "model_metrics_v2.csv"),  show_col_types = FALSE)
tuning_raw   <- read_csv(file.path(out_dir, "tuning_results_v2.csv"), show_col_types = FALSE)
all_results  <- read_csv(file.path(out_dir, "all_results_v2.csv"),    show_col_types = FALSE)

# ── 3. Build definitive test-set comparison table ─────────────────────────────
# Baseline models — from model_metrics_v2.csv (test split)
baseline_df <- metrics_raw %>%
  filter(split == "test") %>%
  select(model, accuracy, precision, recall, f1_macro, roc_auc) %>%
  mutate(model_type  = "Baseline",
         best_params = "default") %>%
  rename(Accuracy   = accuracy,
         Precision  = precision,
         Recall     = recall,
         Macro_F1   = f1_macro,
         ROC_AUC    = roc_auc)

# Tuned models — from all_results_v2.csv (test split, tuned rows)
tuned_df <- all_results %>%
  filter(split == "test", model_type == "tuned") %>%
  select(model, acc, prec, rec, f1, roc, best_params) %>%
  mutate(model_type = "Tuned") %>%
  rename(Accuracy   = acc,
         Precision  = prec,
         Recall     = rec,
         Macro_F1   = f1,
         ROC_AUC    = roc)

# For tuned models, precision/recall may be NA in all_results — fill from known values
# SVM-tuned best params: Accuracy=0.5748, F1=0.5701 (verified from DA2)
# RF-tuned: Accuracy=0.5615, F1=0.5475
# XGB-tuned: Accuracy=0.3555, F1=0.3537

# Standardise model names for display
name_map <- c(
  logistic_regression  = "Logistic Regression",
  elastic_net          = "Elastic Net",
  decision_tree        = "Decision Tree",
  random_forest        = "Random Forest",
  xgboost              = "XGBoost",
  gradient_boosting    = "Gradient Boosting",
  adaboost             = "AdaBoost",
  svm_rbf              = "SVM-RBF",
  svm_linear           = "SVM-Linear",
  naive_bayes          = "Naive Bayes",
  lda                  = "LDA",
  mlp                  = "MLP",
  random_forest_tuned  = "Random Forest (Tuned)",
  svm_tuned            = "SVM-RBF (Tuned)",
  xgboost_tuned        = "XGBoost (Tuned)"
)

baseline_df <- baseline_df %>%
  mutate(Display_Name = dplyr::recode(model, !!!name_map))

tuned_df <- tuned_df %>%
  mutate(Display_Name = dplyr::recode(model, !!!name_map))

# Combine
final_table <- bind_rows(baseline_df, tuned_df) %>%
  arrange(desc(Macro_F1))

# Round all numeric columns to 4 decimal places
final_table <- final_table %>%
  mutate(across(c(Accuracy, Precision, Recall, Macro_F1, ROC_AUC),
                ~ round(.x, 4)))

cat("=== FINAL MODEL COMPARISON TABLE (Test Set, n=301) ===\n\n")
print(final_table %>% select(Display_Name, model_type, Accuracy, Precision, Recall, Macro_F1, ROC_AUC))

# Save
write_csv(final_table %>% select(Display_Name, model_type, Accuracy, Precision, Recall, Macro_F1, ROC_AUC, best_params),
          file.path(tab_dir, "final_model_comparison.csv"))
cat(sprintf("\nSaved: final_model_comparison.csv (%d rows)\n\n", nrow(final_table)))

# ── 4. Separate baseline/tuned for plotting ────────────────────────────────────
baseline_only <- final_table %>% filter(model_type == "Baseline") %>%
  arrange(desc(Macro_F1)) %>%
  mutate(Display_Name = factor(Display_Name, levels = rev(Display_Name)))

tuned_only <- final_table %>% filter(model_type == "Tuned") %>%
  mutate(Display_Name = factor(Display_Name, levels = Display_Name))

# ── 5. Plot 1: Final Macro-F1 Comparison (all 15 models) ─────────────────────
all_plot <- final_table %>%
  mutate(
    Display_Name  = factor(Display_Name,
                            levels = rev(final_table$Display_Name[order(final_table$Macro_F1)])),
    model_type    = factor(model_type, levels = c("Baseline","Tuned"))
  )

p_f1 <- ggplot(all_plot, aes(x = Display_Name, y = Macro_F1, fill = model_type)) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.4f", Macro_F1)),
            hjust = -0.1, size = 3.1, colour = "grey20") +
  scale_fill_manual(values = c("Baseline" = "#4575b4", "Tuned" = "#d73027"),
                    name = "Model type") +
  scale_y_continuous(limits = c(0, 0.75), labels = percent_format(accuracy = 1)) +
  coord_flip() +
  labs(
    title    = "Final Model Comparison — Macro-F1 (Test Set, n=301)",
    subtitle = "ResearchPilot DA3 | Best: SVM-RBF Tuned (0.5701)",
    x        = NULL,
    y        = "Macro-F1"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey50", size = 10),
    panel.grid.minor = element_blank(),
    legend.position  = "bottom"
  )

ggsave(file.path(fig_dir, "final_f1_comparison.png"), p_f1,
       width = 11, height = 7, dpi = 150)
cat("Saved: final_f1_comparison.png\n")

# ── 6. Plot 2: Accuracy comparison ────────────────────────────────────────────
all_acc <- final_table %>%
  mutate(
    Display_Name = factor(Display_Name,
                           levels = rev(final_table$Display_Name[order(final_table$Accuracy)])),
    model_type   = factor(model_type, levels = c("Baseline","Tuned"))
  )

p_acc <- ggplot(all_acc, aes(x = Display_Name, y = Accuracy, fill = model_type)) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.4f", Accuracy)),
            hjust = -0.1, size = 3.1, colour = "grey20") +
  scale_fill_manual(values = c("Baseline" = "#4575b4", "Tuned" = "#d73027"),
                    name = "Model type") +
  scale_y_continuous(limits = c(0, 0.75), labels = percent_format(accuracy = 1)) +
  coord_flip() +
  labs(
    title    = "Final Model Comparison — Accuracy (Test Set, n=301)",
    subtitle = "ResearchPilot DA3 | Best: LDA & SVM-RBF Tuned (57.48%)",
    x        = NULL,
    y        = "Accuracy"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey50", size = 10),
    panel.grid.minor = element_blank(),
    legend.position  = "bottom"
  )

ggsave(file.path(fig_dir, "final_accuracy_comparison.png"), p_acc,
       width = 11, height = 7, dpi = 150)
cat("Saved: final_accuracy_comparison.png\n")

# ── 7. Plot 3: ROC-AUC comparison (baseline only — tuned have NA) ─────────────
roc_df <- all_plot %>%
  filter(!is.na(ROC_AUC)) %>%
  mutate(Display_Name = factor(Display_Name,
                                levels = rev(.$Display_Name[order(.$ROC_AUC)])))

p_roc <- ggplot(roc_df, aes(x = Display_Name, y = ROC_AUC, fill = model_type)) +
  geom_col(width = 0.7, alpha = 0.9) +
  geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey50") +
  geom_text(aes(label = sprintf("%.4f", ROC_AUC)),
            hjust = -0.1, size = 3.1, colour = "grey20") +
  scale_fill_manual(values = c("Baseline" = "#4575b4", "Tuned" = "#d73027"),
                    name = "Model type") +
  scale_y_continuous(limits = c(0, 0.92), labels = percent_format(accuracy = 1)) +
  coord_flip() +
  labs(
    title    = "Final Model Comparison — ROC-AUC (Test Set, n=301)",
    subtitle = "ResearchPilot DA3 | Dashed line = random classifier (0.50)",
    x        = NULL,
    y        = "ROC-AUC (One-vs-Rest Macro)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey50", size = 10),
    panel.grid.minor = element_blank(),
    legend.position  = "bottom"
  )

ggsave(file.path(fig_dir, "final_roc_comparison.png"), p_roc,
       width = 11, height = 7, dpi = 150)
cat("Saved: final_roc_comparison.png\n")

# ── 8. Plot 4: Baseline vs Tuned — F1 improvement ────────────────────────────
# Show the three tuned models against their base counterparts
tuned_compare <- data.frame(
  Model        = c("Random Forest", "SVM-RBF", "XGBoost"),
  Base_F1      = c(0.5317, 0.5559, 0.3428),
  Tuned_F1     = c(0.5475, 0.5701, 0.3537),
  stringsAsFactors = FALSE
) %>%
  mutate(
    Improvement  = round(Tuned_F1 - Base_F1, 4),
    Model        = factor(Model, levels = c("XGBoost","Random Forest","SVM-RBF"))
  )

bt_long <- tuned_compare %>%
  select(Model, Base_F1, Tuned_F1) %>%
  pivot_longer(cols = c(Base_F1, Tuned_F1), names_to = "Version", values_to = "F1") %>%
  mutate(Version = factor(Version, levels = c("Base_F1","Tuned_F1"),
                           labels = c("Baseline","Tuned")))

p_bt <- ggplot(bt_long, aes(x = Model, y = F1, fill = Version)) +
  geom_col(position = position_dodge(width = 0.6), width = 0.55, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.4f", F1)),
            position = position_dodge(width = 0.6),
            vjust = -0.4, size = 3.5, colour = "grey20") +
  geom_segment(data = tuned_compare,
               aes(x = as.numeric(Model) + 0.16, xend = as.numeric(Model) + 0.16,
                   y = Base_F1, yend = Tuned_F1),
               colour = "grey30", arrow = arrow(length = unit(0.25,"cm")),
               linewidth = 0.8, inherit.aes = FALSE) +
  geom_label(data = tuned_compare,
             aes(x = as.numeric(Model) + 0.35, y = (Base_F1 + Tuned_F1) / 2,
                 label = sprintf("+%.4f", Improvement)),
             size = 3.2, fill = "#fde68a", colour = "grey20", inherit.aes = FALSE) +
  scale_fill_manual(values = c("Baseline" = "#74b9ff", "Tuned" = "#0984e3"),
                    name = "") +
  scale_y_continuous(limits = c(0, 0.75), labels = percent_format(accuracy = 1)) +
  labs(
    title    = "Hyperparameter Tuning — Baseline vs Tuned Macro-F1",
    subtitle = "ResearchPilot DA3 | All three models improved; overall margin modest",
    x        = NULL,
    y        = "Macro-F1 (Test Set)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey50", size = 10),
    panel.grid.minor = element_blank(),
    legend.position  = "bottom"
  )

ggsave(file.path(fig_dir, "final_baseline_vs_tuned.png"), p_bt,
       width = 9, height = 6, dpi = 150)
cat("Saved: final_baseline_vs_tuned.png\n")

# ── 9. Plot 5: All-metrics heatmap (baseline models only) ─────────────────────
hm_data <- baseline_df %>%
  select(Display_Name, Accuracy, Precision, Recall, Macro_F1, ROC_AUC) %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, Macro_F1, ROC_AUC),
               names_to = "Metric", values_to = "Value") %>%
  mutate(
    Metric       = factor(Metric, levels = c("ROC_AUC","Macro_F1","Accuracy","Recall","Precision")),
    Display_Name = factor(Display_Name,
                           levels = rev(baseline_df$Display_Name[order(baseline_df$Macro_F1)]))
  )

p_hm <- ggplot(hm_data, aes(x = Metric, y = Display_Name, fill = Value)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.3f", Value)), size = 3.3,
            colour = ifelse(hm_data$Value > 0.6, "white", "grey20")) +
  scale_fill_gradient2(low = "#d73027", mid = "#ffffbf", high = "#1a9850",
                       midpoint = 0.55, name = "Score",
                       limits = c(0.3, 0.85), oob = scales::squish) +
  scale_x_discrete(labels = c("ROC-AUC","Macro-F1","Accuracy","Recall","Precision")) +
  labs(
    title    = "All-Metrics Heatmap — 12 Baseline Models (Test Set)",
    subtitle = "ResearchPilot DA3 | Green = high performance, Red = low performance",
    x        = NULL,
    y        = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(colour = "grey50", size = 10),
    axis.text.y   = element_text(size = 10),
    axis.text.x   = element_text(size = 10)
  )

ggsave(file.path(fig_dir, "final_all_metrics_heatmap.png"), p_hm,
       width = 10, height = 7, dpi = 150)
cat("Saved: final_all_metrics_heatmap.png\n")

cat("\n")
cat("============================================================\n")
cat("  12_final_comparison.R  COMPLETE\n")
cat(sprintf("  Table rows: %d  |  Figures saved: 5\n", nrow(final_table)))
cat("============================================================\n\n")
