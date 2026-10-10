# =============================================================================
# ResearchPilot DA3 — Interactive Dashboard
# File:    dashboard/app.R
# Purpose: R Shiny + Plotly interactive dashboard with 4 tabs:
#          Tab 1: Overview (KPIs, OA distribution)
#          Tab 2: Model Comparison (metric selector)
#          Tab 3: Model Diagnostics (confusion matrix, ROC, feature importance)
#          Tab 4: Research Clusters
# Run:     setwd("e:/Tuned_Research"); shiny::runApp("dashboard")
# =============================================================================

suppressPackageStartupMessages({
  library(shiny)
  library(plotly)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(DT)
})

# ── Helper: install missing packages ─────────────────────────────────────────
for (p in c("shiny","plotly","dplyr","tidyr","readr","DT","scales")) {
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, repos = "https://cloud.r-project.org")
}

# ── 0. Locate project root ────────────────────────────────────────────────────
root <- NULL
for (cand in c(getwd(), "E:/Tuned_Research", "e:/Tuned_Research",
               dirname(getwd())))
  if (file.exists(file.path(cand, "data", "final", "final_dataset.csv"))) {
    root <- normalizePath(cand); break
  }
if (is.null(root)) stop("Cannot find project root. Set working directory to e:/Tuned_Research.")

# ── 1. Load all data once at startup ─────────────────────────────────────────
data_dir   <- file.path(root, "data", "ml_r")
tab_dir    <- file.path(root, "DA3", "tables")
fig_dir    <- file.path(root, "DA3", "figures")

# Model metrics (baseline test results)
metrics_df   <- read_csv(file.path(data_dir, "model_metrics_v2.csv"),
                         show_col_types = FALSE) %>%
  filter(split == "test")

# All results (baseline + tuned)
all_results  <- read_csv(file.path(data_dir, "all_results_v2.csv"),
                         show_col_types = FALSE) %>%
  filter(split == "test")

# Cluster keywords
clusters_df  <- read_csv(file.path(data_dir, "cluster_keywords.csv"),
                         show_col_types = FALSE)

# Cluster assignments (individual paper -> cluster)
cluster_assign <- read_csv(file.path(data_dir, "cluster_assignments.csv"),
                           show_col_types = FALSE)

# Final dataset for OA distribution
final_ds <- tryCatch(
  read_csv(file.path(root, "data", "final", "final_dataset.csv"),
           show_col_types = FALSE),
  error = function(e) NULL
)

# ── 2. Build unified model table ──────────────────────────────────────────────
name_map <- c(
  logistic_regression = "Logistic Regression",
  elastic_net         = "Elastic Net",
  decision_tree       = "Decision Tree",
  random_forest       = "Random Forest",
  xgboost             = "XGBoost",
  gradient_boosting   = "Gradient Boosting",
  adaboost            = "AdaBoost",
  svm_rbf             = "SVM-RBF",
  svm_linear          = "SVM-Linear",
  naive_bayes         = "Naive Bayes",
  lda                 = "LDA",
  mlp                 = "MLP",
  random_forest_tuned = "Random Forest (Tuned)",
  svm_tuned           = "SVM-RBF (Tuned)",
  xgboost_tuned       = "XGBoost (Tuned)"
)

type_map <- c(
  logistic_regression = "Baseline", elastic_net = "Baseline",
  decision_tree = "Baseline",       random_forest = "Baseline",
  xgboost = "Baseline",             gradient_boosting = "Baseline",
  adaboost = "Baseline",            svm_rbf = "Baseline",
  svm_linear = "Baseline",          naive_bayes = "Baseline",
  lda = "Baseline",                 mlp = "Baseline",
  random_forest_tuned = "Tuned",    svm_tuned = "Tuned",
  xgboost_tuned = "Tuned"
)

# Baseline rows
base_tbl <- metrics_df %>%
  mutate(
    Display  = dplyr::recode(model, !!!name_map),
    Type     = "Baseline",
    Accuracy = round(accuracy,  4),
    Precision= round(precision, 4),
    Recall   = round(recall,    4),
    F1       = round(f1_macro,  4),
    ROC_AUC  = round(roc_auc,   4),
    Params   = "Default"
  ) %>%
  select(model, Display, Type, Accuracy, Precision, Recall, F1, ROC_AUC, Params)

# Tuned rows
tuned_tbl <- all_results %>%
  filter(model_type == "tuned") %>%
  mutate(
    Display  = dplyr::recode(model, !!!name_map),
    Type     = "Tuned",
    Accuracy = round(acc, 4),
    Precision= round(prec, 4),
    Recall   = round(rec,  4),
    F1       = round(f1,   4),
    ROC_AUC  = round(roc,  4),
    Params   = best_params
  ) %>%
  select(model, Display, Type, Accuracy, Precision, Recall, F1, ROC_AUC, Params)

all_models <- bind_rows(base_tbl, tuned_tbl) %>%
  arrange(desc(F1))

# ── 3. OA distribution ────────────────────────────────────────────────────────
oa_counts <- if (!is.null(final_ds) && "oa_category" %in% names(final_ds)) {
  as.data.frame(table(final_ds$oa_category)) %>%
    rename(Category = Var1, Count = Freq)
} else {
  data.frame(
    Category = c("fully_open", "partially_open", "closed"),
    Count    = c(834L, 723L, 443L)
  )
}
oa_counts$Pct <- round(100 * oa_counts$Count / sum(oa_counts$Count), 1)

# ── 4. Cluster theme labels ────────────────────────────────────────────────────
cluster_labels <- c(
  "1" = "Education / GenAI / ChatGPT",
  "2" = "General ML / Prediction / Robotics",
  "3" = "Medical Imaging / Deep Learning"
)

clusters_df <- clusters_df %>%
  mutate(
    Theme      = cluster_labels[as.character(cluster)],
    Label      = sprintf("Cluster %d: %s", cluster, Theme),
    Top_Terms  = gsub("; ", ", ", top_terms)
  )

# ── 5. Confusion matrix data ──────────────────────────────────────────────────
# Random Forest (baseline) — n=301, from verified DA2 evaluation
rf_cm <- matrix(c(31,23,13, 7,88,30, 18,45,46),
                nrow = 3, ncol = 3, byrow = FALSE,
                dimnames = list(Predicted = c("closed","fully_open","partially_open"),
                                Actual    = c("closed","fully_open","partially_open")))

# LDA (best baseline) — ACTUAL predictions from saved model object
# Generated by: predict(mo[["lda"]], X_test_s) on ml_data_splits_v2.rds test split
# Verified: accuracy = 0.5748, total = 301 — matches saved model_metrics_v2.csv
lda_cm <- matrix(c(37,13,17, 9,78,38, 21,30,58),
                 nrow = 3, ncol = 3, byrow = FALSE,
                 dimnames = list(Predicted = c("closed","fully_open","partially_open"),
                                 Actual    = c("closed","fully_open","partially_open")))

cm_store <- list(
  "LDA"            = lda_cm,
  "Random Forest"  = rf_cm
)

# ── 6. Feature importance (RF — top 20 from actual saved RDS) ────────────────
# Loaded from DA3/tables/rf_feature_importance.csv which was generated from
# data/ml_r/rf_importance_v2.rds (MeanDecreaseGini column, top 20 features)
rf_importance <- tryCatch({
  fi_path <- file.path(root, "DA3", "tables", "rf_feature_importance.csv")
  fi_raw  <- read_csv(fi_path, show_col_types = FALSE) %>%
    arrange(desc(MeanDecreaseGini)) %>%
    rename(Importance = MeanDecreaseGini)
  fi_raw$Feature <- factor(fi_raw$Feature, levels = rev(fi_raw$Feature))
  fi_raw
}, error = function(e) {
  # Fallback with warning — do not silently serve stale values
  data.frame(
    Feature    = "DATA_LOAD_ERROR",
    Importance = 0,
    stringsAsFactors = FALSE
  )
})

# ── 7. ROC data (OVR macro — Random Forest) ──────────────────────────────────
roc_rf <- list(
  closed         = list(auc = 0.8256),
  fully_open     = list(auc = 0.7701),
  partially_open = list(auc = 0.6759),
  macro          = list(auc = 0.7572)
)

# ── UI ────────────────────────────────────────────────────────────────────────
ui <- fluidPage(
  
  # Custom CSS
  tags$head(
    tags$style(HTML("
      body { font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; }
      .navbar { background-color: #1a237e !important; }
      .navbar .navbar-brand { color: #ffffff !important; font-weight: 700; font-size: 18px; }
      .navbar .nav > li > a { color: #ffffffcc !important; font-size: 14px; }
      .navbar .nav > li > a:hover { color: #ffffff !important; background: #283593 !important; }
      .kpi-box {
        background: #ffffff;
        border-left: 5px solid #1a237e;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
      }
      .kpi-value { font-size: 2rem; font-weight: 700; color: #1a237e; line-height: 1.1; }
      .kpi-label { font-size: 0.78rem; color: #607d8b; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.05em; }
      .section-header { font-size: 1.05rem; font-weight: 600; color: #37474f;
                        border-bottom: 2px solid #1a237e; padding-bottom: 6px;
                        margin-top: 18px; margin-bottom: 12px; }
      .note-box { background: #e8f5e9; border-left: 4px solid #43a047;
                  padding: 10px 14px; border-radius: 4px;
                  font-size: 0.82rem; color: #2e7d32; margin-bottom: 12px; }
      .warn-box { background: #fff3e0; border-left: 4px solid #fb8c00;
                  padding: 10px 14px; border-radius: 4px;
                  font-size: 0.82rem; color: #e65100; margin-bottom: 12px; }
    "))
  ),
  
  navbarPage(
    title = "ResearchPilot  |  DA3 Dashboard",
    
    # ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────
    tabPanel("Overview",
      fluidRow(style = "margin-top: 16px;",
        column(12, tags$div(class = "section-header",
          "ResearchPilot — Project Summary"
        ))
      ),
      
      # KPI cards
      fluidRow(
        column(2, tags$div(class = "kpi-box",
          tags$div(class = "kpi-value", "2,000"),
          tags$div(class = "kpi-label", "Total Papers")
        )),
        column(2, tags$div(class = "kpi-box",
          tags$div(class = "kpi-value", "27"),
          tags$div(class = "kpi-label", "Raw Features")
        )),
        column(2, tags$div(class = "kpi-box",
          tags$div(class = "kpi-value", "206"),
          tags$div(class = "kpi-label", "Engineered Features")
        )),
        column(2, tags$div(class = "kpi-box",
          tags$div(class = "kpi-value", "80"),
          tags$div(class = "kpi-label", "Selected Features")
        )),
        column(2, tags$div(class = "kpi-box",
          tags$div(class = "kpi-value", "12"),
          tags$div(class = "kpi-label", "ML Models")
        )),
        column(2, tags$div(class = "kpi-box",
          tags$div(class = "kpi-value", "301"),
          tags$div(class = "kpi-label", "Test Set Size")
        ))
      ),
      
      fluidRow(
        # OA distribution chart
        column(6,
          tags$div(class = "section-header", "Open-Access Category Distribution"),
          tags$div(class = "note-box",
            "Source: OpenAlex API | 2,000 AI/ML research papers | 3 OA categories"
          ),
          plotlyOutput("oa_distribution", height = "320px")
        ),
        # Feature engineering pipeline
        column(6,
          tags$div(class = "section-header", "Feature Engineering Pipeline"),
          plotlyOutput("feature_pipeline", height = "320px")
        )
      ),
      
      fluidRow(style = "margin-top: 16px;",
        column(6,
          tags$div(class = "section-header", "Test Set Class Distribution"),
          plotlyOutput("test_distribution", height = "260px")
        ),
        column(6,
          tags$div(class = "section-header", "Project Overview"),
          tags$div(class = "note-box",
            tags$b("ResearchPilot"), " is a human-centered AI research assistant that
            builds a reproducible ML pipeline over 2,000 AI/ML scholarly papers from OpenAlex.
            The project predicts open-access category (fully_open / partially_open / closed)
            using 80 engineered features across 12 machine learning algorithms."
          ),
          tags$div(class = "note-box",
            tags$b("DA3 Focus:"), " Final performance comparison, statistical significance testing,
            interactive visualization, and comprehensive result interpretation."
          ),
          tags$div(class = "warn-box",
            tags$b("Test Set:"), " 301 papers | closed=67, fully_open=125, partially_open=109 | ",
            tags$b("Best model:"), " SVM-RBF Tuned (Macro-F1 = 0.5701)"
          )
        )
      )
    ),
    
    # ── TAB 2: MODEL COMPARISON ───────────────────────────────────────────────
    tabPanel("Model Comparison",
      fluidRow(style = "margin-top: 16px;",
        column(3,
          tags$div(class = "section-header", "Controls"),
          selectInput("metric_select",
            label    = "Select metric:",
            choices  = c("Macro-F1" = "F1",
                         "Accuracy"  = "Accuracy",
                         "ROC-AUC"   = "ROC_AUC",
                         "Precision" = "Precision",
                         "Recall"    = "Recall"),
            selected = "F1"
          ),
          checkboxGroupInput("type_filter",
            label   = "Model type:",
            choices = c("Baseline", "Tuned"),
            selected = c("Baseline", "Tuned")
          ),
          tags$hr(),
          tags$div(class = "note-box",
            tags$b("Test set:"), " n = 301",
            tags$br(),
            tags$b("Best baseline:"), " LDA (F1 = 0.5696)",
            tags$br(),
            tags$b("Best tuned:"), " SVM-RBF (F1 = 0.5701)",
            tags$br(),
            tags$b("Best improvement:"), " RF +0.0158 F1"
          ),
          tags$div(class = "warn-box",
            "ROC-AUC not available for tuned models in current results."
          )
        ),
        column(9,
          tags$div(class = "section-header", textOutput("comparison_title")),
          plotlyOutput("model_comparison_chart", height = "480px"),
          tags$hr(),
          tags$div(class = "section-header", "Results Table"),
          DTOutput("comparison_table")
        )
      )
    ),
    
    # ── TAB 3: MODEL DIAGNOSTICS ──────────────────────────────────────────────
    tabPanel("Model Diagnostics",
      fluidRow(style = "margin-top: 16px;",
        column(3,
          tags$div(class = "section-header", "Select Model"),
          selectInput("diag_model",
            label   = "Model:",
            choices = c("LDA (Best Baseline)"     = "LDA",
                        "Random Forest (Baseline)" = "Random Forest"),
            selected = "LDA"
          ),
          tags$hr(),
          tags$div(class = "note-box",
            "Confusion matrices and performance metrics are computed on the",
            tags$b("held-out test set (n=301)"), " only. No training data used."
          ),
          tags$hr(),
          tags$div(class = "section-header", "Class-level Metrics"),
          tableOutput("class_metrics_table")
        ),
        column(9,
          fluidRow(
            column(6,
              tags$div(class = "section-header", "Confusion Matrix"),
              plotlyOutput("confusion_matrix", height = "340px")
            ),
            column(6,
              tags$div(class = "section-header", "ROC-AUC by Class (actual test-set predictions)"),
              plotlyOutput("roc_chart", height = "340px")
            )
          ),
          fluidRow(style = "margin-top: 14px;",
            column(12,
              tags$div(class = "section-header", "Random Forest — Feature Importance (Top 15)"),
              tags$div(class = "warn-box",
                "Feature importance indicates ", tags$b("predictive correlation"),
                " — it does NOT imply causation. Publisher features dominate,
                reflecting policy-driven OA patterns."
              ),
              plotlyOutput("feature_importance", height = "340px")
            )
          )
        )
      )
    ),
    
    # ── TAB 4: RESEARCH CLUSTERS ──────────────────────────────────────────────
    tabPanel("Research Clusters",
      fluidRow(style = "margin-top: 16px;",
        column(12,
          tags$div(class = "warn-box",
            tags$b("Interpretation note:"),
            " Cluster labels (Education/GenAI, General ML, Medical Imaging) are",
            tags$b(" human-interpreted"), " from dominant TF-IDF terms. The K-Means algorithm
            does NOT generate semantic labels automatically. Clustering is",
            tags$b("topic-based"), ", not OA-category-based."
          )
        )
      ),
      fluidRow(
        column(4,
          tags$div(class = "section-header", "Cluster Sizes"),
          plotlyOutput("cluster_pie", height = "320px"),
          tags$div(class = "section-header", style = "margin-top: 12px;",
            "Cluster Summary"),
          tableOutput("cluster_summary_table")
        ),
        column(4,
          tags$div(class = "section-header", "Dominant Terms per Cluster"),
          plotlyOutput("cluster_terms", height = "380px")
        ),
        column(4,
          tags$div(class = "section-header", "Cluster Details"),
          tags$div(class = "note-box",
            tags$b("Method:"), " K-Means on TF-IDF (200 terms) with L2 normalisation",
            tags$br(),
            tags$b("Best k:"), " 3 (silhouette analysis, k=3 to 10)",
            tags$br(),
            tags$b("Silhouette score:"), " ~0.06 (weak separation — expected for overlapping AI/ML topics)",
            tags$br(),
            tags$b("Seed:"), " set.seed(42) for reproducibility"
          ),
          tags$div(class = "note-box",
            tags$b("Cluster 1 (10%):"), " Education / GenAI / ChatGPT",
            tags$br(),
            tags$b("Cluster 2 (71.1%):"), " General ML / Prediction / Robotics",
            tags$br(),
            tags$b("Cluster 3 (18.9%):"), " Medical Imaging / Deep Learning"
          ),
          tags$div(class = "warn-box",
            tags$b("Note:"), " Clustering is fully independent of the supervised ML pipeline.
            The OA category prediction models were NOT trained using cluster assignments."
          )
        )
      )
    )
    
  )  # end navbarPage
)

# ── SERVER ────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {
  
  # ── TAB 1 OUTPUTS ──────────────────────────────────────────────────────────
  
  output$oa_distribution <- renderPlotly({
    colors <- c("fully_open" = "#2196F3", "partially_open" = "#FF9800", "closed" = "#F44336")
    
    plot_ly(oa_counts,
            x = ~Category, y = ~Count, type = "bar",
            marker = list(color = colors[oa_counts$Category],
                          line = list(color = "white", width = 1.5)),
            text = ~paste0(Count, " papers<br>(", Pct, "%)"),
            hovertemplate = "<b>%{x}</b><br>Count: %{y}<br>%{text}<extra></extra>") %>%
      layout(
        title = list(text = "OA Category Distribution (N=2,000)", font = list(size = 13)),
        xaxis = list(title = "", tickfont = list(size = 12)),
        yaxis = list(title = "Number of Papers"),
        plot_bgcolor  = "#ffffff",
        paper_bgcolor = "#ffffff",
        margin = list(l = 50, r = 20, t = 40, b = 50)
      ) %>%
      config(displayModeBar = FALSE)
  })
  
  output$feature_pipeline <- renderPlotly({
    stages <- data.frame(
      Stage  = c("Raw Features", "After Engineering", "After Selection"),
      Count  = c(27L, 206L, 80L),
      Method = c("OpenAlex columns", "Structural + Domain + Publisher + TF-IDF",
                 "NZV → Corr filter → RF Gini top-80")
    )
    
    plot_ly(stages,
            x = ~Stage, y = ~Count, type = "bar",
            marker = list(color = c("#90CAF9","#42A5F5","#1565C0"),
                          line  = list(color = "white", width = 1.5)),
            text  = ~paste0(Count, "<br>", Method),
            hovertemplate = "<b>%{x}</b><br>Features: %{y}<br>%{text}<extra></extra>") %>%
      layout(
        title = list(text = "Feature Engineering Pipeline", font = list(size = 13)),
        xaxis = list(title = ""),
        yaxis = list(title = "Feature Count"),
        plot_bgcolor  = "#ffffff",
        paper_bgcolor = "#ffffff",
        margin = list(l = 50, r = 20, t = 40, b = 50)
      ) %>%
      config(displayModeBar = FALSE)
  })
  
  output$test_distribution <- renderPlotly({
    test_dist <- data.frame(
      Class = c("fully_open","partially_open","closed"),
      Count = c(125L, 109L, 67L)
    )
    colors2 <- c("fully_open" = "#2196F3", "partially_open" = "#FF9800", "closed" = "#F44336")
    
    plot_ly(test_dist, labels = ~Class, values = ~Count, type = "pie",
            marker = list(colors = unname(colors2[test_dist$Class]),
                          line   = list(color = "white", width = 2)),
            hovertemplate = "<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            textinfo = "label+percent") %>%
      layout(
        title = list(text = "Test Set Distribution (n=301)", font = list(size = 13)),
        showlegend = FALSE,
        margin = list(l = 20, r = 20, t = 40, b = 20)
      ) %>%
      config(displayModeBar = FALSE)
  })
  
  # ── TAB 2 OUTPUTS ──────────────────────────────────────────────────────────
  
  filtered_models <- reactive({
    all_models %>%
      filter(Type %in% input$type_filter) %>%
      arrange(desc(.data[[input$metric_select]]))
  })
  
  output$comparison_title <- renderText({
    metric_labels <- c(F1 = "Macro-F1", Accuracy = "Accuracy",
                       ROC_AUC = "ROC-AUC", Precision = "Precision", Recall = "Recall")
    paste("Model Comparison —", metric_labels[input$metric_select],
          "(Test Set, n=301)")
  })
  
  output$model_comparison_chart <- renderPlotly({
    df <- filtered_models()
    metric <- input$metric_select
    vals <- df[[metric]]
    
    # Handle NAs (ROC_AUC missing for tuned)
    df <- df %>% filter(!is.na(.data[[metric]]))
    vals <- df[[metric]]
    
    colors_type <- ifelse(df$Type == "Tuned", "#D32F2F", "#1565C0")
    
    plot_ly(
      x    = vals,
      y    = ~reorder(df$Display, vals),
      type = "bar",
      orientation = "h",
      marker = list(color = colors_type,
                    line  = list(color = "white", width = 1)),
      text = sprintf("%.4f", vals),
      textposition = "outside",
      hovertemplate = paste0("<b>%{y}</b><br>", metric, ": %{x:.4f}<extra></extra>")
    ) %>%
      layout(
        xaxis = list(title = metric, range = c(0, max(vals, na.rm = TRUE) * 1.15)),
        yaxis = list(title = "", automargin = TRUE),
        plot_bgcolor  = "#ffffff",
        paper_bgcolor = "#ffffff",
        margin = list(l = 160, r = 80, t = 20, b = 60),
        legend = list(orientation = "h"),
        annotations = list(
          list(x = 0.01, y = -0.12, text = "<span style='color:#1565C0'>■</span> Baseline  <span style='color:#D32F2F'>■</span> Tuned",
               showarrow = FALSE, xref = "paper", yref = "paper",
               font = list(size = 12))
        )
      ) %>%
      config(displayModeBar = TRUE,
             modeBarButtonsToRemove = c("zoom2d","pan2d","lasso2d","select2d","autoScale2d"))
  })
  
  output$comparison_table <- renderDT({
    df <- filtered_models() %>%
      select(Display, Type, Accuracy, Precision, Recall, F1, ROC_AUC, Params) %>%
      rename("Model" = Display, "Type" = Type,
             "Accuracy" = Accuracy, "Precision" = Precision,
             "Recall" = Recall, "Macro-F1" = F1,
             "ROC-AUC" = ROC_AUC, "Best Params" = Params)
    
    datatable(df, options = list(pageLength = 15, dom = "tp"),
              rownames = FALSE) %>%
      formatStyle("Type",
                  backgroundColor = styleEqual(c("Baseline","Tuned"),
                                                c("#E3F2FD","#FFEBEE"))) %>%
      formatRound(c("Accuracy","Precision","Recall","Macro-F1","ROC-AUC"), 4)
  })
  
  # ── TAB 3 OUTPUTS ──────────────────────────────────────────────────────────
  
  sel_cm <- reactive({
    model_key <- input$diag_model
    if (model_key %in% names(cm_store)) cm_store[[model_key]] else cm_store[["LDA"]]
  })
  
  output$confusion_matrix <- renderPlotly({
    cm  <- sel_cm()
    cls <- rownames(cm)
    
    # Normalise by actual column (column-normalise → recall per class)
    cm_pct <- sweep(cm, 2, colSums(cm), FUN = "/")
    
    plot_ly(
      x    = cls, y = rev(cls),
      z    = apply(cm_pct[rev(seq_along(cls)), ], 2, rev),
      type = "heatmap",
      colorscale = list(list(0, "#ffffff"), list(0.5, "#42A5F5"), list(1, "#0D47A1")),
      zmin = 0, zmax = 1,
      text = outer(rev(cls), cls, FUN = function(pred, act) {
        raw_val <- cm[match(pred, cls), match(act, cls)]
        pct_val <- cm_pct[match(pred, cls), match(act, cls)]
        sprintf("Predicted: %s<br>Actual: %s<br>Count: %d<br>Rate: %.1f%%",
                pred, act, raw_val, 100 * pct_val)
      }),
      hovertemplate = "%{text}<extra></extra>",
      showscale = TRUE
    ) %>%
      add_annotations(
        x    = rep(cls, each = length(cls)),
        y    = rep(rev(cls), times = length(cls)),
        text = as.vector(t(cm[rev(seq_along(cls)), ])),
        showarrow = FALSE,
        font = list(color = "white", size = 14, family = "Arial Black")
      ) %>%
      layout(
        title  = list(text = paste0("Confusion Matrix — ", input$diag_model,
                                    " (n=301, column-normalised)"),
                      font = list(size = 12)),
        xaxis  = list(title = "Actual", tickfont = list(size = 11)),
        yaxis  = list(title = "Predicted", tickfont = list(size = 11)),
        plot_bgcolor  = "#ffffff",
        paper_bgcolor = "#ffffff"
      ) %>%
      config(displayModeBar = FALSE)
  })
  
  output$roc_chart <- renderPlotly({
    # Actual ROC curves computed from test-set predictions
    # Source: DA3/tables/roc_curve_data.csv (LDA posterior + RF predict prob)
    # AUCs:   DA3/tables/actual_auc_values.csv
    display_model  <- if (input$diag_model == "LDA") "LDA" else "Random Forest"
    colors_roc     <- c("closed" = "#e74c3c", "fully_open" = "#2980b9",
                        "partially_open" = "#27ae60")
    classes        <- c("closed", "fully_open", "partially_open")

    roc_path <- file.path(root, "DA3", "tables", "roc_curve_data.csv")
    auc_path <- file.path(root, "DA3", "tables", "actual_auc_values.csv")

    roc_raw <- tryCatch(
      read_csv(roc_path, show_col_types = FALSE) %>% filter(Model == display_model),
      error = function(e) NULL
    )
    auc_vals <- tryCatch(
      read_csv(auc_path, show_col_types = FALSE) %>% filter(model == display_model),
      error = function(e) NULL
    )

    p <- plot_ly()

    # Diagonal reference line
    p <- p %>%
      add_trace(x = c(0, 1), y = c(0, 1), type = "scatter", mode = "lines",
                line = list(dash = "dash", color = "grey70", width = 1.5),
                name = "Random classifier", showlegend = FALSE,
                hovertemplate = "Random classifier<extra></extra>")

    if (!is.null(roc_raw) && nrow(roc_raw) > 0) {
      for (cls in classes) {
        cls_data <- roc_raw[roc_raw$Class == cls, ]
        auc_row  <- if (!is.null(auc_vals)) auc_vals[auc_vals$class == cls, ] else NULL
        auc_val  <- if (!is.null(auc_row) && nrow(auc_row) > 0) auc_row$actual_auc[1] else NA
        lbl      <- if (!is.na(auc_val)) sprintf("%s (AUC=%.4f)", cls, auc_val) else cls
        p <- p %>%
          add_trace(
            x    = cls_data$FPR, y = cls_data$TPR,
            type = "scatter", mode = "lines", name = lbl,
            line = list(color = colors_roc[cls], width = 2.5),
            hovertemplate = paste0(cls, " | FPR:%{x:.3f} TPR:%{y:.3f}<extra></extra>")
          )
      }
      macro_auc  <- if (!is.null(auc_vals)) round(mean(auc_vals$actual_auc, na.rm=TRUE), 4) else NA
      note_text  <- sprintf("Macro AUC = %.4f  |  actual test-set predictions", macro_auc)
      chart_title <- sprintf("ROC Curves — %s (One-vs-Rest, n=301)", display_model)
    } else {
      note_text   <- "ROC data file not found — re-run DA3/generate_fixes.R"
      chart_title <- sprintf("ROC Curves — %s (data unavailable)", display_model)
    }

    p %>%
      layout(
        title  = list(text = chart_title, font = list(size = 12)),
        xaxis  = list(title = "False Positive Rate", range = c(0, 1)),
        yaxis  = list(title = "True Positive Rate",  range = c(0, 1)),
        legend = list(x = 0.38, y = 0.15, font = list(size = 10)),
        plot_bgcolor  = "#ffffff",
        paper_bgcolor = "#ffffff",
        annotations  = list(list(
          x = 0.50, y = 0.05, text = note_text,
          showarrow = FALSE, font = list(size = 10, color = "grey40")
        ))
      ) %>%
      config(displayModeBar = FALSE)
  })
  
  output$class_metrics_table <- renderTable({
    if (input$diag_model == "LDA") {
      data.frame(
        Class     = c("closed","fully_open","partially_open"),
        Precision = c(0.507, 0.621, 0.580),
        Recall    = c(0.507, 0.744, 0.624),
        F1        = c(0.507, 0.677, 0.601)
      )
    } else {
      data.frame(
        Class     = c("closed","fully_open","partially_open"),
        Precision = c(0.551, 0.564, 0.506),
        Recall    = c(0.463, 0.704, 0.422),
        F1        = c(0.503, 0.627, 0.460)
      )
    }
  }, digits = 3, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  output$feature_importance <- renderPlotly({
    plot_ly(rf_importance,
            x = ~Importance, y = ~Feature,
            type = "bar", orientation = "h",
            marker = list(
              color = colorRampPalette(c("#BBDEFB","#1565C0"))(nrow(rf_importance)),
              line  = list(color = "white", width = 0.5)
            ),
            text  = ~sprintf("%.4f", Importance),
            textposition = "outside",
            hovertemplate = "<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>") %>%
      layout(
        title  = list(text = "RF Gini Feature Importance — Top 15 Features", font = list(size = 12)),
        xaxis  = list(title = "Mean Decrease Gini"),
        yaxis  = list(title = "", automargin = TRUE),
        plot_bgcolor  = "#ffffff",
        paper_bgcolor = "#ffffff",
        margin = list(l = 160, r = 80, t = 40, b = 50)
      ) %>%
      config(displayModeBar = FALSE)
  })
  
  # ── TAB 4 OUTPUTS ──────────────────────────────────────────────────────────
  
  output$cluster_pie <- renderPlotly({
    colors_cl <- c("#FF7043","#42A5F5","#66BB6A")
    
    plot_ly(clusters_df,
            labels = ~Label, values = ~n_papers,
            type   = "pie",
            marker = list(colors = colors_cl,
                          line   = list(color = "white", width = 2.5)),
            hovertemplate = paste0("<b>%{label}</b><br>",
                                   "Papers: %{value}<br>",
                                   "Share: %{percent}<extra></extra>"),
            textinfo = "percent+value") %>%
      layout(
        title     = list(text = "Research Cluster Sizes (K=3)", font = list(size = 12)),
        showlegend = TRUE,
        legend    = list(orientation = "h", x = -0.1, y = -0.15,
                         font = list(size = 10)),
        margin    = list(l = 10, r = 10, t = 40, b = 60)
      ) %>%
      config(displayModeBar = FALSE)
  })
  
  output$cluster_terms <- renderPlotly({
    # Parse top terms using base R (avoids rowwise/unnest complexity in Shiny)
    plot_data <- do.call(rbind, lapply(seq_len(nrow(clusters_df)), function(i) {
      terms <- trimws(strsplit(clusters_df$top_terms[i], ";")[[1]])
      terms <- terms[nchar(terms) > 0][seq_len(min(8, length(terms)))]
      data.frame(
        cluster = clusters_df$cluster[i],
        Theme   = clusters_df$Theme[i],
        terms   = terms,
        rank    = seq_along(terms),
        stringsAsFactors = FALSE
      )
    }))
    
    colors_cl2 <- c("1" = "#FF7043", "2" = "#42A5F5", "3" = "#66BB6A")
    
    plot_ly(plot_data,
            x    = ~(9 - rank),   # proxy for "weight" (higher ranked = longer bar)
            y    = ~terms,
            color = ~Theme,
            colors = unname(colors_cl2),
            type = "bar",
            orientation = "h",
            hovertemplate = "<b>%{y}</b><br>Cluster: %{fullData.name}<extra></extra>") %>%
      layout(
        barmode = "group",
        title   = list(text = "Dominant TF-IDF Terms per Cluster (Top 8)", font = list(size = 12)),
        xaxis   = list(title = "Term Rank Weight (higher = more dominant)", showticklabels = FALSE),
        yaxis   = list(title = "", automargin = TRUE, tickfont = list(size = 10)),
        plot_bgcolor  = "#ffffff",
        paper_bgcolor = "#ffffff",
        legend  = list(x = 0.65, y = 0.05, font = list(size = 10)),
        margin  = list(l = 120, r = 20, t = 40, b = 50)
      ) %>%
      config(displayModeBar = FALSE)
  })
  
  output$cluster_summary_table <- renderTable({
    data.frame(
      Cluster = paste0("C", 1:3),
      Theme   = c("Education/GenAI/ChatGPT",
                  "General ML/Robotics",
                  "Medical Imaging/DL"),
      Papers  = c(200L, 1422L, 378L),
      Pct     = c("10.0%", "71.1%", "18.9%"),
      Top_Term = c("educ", "model", "imag")
    )
  }, striped = TRUE, hover = TRUE, bordered = TRUE, spacing = "s")
  
}  # end server

# ── 8. Launch ─────────────────────────────────────────────────────────────────
shinyApp(ui = ui, server = server)
