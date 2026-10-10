# =============================================================================
# DA3 Fix 3 — Create presentation as self-contained HTML
# No external packages needed — uses only base R
# Can be: opened in browser, printed to PDF, or converted to PPTX manually
# =============================================================================
setwd("e:/Tuned_Research")
out_html <- "DA3/DA3_Presentation.html"

slides <- list(

  list(
    num = 1, title = "ResearchPilot",
    subtitle = "A Human-Centered AI Research Assistant",
    body = c(
      "<p style='font-size:1.2em;text-align:center;margin-top:20px'>DA3 — Final Evaluation &amp; Project Demonstration</p>",
      "<p style='text-align:center;color:#555'>October 16, 2026</p>",
      "<hr>",
      "<table style='margin:20px auto;font-size:1.05em;border-collapse:collapse'>",
      "  <tr><td style='padding:6px 20px;background:#e3f2fd;border-radius:4px'><b>2,000 papers</b></td>",
      "      <td style='padding:6px 20px;background:#e8f5e9;border-radius:4px'><b>12 ML models</b></td>",
      "      <td style='padding:6px 20px;background:#fff3e0;border-radius:4px'><b>Friedman p = 0.000009</b></td></tr>",
      "  <tr><td style='padding:6px 20px;background:#f3e5f5;border-radius:4px'><b>80 features</b></td>",
      "      <td style='padding:6px 20px;background:#fce4ec;border-radius:4px'><b>Best F1 = 0.5701</b></td>",
      "      <td style='padding:6px 20px;background:#e0f2f1;border-radius:4px'><b>R Shiny Dashboard</b></td></tr>",
      "</table>",
      "<p style='text-align:center;color:#888;font-size:0.9em'>Dataset: OpenAlex API &nbsp;|&nbsp; Language: R &nbsp;|&nbsp; DB: SQLite</p>"
    )
  ),

  list(
    num = 2, title = "Problem Statement and DA3 Objectives",
    body = c(
      "<div style='display:flex;gap:30px'>",
      "<div style='flex:1;background:#e3f2fd;padding:15px;border-radius:8px'>",
      "<b>THE PROBLEM</b><br>Can we automatically predict whether a research paper<br>is freely accessible — from metadata alone?<br><br>",
      "<b>Target: oa_category (3 classes)</b><br>",
      "🟢 fully_open &nbsp;&nbsp;&nbsp; 834 papers (41.7%)<br>",
      "🟡 partially_open &nbsp; 723 papers (36.2%)<br>",
      "🔴 closed &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 443 papers (22.2%)",
      "</div>",
      "<div style='flex:1;background:#f3e5f5;padding:15px;border-radius:8px'>",
      "<b>DA3 OBJECTIVES — Official Rubric (10 Marks)</b><br><br>",
      "① Final performance comparison + statistical analysis &nbsp;<b>[2]</b><br>",
      "② Interactive R Shiny + Plotly dashboard &nbsp;<b>[2]</b><br>",
      "③ Result interpretation and discussion &nbsp;<b>[2]</b><br>",
      "④ Conclusion, limitations, future scope &nbsp;<b>[1]</b><br>",
      "⑤ Final report quality + code organization &nbsp;<b>[2]</b><br>",
      "⑥ Presentation + viva &nbsp;<b>[1]</b>",
      "</div></div>"
    )
  ),

  list(
    num = 3, title = "ResearchPilot — Complete Pipeline",
    body = c(
      "<div style='font-size:0.95em'>",
      "<div style='background:#e3f2fd;padding:10px;border-radius:6px;margin-bottom:8px'><b>DA1 — DATA FOUNDATION</b><br>OpenAlex API → 2,000 AI/ML papers → EDA → <code>final_dataset.csv</code></div>",
      "<div style='background:#e8f5e9;padding:10px;border-radius:6px;margin-bottom:8px'><b>DA2 — INTELLIGENCE LAYER</b><br>",
      "Feature Engineering: 27 raw → 206 engineered → 80 selected<br>",
      "SQLite Database: 4 tables (papers, ml_features, oa_features, model_results)<br>",
      "12 ML Models trained + 3 hyperparameter-tuned + clustering (k=3)</div>",
      "<div style='background:#fff3e0;padding:10px;border-radius:6px;margin-bottom:8px'><b>DA3 — FINAL EVALUATION ← TODAY</b><br>",
      "Friedman test (χ²=43.46, p=0.000009) + Nemenyi post-hoc<br>",
      "Final model comparison (15 models) + R Shiny dashboard (4 tabs)</div>",
      "<div style='background:#fce4ec;padding:10px;border-radius:6px'><b>FUTURE SCOPE — ❌ NOT IMPLEMENTED</b><br>",
      "RAG research assistant &nbsp;|&nbsp; Ollama local LLM &nbsp;|&nbsp; SciBERT embeddings</div>",
      "</div>"
    )
  ),

  list(
    num = 4, title = "Dataset and Feature Engineering",
    body = c(
      "<div style='display:flex;gap:20px;font-size:0.93em'>",
      "<div style='flex:1'>",
      "<b>DATASET</b><br>Source: OpenAlex API<br>",
      "Size: 2,000 papers × 27 raw columns<br>",
      "Splits: 70% train (1400) | 15% val (299) | 15% test (301)<br>",
      "Seed: set.seed(42) — fully reproducible<br><br>",
      "<b>FEATURE ENGINEERING (27 → 206)</b><br>",
      "• Structural/text-length (12): abstract_length, text_richness...<br>",
      "• Domain binary flags (25): dom_medical, dom_llm...<br>",
      "• Publisher one-hot (18): pub_ieee, pub_mdpi, pub_springer...<br>",
      "• TF-IDF terms (~150): tf_model, tf_deep, tf_learning...",
      "</div>",
      "<div style='flex:1;background:#f9fbe7;padding:12px;border-radius:8px'>",
      "<b>FEATURE SELECTION (206 → 80)</b><br>",
      "<i>All steps fitted on TRAINING data only — no leakage</i><br><br>",
      "Step 1: Near-Zero Variance filter<br>",
      "Step 2: Pearson corr filter (TF-IDF, |r|>0.95)<br>",
      "Step 3: RF Gini importance → top 80 retained<br><br>",
      "<b>TOP FEATURES (MeanDecreaseGini)</b><br>",
      "1. pub_ieee &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 43.78<br>",
      "2. pub_mdpi &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 38.48<br>",
      "3. title_to_abstract_ratio &nbsp; 29.03<br>",
      "4. abstract_length &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 28.06<br>",
      "5. title_length &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 27.43",
      "</div></div>"
    )
  ),

  list(
    num = 5, title = "12 Machine Learning Models",
    body = c(
      "<div style='display:flex;gap:15px;font-size:0.93em'>",
      "<div style='flex:1;background:#e3f2fd;padding:12px;border-radius:8px'>",
      "<b>Linear / Discriminant</b><br>✦ LDA (best baseline)<br>✦ Logistic Regression<br>✦ Elastic Net<br>✦ SVM-Linear<br>✦ SVM-RBF</div>",
      "<div style='flex:1;background:#e8f5e9;padding:12px;border-radius:8px'>",
      "<b>Tree-Based / Ensemble</b><br>✦ Decision Tree<br>✦ Random Forest<br>✦ Gradient Boosting<br>✦ AdaBoost<br>✦ XGBoost (worst)</div>",
      "<div style='flex:1;background:#fff3e0;padding:12px;border-radius:8px'>",
      "<b>Other</b><br>✦ Naive Bayes<br>✦ MLP (neural, 50 units)<br><br>",
      "<b>Evaluation</b><br>Same test set (n=301)<br>Primary: Macro-F1<br>No leakage</div>",
      "</div>",
      "<div style='margin-top:15px;background:#fce4ec;padding:12px;border-radius:8px;font-size:0.93em'>",
      "<b>HYPERPARAMETER TUNING (5-fold CV grid search on training only)</b><br>",
      "Random Forest: ntree=500, mtry=8 → F1: 0.5317 → <b>0.5475</b> (+0.0158)<br>",
      "SVM-RBF: cost=10, gamma=0.001 → F1: 0.5559 → <b>0.5701</b> (+0.0142)<br>",
      "XGBoost: depth=6, eta=0.10 → F1: 0.3428 → <b>0.3537</b> (+0.0109)</div>"
    )
  ),

  list(
    num = 6, title = "Final Performance Comparison (Test Set, n=301)",
    body = c(
      "<table style='width:100%;border-collapse:collapse;font-size:0.88em'>",
      "<tr style='background:#1a237e;color:white'><th style='padding:6px'>Model</th><th>Accuracy</th><th>Macro-F1</th><th>ROC-AUC</th><th>Type</th></tr>",
      "<tr style='background:#ffecb3'><td><b>SVM-RBF (Tuned) ★</b></td><td>57.48%</td><td><b>0.5701</b></td><td>—</td><td>Tuned</td></tr>",
      "<tr style='background:#e8f5e9'><td><b>LDA ★ (best baseline)</b></td><td>57.48%</td><td><b>0.5696</b></td><td>0.7577</td><td>Baseline</td></tr>",
      "<tr><td>MLP</td><td>56.81%</td><td>0.5657</td><td>0.7659</td><td>Baseline</td></tr>",
      "<tr style='background:#f5f5f5'><td>Logistic Regression</td><td>56.48%</td><td>0.5631</td><td>0.7565</td><td>Baseline</td></tr>",
      "<tr><td>Elastic Net</td><td>55.81%</td><td>0.5539</td><td>0.7584</td><td>Baseline</td></tr>",
      "<tr style='background:#fff3e0'><td>RF (Tuned)</td><td>56.15%</td><td>0.5475</td><td>—</td><td>Tuned</td></tr>",
      "<tr><td>SVM-Linear</td><td>55.15%</td><td>0.5441</td><td>0.7587</td><td>Baseline</td></tr>",
      "<tr style='background:#f5f5f5'><td>Gradient Boosting</td><td>53.82%</td><td>0.5384</td><td>0.7402</td><td>Baseline</td></tr>",
      "<tr><td>Random Forest</td><td>54.82%</td><td>0.5317</td><td>0.7572</td><td>Baseline</td></tr>",
      "<tr style='background:#f5f5f5'><td>AdaBoost</td><td>51.50%</td><td>0.5123</td><td>0.6963</td><td>Baseline</td></tr>",
      "<tr><td>Decision Tree</td><td>50.83%</td><td>0.5017</td><td>0.6924</td><td>Baseline</td></tr>",
      "<tr style='background:#f5f5f5'><td>Naive Bayes</td><td>41.86%</td><td>0.4081</td><td>0.6840</td><td>Baseline</td></tr>",
      "<tr style='background:#fff3e0'><td>XGBoost (Tuned)</td><td>35.55%</td><td>0.3537</td><td>—</td><td>Tuned</td></tr>",
      "<tr style='background:#ffebee'><td><b>XGBoost (worst) ✗</b></td><td>34.55%</td><td><b>0.3428</b></td><td>0.5243</td><td>Baseline</td></tr>",
      "</table>",
      "<p style='font-size:0.85em;color:#555;margin-top:8px'>Tuning margin over best baseline: +0.0005 F1 (practically negligible) &nbsp;|&nbsp; See: DA3/figures/final_f1_comparison.png</p>"
    )
  ),

  list(
    num = 7, title = "Statistical Analysis — Friedman Test + Nemenyi Post-Hoc",
    body = c(
      "<div style='display:flex;gap:20px;font-size:0.95em'>",
      "<div style='flex:1'>",
      "<b>METHOD: Friedman non-parametric test</b><br>",
      "Why? Repeated measures (same 5 folds for all models)<br>",
      "Non-parametric (cannot assume normality at K=5)<br><br>",
      "<b>HYPOTHESES</b><br>",
      "H₀: All 12 models equal median Macro-F1 (differences = chance)<br>",
      "H₁: At least one model differs significantly<br>",
      "α = 0.05<br><br>",
      "<div style='background:#e8f5e9;padding:10px;border-radius:6px'>",
      "<b>RESULTS</b><br>",
      "χ²(11) = <b>43.4615</b><br>",
      "p-value = <b>0.000009</b><br>",
      "<span style='color:#2e7d32;font-size:1.1em'><b>→ REJECT H₀</b></span><br>",
      "Differences are STATISTICALLY REAL",
      "</div>",
      "</div>",
      "<div style='flex:1'>",
      "<b>POST-HOC: Nemenyi test → 7 significant pairs</b><br>",
      "<table style='width:100%;font-size:0.9em;border-collapse:collapse;margin-top:8px'>",
      "<tr style='background:#ffcdd2'><td>XGBoost</td><td>vs Random Forest</td><td>p=0.0045</td></tr>",
      "<tr style='background:#ffcdd2'><td>LDA</td><td>vs XGBoost</td><td>p=0.0045</td></tr>",
      "<tr style='background:#ffcdd2'><td>XGBoost</td><td>vs Elastic Net</td><td>p=0.0064</td></tr>",
      "<tr style='background:#ffe0b2'><td>Naive Bayes</td><td>vs Random Forest</td><td>p=0.0229</td></tr>",
      "<tr style='background:#ffe0b2'><td>LDA</td><td>vs Naive Bayes</td><td>p=0.0229</td></tr>",
      "<tr style='background:#fff9c4'><td>XGBoost</td><td>vs Logistic Reg.</td><td>p=0.0307</td></tr>",
      "<tr style='background:#fff9c4'><td>Naive Bayes</td><td>vs Elastic Net</td><td>p=0.0307</td></tr>",
      "</table>",
      "<p style='font-size:0.85em;margin-top:8px;color:#555'>",
      "CD = 7.45 &nbsp;|&nbsp; Top models (LDA, RF, EN, LR) do NOT significantly differ<br>",
      "See: DA3/figures/stat_cd_diagram.png",
      "</p>",
      "</div></div>"
    )
  ),

  list(
    num = 8, title = "Interactive Dashboard — R Shiny + Plotly",
    body = c(
      "<p><b>Launch:</b> <code>setwd('e:/Tuned_Research'); shiny::runApp('dashboard')</code> &nbsp; OR double-click <code>LAUNCH_DASHBOARD.bat</code></p>",
      "<div style='display:flex;gap:12px;font-size:0.92em;margin-top:10px'>",
      "<div style='flex:1;background:#e3f2fd;padding:12px;border-radius:8px'>",
      "<b>TAB 1 — OVERVIEW</b><br>",
      "KPI cards: 2000 papers,<br>27→206→80 features, 12 models<br>",
      "Interactive OA distribution chart<br>Feature pipeline visualization</div>",
      "<div style='flex:1;background:#e8f5e9;padding:12px;border-radius:8px'>",
      "<b>TAB 2 — MODEL COMPARISON ★</b><br>",
      "Metric selector dropdown:<br>F1 / Accuracy / AUC / Precision / Recall<br>",
      "Chart updates instantly<br>Baseline (blue) vs Tuned (red)<br>Hover for exact values</div>",
      "<div style='flex:1;background:#fff3e0;padding:12px;border-radius:8px'>",
      "<b>TAB 3 — DIAGNOSTICS ★</b><br>",
      "Model selector: LDA / Random Forest<br>",
      "Confusion matrix heatmap<br><i>(ACTUAL test-set predictions)</i><br>",
      "ROC curves <i>(ACTUAL from model)</i><br>",
      "Feature importance top-20<br><i>(from rf_importance_v2.rds)</i></div>",
      "<div style='flex:1;background:#f3e5f5;padding:12px;border-radius:8px'>",
      "<b>TAB 4 — CLUSTERS</b><br>",
      "Cluster size pie chart<br>Dominant TF-IDF terms<br>Theme descriptions</div>",
      "</div>",
      "<p style='font-size:0.85em;color:#555;margin-top:10px'>All data loaded from project files — no fabricated values in any tab</p>"
    )
  ),

  list(
    num = 9, title = "Key Findings and Interpretation",
    body = c(
      "<div style='font-size:0.93em'>",
      "<div style='background:#e8f5e9;padding:10px;border-radius:6px;margin-bottom:8px'>",
      "<b>1. PUBLISHER FEATURES DOMINATE</b><br>",
      "pub_ieee (43.78), pub_mdpi (38.48) = top RF Gini features<br>",
      "OA status = publisher POLICY — not paper content<br>",
      "⚠️ Importance = predictive correlation, NOT causation</div>",
      "<div style='background:#e3f2fd;padding:10px;border-radius:6px;margin-bottom:8px'>",
      "<b>2. LINEAR MODELS WIN</b><br>",
      "LDA > MLP > Logistic Reg > SVM-RBF > ... > XGBoost<br>",
      "Feature space is approximately linearly separable<br>",
      "~1,400 training samples insufficient for XGBoost's complexity</div>",
      "<div style='background:#fff3e0;padding:10px;border-radius:6px;margin-bottom:8px'>",
      "<b>3. PARTIALLY_OPEN IS HARDEST</b><br>",
      "Lowest class AUC: partially_open = 0.6649 (LDA) vs closed = 0.8379<br>",
      "Definitional ambiguity → inherent label noise</div>",
      "<div style='background:#f3e5f5;padding:10px;border-radius:6px;margin-bottom:8px'>",
      "<b>4. TUNING = DIMINISHING RETURNS</b><br>",
      "Best individual gain: RF +0.0158 F1<br>",
      "Overall best-vs-best margin: +0.0005 F1 (LDA already near-optimal)</div>",
      "<div style='background:#fce4ec;padding:10px;border-radius:6px'>",
      "<b>5. STATISTICALLY VALIDATED</b><br>",
      "XGBoost and Naive Bayes provably inferior (Nemenyi p&lt;0.05)<br>",
      "Top models are statistically interchangeable at K=5 folds</div>",
      "</div>"
    )
  ),

  list(
    num = 10, title = "Topic Clustering — K-Means (k=3) on TF-IDF",
    body = c(
      "<div style='display:flex;gap:15px;font-size:0.93em;margin-bottom:12px'>",
      "<div style='flex:1;background:#fff3e0;padding:12px;border-radius:8px;text-align:center'>",
      "<div style='font-size:2em'>🎓</div>",
      "<b>Cluster 1 — 10.0%</b><br>200 papers<br><br>",
      "<i>Education / GenAI / ChatGPT</i><br><br>",
      "Terms: educ, chatgpt, student,<br>generat, tool, artifici</div>",
      "<div style='flex:1;background:#e3f2fd;padding:12px;border-radius:8px;text-align:center'>",
      "<div style='font-size:2em'>🤖</div>",
      "<b>Cluster 2 — 71.1%</b><br>1,422 papers<br><br>",
      "<i>General ML / Prediction / Robotics</i><br><br>",
      "Terms: model, learn, data,<br>predict, robot, system</div>",
      "<div style='flex:1;background:#e8f5e9;padding:12px;border-radius:8px;text-align:center'>",
      "<div style='font-size:2em'>🏥</div>",
      "<b>Cluster 3 — 18.9%</b><br>378 papers<br><br>",
      "<i>Medical Imaging / Deep Learning</i><br><br>",
      "Terms: imag, detect, segment,<br>medic, network, diseas</div>",
      "</div>",
      "<div style='background:#fff9c4;padding:10px;border-radius:6px;font-size:0.9em'>",
      "⚠️ <b>Labels are HUMAN-INTERPRETED</b> from top TF-IDF terms — K-Means produces numbers, not semantic labels<br>",
      "Method: L2-normalised TF-IDF | Silhouette ~0.06 (expected weak separation for overlapping AI/ML topics)<br>",
      "Clustering is fully independent of the supervised ML pipeline",
      "</div>"
    )
  ),

  list(
    num = 11, title = "Limitations and Future Scope",
    body = c(
      "<div style='display:flex;gap:20px;font-size:0.88em'>",
      "<div style='flex:1'>",
      "<b>GENUINE LIMITATIONS (10 verified)</b><br>",
      "<ol style='margin:5px 0;padding-left:18px'>",
      "<li>Dataset: 2,000 papers, AI/ML only, static snapshot</li>",
      "<li>Performance: ~57% accuracy — 3-class overlap</li>",
      "<li>partially_open: ambiguous definition → label noise</li>",
      "<li>Features: TF-IDF only, no semantic embeddings</li>",
      "<li>No full-text: titles + abstracts only</li>",
      "<li>Metadata dependency: missing DOIs degrade predictions</li>",
      "<li>Static dataset: no continuous update pipeline</li>",
      "<li>Cluster labels: human-interpreted, not algorithmic</li>",
      "<li>CV K=5: limited power for detecting small differences</li>",
      "<li>No causal analysis — correlation only</li>",
      "</ol></div>",
      "<div style='flex:1'>",
      "<b>FUTURE SCOPE — ALL ❌ NOT IMPLEMENTED</b><br><br>",
      "<div style='background:#ffebee;padding:8px;border-radius:6px;margin-bottom:6px'>",
      "<b>Tier 1 (High Priority)</b><br>",
      "❌ RAG-based research assistant<br>",
      "❌ Local LLM via Ollama (hardware-dependent)<br>",
      "❌ LoRA/QLoRA instruction fine-tuning</div>",
      "<div style='background:#fff3e0;padding:8px;border-radius:6px;margin-bottom:6px'>",
      "<b>Tier 2 (Medium Priority)</b><br>",
      "❌ SciBERT / Sentence-BERT embeddings<br>",
      "❌ Paper similarity search<br>",
      "❌ Research gap detection</div>",
      "<div style='background:#f3e5f5;padding:8px;border-radius:6px'>",
      "<b>Tier 3 (Long-term)</b><br>",
      "❌ Multi-domain corpus (10k+ papers)<br>",
      "❌ Real-time data pipeline<br>",
      "❌ Explainable AI (SHAP/LIME)</div>",
      "</div></div>"
    )
  ),

  list(
    num = 12, title = "Conclusion",
    body = c(
      "<div style='display:flex;gap:20px;font-size:0.95em'>",
      "<div style='flex:1'>",
      "<div style='background:#e8f5e9;padding:15px;border-radius:8px;margin-bottom:12px'>",
      "<b>DA3 ACHIEVEMENTS</b><br>",
      "✅ 12 models trained + 3 tuned + evaluated (n=301)<br>",
      "✅ Best: SVM-RBF Tuned (F1=0.5701, Acc=57.48%)<br>",
      "✅ Friedman χ²=43.46, p=0.000009 → REJECT H₀<br>",
      "✅ 7 significant Nemenyi pairs<br>",
      "✅ R Shiny dashboard: 4 tabs, actual data throughout<br>",
      "✅ 3 research themes (K-Means clustering)<br>",
      "✅ Complete report + viva + presentation</div>",
      "<div style='background:#e3f2fd;padding:15px;border-radius:8px'>",
      "<b>KEY INSIGHT</b><br>",
      "OA status is a <b>publisher policy decision</b><br>",
      "57.5% accuracy = 73% lift over random (33%)<br>",
      "LDA recommended: simple, interpretable,<br>",
      "statistically equivalent to SVM-RBF tuned</div>",
      "</div>",
      "<div style='flex:1'>",
      "<div style='background:#fff3e0;padding:15px;border-radius:8px;margin-bottom:12px'>",
      "<b>CODE ORGANIZATION</b><br>",
      "<code>r/scripts/phase3/</code> — DA3 R scripts<br>",
      "<code>dashboard/app.R</code> — Shiny dashboard<br>",
      "<code>DA3/</code> — all deliverable documents<br>",
      "<code>DA3/figures/</code> — 11 publication figures<br>",
      "<code>DA3/tables/</code> — 8 output tables</div>",
      "<div style='background:#f3e5f5;padding:15px;border-radius:8px;text-align:center'>",
      "<br>",
      "<span style='font-size:1.3em'><b>Thank you</b></span><br><br>",
      "<i>Questions?</i><br><br>",
      "<span style='color:#888;font-size:0.85em'>ResearchPilot DA3 | October 2026</span>",
      "</div></div></div>"
    )
  )
)

# ── Build HTML ─────────────────────────────────────────────────────────────────
html_parts <- c(
  "<!DOCTYPE html>",
  "<html lang='en'>",
  "<head>",
  "  <meta charset='UTF-8'>",
  "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
  "  <title>ResearchPilot DA3 Presentation</title>",
  "  <style>",
  "    * { box-sizing: border-box; }",
  "    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #263238; }",
  "    .slide {",
  "      width: 960px; min-height: 540px; margin: 30px auto;",
  "      background: white; border-radius: 10px;",
  "      box-shadow: 0 4px 20px rgba(0,0,0,0.4);",
  "      overflow: hidden; page-break-after: always;",
  "    }",
  "    .slide-header {",
  "      background: linear-gradient(135deg, #1a237e 0%, #283593 100%);",
  "      color: white; padding: 18px 30px;",
  "    }",
  "    .slide-num { font-size: 0.75em; opacity: 0.7; }",
  "    .slide-title { font-size: 1.5em; font-weight: 700; margin: 4px 0 0 0; }",
  "    .slide-subtitle { font-size: 1.1em; opacity: 0.85; margin: 2px 0 0 0; }",
  "    .slide-body { padding: 20px 30px; }",
  "    .title-slide .slide-header { min-height: 540px; display: flex; flex-direction: column; justify-content: center; text-align: center; }",
  "    .title-slide .slide-title { font-size: 2.2em; }",
  "    @media print {",
  "      body { background: white; }",
  "      .slide { box-shadow: none; margin: 0; border-radius: 0; border: 1px solid #ddd; }",
  "    }",
  "  </style>",
  "</head>",
  "<body>",
  "<div style='text-align:center;color:#cfd8dc;padding:15px;font-size:0.9em'>",
  "  ResearchPilot DA3 Presentation &nbsp;|&nbsp; October 2026 &nbsp;|&nbsp;",
  "  Press Ctrl+P to print/save as PDF &nbsp;|&nbsp; 12 slides",
  "</div>"
)

for (sl in slides) {
  is_title <- sl$num == 1
  cls      <- if (is_title) "slide title-slide" else "slide"
  html_parts <- c(html_parts, sprintf("<div class='%s' id='slide%d'>", cls, sl$num))
  html_parts <- c(html_parts, "  <div class='slide-header'>")
  html_parts <- c(html_parts, sprintf("    <div class='slide-num'>Slide %d of 12</div>", sl$num))
  html_parts <- c(html_parts, sprintf("    <div class='slide-title'>%s</div>", sl$title))
  if (!is.null(sl$subtitle)) {
    html_parts <- c(html_parts, sprintf("    <div class='slide-subtitle'>%s</div>", sl$subtitle))
  }
  html_parts <- c(html_parts, "  </div>")
  if (!is_title) {
    html_parts <- c(html_parts, "  <div class='slide-body'>")
    html_parts <- c(html_parts, sl$body)
    html_parts <- c(html_parts, "  </div>")
  }
  html_parts <- c(html_parts, "</div>")
}

html_parts <- c(html_parts, "</body>", "</html>")

writeLines(html_parts, out_html)
sz <- file.info(out_html)$size
cat(sprintf("Saved: %s  (%d KB)\n", out_html, as.integer(sz / 1024)))
cat(sprintf("Slides: %d\n", length(slides)))
cat("Open in any browser. Press Ctrl+P -> Save as PDF to create a PDF version.\n")
cat("Fix 3 COMPLETE\n")
