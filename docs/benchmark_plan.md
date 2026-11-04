# ResearchPilot — Benchmark Plan

**Status:** Planning

---

## 1. Purpose

Define how quality is measured across phases so decisions stay evidence-based.

| Phase | Focus |
|-------|--------|
| **1** | Completeness, validity, duplicates, missingness, reproducibility |
| **2** | Held-out ML metrics, cross-validation, error analysis, model comparison |
| **3** | Assistant faithfulness, RAG grounding, usefulness, dashboard usability |

---

## 2. Phase 2 Metrics (examples)

Choose metrics that match the task:

- Classification: accuracy, precision, recall, F1, ROC-AUC / PR-AUC, confusion matrix  
- Regression: MAE, RMSE, R²  
- Clustering: silhouette / qualitative validation  

Always report split strategy, class balance, and baselines.

---

## 3. Phase 3 Module Evaluation

| Module | Signals |
|--------|---------|
| Translator | Faithfulness, clarity, takeaway coverage, BERTScore vs reference |
| Statistical Advisor | Correct test set match, assumption coverage, rationale quality |
| Abstract Improver | Meaning preservation, tone/clarity, no invented claims |
| Gap Finder | Grounding in inputs, usefulness, non-generic gaps |

Automatic tools (planned): `evaluate`, `bert-score`.

---

## 4. Reporting

```text
evaluation/reports/
└── <date>_<run_id>/
    ├── summary.md
    ├── metrics.json
    └── figures/
```

---

## 5. Related Docs

- [dataset_plan.md](dataset_plan.md)  
- [development_roadmap.md](development_roadmap.md)  
- [architecture.md](architecture.md)  
