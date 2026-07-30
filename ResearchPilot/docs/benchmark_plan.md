# ResearchPilot — Benchmark Plan

**Document type:** Evaluation design  
**Phases:** DA1 data quality, DA2 model evaluation, DA3 assistant evaluation  
**Status:** Planning

---

## 1. Purpose

Define how ResearchPilot will measure quality across the full course workflow: DA1 data quality, DA2 predictive models, and DA3 assistant modules. This keeps model and product decisions evidence-based and comparable over time.

Benchmarks live under `data/benchmark/`. Reports are written to `evaluation/reports/`. Metric code belongs in `evaluation/metrics/` (implemented in later phases).

---

## 2. Evaluation Principles

1. **Frozen test sets** — Benchmark examples are never used for training.  
2. **Multi-signal scoring** — Automatic metrics + human rubrics.  
3. **Faithfulness first** — Prefer grounded answers over fluent hallucination.  
4. **Module-specific criteria** — One size does not fit all four skills.  
5. **Reproducibility** — Fixed seeds, documented model IDs, config snapshots.  

---

## 3. Cross-Phase Evaluation

| Phase | Evaluation focus |
|-------|------------------|
| **DA1** | Completeness, validity, duplicates, missingness, preprocessing checks, and reproducibility |
| **DA2** | Held-out predictive metrics, cross-validation, error analysis, efficiency, and comparative visualizations |
| **DA3** | Assistant faithfulness, usefulness, grounding, latency, and dashboard usability |

DA2 metrics must match the selected task. Classification may use precision, recall, F1, ROC-AUC/PR-AUC, and confusion matrices; regression may use MAE, RMSE, and R²; clustering may use internal scores plus qualitative validation. Class imbalance and split strategy must be reported.

## 4. Benchmark Assets (Planned)

| Asset | Location | Description |
|-------|----------|-------------|
| Gold JSONL per module | `data/benchmark/<module>/` | Inputs + reference outputs |
| Rubric definitions | `data/benchmark/rubrics/` | Human scoring guidelines |
| Split manifest | `data/metadata/` | IDs reserved for benchmark |
| Report templates | `evaluation/reports/` | Markdown/HTML summaries |

---

## 5. DA3 Per-Module Evaluation Design

### 4.1 Adaptive Research Translator

| Dimension | Method |
|-----------|--------|
| **Semantic faithfulness** | Human 1–5; optional NLI / factuality checks |
| **Simplification quality** | Readability proxies + human “beginner clarity” |
| **Coverage of takeaways** | Checklist vs gold key points |
| **Automatic similarity** | BERTScore (`bert-score`) vs reference rewrite |

**Pass guideline (draft):** Mean human faithfulness ≥ 4.0 and BERTScore F1 competitive with baseline + documented delta.

### 4.2 Statistical Advisor

| Dimension | Method |
|-----------|--------|
| **Correct primary test** | Exact / acceptable-set match vs gold |
| **Rationale quality** | Human rubric (justification completeness) |
| **Assumption coverage** | Recall of critical assumptions |
| **Alternatives** | Presence and appropriateness when required |

**Pass guideline (draft):** Primary recommendation accuracy ≥ agreed threshold on gold set; zero tolerance for unsafe absolute medical claims.

### 4.3 Abstract Improver

| Dimension | Method |
|-----------|--------|
| **Meaning preservation** | Human faithfulness; diff-based review |
| **Grammar / fluency** | Human + optional grammar tools |
| **Academic tone / clarity** | Rubric scores |
| **Novelty framing** | Rubric: improves contribution clarity without inventing claims |
| **Automatic similarity** | BERTScore vs expert-edited reference (where available) |

**Pass guideline (draft):** No increase in hallucinated claims; clarity/tone scores beat baseline.

### 4.4 Research Gap Finder

| Dimension | Method |
|-----------|--------|
| **Grounding** | Gaps must map to input limitations / contrasts |
| **Usefulness** | Human “actionable research idea” score |
| **Diversity** | Non-duplicate gap ideas |
| **Coverage** | Hits major limitations present in sources |

**Pass guideline (draft):** High grounding score; low rate of generic unsupported gaps.

---

## 6. Automatic Metrics Stack (Planned)

| Library / tool | Use |
|----------------|-----|
| `bert-score` | Semantic overlap with references |
| `evaluate` (Hugging Face) | Metric orchestration / standard metrics |
| Task-specific scripts | Accuracy of stats test labels; checklist scores |
| `matplotlib` / `plotly` | Result visualization in reports |

Custom research metrics (e.g., assumption recall) will be implemented under `evaluation/metrics/`.

---

## 7. Human Evaluation Protocol

### Sample sizes (guidelines)

- **Pilot:** 20–30 items/module for rubric calibration  
- **Main:** 50–100 items/module (budget permitting)  
- **Critical stats items:** Oversample ambiguous designs  

### Annotator guidance

- Score independently, then adjudicate disagreements on faithfulness.  
- Mark hallucinations explicitly.  
- For Gap Finder, require pointer to supporting span when possible.  

Rubrics will be stored with the DA3 benchmark release.

---

## 8. Baselines

Compare ResearchPilot models against:

1. Untuned base LLM with the same prompts  
2. Untuned base LLM + retrieval (when RAG is enabled)  
3. (Optional) Strong general-purpose API model for reference ceiling—not a hard dependency  

Report **deltas**, not only absolute scores.

---

## 9. Reporting Format

Each evaluation run should produce:

```text
evaluation/reports/
└── YYYYMMDD_<model_id>_<benchmark_version>/
    ├── summary.md
    ├── metrics.json
    ├── per_module/
    └── configs_snapshot/
```

`summary.md` must include: model ID, data version, metric versions, hardware notes, and known limitations.

---

## 10. Go / No-Go for Demo Release

Before the DA3 final demonstration:

- [ ] Benchmark version pinned in metadata  
- [ ] All four modules evaluated  
- [ ] Human sample completed for faithfulness-critical modules  
- [ ] Failure modes documented (e.g., rare stats designs)  
- [ ] No unresolved license issues in benchmark sources  

---

## 11. Phase Alignment

| Course phase | Benchmark activity |
|--------------|--------------------|
| Foundation | Plan only |
| DA1 | Data-quality tests and EDA validation |
| DA2 | Predictive benchmark, baselines, and model comparison |
| DA3 | Assistant benchmark, usability, and end-to-end evaluation |

---

## 12. Related Documents

- [dataset_plan.md](dataset_plan.md)  
- [architecture.md](architecture.md)  
- [development_roadmap.md](development_roadmap.md)  
- [project_scope.md](project_scope.md)  
