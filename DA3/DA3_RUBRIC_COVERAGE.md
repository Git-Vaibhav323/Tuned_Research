# DA3 RUBRIC COVERAGE TABLE

**Project:** ResearchPilot  
**Assessment:** DA3 Final Evaluation  
**Total Marks:** 10  
**Date:** October 6, 2026

---

## OFFICIAL DA3 RUBRIC (AUTHORITATIVE)

| # | Criterion | Marks | Weight |
|---|-----------|-------|--------|
| 1 | Final performance comparison and statistical analysis | 2 | 20% |
| 2 | Interactive visualization/dashboard (Plotly/Shiny/Tableau/Power BI) | 2 | 20% |
| 3 | Result interpretation and discussion | 2 | 20% |
| 4 | Conclusion, limitations and future scope | 1 | 10% |
| 5 | Final report quality and code organization | 2 | 20% |
| 6 | Final presentation and viva | 1 | 10% |
| **TOTAL** | | **10** | **100%** |

---

## CRITERION 1: PERFORMANCE COMPARISON & STATISTICAL ANALYSIS (2 Marks)

### Part A: Final Performance Comparison

| Component | Status | Evidence | Gap |
|-----------|--------|----------|-----|
| Model comparison table | ✅ EXISTS | `DA2/m6_comparative_metrics.csv`, `data/ml_r/all_results_v2.csv` | — |
| All 12 baseline models | ✅ COMPLETE | Logistic, Elastic Net, DT, RF, XGB, GB, AdaBoost, SVM-RBF, SVM-Linear, NB, LDA, MLP | — |
| 3 tuned models | ✅ COMPLETE | RF-tuned, SVM-tuned, XGB-tuned | — |
| Test set evaluation | ✅ PROPER | n=301, stratified split, no leakage | — |
| Multiple metrics | ✅ COMPLETE | Accuracy, Precision, Recall, Macro-F1, ROC-AUC | — |
| Baseline vs tuned comparison | ✅ CLEAR | Table distinguishes baseline/tuned | — |
| Best model identification | ✅ CLEAR | Best baseline: LDA (F1=0.5696), Best tuned: SVM-RBF (F1=0.5701) | — |
| Publication-ready visualizations | ⚠️ PARTIAL | Existing figures in DA2/figures/m6/, need DA3-specific finals | Minor gap |

**Part A Assessment:** 90% complete. Need DA3-specific final comparison visualizations.

### Part B: Statistical Analysis

| Component | Status | Evidence | Gap |
|-----------|--------|----------|-----|
| Statistical significance test | ❌ MISSING | — | **CRITICAL GAP** |
| Fold-level CV data | ⚠️ UNCLEAR | May exist in model_objects_v2.rds, needs verification | **CRITICAL GAP** |
| Hypothesis testing | ❌ MISSING | No null/alternative hypotheses documented | **CRITICAL GAP** |
| Test statistic | ❌ MISSING | — | **CRITICAL GAP** |
| p-value | ❌ MISSING | — | **CRITICAL GAP** |
| Post-hoc analysis | ❌ MISSING | No pairwise comparisons (Nemenyi/Wilcoxon) | **CRITICAL GAP** |
| Statistical visualizations | ❌ MISSING | No CD diagram, no distribution plots | **CRITICAL GAP** |
| Interpretation document | ❌ MISSING | — | **CRITICAL GAP** |

**Part B Assessment:** 0% complete. Entire statistical analysis component missing.

### Criterion 1 Overall

**Current Completion:** ~45%  
**Expected Mark:** 0.9 / 2.0 (if submitted as-is)  
**To Achieve Full 2.0 Marks:** Implement statistical analysis (Friedman test, post-hoc, interpretation)  
**Priority:** 🔴 HIGHEST (must implement immediately)

---

## CRITERION 2: INTERACTIVE DASHBOARD (2 Marks)

| Component | Status | Evidence | Gap |
|-----------|--------|----------|-----|
| R Shiny app exists | ❌ MISSING | No `dashboard/app.R` | **CRITICAL GAP** |
| Interactive (not static) | ❌ MISSING | — | **CRITICAL GAP** |
| Plotly charts | ❌ MISSING | — | **CRITICAL GAP** |
| Multiple tabs/sections | ❌ MISSING | — | **CRITICAL GAP** |
| User controls (selectors) | ❌ MISSING | — | **CRITICAL GAP** |
| Overview/KPI section | ❌ MISSING | — | **CRITICAL GAP** |
| Model comparison section | ❌ MISSING | — | **CRITICAL GAP** |
| Model diagnostics section | ❌ MISSING | — | **CRITICAL GAP** |
| Clustering visualization | ❌ MISSING | — | **CRITICAL GAP** |
| Data loaded dynamically | ❌ MISSING | — | **CRITICAL GAP** |
| Responsive UI | ❌ MISSING | — | **CRITICAL GAP** |
| Run instructions | ❌ MISSING | — | **CRITICAL GAP** |

**Alternative Technologies:**
- Plotly Dash (Python): ❌ Not implemented
- Tableau: ❌ Not implemented
- Power BI: ❌ Not implemented

### Criterion 2 Overall

**Current Completion:** 0%  
**Expected Mark:** 0.0 / 2.0 (if submitted as-is)  
**To Achieve Full 2.0 Marks:** Build complete R Shiny + Plotly dashboard with 4 tabs  
**Priority:** 🔴 HIGH (second priority after statistical analysis)

---

## CRITERION 3: RESULT INTERPRETATION & DISCUSSION (2 Marks)

| Component | Status | Evidence | Gap |
|-----------|--------|----------|-----|
| Dataset distribution discussion | ⚠️ PARTIAL | Exists in DA2 M1-M7 reports | Needs DA3 integration |
| Feature engineering impact | ⚠️ PARTIAL | Exists in DA2 M2 report | Needs interpretation depth |
| Feature selection discussion | ⚠️ PARTIAL | Exists in DA2 M3 report | Needs interpretation depth |
| Model comparison discussion | ⚠️ PARTIAL | Exists in DA2 M4-M6 reports | Needs DA3 synthesis |
| Hyperparameter tuning discussion | ⚠️ PARTIAL | Exists in DA2 M5 report | Needs diminishing returns discussion |
| Statistical analysis interpretation | ❌ MISSING | No statistical test performed yet | **Depends on Criterion 1** |
| Confusion matrix analysis | ⚠️ PARTIAL | CM exists in DA2/figures/m6/ | Needs deep error pattern analysis |
| ROC-AUC interpretation | ⚠️ PARTIAL | ROC curves exist | Needs discrimination ability discussion |
| Feature importance discussion | ⚠️ PARTIAL | Importance plots exist | Needs causation warning, domain insights |
| Clustering interpretation | ⚠️ PARTIAL | Exists in DA2 M7 report | Needs theme discussion, OA relationship |
| Integrated discussion document | ❌ MISSING | Fragmented across DA2 reports | **CRITICAL GAP** |

### Criterion 3 Overall

**Current Completion:** ~40%  
**Raw material exists but not synthesized**  
**Expected Mark:** 0.8 / 2.0 (if submitted as-is)  
**To Achieve Full 2.0 Marks:** Write comprehensive DA3 results discussion (8000+ words, 10 sections)  
**Priority:** 🟡 HIGH (third priority)

---

## CRITERION 4: CONCLUSION, LIMITATIONS, FUTURE SCOPE (1 Mark)

### Part A: Conclusion

| Component | Status | Evidence | Gap |
|-----------|--------|----------|-----|
| DA3 achievements summary | ⚠️ PARTIAL | README mentions status | Needs formal conclusion |
| Key findings stated | ⚠️ PARTIAL | Best models identified in DA2 | Needs DA3 synthesis with stats |
| Contribution statement | ⚠️ PARTIAL | README mentions pipeline | Needs DA3 contribution articulation |

### Part B: Limitations

| Component | Status | Evidence | Gap |
|-----------|--------|----------|-----|
| Limitations list | ⚠️ PARTIAL | README has brief limitations section | Needs structured expansion |
| Dataset size limitation | ✅ EXISTS | Mentioned in README | Needs justification |
| Domain scope limitation | ✅ EXISTS | Mentioned in README | Needs justification |
| Performance limitation | ✅ EXISTS | 57% accuracy acknowledged | Needs explanation |
| Feature type limitation (TF-IDF) | ✅ EXISTS | Mentioned | Needs impact discussion |
| Static dataset limitation | ❌ MISSING | — | Add |
| Metadata dependency | ❌ MISSING | — | Add |
| Clustering subjectivity | ❌ MISSING | — | Add |
| No full-text analysis | ❌ MISSING | — | Add |
| CV fold count | ❌ MISSING | — | Add |

### Part C: Future Scope

| Component | Status | Evidence | Gap |
|-----------|--------|----------|-----|
| Future scope list | ⚠️ PARTIAL | README has "Future roadmap" | Needs structure |
| RAG clearly marked NOT IMPLEMENTED | ⚠️ UNCLEAR | README says "not implemented" | Needs emphasis |
| Ollama/LLM clearly marked NOT IMPLEMENTED | ❌ MISSING | Not mentioned in README | **Add explicitly** |
| Fine-tuning clearly marked NOT IMPLEMENTED | ❌ MISSING | Not mentioned | **Add explicitly** |
| Semantic embeddings (future) | ⚠️ PARTIAL | Mentioned briefly | Expand |
| Larger dataset (future) | ⚠️ PARTIAL | Mentioned briefly | Expand |
| Priority levels | ❌ MISSING | No prioritization | Add (Tier 1/2/3) |

### Criterion 4 Overall

**Current Completion:** ~50%  
**Expected Mark:** 0.5 / 1.0 (if submitted as-is)  
**To Achieve Full 1.0 Mark:** Formalize conclusion, expand limitations (8-10 items), structure future scope with priorities  
**Priority:** 🟢 MEDIUM (fourth priority)

---

## CRITERION 5: FINAL REPORT QUALITY & CODE ORGANIZATION (2 Marks)

### Part A: Final Report

| Component | Status | Evidence | Gap |
|-----------|--------|----------|-----|
| DA3-specific final report | ❌ MISSING | DA2 report exists, not DA3 | **CRITICAL GAP** |
| Report structure/ToC | ⚠️ PARTIAL | DA2 report well-structured | Needs DA3 version |
| Abstract/summary | ⚠️ PARTIAL | DA2 report has summary | Needs DA3 version |
| All 6 DA3 criteria covered | ❌ MISSING | DA2 report doesn't cover DA3 criteria | **CRITICAL GAP** |
| Figures embedded | ⚠️ PARTIAL | DA2 report has figures | Needs DA3 figures added |
| Tables formatted | ⚠️ PARTIAL | DA2 tables exist | Needs DA3 tables |
| Professional writing | ✅ GOOD | DA2 report is well-written | Maintain quality |
| References cited | ⚠️ PARTIAL | DA2 has some references | Expand for DA3 |
| Length appropriate (12k-15k words) | ⚠️ PARTIAL | DA2 report ~10k words | Needs expansion for DA3 |

### Part B: Code Organization

| Component | Status | Evidence | Gap |
|-----------|--------|----------|-----|
| Clear directory structure | ✅ GOOD | Phase 2 well-organized | Needs DA3 addition |
| DA3-specific directory | ❌ MISSING | No `DA3/` folder with deliverables | **CRITICAL GAP** |
| Dashboard directory | ❌ MISSING | No `dashboard/` folder | **CRITICAL GAP** |
| Code comments/documentation | ⚠️ PARTIAL | Some scripts commented | Needs improvement |
| Header comments in scripts | ⚠️ PARTIAL | Some have headers | Needs standardization |
| Reproducibility instructions | ✅ GOOD | `DA2/REPRODUCIBILITY.md` exists | Extend for DA3 |
| README updated | ⚠️ PARTIAL | README mentions DA2 complete | Needs DA3 status update |
| No code duplication | ✅ GOOD | Clean code structure | Maintain |
| File naming conventions | ✅ GOOD | Consistent naming | Maintain |

### Criterion 5 Overall

**Current Completion:** ~35%  
**Expected Mark:** 0.7 / 2.0 (if submitted as-is)  
**To Achieve Full 2.0 Marks:** Write DA3 final report (12k-15k words), create DA3 directory structure, document all code  
**Priority:** 🟡 MEDIUM (fifth priority)

---

## CRITERION 6: FINAL PRESENTATION & VIVA (1 Mark)

| Component | Status | Evidence | Gap |
|-----------|--------|----------|-----|
| Presentation slides | ❌ MISSING | DA2 demo script exists, not slides | **CRITICAL GAP** |
| 10-15 minute presentation | ❌ MISSING | — | **CRITICAL GAP** |
| Covers all DA3 criteria | ❌ MISSING | — | **CRITICAL GAP** |
| Live demonstration | ⚠️ UNCLEAR | Can demo DA2, but no DA3 dashboard yet | **Depends on Criterion 2** |
| Viva Q&A preparation | ⚠️ PARTIAL | `DA2/DA2_Viva_QA.md` exists for DA2 | Needs DA3 extension |
| Statistical results presentation | ❌ MISSING | No stats to present yet | **Depends on Criterion 1** |
| Dashboard demonstration | ❌ MISSING | No dashboard yet | **Depends on Criterion 2** |
| Professional delivery readiness | ⚠️ UNCLEAR | Cannot assess until materials prepared | TBD |

### Criterion 6 Overall

**Current Completion:** ~15%  
**Expected Mark:** 0.15 / 1.0 (if submitted as-is)  
**To Achieve Full 1.0 Mark:** Create presentation (12 slides), prepare live demo script, extend viva Q&A for DA3  
**Priority:** 🟢 MEDIUM (sixth priority)

---

## OVERALL DA3 RUBRIC COVERAGE

| Criterion | Weight | Current % | Expected Mark | Full Mark | Gap |
|-----------|--------|-----------|---------------|-----------|-----|
| 1. Performance Comparison & Stats | 20% | 45% | 0.9 | 2.0 | -1.1 |
| 2. Interactive Dashboard | 20% | 0% | 0.0 | 2.0 | -2.0 |
| 3. Result Interpretation | 20% | 40% | 0.8 | 2.0 | -1.2 |
| 4. Conclusion/Limitations/Future | 10% | 50% | 0.5 | 1.0 | -0.5 |
| 5. Report & Code Organization | 20% | 35% | 0.7 | 2.0 | -1.3 |
| 6. Presentation & Viva | 10% | 15% | 0.15 | 1.0 | -0.85 |
| **TOTAL** | **100%** | **30%** | **3.05** | **10.0** | **-6.95** |

### Summary

- **Current Project Status:** 30% DA3 completion
- **Expected Grade (if submitted now):** 3.05 / 10 marks (30.5%)
- **Missing Marks:** 6.95 / 10 (69.5%)
- **Critical Gaps:** Statistical analysis (Criterion 1B), Interactive dashboard (Criterion 2)
- **Feasibility to Achieve 10/10:** HIGH (with focused 10-day effort)

---

## IMPLEMENTATION PRIORITIES TO CLOSE GAPS

### 🔴 HIGHEST PRIORITY (Must Have for Passing Grade)

1. **Statistical Analysis** (Criterion 1B)
   - Implement Friedman test
   - Document hypothesis testing
   - Create statistical visualizations
   - **Impact:** +1.1 marks
   - **Effort:** 6-8 hours

2. **Interactive Dashboard** (Criterion 2)
   - Build R Shiny app with 4 tabs
   - Implement Plotly charts
   - Add user controls
   - **Impact:** +2.0 marks
   - **Effort:** 14-18 hours

**Combined Impact:** +3.1 marks (brings total to 6.15/10, passing grade)

### 🟡 HIGH PRIORITY (For Strong Grade)

3. **Results Discussion** (Criterion 3)
   - Write comprehensive interpretation
   - Integrate statistical findings
   - Deep-dive error analysis
   - **Impact:** +1.2 marks
   - **Effort:** 8-10 hours

4. **Conclusion & Limitations** (Criterion 4)
   - Formalize conclusion
   - Expand limitations (8-10 items)
   - Structure future scope
   - **Impact:** +0.5 marks
   - **Effort:** 4-5 hours

**Combined Impact:** +1.7 marks (brings total to 7.85/10, good grade)

### 🟢 MEDIUM PRIORITY (For Excellent Grade)

5. **DA3 Final Report** (Criterion 5A)
   - Write comprehensive report (12k-15k words)
   - Integrate all sections
   - Professional formatting
   - **Impact:** +0.65 marks
   - **Effort:** 8-10 hours

6. **Code Organization** (Criterion 5B)
   - Create DA3 directory structure
   - Document all scripts
   - Update README
   - **Impact:** +0.65 marks
   - **Effort:** 3-4 hours

7. **Presentation & Viva** (Criterion 6)
   - Create slides (12 slides)
   - Prepare live demo
   - Extend viva Q&A
   - **Impact:** +0.85 marks
   - **Effort:** 6-8 hours

**Combined Impact:** +2.15 marks (brings total to 10.0/10, excellent grade)

---

## TOTAL EFFORT REQUIRED

| Priority | Tasks | Estimated Hours | Impact (Marks) |
|----------|-------|-----------------|----------------|
| 🔴 Highest | Statistical Analysis + Dashboard | 20-26 hours | +3.1 |
| 🟡 High | Results Discussion + Conclusion/Limits | 12-15 hours | +1.7 |
| 🟢 Medium | Report + Code Org + Presentation | 17-22 hours | +2.15 |
| **TOTAL** | **All DA3 Implementation** | **49-63 hours** | **+6.95** |

**Available Time:** 10 days (80-100 working hours if full-time)  
**Feasibility:** ✅ HIGHLY FEASIBLE (50% time buffer available)

---

## RECOMMENDED APPROACH

### Minimum Viable DA3 (7/10 marks, 35 hours)
Focus on: Statistical Analysis + Dashboard + Basic Discussion + Basic Conclusion
- Effort: 35-40 hours
- Mark: ~7.0-7.5 / 10
- Risk: LOW

### Complete DA3 (10/10 marks, 60 hours)
Implement all components according to implementation plan
- Effort: 50-60 hours
- Mark: 9.5-10.0 / 10
- Risk: LOW-MEDIUM (time management required)

### Recommended Strategy
1. Days 1-5: Focus on highest priority (statistical analysis + dashboard) → secure 6/10
2. Days 6-8: Focus on high priority (discussion + conclusion) → reach 8/10
3. Days 9-10: Focus on medium priority (report + presentation) → reach 10/10

**Conclusion:** With disciplined execution of the implementation plan, achieving full 10/10 marks is realistic and achievable.

---

## FILES REQUIRING CREATION/MODIFICATION

### To Create (15 New Files)

1. `DA3/DA3_Final_Report.md`
2. `DA3/DA3_Statistical_Analysis.md`
3. `DA3/DA3_Results_Discussion.md`
4. `DA3/DA3_Conclusion_Limitations_Future.md`
5. `DA3/DA3_Presentation.pdf` (or .pptx)
6. `DA3/DA3_Viva_QA.md`
7. `DA3/README.md`
8. `dashboard/app.R`
9. `dashboard/global.R`
10. `dashboard/README.md`
11. `r/scripts/phase3/11_statistical_analysis.R`
12. `r/scripts/phase3/12_final_comparison.R`
13. `r/scripts/phase3/13_export_for_dashboard.R`
14. `data/ml_r/cv_fold_results.csv`
15. `DA3_CHECKLIST.md`

### To Modify (3 Existing Files)

1. `README.md` (add DA3 status)
2. `DA2/DA2_Viva_QA.md` (extend with DA3 questions) OR create separate DA3 version
3. `.gitignore` (if dashboard generates temp files)

---

**Coverage Assessment Complete**  
**Next Action:** Begin implementation according to DA3_IMPLEMENTATION_PLAN.md  
**Start With:** Statistical Analysis (Task 1.1)

