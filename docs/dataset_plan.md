# ResearchPilot — Dataset Plan (Phase 1)

**Status:** Planning — no dataset created yet

---

## 1. Goals

Phase 1 produces a documented, cleaned, analysis-ready dataset that supports:

- Exploratory data analysis and visualization  
- Phase 2 machine learning / comparative analysis  
- Phase 3 fine-tuning and RAG corpora derived from the same provenance  

---

## 2. Selection Criteria

The dataset should:

- Relate to research papers, scholarly metadata, abstracts, citations, or methods  
- Have a clear license and source  
- Contain enough rows/columns for nontrivial EDA and modeling  
- Include realistic quality issues worth documenting  
- Avoid sensitive personal data unless governance is explicit  

---

## 3. Data Zones

| Zone | Path | Rule |
|------|------|------|
| Raw | `data/raw/`, `data/raw_papers/` | Immutable |
| External | `data/external/` | Immutable third-party |
| Interim | `data/interim/` | Regenerable intermediates |
| Processed | `data/processed/` | Analysis-ready |
| Metadata | `data/metadata/` | Dictionary, licenses, manifests |
| Benchmark | `data/benchmark/` | Frozen eval (Phase 2–3) |
| Instruction | `data/instruction_dataset/` | Phase 3 only |

---

## 4. Phase 1 Workflow

```text
Define research questions
  → Select and license-screen sources
  → Collect into raw/external
  → Profile quality
  → Document data dictionary
  → Clean + preprocess
  → EDA + visualizations
  → Publish processed manifest and findings
```

Scripts: `scripts/phase1/`  
Configs: `configs/phase1/`  
Notebooks: `notebooks/phase1_eda/`

---

## 5. Required Documentation

- Source URL/API, retrieval date, version  
- License and redistribution constraints  
- Unit of observation  
- Row/column counts before and after cleaning  
- Data dictionary (name, type, meaning, missingness)  
- Cleaning log and known biases  

---

## 6. Cleaning Checklist

- Duplicates and identifier conflicts  
- Missing values and invalid types  
- Text normalization / markup removal  
- Outliers with domain-aware handling  
- Consistency across title, abstract, year, venue, DOI  
- Columns that would leak the Phase 2 target  

---

## 7. EDA Checklist

- Shape, dtypes, missingness, uniqueness  
- Distributions and category frequencies  
- Relationships among candidate predictors/outcomes  
- Temporal / domain patterns if present  
- Findings that motivate Phase 2 features and models  

---

## 8. Phase 1 Exit Checklist

- [ ] Dataset choice justified  
- [ ] License/provenance recorded  
- [ ] Data dictionary complete  
- [ ] Raw data preserved  
- [ ] Preprocessing reproducible  
- [ ] EDA completed and interpreted  
- [ ] Phase 2 target and split strategy proposed  
