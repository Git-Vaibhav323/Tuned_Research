# ResearchPilot — Dataset and Data Science Plan

**Primary phase:** DA1  
**Downstream consumers:** DA2 database/models and DA3 assistant  
**Status:** Planning; no dataset created yet

## 1. Purpose

DA1 establishes the empirical foundation of ResearchPilot. The selected dataset must support meaningful cleaning, preprocessing, EDA, R visualization, database integration, and at least one defensible DA2 modeling task. LLM instruction data is optional future material for DA3 and is not the only dataset concern.

## 2. Dataset Selection Criteria

The final dataset should:

- Relate directly to research papers, scholarly metadata, research methods, citations, abstracts, or researcher workflows.
- Have a legal and documented source/license.
- Include enough rows and variables for non-trivial EDA and feature engineering.
- Contain realistic quality issues suitable for documented cleaning.
- Support relational or structured database storage.
- Support at least one measurable ML/DL task in DA2.
- Avoid sensitive personal data unless governance requirements are explicitly addressed.

Candidate sources may include open scholarly metadata, open-access paper metadata/abstracts, citation data, publication venues, fields of study, and author-approved text.

## 3. Data Zones

| Zone | Location | Rule |
|------|----------|------|
| Raw | `data/raw_papers/` | Immutable source documents |
| External | `data/external/` | Immutable third-party structured datasets |
| Interim | `data/interim/` | Reproducible intermediate transformations |
| Processed | `data/processed/` | Analysis-ready, validated datasets |
| Metadata | `data/metadata/` | Data dictionary, provenance, licenses, manifests |
| Benchmark | `data/benchmark/` | Frozen DA2/DA3 evaluation records |
| Instruction | `data/instruction_dataset/` | DA3-only instruction examples, if required |

Large or restricted data should not be committed to Git. Store acquisition instructions, checksums, schemas, and small license-cleared samples instead.

## 4. DA1 Workflow

```text
Define research questions
  ↓
Select and license-screen data source
  ↓
Collect immutable raw/external data
  ↓
Profile schema and quality
  ↓
Document data dictionary and provenance
  ↓
Clean missing values, duplicates, types, text, and outliers
  ↓
Preprocess and validate
  ↓
Exploratory Data Analysis
  ├── Python analysis and visualization
  └── R analysis and ggplot visualizations
  ↓
Publish processed data manifest and DA1 findings
```

## 5. Required Dataset Documentation

The DA1 metadata package should include:

- Dataset title, source URL/API, retrieval date, and version
- License and redistribution constraints
- Unit of observation
- Row and column counts before and after cleaning
- Data dictionary: name, type, meaning, allowed values, missingness
- Collection method and sampling limitations
- Known quality issues and potential biases
- Cleaning log and transformation summary
- File checksums or reproducible acquisition instructions

## 6. Cleaning and Preprocessing Plan

Cleaning decisions must be based on profiling, not assumptions. Candidate checks include:

- Duplicate records and duplicate paper identifiers
- Missing identifiers, abstracts, dates, categories, or outcomes
- Invalid types, date ranges, encodings, and categorical labels
- Text normalization and whitespace/markup cleanup
- Outlier detection with domain-aware treatment
- Consistency across DOI, title, author, venue, and year fields
- Class imbalance for the selected DA2 target
- Leakage-prone columns that reveal the target

Reusable transformations belong in `src/researchpilot/data/`; notebooks should call these functions rather than hide the complete pipeline in notebook cells.

## 7. EDA Plan

DA1 analysis should cover:

- Dataset shape, types, missingness, and uniqueness
- Distribution of numerical and categorical variables
- Temporal, domain, venue, and publication patterns where available
- Relationships among candidate predictors and outcomes
- Correlation/association analysis with appropriate caveats
- Outliers, imbalance, and bias indicators
- Findings that motivate DA2 feature engineering and model selection

Each chart must answer a stated question and include a short interpretation.

## 8. R Visualization Requirement

R work belongs under `r/` and should:

- Read the same processed dataset used by Python EDA
- Use reproducible scripts or R Markdown
- Include several meaningful `ggplot2` visualizations
- Export figures to `r/visualizations/` or `reports/figures/`
- Document package requirements and execution order in `r/README.md`
- Explain any differences between Python and R analysis results

## 9. DA2 Readiness

Before DA2 begins, the DA1 dataset must have:

- A stable processed schema
- A candidate database design and keys
- A clearly defined modeling target or unsupervised objective
- A leakage-safe split strategy
- Baseline metric choice
- Documented class balance and sampling concerns

The database should load processed data; it should not become an undocumented alternate cleaning pipeline.

## 10. DA3 Data Extension

DA3 may derive:

- Retrieval chunks from processed paper text
- A frozen assistant benchmark
- Instruction-response examples
- Source-attribution metadata

These assets must preserve links to the DA1 provenance records. DA3 data preparation must not overwrite DA1 or DA2 artifacts.

## 11. DA1 Exit Checklist

- [ ] Research questions and dataset selection justified
- [ ] License/provenance recorded
- [ ] Data dictionary completed
- [ ] Raw data preserved
- [ ] Cleaning and preprocessing reproducible
- [ ] Processed schema validated
- [ ] Python EDA completed and interpreted
- [ ] R visualizations completed and interpreted
- [ ] Bias, limitations, and missingness documented
- [ ] DA2 target, keys, and split strategy proposed
