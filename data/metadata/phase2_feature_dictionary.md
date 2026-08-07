# Phase 2 Feature Dictionary (M2)

**Source:** `data/final/final_dataset.csv` (Phase 1, read-only)  
**Feature version:** `v1`  
**Config:** `configs/phase2/features.yaml`

---

## Targets

| Track | Target | Type | Notes |
|-------|--------|------|-------|
| Primary | `oa_category` | Multiclass | `fully_open` / `partially_open` / `closed` — from Phase 1 |
| Secondary | `impact_tier` | Multiclass | `low` / `medium` / `high` — derived in M2 from train tertiles of `citation_per_year` |

### Impact tier rule

1. Stratified split by `oa_category` (70% / 15% / 15%).
2. Compute train quantiles of `citation_per_year` at ~33% and ~66%.
3. Apply those thresholds to **all** rows → `impact_tier`.

This avoids fitting thresholds on the full dataset (label construction leakage).

---

## Structural features (Phase 1 engineered / metadata)

| Feature | Used in OA track | Used in impact track | Description |
|---------|------------------|----------------------|-------------|
| `publication_year` | yes | yes | Year of publication |
| `paper_age` | yes | yes | Years since publication (ref 2026) |
| `title_length` | yes | yes | Title character length |
| `abstract_length` | yes | yes | Abstract character length |
| `keyword_count` | yes | yes | Count of keywords |
| `concept_count` | yes | yes | Count of concepts |
| `recent_paper` | yes | yes | 1 if age ≤ 2 |
| `has_doi` | **dropped** | **dropped** | Near-constant (~99.9%); no signal |
| `is_open_access` | **no** | yes | OA flag (leakage for OA track) |
| `has_fulltext` | **no** | yes | Fulltext flag (leakage for OA track) |
| `oa_cat_*` one-hots | **no** | yes | Encoded `oa_category` for impact track |

---

## Text features

TF-IDF on `title + abstract` (fit on **train** only):

- `max_features`: 100  
- `ngram_range`: (1, 2)  
- English stop words  
- Columns named `tfidf_<token>`

Vectorizer path: `data/ml/tfidf_vectorizer.joblib`

---

## Leakage exclusions

### OA category track — never use as inputs

`is_open_access`, `oa_status`, `oa_url`, `oa_category`, `open_access`, `has_fulltext`, plus raw identifiers/JSON/text columns (text enters only via TF-IDF).

### Impact tier track — never use as inputs

`cited_by_count`, `citation_per_year`, `citation_log`, `impact_tier`, and raw OA JSON / status / URL (category one-hots are allowed).

---

## Feature selection (primary track)

- Method: mutual information (`SelectKBest`)
- Fit on **train** only, target = `oa_category`
- Keep top **k = 30** features → `data/ml/X_oa_category_selected.*`
- Scores: `data/ml/feature_selection_scores_oa.csv`

---

## Artifacts

| File | Role |
|------|------|
| `data/ml/train.csv` / `val.csv` / `test.csv` | Split exports with labels + matrices |
| `data/ml/feature_lists.json` | Feature lists + leakage rules |
| `data/ml/label_maps.json` | Class maps + counts |
| `data/ml/m2_build_summary.json` | Build summary |
| SQLite `ml_features` | Per-paper summary + split + tiers |
