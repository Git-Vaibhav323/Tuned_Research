# ResearchPilot — Reproducibility Guide (DA2 / M8)

**Project root:** `E:\Tuned_Research` (or your clone path)  
**Do not commit API keys, `.env` files, or `*.db` dumps.**

---

## Environment

| Component | Guidance |
|-----------|----------|
| OS | Windows 10/11 tested; Linux/macOS should work with path adjustments |
| Python | **3.10+** recommended (project developed with modern CPython) |
| R | Optional for EDA (`r/r_packages.txt`) |
| GPU | **Not required**. XGBoost forced CPU-safe in M7 |

Install Python deps:

```powershell
cd E:\Tuned_Research
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` lists planned packages across phases; for Phase 2 ensure at least:  
`pandas numpy scikit-learn matplotlib seaborn pyyaml joblib xgboost lightgbm sqlalchemy pyarrow requests tqdm`.

R (optional):

```r
source("r/scripts/00_install_packages.R")
```

### Secrets / API

OpenAlex collection may use a polite contact email. If you use `.env`, keep it gitignored. **Never paste keys into reports.**

---

## Inputs / outputs

| Stage | Inputs | Outputs |
|-------|--------|---------|
| Phase 1 | OpenAlex API | `data/raw/` → `data/final/final_dataset.csv` |
| M1 | final CSV | `database/researchpilot.db` |
| M2 | final CSV | `data/ml/*`, `label_maps.json` |
| M3 | M2 matrices | `data/ml/m3/*` |
| M4 | M3 OA matrices | `evaluation/reports/m4/`, `models/checkpoints/m4/` |
| M5 | M4 + matrices | `evaluation/reports/m5/`, `models/checkpoints/m5/` |
| M6 | M4/M5 preds | `evaluation/reports/m6/`, `reports/figures/m6/` |
| M7 | M3 impact matrices | `evaluation/reports/m7/`, `reports/figures/m7/`, `models/checkpoints/m7/` |

Database location: `database/researchpilot.db` (gitignored).

---

## Exact commands

### Phase 1

```powershell
python scripts/phase1/collect_openalex.py
python scripts/phase1/preprocess_papers.py
python scripts/phase1/feature_extraction.py
python scripts/phase1/feature_engineering.py
```

Optional R EDA: `r/scripts/01_eda_final_dataset.R` (and related scripts).

### Phase 2 — M1 to M7

```powershell
python scripts/phase2/01_load_to_db.py
python scripts/phase2/02_build_ml_features.py
python scripts/phase2/03_feature_selection.py
python scripts/phase2/04_train_models.py
python scripts/phase2/05_tune_hyperparameters.py
python scripts/phase2/06_comparative_visualizations.py
python scripts/phase2/07_impact_and_clustering.py
```

### M8 helpers (docs / demo; no metric overwrite intent)

```powershell
python scripts/phase2/08_da2_database_demo.py
python scripts/phase2/08_generate_architecture_diagram.py
python scripts/phase2/run_db_queries.py
```

---

## Frozen metrics policy

Verified M4–M7 numbers in DA2 reports are authoritative. Re-running training scripts regenerates local `evaluation/reports/` artifacts; only update published “best result” tables if you intentionally accept a full rebuild and re-QA.

Canonical frozen values are recorded in:

- `DA2/M6_Comparative_Analysis_Report.md` / `DA2/m6_comparative_metrics.csv`
- `DA2/M7_Impact_and_Clustering_Report.md` / `reports/m7_final_qa.md`

---

## Configs

| Milestone | Config |
|-----------|--------|
| M1 | `configs/phase2/db.yaml` |
| M2 | `configs/phase2/features.yaml` |
| M3 | `configs/phase2/feature_selection.yaml` |
| M4 | `configs/phase2/models.yaml` |
| M5 | `configs/phase2/tuning.yaml` |
| M6 | `configs/phase2/comparative.yaml` |
| M7 | `configs/phase2/m7_secondary.yaml` |

---

## Smoke checks

```powershell
python scripts/phase2/08_da2_database_demo.py
# Expect: Connected OK | papers rows = 2000 | three query blocks
```

Impact thresholds should remain train-only: `data/ml/label_maps.json` → `q_low≈87.3333`, `q_high≈134.0`.
