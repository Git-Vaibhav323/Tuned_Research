# Scripts

Command-line entrypoints organized by phase. All scripts are scaffolds until their phase begins.

| Path | Purpose |
|------|---------|
| `phase1/collect_openalex.py` | Fetch ~2000 OpenAlex papers (2022–2025, AI/ML) → `data/raw/raw_papers.csv` |
| `phase1/collect_data.py` | Generic collection entrypoint (scaffold) |
| `phase1/preprocess_papers.py` | Clean papers → `data/cleaned/cleaned_papers.csv` + report |
| `phase1/preprocess_data.py` | Alias entrypoint for the preprocessing pipeline |
| `phase1/feature_extraction.py` | Parse JSON fields → `data/processed/feature_extracted.csv` |
| `phase1/feature_engineering.py` | Engineer ML features → `data/final/final_dataset.csv` + report |
| `phase2/train_models.py` | Train ML models |
| `phase2/compare_models.py` | Comparative metrics and plots |
| `phase3/build_index.py` | Build RAG vector index |
| `phase3/finetune_model.py` | Fine-tune / PEFT training |
| `phase3/run_dashboard.py` | Launch dashboard |

### OpenAlex collection

```bash
# from project root, with OpenAlex_API_KEY set in .env
python scripts/phase1/collect_openalex.py
```

Requires: `pandas`, `requests`, `python-dotenv`.

Configs live under `configs/phase1|phase2|phase3/`.
