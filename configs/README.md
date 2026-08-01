# Configs

YAML configuration files for reproducible runs.

| Path | Phase | Purpose |
|------|-------|---------|
| `phase1/data_collection.yaml` | 1 | Sources and output paths |
| `phase1/preprocessing.yaml` | 1 | Cleaning / preprocessing rules |
| `phase2/ml_experiment.yaml` | 2 | Models, splits, metrics |
| `phase3/rag.yaml` | 3 | Chunking, embeddings, retrieval |
| `phase3/finetune.yaml` | 3 | PEFT / training hyperparameters |
| `phase3/dashboard.yaml` | 3 | Frontend and API settings |

Values are placeholders until implementation starts.
