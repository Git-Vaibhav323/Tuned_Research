# DA2 — ResearchPilot Phase 2 Documentation Pack

**Status covered:** M1 · M2 · M3 · M4 · M5 · **M6** (complete)  
**Next:** M7 (impact-tier track + clustering) / M8 (final demo pack)

## Start here

1. [DA2_Progress_Report_M1_M2_M3.md](DA2_Progress_Report_M1_M2_M3.md) — foundation  
2. [M4_Model_Training_Report.md](M4_Model_Training_Report.md) — 14 models  
3. [M5_Hyperparameter_Tuning_Report.md](M5_Hyperparameter_Tuning_Report.md) — tuning  
4. [M6_Comparative_Analysis_Report.md](M6_Comparative_Analysis_Report.md) — ROC / PR / CM / importance  

## Detailed milestone reports

| Milestone | Report |
|-----------|--------|
| M1 Database connectivity | [M1_Database_Connectivity_Report.md](M1_Database_Connectivity_Report.md) |
| M2 Feature engineering | [M2_Feature_Engineering_Report.md](M2_Feature_Engineering_Report.md) |
| M3 Feature selection | [M3_Feature_Selection_Report.md](M3_Feature_Selection_Report.md) |
| M4 Model training | [M4_Model_Training_Report.md](M4_Model_Training_Report.md) |
| M5 Hyperparameter tuning | [M5_Hyperparameter_Tuning_Report.md](M5_Hyperparameter_Tuning_Report.md) |
| M6 Comparative visualizations | [M6_Comparative_Analysis_Report.md](M6_Comparative_Analysis_Report.md) |

## Key figures (M6)

Folder: [figures/m6/](figures/m6/)

- `cm_grid_comparative.png` — confusion matrices  
- `roc_auc_comparison.png` / `roc_ovr_*.png` — ROC  
- `pr_ap_comparison.png` / `pr_*.png` — Precision–Recall  
- `metrics_grouped_bars.png` — metric overview  
- `feature_importance_*.png` — importances  

Also: [m6_comparative_metrics.md](m6_comparative_metrics.md)

## Quick rebuild commands

```powershell
cd E:\Tuned_Research
python scripts/phase2/04_train_models.py
python scripts/phase2/05_tune_hyperparameters.py
python scripts/phase2/06_comparative_visualizations.py
```
