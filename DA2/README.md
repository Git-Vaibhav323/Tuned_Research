# DA2 — ResearchPilot Phase 2 Documentation Pack

**Status covered:** M1 · M2 · M3 · M4 · **M5** (complete)  
**Next:** M6 (comparative visualizations)

## Start here

1. [DA2_Progress_Report_M1_M2_M3.md](DA2_Progress_Report_M1_M2_M3.md) — foundation through features  
2. [M4_Model_Training_Report.md](M4_Model_Training_Report.md) — 14-model training  
3. [M5_Hyperparameter_Tuning_Report.md](M5_Hyperparameter_Tuning_Report.md) — tuning results  

## Detailed milestone reports

| Milestone | Report |
|-----------|--------|
| M1 Database connectivity | [M1_Database_Connectivity_Report.md](M1_Database_Connectivity_Report.md) |
| M2 Feature engineering | [M2_Feature_Engineering_Report.md](M2_Feature_Engineering_Report.md) |
| M3 Feature selection | [M3_Feature_Selection_Report.md](M3_Feature_Selection_Report.md) |
| M4 Model training | [M4_Model_Training_Report.md](M4_Model_Training_Report.md) |
| M5 Hyperparameter tuning | [M5_Hyperparameter_Tuning_Report.md](M5_Hyperparameter_Tuning_Report.md) |

## Figures & tables

- [figures/m3_feature_selection_agreement.png](figures/m3_feature_selection_agreement.png)
- [figures/m4_model_comparison_f1.png](figures/m4_model_comparison_f1.png)
- [figures/m5_vs_m4_test_f1.png](figures/m5_vs_m4_test_f1.png)
- [m4_leaderboard.md](m4_leaderboard.md)
- [m5_tuning_leaderboard.md](m5_tuning_leaderboard.md)

## Quick rebuild commands

```powershell
cd E:\Tuned_Research
python scripts/phase2/01_load_to_db.py
python scripts/phase2/02_build_ml_features.py
python scripts/phase2/03_feature_selection.py
python scripts/phase2/04_train_models.py
python scripts/phase2/05_tune_hyperparameters.py
```
