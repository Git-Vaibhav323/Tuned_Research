# DA2 — ResearchPilot Phase 2 Documentation Pack

**Status covered:** M1 · M2 · M3 (complete)  
**Next:** M4 (train 10–15 algorithms)

## Start here

1. **[DA2_Progress_Report_M1_M2_M3.md](DA2_Progress_Report_M1_M2_M3.md)** — consolidated progress report (read this first)

## Detailed milestone reports

| Milestone | Report |
|-----------|--------|
| M1 Database connectivity | [M1_Database_Connectivity_Report.md](M1_Database_Connectivity_Report.md) |
| M2 Feature engineering | [M2_Feature_Engineering_Report.md](M2_Feature_Engineering_Report.md) |
| M3 Feature selection | [M3_Feature_Selection_Report.md](M3_Feature_Selection_Report.md) |

## Figures

- [figures/m3_feature_selection_agreement.png](figures/m3_feature_selection_agreement.png)

## Quick rebuild commands

```powershell
cd E:\Tuned_Research
python scripts/phase2/01_load_to_db.py
python scripts/phase2/02_build_ml_features.py
python scripts/phase2/03_feature_selection.py
```
