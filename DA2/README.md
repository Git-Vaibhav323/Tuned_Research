# DA2 README

**Status:** M1–M7 complete and frozen · **M8 documentation/demo pack complete**

## Start here

1. [ResearchPilot_DA2_Final_Report.md](ResearchPilot_DA2_Final_Report.md) — consolidated DA2 report  
2. [DA2_FINAL_CHECKLIST.md](DA2_FINAL_CHECKLIST.md) — marks ↔ evidence  
3. [DA2_Demo_Script.md](DA2_Demo_Script.md) — 7–10 min live demo  
4. [DA2_Viva_QA.md](DA2_Viva_QA.md) — ≥40 viva questions  
5. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — exact commands  
6. [DA2_Model_Inventory.md](DA2_Model_Inventory.md) — algorithms + metrics  
7. [M8_Cleanup_Recommendations.md](M8_Cleanup_Recommendations.md) — no deletes until approved  

## Milestone reports

| Milestone | Report |
|-----------|--------|
| Audit | [M8_Repository_Audit.md](M8_Repository_Audit.md) |
| M1 | [M1_Database_Connectivity_Report.md](M1_Database_Connectivity_Report.md) |
| M2 | [M2_Feature_Engineering_Report.md](M2_Feature_Engineering_Report.md) |
| M3 | [M3_Feature_Selection_Report.md](M3_Feature_Selection_Report.md) |
| M4 | [M4_Model_Training_Report.md](M4_Model_Training_Report.md) |
| M5 | [M5_Hyperparameter_Tuning_Report.md](M5_Hyperparameter_Tuning_Report.md) |
| M6 | [M6_Comparative_Analysis_Report.md](M6_Comparative_Analysis_Report.md) |
| M7 | [M7_Impact_and_Clustering_Report.md](M7_Impact_and_Clustering_Report.md) |

## Frozen leaders (do not invent)

- OA Acc/F1: **m4_adaboost** 0.482 / 0.452  
- OA ROC-AUC: **m5_extra_trees_tuned** 0.628  
- Impact: AdaBoost test F1 **0.543**, Acc **0.551**, ROC **0.680**  
- Clusters: k=8, silhouette **0.064**

## Live demo commands

```powershell
python scripts/phase2/08_da2_database_demo.py
python scripts/phase2/08_generate_architecture_diagram.py
```
