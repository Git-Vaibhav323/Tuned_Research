# Preprocessing Report

**Input:** `data/raw/raw_papers.csv`  
**Output:** `data/cleaned/cleaned_papers_r.csv`  

## Pipeline

1. Load raw data
2. Missing value audit
3. Title and abstract cleaning
4. Type conversion
5. Language filter
6. DOI deduplication
7. Categorical encoding
8. Min-max normalization
9. Save cleaned dataset

## Row impact

| Step | Detail | Before | After | Removed |
|------|--------|-------:|------:|--------:|
| load_raw | Read raw_papers.csv | 2000 | 2000 | 0 |
| clean_title | Remove newlines and extra spaces in title | 2000 | 2000 | 0 |
| clean_abstract | Normalize whitespace; remove empty abstracts | 2000 | 2000 | 0 |
| type_conversion | Convert year and citations to integer | 2000 | 2000 | 0 |
| language_filter | Keep English records only | 2000 | 2000 | 0 |
| doi_dedup | Remove duplicate DOIs; keep highest citations | 2000 | 2000 | 0 |

