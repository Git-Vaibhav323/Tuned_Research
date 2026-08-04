# Preprocessing Report (R Demonstration)

**Project:** ResearchPilot  
**Input:** `data/raw/raw_papers.csv`  
**Output:** `data/cleaned/cleaned_papers_r.csv`  

## Steps demonstrated

1. Load raw data
2. Missing-value audit (before)
3. Title cleaning (newlines / whitespace)
4. Abstract cleaning + drop empty abstracts
5. Type conversion (`publication_year`, `cited_by_count` → integer)
6. Language filter (English only)
7. DOI deduplication
8. Encoding example (`language_code`, `type_code`)
9. Normalization example (min-max on citations)
10. Missing-value audit (after) + save

## Row impact

| Step | Rows before | Rows after | Removed |
|------|------------:|-----------:|--------:|
| 1_load_raw | 2000 | 2000 | 0 |
| 3_clean_title | 2000 | 2000 | 0 |
| 4_clean_abstract | 2000 | 2000 | 0 |
| 5_type_conversion | 2000 | 2000 | 0 |
| 6_language_filter | 2000 | 2000 | 0 |
| 7_doi_dedup | 2000 | 2000 | 0 |

## Notes for evaluation

- Collection already applied year/language/type/abstract filters via OpenAlex API.
- This R script re-validates and demonstrates preprocessing transparently.
- Production engineered features (log citations, age-normalized citations) are in `feature_engineering.py` / `final_dataset.csv`.

