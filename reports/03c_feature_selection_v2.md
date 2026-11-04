# Phase 2 DA2 — Feature Selection Report v2 (Enriched Features)

> **Date**: 2026-09-22  |  **Script**: `r/scripts/phase2/02_feature_selection_v2.R`

## Summary

| Stage | Features |
|-------|---------|
| Input (enriched matrix) | 205 |
| After NZV filter | 164 |
| After correlation filter | 164 |
| Final (RF top-80) | 80 |

## Selected Feature Groups

| Group | Count |
|-------|-------|
| domain | 1 |
| publisher | 5 |
| structural | 10 |
| TF-IDF | 64 |

## Top 20 Selected Features

| Rank | Feature | Importance | Group |
|------|---------|------------|-------|
| 1 | `pub_ieee` | 37.8445 | Publisher |
| 2 | `pub_mdpi` | 34.8944 | Publisher |
| 3 | `title_to_abstract_ratio` | 22.2110 | Structural |
| 4 | `text_richness` | 20.5306 | Structural |
| 5 | `title_length` | 19.6290 | Structural |
| 6 | `abstract_length` | 19.2832 | Structural |
| 7 | `abstract_word_count` | 18.1713 | Structural |
| 8 | `title_word_count` | 12.9165 | Structural |
| 9 | `concept_count` | 12.6396 | Structural |
| 10 | `tf_abstract` | 12.4648 | TF-IDF |
| 11 | `keyword_count` | 12.1241 | Structural |
| 12 | `tf_data` | 11.0038 | TF-IDF |
| 13 | `tf_based` | 10.5688 | TF-IDF |
| 14 | `tf_analysis` | 10.2160 | TF-IDF |
| 15 | `tf_model` | 9.9901 | TF-IDF |
| 16 | `tf_learning` | 9.7189 | TF-IDF |
| 17 | `tf_paper` | 9.5043 | TF-IDF |
| 18 | `pub_springer` | 9.4033 | Publisher |
| 19 | `tf_research` | 8.9675 | TF-IDF |
| 20 | `tf_artificial` | 8.9049 | TF-IDF |

## Why These Features Work

The publisher DOI prefix features (pub_ieee, pub_mdpi, pub_elsevier etc.)
are the most discriminative because OA status is largely determined by
journal/publisher policy, which is encoded in the DOI prefix.
Domain features (dom_medical, dom_education) capture field-level OA
mandates (e.g. NIH mandate for medical research, open pedagogy in education).
TF-IDF terms capture vocabulary patterns correlated with specific venues.

*Generated: 2026-09-22 18:56:21.25983*
