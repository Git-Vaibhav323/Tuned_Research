-- Q5: Citation and text-feature summary by OA category
-- Purpose: Exploratory signal check before Phase 2 feature engineering (M2).

SELECT
    oa_category,
    COUNT(*) AS n,
    ROUND(AVG(cited_by_count), 1) AS avg_citations,
    ROUND(AVG(citation_per_year), 1) AS avg_cite_per_year,
    ROUND(AVG(title_length), 1) AS avg_title_len,
    ROUND(AVG(abstract_length), 1) AS avg_abstract_len,
    ROUND(AVG(keyword_count), 2) AS avg_keywords,
    ROUND(AVG(concept_count), 2) AS avg_concepts,
    ROUND(AVG(has_fulltext), 3) AS frac_fulltext
FROM papers
GROUP BY oa_category
ORDER BY oa_category;
